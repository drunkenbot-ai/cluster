"""Distributed Cluster Worker Daemon for Local SGD Training.

Runs as a lightweight headless service on worker compute nodes:
1. Discovers local hardware (GPU, VRAM).
2. Registers with the central storage bus and emits periodic heartbeats.
3. Claims a data shard and executes K steps of local training.
4. Atomically deposits weight checkpoints to central storage.
5. Synchronizes from the coordinator's global averaged model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import socket
import threading
import time
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .bus import ClusterStorageBus
from .sharding import ShardedTokenDataset


def get_hardware_info(device_preference: Optional[str] = None) -> tuple[str, str, float]:
    """Detect available compute device, GPU model, and VRAM in GB.

    Args:
        device_preference: Optional explicit device string (e.g. 'cuda:0', 'cpu').

    Returns:
        (device_str, gpu_name, vram_gb)
    """
    if device_preference:
        device_str = device_preference
    elif torch.cuda.is_available():
        device_str = "cuda:0"
    else:
        device_str = "cpu"

    if device_str.startswith("cuda") and torch.cuda.is_available():
        device_idx = 0
        if ":" in device_str:
            try:
                device_idx = int(device_str.split(":")[1])
            except ValueError:
                device_idx = 0
        gpu_name = torch.cuda.get_device_name(device_idx)
        props = torch.cuda.get_device_properties(device_idx)
        vram_gb = round(props.total_memory / (1024 ** 3), 2)
    else:
        gpu_name = platform.processor() or "CPU"
        vram_gb = 0.0

    return device_str, gpu_name, vram_gb


def build_model_from_config(model_config: dict[str, Any], device: str) -> nn.Module:
    """Instantiate a transformer model from configuration.

    Attempts import from `engine.model_transformer.TransformerModel` first.
    Falls back to a standard PyTorch TransformerLM if engine is not installed.
    """
    try:
        from engine.config import ModelConfig
        from engine.model import MicroGPT

        # Filter config keys recognized by ModelConfig
        valid_keys = ModelConfig.__dataclass_fields__.keys()
        filtered = {k: v for k, v in model_config.items() if k in valid_keys}
        cfg = ModelConfig(**filtered)
        model = MicroGPT(cfg)
        return model.to(device)
    except (ImportError, Exception):
        # Fallback minimal transformer language model
        vocab_size = int(model_config.get("vocab_size", 1000))
        embed_dim = int(model_config.get("embedding_size", 128))
        num_heads = int(model_config.get("head_count", 4))
        num_layers = int(model_config.get("layer_count", 2))
        dropout = float(model_config.get("dropout", 0.1))

        class FallbackLM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.tok_embed = nn.Embedding(vocab_size, embed_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, s = x.shape
                # Causal mask
                mask = torch.triu(torch.full((s, s), float("-inf"), device=x.device), diagonal=1)
                h = self.tok_embed(x)
                out = self.transformer(h, mask=mask, is_causal=True)
                return self.lm_head(out)

        return FallbackLM().to(device)


class ClusterWorker:
    """Worker daemon executing distributed Local SGD rounds on a worker node."""

    def __init__(
        self,
        bus: ClusterStorageBus,
        worker_id: Optional[str] = None,
        device: Optional[str] = None,
        heartbeat_interval: float = 5.0,
    ) -> None:
        """Initialize worker daemon.

        Args:
            bus: Shared storage bus instance.
            worker_id: Unique worker node identifier (defaults to hostname-pid).
            device: Compute device ('cuda:0', 'cpu', etc.).
            heartbeat_interval: Heartbeat interval in seconds.
        """
        self.bus = bus
        self.hostname = socket.gethostname()
        self.worker_id = worker_id or f"{self.hostname}_{os.getpid()}"
        self.device_str, self.gpu_name, self.vram_gb = get_hardware_info(device)
        self.heartbeat_interval = heartbeat_interval
        self._stop_event = threading.Event()
        self._current_job_id: Optional[str] = None
        self._current_status: str = "IDLE"

        # Register worker in SQLite database
        self.bus.register_worker(
            worker_id=self.worker_id,
            hostname=self.hostname,
            gpu_name=self.gpu_name,
            vram_gb=self.vram_gb,
        )

        # Start background heartbeat daemon thread
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._hb_thread.start()

    def log(self, message: str, level: str = "INFO") -> None:
        """Write diagnostic log to stdout and central SQLite worker_logs table."""
        timestamp_str = time.strftime("%H:%M:%S")
        print(f"[{timestamp_str}] [{level}] [Worker {self.worker_id}] {message}", flush=True)
        try:
            self.bus.write_worker_log(self.worker_id, message, level=level)
        except Exception:
            pass

    def _heartbeat_loop(self) -> None:
        """Background thread updating heartbeat periodically."""
        while not self._stop_event.is_set():
            try:
                self.bus.heartbeat(
                    worker_id=self.worker_id,
                    status=self._current_status,
                    current_job_id=self._current_job_id,
                )
            except Exception:
                pass
            self._stop_event.wait(self.heartbeat_interval)

    def stop(self) -> None:
        """Signal worker to gracefully stop."""
        self._stop_event.set()
        if self._hb_thread.is_alive():
            self._hb_thread.join(timeout=2.0)
        try:
            self.bus.heartbeat(self.worker_id, status="OFFLINE", current_job_id=None)
        except Exception:
            pass

    def run_training_round(
        self,
        job_id: str,
        round_num: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        dataloader_iter: Any,
        steps_per_round: int,
        job: Optional[dict[str, Any]] = None,
        log_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> tuple[float, Any]:
        """Execute K steps of local SGD training for a single round.

        Args:
            job_id: Distributed job ID.
            round_num: Current synchronization round.
            model: PyTorch model module.
            optimizer: Optimizer instance.
            dataloader: Sharded token data loader.
            dataloader_iter: Active DataLoader iterator.
            steps_per_round: Local training steps before sync.
            job: Optional full job configuration dict.
            log_callback: Optional progress reporter.

        Returns:
            (average_loss, dataloader_iter)
        """
        model.train()
        total_loss = 0.0
        steps_done = 0
        tokens_processed = 0
        start_time = time.time()

        # Mixed precision (AMP) configuration
        is_cuda = self.device_str.startswith("cuda") and torch.cuda.is_available()
        training_cfg = job.get("training_config", {}) if isinstance(job, dict) else {}
        precision = str(training_cfg.get("precision", "fp16")).lower()
        use_amp = bool(training_cfg.get("use_amp", True))
        amp_enabled = is_cuda and use_amp and (precision in ("fp16", "bf16"))
        amp_dtype = torch.bfloat16 if (precision == "bf16" and torch.cuda.is_bf16_supported()) else torch.float16

        scaler = None
        if amp_enabled and amp_dtype == torch.float16:
            if not hasattr(self, "_scaler") or self._scaler is None:
                self._scaler = torch.amp.GradScaler('cuda')
            scaler = self._scaler

        for step in range(steps_per_round):
            if self._stop_event.is_set() or self.bus.is_stopped(job_id):
                break

            # Handle cooperative pause
            while self.bus.is_paused(job_id) and not self._stop_event.is_set():
                self._current_status = "PAUSED"
                time.sleep(1.0)
                if self.bus.is_stopped(job_id):
                    break

            self._current_status = "TRAINING"

            # Fetch next batch (looping iterator if exhausted)
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            x, y = batch
            tokens_processed += int(x.numel())
            if hasattr(model, "token_embedding") and hasattr(model.token_embedding, "num_embeddings"):
                n_emb = model.token_embedding.num_embeddings
                x = torch.clamp(x, 0, n_emb - 1)
                y = torch.clamp(y, 0, n_emb - 1)
            x = x.to(self.device_str, non_blocking=True)
            y = y.to(self.device_str, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_val = float(loss.item())
            total_loss += loss_val
            steps_done += 1

            if log_callback:
                log_callback({
                    "job_id": job_id,
                    "round": round_num,
                    "step": step + 1,
                    "steps_per_round": steps_per_round,
                    "loss": loss_val,
                })

        elapsed = max(time.time() - start_time, 1e-4)
        tokens_per_sec = tokens_processed / elapsed
        avg_loss = (total_loss / steps_done) if steps_done > 0 else 0.0

        self._last_round_metrics = {
            "avg_loss": avg_loss,
            "tokens_processed": tokens_processed,
            "tokens_per_sec": tokens_per_sec,
        }
        return avg_loss, dataloader_iter

    def execute_job(
        self,
        job: dict[str, Any],
        poll_interval: float = 2.0,
        log_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> bool:
        """Execute a full distributed training job across rounds.

        Args:
            job: Job metadata dictionary from database.
            poll_interval: Polling frequency during synchronization waits.
            log_callback: Optional status and telemetry logger.

        Returns:
            True if completed successfully, False if aborted or stopped.
        """
        job_id = job["job_id"]
        self._current_job_id = job_id
        self._current_status = "PREPARING"
        self.log(f">>> Claimed job: {job_id}")

        model_config = job.get("model_config", {})
        training_config = job.get("training_config", {})
        dataset_path = job.get("dataset_path", "")
        max_rounds = int(job.get("max_rounds", 10))
        sync_interval_steps = int(job.get("sync_interval_steps", 250))
        batch_size = int(training_config.get("batch_size", 4))
        lr = float(training_config.get("learning_rate", 3e-4))
        context_length = int(model_config.get("context_length", 512))

        model = None
        optimizer = None
        dataloader = None
        dataloader_iter = None
        dataset = None
        token_array = None

        try:
            try:
                # Claim data shard slot
                shard_idx, total_shards = self.bus.claim_job_slot(job_id, self.worker_id)
                self.log(f"Assigned data shard slot {shard_idx + 1} of {total_shards} total nodes")

                # Prepare dataset
                if not os.path.exists(dataset_path):
                    err = f"Dataset path not found on shared storage: {dataset_path}"
                    self.log(err, level="ERROR")
                    self._current_status = "ERROR"
                    return False

                token_array = np.load(dataset_path, mmap_mode="r")
                vocab_size = int(model_config.get("vocab_size", 0) or 0)

                # 1. Inspect metadata files if present on shared storage or dataset dir
                detected_vocab = 0
                for sdir in [Path(dataset_path).parent, self.bus.shared_dir]:
                    summary_file = sdir / "dataset_summary.json"
                    if summary_file.exists():
                        try:
                            import json
                            meta = json.loads(summary_file.read_text(encoding="utf-8"))
                            detected_vocab = int(meta.get("tokenizer_vocab_size", 0) or 0)
                            if detected_vocab > 0:
                                break
                        except Exception:
                            pass
                    tok_file = sdir / "tokenizer.json"
                    if tok_file.exists() and detected_vocab <= 0:
                        try:
                            import json
                            tok_data = json.loads(tok_file.read_text(encoding="utf-8"))
                            vocab_dict = tok_data.get("model", {}).get("vocab", {})
                            if vocab_dict:
                                detected_vocab = len(vocab_dict)
                                break
                        except Exception:
                            pass

                # 2. Inspect dataset tokens for true maximum token ID
                max_token_id = 0
                try:
                    arr_len = len(token_array)
                    if arr_len <= 10_000_000:
                        max_token_id = int(np.max(token_array))
                    else:
                        slices = [
                            token_array[:200000],
                            token_array[arr_len // 4 : (arr_len // 4) + 200000],
                            token_array[arr_len // 2 : (arr_len // 2) + 200000],
                            token_array[(3 * arr_len) // 4 : ((3 * arr_len) // 4) + 200000],
                            token_array[-200000:],
                        ]
                        max_token_id = max(int(np.max(s)) for s in slices if len(s) > 0)
                except Exception:
                    max_token_id = 0

                # 3. Determine safe aligned vocab_size
                min_safe_vocab = 256
                if max_token_id > 0:
                    min_safe_vocab = ((max_token_id + 1 + 255) // 256) * 256
                    if 31000 <= max_token_id < 32000:
                        min_safe_vocab = max(min_safe_vocab, 32000)

                target_vocab = max(vocab_size, detected_vocab, min_safe_vocab)
                if vocab_size < target_vocab:
                    self.log(
                        f"Model vocab_size ({vocab_size}) is smaller than required ({target_vocab}, detected: {detected_vocab}, max token: {max_token_id}). "
                        f"Auto-adjusting vocab_size to {target_vocab}.",
                        level="WARNING",
                    )
                    vocab_size = target_vocab
                    model_config["vocab_size"] = vocab_size

                dataset = ShardedTokenDataset(
                    token_array=token_array,
                    context_length=context_length,
                    shard_index=shard_idx,
                    total_shards=total_shards,
                    vocab_size=vocab_size,
                )
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                dataloader_iter = iter(dataloader)

                # Build model and optimizer
                model = build_model_from_config(model_config, self.device_str)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                self.log(f"Model initialized on device '{self.device_str}'. Ready to train {max_rounds} rounds.")
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                self.log(f"Preparation failed for job {job_id}:\n{tb}", level="ERROR")
                self._current_status = "ERROR"
                return False

            current_round = int(job.get("current_round", 0))

            while current_round < max_rounds and not self._stop_event.is_set():
                if self.bus.is_stopped(job_id):
                    self._current_status = "STOPPED"
                    self.log(f"Job {job_id} stopped.")
                    return False

                # If round > 0, wait for and synchronize from previous global averaged model
                if current_round > 0:
                    self._current_status = "SYNC_WAIT"
                    while not self.bus.is_global_weights_ready(job_id, current_round - 1):
                        if self._stop_event.is_set() or self.bus.is_stopped(job_id):
                            return False
                        time.sleep(poll_interval)

                    global_state = self.bus.load_global_weights(
                        job_id, current_round - 1, device=self.device_str
                    )
                    model.load_state_dict(global_state)

                # Train locally for K steps
                self._current_status = "TRAINING"
                self.log(f"Starting training round {current_round + 1}/{max_rounds} ({sync_interval_steps} local steps)...")
                try:
                    avg_loss, dataloader_iter = self.run_training_round(
                        job_id=job_id,
                        round_num=current_round,
                        model=model,
                        optimizer=optimizer,
                        dataloader=dataloader,
                        dataloader_iter=dataloader_iter,
                        steps_per_round=sync_interval_steps,
                        job=job,
                        log_callback=log_callback,
                    )
                    round_metrics = getattr(self, "_last_round_metrics", {})
                    tokens_per_sec = round_metrics.get("tokens_per_sec", 0.0)
                    tokens_processed = round_metrics.get("tokens_processed", 0)

                    self.log(
                        f"Round {current_round + 1}/{max_rounds} completed. "
                        f"Avg loss: {avg_loss:.4f}, Speed: {tokens_per_sec:,.0f} tok/s. Depositing weights & telemetry..."
                    )
                except Exception as exc:
                    import traceback
                    tb = traceback.format_exc()
                    self.log(f"Training round {current_round} crashed:\n{tb}", level="ERROR")
                    self._current_status = "ERROR"
                    return False

                # Atomically deposit local worker weights
                self.bus.save_worker_weights(
                    job_id=job_id,
                    round_num=current_round,
                    worker_id=self.worker_id,
                    state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                )

                # Atomically deposit local worker telemetry
                self.bus.save_worker_telemetry(
                    job_id=job_id,
                    round_num=current_round,
                    worker_id=self.worker_id,
                    telemetry={
                        "worker_id": self.worker_id,
                        "round": current_round,
                        "steps_completed": sync_interval_steps,
                        "avg_loss": round(avg_loss, 4),
                        "tokens_processed": tokens_processed,
                        "tokens_per_sec": round(tokens_per_sec, 1),
                        "timestamp": time.time(),
                    },
                )

                self._current_status = "SYNC_WAIT"

                # Wait for coordinator to publish global model for this round
                while not self.bus.is_global_weights_ready(job_id, current_round):
                    if self._stop_event.is_set() or self.bus.is_stopped(job_id):
                        return False
                    time.sleep(poll_interval)

                current_round += 1

            self._current_status = "COMPLETED"
            self._current_job_id = None
            self.log(f"Job {job_id} successfully completed all {max_rounds} rounds.")
            return True
        finally:
            try:
                if model is not None:
                    model.to("cpu")
            except Exception:
                pass

            model = None
            optimizer = None
            dataloader = None
            dataloader_iter = None
            dataset = None
            token_array = None

            import gc
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    def run_daemon(self, poll_interval: float = 3.0) -> None:
        """Run persistent background loop polling for jobs and executing them."""
        from .cluster_worker import acquire_singleton_lock, release_singleton_lock, get_device_tag

        device_tag = get_device_tag(self.device_str)
        if not acquire_singleton_lock(device_tag):
            self.log(f"A worker is already running for device '{self.device_str}' on this machine. Exiting.", level="WARNING")
            return

        wid_tag = f"wid_{self.worker_id}"
        if not acquire_singleton_lock(wid_tag):
            release_singleton_lock(device_tag)
            self.log(f"A worker is already running for worker ID '{self.worker_id}' on this machine. Exiting.", level="WARNING")
            return

        self.log(f"Worker daemon started. Listening for jobs on shared drive...")
        try:
            while not self._stop_event.is_set():
                self._current_status = "IDLE"
                self._current_job_id = None

                try:
                    active_job = self.bus.get_active_job()
                    if active_job and active_job.get("status") in {"RUNNING", "QUEUED"}:
                        self.execute_job(active_job, poll_interval=poll_interval)
                except Exception as exc:
                    import traceback
                    tb = traceback.format_exc()
                    self.log(f"Error in worker daemon:\n{tb}", level="ERROR")
                    time.sleep(poll_interval * 2)

                self._stop_event.wait(poll_interval)
        finally:
            release_singleton_lock(device_tag)
            release_singleton_lock(wid_tag)
            self._current_status = "OFFLINE"
            self._current_job_id = None
            try:
                self.bus.heartbeat(self.worker_id, status="OFFLINE", current_job_id=None)
            except Exception:
                pass
