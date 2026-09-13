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
import subprocess
import sys
import shutil
import tempfile
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


def is_network_path(path_str: str) -> bool:
    """Determine if a file path resides on a remote network share (SMB/NFS/UNC)."""
    p = str(path_str).strip()
    if p.startswith(("\\\\", "//")):
        return True
    if sys.platform == "win32" and len(p) >= 2 and p[1] == ":":
        try:
            import ctypes
            drive_root = p[:2] + "\\"
            # DRIVE_REMOTE = 4 in Windows API
            return ctypes.windll.kernel32.GetDriveTypeW(drive_root) == 4
        except Exception:
            pass
    return False


def purge_local_dataset_cache(
    cache_dir: Optional[Path] = None,
    keep_files: Optional[set[str]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> int:
    """Remove/delete stale .npy files from local cache directory to keep local storage clean.

    Args:
        cache_dir: Optional cache directory path. Defaults to LOCALAPPDATA/llm_cluster_cache.
        keep_files: Optional set of filenames to preserve (e.g. active train/val tokens).
        log_fn: Optional logger callable.

    Returns:
        Number of .npy files removed.
    """
    log = log_fn or (lambda msg, **kw: print(f"[DatasetCache] {msg}"))
    if cache_dir is None:
        cache_dir = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "llm_cluster_cache"
    if not cache_dir.exists():
        return 0

    keep = keep_files or set()
    log(f"[DatasetCache] Cleaning local SSD cache directory ({cache_dir})...")
    purged_count = 0
    for f in cache_dir.glob("*.npy"):
        if f.name in keep:
            continue
        try:
            f.unlink(missing_ok=True)
            log(f"[DatasetCache] Removed old cache file: {f.name}")
            purged_count += 1
        except Exception as exc:
            log(f"[DatasetCache] Notice: could not remove {f.name}: {exc}")

    if purged_count > 0:
        log(f"[DatasetCache] Purged {purged_count} stale .npy file(s) before caching new dataset.")
    else:
        log(f"[DatasetCache] Local cache is clean (0 stale .npy files found).")
    return purged_count


def cache_dataset_to_local(
    remote_path: str,
    log_fn: Optional[Callable[[str], None]] = None,
    keep_files: Optional[set[str]] = None,
    progress_callback: Optional[Callable[[float, float, float], None]] = None,
) -> str:
    """Safely cache remote/network dataset .npy file to fast local SSD storage.

    Purges stale .npy files before copying new files to keep local disk clean.
    Prevents unrecoverable Windows STATUS_IN_PAGE_ERROR (0xC0000006) on network shares.

    Args:
        remote_path: Path to dataset .npy on network share (SMB/NFS).
        log_fn: Optional logger callable.
        keep_files: Optional set of filenames to preserve during cache cleanup.
        progress_callback: Optional callback receiving (pct, speed_mb_s, eta_seconds).

    Returns:
        Local path to the cached .npy file (or original remote path if caching is skipped/impossible).
    """
    log = log_fn or (lambda msg, **kw: print(f"[DatasetCache] {msg}"))

    if not remote_path or not os.path.exists(remote_path):
        return remote_path

    # If it's already on a local non-network drive, no caching needed
    if not is_network_path(remote_path):
        return remote_path

    cache_dir = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "llm_cluster_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        log(f"[DatasetCache] Could not create local cache directory {cache_dir}: {exc}", level="WARNING")
        return remote_path

    remote_file = Path(remote_path)
    local_file = cache_dir / remote_file.name
    lock_file = cache_dir / f".{remote_file.name}.lock"
    temp_file = cache_dir / f".{remote_file.name}.tmp_{os.getpid()}"

    try:
        remote_stat = remote_file.stat()
        remote_size = remote_stat.st_size
        remote_size_gb = remote_size / (1024 ** 3)
        remote_mtime = remote_stat.st_mtime

        # Fast path: check if local file is already fully cached and valid
        if local_file.exists():
            try:
                local_stat = local_file.stat()
                if local_stat.st_size == remote_size and abs(local_stat.st_mtime - remote_mtime) < 2.0:
                    log(f"[DatasetCache] Verified existing local SSD dataset cache: {local_file.name} ({remote_size_gb:.2f} GB)")
                    return str(local_file)
            except Exception:
                pass

        # Multi-process coordination: if another slot process on this node is currently caching, wait for it
        if lock_file.exists():
            try:
                lock_age = time.time() - lock_file.stat().st_mtime
                if lock_age < 600:
                    log(f"[DatasetCache] Another slot on this node is currently caching {remote_file.name}. Waiting for local transfer to finish...")
                    wait_start = time.time()
                    while lock_file.exists() and (time.time() - wait_start < 180):
                        time.sleep(1.0)
                        if local_file.exists():
                            try:
                                if local_file.stat().st_size == remote_size:
                                    log(f"[DatasetCache] Local SSD dataset ready from concurrent slot: {local_file.name} ({remote_size_gb:.2f} GB)")
                                    return str(local_file)
                            except Exception:
                                pass
            except Exception:
                pass

        # Check again if file became available while checking/waiting
        if local_file.exists():
            try:
                if local_file.stat().st_size == remote_size:
                    return str(local_file)
            except Exception:
                pass

        # Claim the lock for this process
        try:
            lock_file.write_text(str(os.getpid()), encoding="utf-8")
        except Exception:
            pass

        # Purge stale .npy files from cache before copying new to keep local storage clean
        effective_keep = set(keep_files or ())
        effective_keep.add(remote_file.name)
        purge_local_dataset_cache(cache_dir=cache_dir, keep_files=effective_keep, log_fn=log)

        # Check free disk space on local cache drive
        free_space = shutil.disk_usage(str(cache_dir)).free
        free_gb = free_space / (1024 ** 3)
        if free_space < remote_size * 1.15:
            log(
                f"[DatasetCache] Insufficient local disk space to cache dataset "
                f"({free_gb:.1f} GB free vs {remote_size_gb:.1f} GB required). Falling back to direct network share.",
                level="WARNING",
            )
            try:
                if lock_file.exists():
                    lock_file.unlink(missing_ok=True)
            except Exception:
                pass
            return remote_path

        log(f"[DatasetCache] Streaming network dataset {remote_file.name} ({remote_size_gb:.2f} GB) to local SSD ({local_file})...")
        t0 = time.time()

        # Safe 2MB buffer chunks matching native Windows SMB buffers with live transfer progress
        buf_size = 2 * 1024 * 1024
        transferred = 0
        last_log_time = time.time()
        try:
            with open(str(remote_file), "rb") as src, open(str(temp_file), "wb") as dst:
                with memoryview(bytearray(buf_size)) as mv:
                    while True:
                        n = src.readinto(mv)
                        if not n:
                            break
                        dst.write(mv[:n])
                        transferred += n
                        now = time.time()
                        if now - last_log_time >= 4.0:
                            pct = (transferred / max(remote_size, 1)) * 100.0
                            elapsed_cur = max(now - t0, 0.001)
                            speed_cur = (transferred / (1024 * 1024)) / elapsed_cur
                            rem_bytes = max(remote_size - transferred, 0)
                            eta_s = rem_bytes / max(speed_cur * 1024 * 1024, 1)
                            log(
                                f"[DatasetCache] Caching {remote_file.name}: "
                                f"{transferred / (1024**3):.2f}/{remote_size_gb:.2f} GB ({pct:.1f}%) "
                                f"@ {speed_cur:.1f} MB/s (ETA: {eta_s:.0f}s)"
                            )
                            if progress_callback:
                                try:
                                    progress_callback(pct, speed_cur, eta_s)
                                except Exception:
                                    pass
                            last_log_time = now
        except OSError as os_err:
            log(f"[DatasetCache] Chunked stream encountered {os_err}, attempting fallback via shutil.copyfile...", level="WARNING")
            shutil.copyfile(str(remote_file), str(temp_file))

        # Atomically move temp_file into local_file
        temp_file.replace(local_file)
        try:
            os.utime(str(local_file), (remote_mtime, remote_mtime))
        except Exception:
            pass

        try:
            if lock_file.exists():
                lock_file.unlink(missing_ok=True)
        except Exception:
            pass

        elapsed = max(time.time() - t0, 0.001)
        speed_mb_s = (remote_size / (1024 * 1024)) / elapsed
        log(f"[DatasetCache] Cache transfer completed in {elapsed:.1f}s ({speed_mb_s:.1f} MB/s). Ready for zero-latency local training.")
        return str(local_file)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        log(f"[DatasetCache] Failed to cache dataset to local SSD ({exc}). Falling back to network share.\nTraceback:\n{tb}", level="WARNING")
        try:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            if lock_file.exists():
                lock_file.unlink(missing_ok=True)
        except Exception:
            pass
        return remote_path


def calculate_optimal_worker_slots(
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    device_str: str,
) -> tuple[int, float, float]:
    """Calculate optimal number of worker processes to spawn on this node based on VRAM capacity.

    Args:
        model_cfg: Model architecture parameters.
        training_cfg: Training parameters (batch size, precision).
        device_str: Device string (e.g. 'cuda:0', 'cpu').

    Returns:
        (optimal_slots, estimated_job_gb, node_vram_gb)
    """
    if not device_str.startswith("cuda") or not torch.cuda.is_available():
        return 1, 0.0, 0.0

    try:
        dev_idx = 0
        if ":" in device_str:
            try:
                dev_idx = int(device_str.split(":")[1])
            except ValueError:
                dev_idx = 0
        total_vram_bytes = torch.cuda.get_device_properties(dev_idx).total_memory
        node_vram_gb = total_vram_bytes / (1024 ** 3)
    except Exception:
        return 1, 0.0, 0.0

    explicit_gb = float(training_cfg.get("vram_required_gb") or model_cfg.get("vram_required_gb") or 0.0)
    if explicit_gb > 0:
        estimated_job_gb = explicit_gb
    else:
        vocab_size = int(model_cfg.get("vocab_size") or 32000)
        emb_size = int(model_cfg.get("embedding_size") or 768)
        n_layers = int(model_cfg.get("layer_count") or 12)
        ctx_len = int(model_cfg.get("context_length") or 1024)
        batch_size = int(training_cfg.get("batch_size") or 2)

        # Estimate parameter count: token embeddings + transformer layers + head
        approx_params = (vocab_size * emb_size * 2) + (n_layers * 12 * emb_size * emb_size)
        prec = str(training_cfg.get("precision", "float16")).lower()
        bytes_per_param = 16 if "16" in prec else 24
        param_bytes = approx_params * bytes_per_param
        activation_bytes = batch_size * ctx_len * emb_size * n_layers * 16
        cuda_overhead = 1024 ** 3

        total_job_bytes = param_bytes + activation_bytes + cuda_overhead
        # In practice, each active worker requires at least 6.5 GB VRAM for model weights,
        # fp32 AdamW moments, master weights, gradients, PyTorch CUDA context, and activation cache.
        estimated_job_gb = max(round(total_job_bytes / (1024 ** 3), 2), 6.5)

    # Calculate slots: allow 1.10x safety headroom
    slots = max(1, min(4, int(node_vram_gb // (estimated_job_gb * 1.10))))
    return slots, estimated_job_gb, node_vram_gb


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
        # Active preflight test: verify that CUDA kernels can actually execute
        try:
            test_t = torch.zeros(1, device=device_str)
            del test_t
        except Exception as exc:
            print(f"[Worker] WARNING: CUDA device '{device_str}' ({gpu_name}) failed kernel execution preflight test: {exc}")
    else:
        gpu_name = platform.processor() or "CPU"
        vram_gb = 0.0

    return device_str, gpu_name, vram_gb


def collect_system_metrics(device_str: str = "cpu", total_vram_gb: float = 0.0) -> dict[str, Any]:
    """Collect current real-time CPU, RAM, and GPU VRAM usage."""
    metrics: dict[str, Any] = {
        "cpu_percent": 0.0,
        "ram_used_gb": 0.0,
        "ram_total_gb": 0.0,
        "vram_used_gb": 0.0,
        "vram_total_gb": float(total_vram_gb or 0.0),
    }

    # 1. RAM & CPU
    try:
        import psutil
        metrics["cpu_percent"] = round(float(psutil.cpu_percent(interval=None)), 1)
        vm = psutil.virtual_memory()
        metrics["ram_used_gb"] = round(float(vm.used) / (1024 ** 3), 1)
        metrics["ram_total_gb"] = round(float(vm.total) / (1024 ** 3), 1)
    except Exception:
        if sys.platform == "win32":
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                tot = float(stat.ullTotalPhys) / (1024 ** 3)
                avail = float(stat.ullAvailPhys) / (1024 ** 3)
                metrics["ram_total_gb"] = round(tot, 1)
                metrics["ram_used_gb"] = round(tot - avail, 1)
                metrics["cpu_percent"] = float(stat.dwMemoryLoad)
            except Exception:
                pass

    # 2. VRAM
    try:
        if device_str.startswith("cuda") and torch.cuda.is_available():
            dev_idx = 0
            if ":" in device_str:
                try:
                    dev_idx = int(device_str.split(":")[1])
                except ValueError:
                    dev_idx = 0
            free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
            used_bytes = max(total_bytes - free_bytes, 0)
            metrics["vram_used_gb"] = round(float(used_bytes) / (1024 ** 3), 2)
            metrics["vram_total_gb"] = round(float(total_bytes) / (1024 ** 3), 2)
    except Exception:
        pass

    return metrics


def build_model_from_config(
    model_config: dict[str, Any],
    device: str,
    logger: Optional[Callable[[str], None]] = None,
) -> nn.Module:
    """Instantiate a transformer model from configuration with detailed debug tracking.

    Attempts import from `engine.model_transformer.TransformerModel` first.
    Falls back to a standard PyTorch TransformerLM if engine is not installed.
    """
    log = logger or (lambda msg: print(f"[ModelBuilder] {msg}"))
    try:
        log(f"Importing engine components (ModelConfig, MicroGPT)...")
        from engine.config import ModelConfig
        from engine.model import MicroGPT

        # Filter config keys recognized by ModelConfig
        valid_keys = ModelConfig.__dataclass_fields__.keys()
        filtered = {k: v for k, v in model_config.items() if k in valid_keys}
        cfg = ModelConfig(**filtered)
        log(f"Instantiating MicroGPT on CPU (layers={cfg.layer_count}, heads={cfg.head_count}, embd={cfg.embedding_size}, vocab={cfg.vocab_size})...")
        t0 = time.time()
        model = MicroGPT(cfg)
        log(f"MicroGPT constructed on CPU in {time.time() - t0:.2f}s. Transferring parameters to {device}...")
        t1 = time.time()
        model = model.to(device)
        log(f"Model parameters successfully transferred to {device} in {time.time() - t1:.2f}s.")
        return model
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        log(f"MicroGPT initialization failed ({exc}). Falling back to minimal TransformerLM. Traceback:\n{tb}")

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

        log(f"Constructing FallbackLM on {device}...")
        return FallbackLM().to(device)


# -----------------------------------------------------------------------------
# Parameter-Efficient Fine-Tuning (LoRA) Support
# -----------------------------------------------------------------------------

try:
    from engine.model_norm_lora import (
        LoRALinear,
        apply_lora_adapters,
        freeze_non_lora_parameters,
        lora_state_dict,
        merged_lora_state_dict,
    )
except ImportError:
    import math

    class LoRALinear(nn.Module):
        """Linear layer with trainable low-rank LoRA adapters."""
        def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
            super().__init__()
            self.base = base
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            self.dropout = nn.Dropout(dropout)
            self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)
            self.base.weight.requires_grad_(False)
            if self.base.bias is not None:
                self.base.bias.requires_grad_(False)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            update = F.linear(F.linear(self.dropout(value), self.lora_a), self.lora_b) * self.scaling
            return self.base(value) + update

    def _set_nested_module(root: nn.Module, module_name: str, module: nn.Module) -> None:
        parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
        parent = root.get_submodule(parent_name) if parent_name else root
        setattr(parent, child_name, module)

    def _lora_target_names(model: nn.Module, target_modules: str) -> set[str]:
        groups = {part.strip().lower() for part in target_modules.split(",") if part.strip()}
        if "all" in groups:
            groups.update({"attention", "mlp"})
        names: set[str] = set()
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name.endswith("lm_head"):
                continue
            is_attention = ".attn." in name or "attn" in name or "attention" in name
            is_mlp = ".mlp." in name or "mlp" in name
            if ("attention" in groups and is_attention) or ("mlp" in groups and is_mlp):
                names.add(name)
        return names

    def apply_lora_adapters(model: nn.Module, rank: int, alpha: float, dropout: float, target_modules: str) -> int:
        names = _lora_target_names(model, target_modules)
        for name in sorted(names):
            try:
                module = model.get_submodule(name)
                if isinstance(module, nn.Linear):
                    _set_nested_module(model, name, LoRALinear(module, rank, alpha, dropout))
            except Exception:
                pass
        return len(names)

    def freeze_non_lora_parameters(model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(("lora_a" in name) or ("lora_b" in name))

    def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
            if ".lora_a" in name or ".lora_b" in name
        }

    def merged_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
        replacements: dict[str, tuple[str, torch.Tensor]] = {}
        adapter_keys: set[str] = set()
        for module_name, module in model.named_modules():
            if not isinstance(module, LoRALinear):
                continue
            prefix = f"{module_name}."
            replacements[f"{prefix}base.weight"] = (
                f"{prefix}weight",
                module.base.weight.detach() + (module.lora_b.detach() @ module.lora_a.detach()) * module.scaling,
            )
            if module.base.bias is not None:
                replacements[f"{prefix}base.bias"] = (
                    f"{prefix}bias",
                    module.base.bias.detach(),
                )
            adapter_keys.update({f"{prefix}lora_a", f"{prefix}lora_b"})
        merged: dict[str, torch.Tensor] = {}
        for name, tensor in model.state_dict().items():
            replacement = replacements.get(name)
            if replacement is not None:
                target_name, value = replacement
                merged[target_name] = value.cpu()
            elif name not in adapter_keys:
                merged[name] = tensor.detach().cpu()
        return merged


class ClusterWorker:
    """Worker daemon executing distributed Local SGD rounds on a worker node."""

    def __init__(
        self,
        bus: ClusterStorageBus,
        worker_id: Optional[str] = None,
        device: Optional[str] = None,
        heartbeat_interval: float = 5.0,
        ephemeral_job_id: Optional[str] = None,
        allow_shared_device: bool = False,
    ) -> None:
        """Initialize worker daemon.

        Args:
            bus: Shared storage bus instance.
            worker_id: Unique worker node identifier (defaults to hostname-pid).
            device: Compute device ('cuda:0', 'cpu', etc.).
            heartbeat_interval: Heartbeat interval in seconds.
            ephemeral_job_id: If set, worker exits automatically after completing this job.
            allow_shared_device: If True, bypasses per-device singleton lock to permit auxiliary slots.
        """
        self.bus = bus
        self.hostname = socket.gethostname()
        self.worker_id = worker_id or f"{self.hostname}_{os.getpid()}"
        self.device_str, self.gpu_name, self.vram_gb = get_hardware_info(device)
        self.heartbeat_interval = heartbeat_interval
        self.ephemeral_job_id = ephemeral_job_id
        self.allow_shared_device = allow_shared_device
        self._child_worker_procs: list[tuple[subprocess.Popen, str]] = []
        self._stop_event = threading.Event()
        self._current_job_id: Optional[str] = None
        self._current_status = "IDLE"
        self._last_round_metrics: dict[str, Any] = {}
        self._preflight_ok: bool = True
        self._incompatible_jobs_reported: set[str] = set()

        # Hardware preflight check
        if self.device_str.startswith("cuda") and torch.cuda.is_available():
            try:
                t = torch.zeros(1, device=self.device_str)
                del t
            except Exception as exc:
                self.log(f"CRITICAL PREFLIGHT WARNING: CUDA device '{self.device_str}' ({self.gpu_name}) failed tensor kernel execution: {exc}", level="ERROR")

        # Register worker in SQLite database and clear any stale pending command
        self.bus.register_worker(
            worker_id=self.worker_id,
            hostname=self.hostname,
            gpu_name=self.gpu_name,
            vram_gb=self.vram_gb,
        )
        self.bus.set_worker_command(self.worker_id, None)

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
        """Background thread updating heartbeat periodically and listening for STOP/RESTART commands."""
        while not self._stop_event.is_set():
            try:
                cmd = self.bus.get_worker_command(self.worker_id)
                if cmd == "STOP":
                    self.log("Received remote STOP command. Shutting down worker...")
                    self.bus.set_worker_command(self.worker_id, None)
                    self._stop_event.set()
                    self._current_status = "OFFLINE"
                    try:
                        self.bus.heartbeat(
                            worker_id=self.worker_id,
                            status="OFFLINE",
                            current_job_id=None,
                        )
                    except Exception:
                        pass
                    from .cluster_worker import release_singleton_lock, get_device_tag
                    if not self.allow_shared_device:
                        release_singleton_lock(get_device_tag(self.device_str))
                    release_singleton_lock(f"wid_{self.worker_id}")
                    os._exit(0)
                elif cmd == "RESTART":
                    self.log("Received remote RESTART command. Respawning worker process...")
                    self.bus.set_worker_command(self.worker_id, None)
                    self._stop_event.set()
                    self._current_status = "OFFLINE"
                    try:
                        self.bus.heartbeat(
                            worker_id=self.worker_id,
                            status="OFFLINE",
                            current_job_id=None,
                        )
                    except Exception:
                        pass
                    from .cluster_worker import release_singleton_lock, get_device_tag
                    if not self.allow_shared_device:
                        release_singleton_lock(get_device_tag(self.device_str))
                    release_singleton_lock(f"wid_{self.worker_id}")
                    time.sleep(0.3)
                    flags = 0
                    if sys.platform == "win32":
                        DETACHED_PROCESS = 0x00000008
                        CREATE_NEW_PROCESS_GROUP = 0x00000200
                        CREATE_NO_WINDOW = 0x08000000
                        flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
                    subprocess.Popen(
                        [sys.executable] + sys.argv,
                        creationflags=flags,
                        close_fds=True,
                    )
                    os._exit(0)

                metrics = collect_system_metrics(self.device_str, getattr(self, "vram_gb", 0.0))
                self.bus.heartbeat(
                    worker_id=self.worker_id,
                    status=self._current_status,
                    current_job_id=self._current_job_id,
                    metrics=metrics,
                )
            except Exception:
                pass
            self._stop_event.wait(self.heartbeat_interval)

    def stop(self) -> None:
        """Signal worker to gracefully stop."""
        self._stop_event.set()
        for proc, aux_wid in self._child_worker_procs:
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
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

            max_grad = float(training_cfg.get("max_gradient") or training_cfg.get("max_grad") or 1.0)
            if scaler is not None:
                scaler.scale(loss).backward()
                if max_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if max_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
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
        is_compat, reason = self.is_job_compatible(job)
        if not is_compat:
            self.log(f"Worker {self.worker_id} cannot execute job '{job_id}': {reason}. Refusing claim.", level="WARNING")
            return False

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
        val_loader = None
        dataset = None
        token_array = None
        val_token_array = None

        try:
            try:
                # 1. Resolve dataset paths and cache remote .npy files to local SSD storage
                val_dataset_path = job.get("val_dataset_path")
                if not val_dataset_path or not os.path.exists(val_dataset_path):
                    cand_val = Path(dataset_path).parent / "val_tokens.npy"
                    if cand_val.exists():
                        val_dataset_path = str(cand_val)
                    else:
                        cand_shared_val = self.bus.shared_dir / "val_tokens.npy"
                        if cand_shared_val.exists():
                            val_dataset_path = str(cand_shared_val)

                target_cache_files = {Path(dataset_path).name}
                if val_dataset_path and os.path.exists(val_dataset_path):
                    target_cache_files.add(Path(val_dataset_path).name)

                # Primary worker purges stale .npy files from local SSD cache before copying new
                if self.ephemeral_job_id is None:
                    purge_local_dataset_cache(keep_files=target_cache_files, log_fn=self.log)

                def _on_cache_progress(pct: float, speed: float, eta: float) -> None:
                    self._current_status = f"PREPARING (Caching {pct:.0f}%)"

                # Safely cache remote/network dataset .npy to fast local SSD storage
                local_dataset_path = cache_dataset_to_local(
                    dataset_path,
                    log_fn=self.log,
                    keep_files=target_cache_files,
                    progress_callback=_on_cache_progress,
                )
                self._current_status = "PREPARING"
                local_val_dataset_path = None
                if val_dataset_path and os.path.exists(val_dataset_path):
                    local_val_dataset_path = cache_dataset_to_local(
                        val_dataset_path,
                        log_fn=self.log,
                        keep_files=target_cache_files,
                        progress_callback=_on_cache_progress,
                    )
                self._current_status = "PREPARING"

                # 2. Dynamic multi-worker: spawn auxiliary workers if node has excess VRAM (primary daemon only)
                if self.ephemeral_job_id is None and not self._child_worker_procs:
                    optimal_slots, est_gb, node_gb = calculate_optimal_worker_slots(
                        model_config, training_config, self.device_str
                    )
                    if optimal_slots > 1:
                        self.log(
                            f"Dynamic multi-worker: Node VRAM ({node_gb:.1f} GB) supports {optimal_slots} concurrent slots "
                            f"(Job estimate: {est_gb:.1f} GB/slot). Spawning {optimal_slots - 1} auxiliary worker process(es)..."
                        )
                        for slot_idx in range(2, optimal_slots + 1):
                            aux_worker_id = f"{self.worker_id}_slot{slot_idx}"
                            if sys.argv[0].endswith("cluster_worker.py"):
                                cmd = [
                                    sys.executable,
                                    sys.argv[0],
                                    "--shared-dir",
                                    str(self.bus.shared_dir),
                                    "--worker-id",
                                    aux_worker_id,
                                    "--device",
                                    self.device_str,
                                    "--ephemeral-job-id",
                                    job_id,
                                    "--allow-shared-device",
                                ]
                            else:
                                cmd = [
                                    sys.executable,
                                    "-m",
                                    "cluster.cli",
                                    "worker",
                                    "--shared-dir",
                                    str(self.bus.shared_dir),
                                    "--worker-id",
                                    aux_worker_id,
                                    "--device",
                                    self.device_str,
                                    "--ephemeral-job-id",
                                    job_id,
                                    "--allow-shared-device",
                                ]
                            try:
                                self.log(f"Launching auxiliary worker slot {slot_idx}: {aux_worker_id}...")
                                p = subprocess.Popen(
                                    cmd,
                                    stdout=None,
                                    stderr=None,
                                    stdin=subprocess.DEVNULL,
                                )
                                self._child_worker_procs.append((p, aux_worker_id))
                            except Exception as e:
                                self.log(f"Failed to spawn auxiliary worker slot {slot_idx}: {e}", level="WARNING")

                        if self._child_worker_procs:
                            time.sleep(1.5)

                # 3. Claim data shard slot
                shard_idx, total_shards = self.bus.claim_job_slot(job_id, self.worker_id)
                self.log(f"Assigned data shard slot {shard_idx + 1} of {total_shards} total nodes")

                # 4. Prepare local dataset from cached SSD file
                if not os.path.exists(local_dataset_path):
                    err = f"Dataset path not found: {local_dataset_path}"
                    self.log(err, level="ERROR")
                    self._current_status = "ERROR"
                    return False

                self.log(f"Mapping dataset: {local_dataset_path}...")
                token_array = np.load(local_dataset_path, mmap_mode="r")
                vocab_size = int(model_config.get("vocab_size", 0) or 0)

                if vocab_size <= 0:
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

                    max_token_id = 0
                    if detected_vocab <= 0:
                        try:
                            sample = token_array[:50000]
                            max_token_id = int(np.max(sample)) if len(sample) > 0 else 0
                        except Exception:
                            max_token_id = 0

                    min_safe_vocab = 256
                    if max_token_id > 0:
                        min_safe_vocab = ((max_token_id + 1 + 255) // 256) * 256
                        if 31000 <= max_token_id < 32000:
                            min_safe_vocab = max(min_safe_vocab, 32000)

                    vocab_size = max(detected_vocab, min_safe_vocab, 256)
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

                # 5. Prepare validation dataset if available alongside train tokens
                if local_val_dataset_path and os.path.exists(local_val_dataset_path):
                    try:
                        val_token_array = np.load(local_val_dataset_path, mmap_mode="r")
                        if len(val_token_array) > context_length:
                            val_dataset = ShardedTokenDataset(
                                token_array=val_token_array,
                                context_length=context_length,
                                shard_index=shard_idx,
                                total_shards=total_shards,
                                vocab_size=vocab_size,
                            )
                            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                            self.log(f"Validation dataset mapped ({len(val_token_array):,} tokens, {len(val_dataset):,} samples).")
                    except Exception as e:
                        self.log(f"Could not initialize validation dataset: {e}", level="WARNING")

                # 6. Build model and optimizer
                self.log(f"Dataset mapped ({len(token_array):,} tokens, vocab_size={vocab_size}). Building model for {self.device_str}...")
                model = build_model_from_config(model_config, self.device_str, logger=self.log)

                # Check if this is a fine-tuning job and load base model weights
                job_type = str(job.get("job_type") or training_config.get("training_mode") or "pretrain").lower()
                is_fine_tune = (job_type == "fine_tune")

                if is_fine_tune:
                    self.log("Fine-tuning job detected. Loading base checkpoint weights from shared storage...")
                    base_weights = self.bus.load_base_model_weights(job_id, device=self.device_str)
                    if base_weights is not None:
                        try:
                            missing, unexpected = model.load_state_dict(base_weights, strict=False)
                            self.log(f"Base model weights loaded successfully for fine-tuning. (Missing keys: {len(missing)}, unexpected keys: {len(unexpected)})")
                        except Exception as e:
                            self.log(f"Warning: Failed to load some base weights: {e}", level="WARNING")
                    else:
                        self.log("Notice: No base model weights found on shared storage. Initializing from scratch.", level="WARNING")

                # Apply LoRA if configured
                peft_method = str(job.get("peft_method") or training_config.get("peft_method") or "none").lower()
                if peft_method == "lora":
                    lora_cfg = job.get("lora_config") or training_config.get("lora_config") or {}
                    l_rank = int(lora_cfg.get("rank", training_config.get("lora_rank", 8)))
                    l_alpha = float(lora_cfg.get("alpha", training_config.get("lora_alpha", 16.0)))
                    l_dropout = float(lora_cfg.get("dropout", training_config.get("lora_dropout", 0.05)))
                    l_targets = str(lora_cfg.get("target_modules", training_config.get("lora_target_modules", "attention")))
                    num_lora = apply_lora_adapters(model, rank=l_rank, alpha=l_alpha, dropout=l_dropout, target_modules=l_targets)
                    freeze_non_lora_parameters(model)
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    self.log(f"LoRA adapters applied to {num_lora} modules. Trainable parameters: {trainable_params:,} (base model frozen).")

                # Create optimizer only over trainable parameters
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                self.log(f"Model ready. Creating AdamW optimizer for {len(trainable_params)} tensor(s) (lr={lr})...")
                t_opt = time.time()
                optimizer = torch.optim.AdamW(trainable_params, lr=lr)
                self.log(f"Optimizer created in {time.time() - t_opt:.2f}s on '{self.device_str}'. Ready to train {max_rounds} rounds.")
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

                # Operator disable guard: exit cleanly if worker was disabled mid-job
                if not getattr(self.bus, "is_worker_enabled", lambda wid: True)(self.worker_id):
                    self.log(f"Worker {self.worker_id} was disabled by operator. Pausing participation in job {job_id}.")
                    self._current_status = "DISABLED"
                    self._current_job_id = None
                    try:
                        self.bus.heartbeat(self.worker_id, status="DISABLED", current_job_id=None)
                    except Exception:
                        pass
                    return True

                # Dynamic shard reassignment: verify active worker pool and take over dropped worker slots
                try:
                    new_shard_idx, new_total_shards = self.bus.get_worker_shard_assignment(
                        job_id, self.worker_id, current_round
                    )
                    if new_shard_idx != shard_idx or new_total_shards != total_shards:
                        self.log(f"Active worker pool updated. Shard reallocated: {new_shard_idx + 1} of {new_total_shards} nodes.")
                        shard_idx, total_shards = new_shard_idx, new_total_shards
                        dataset = ShardedTokenDataset(
                            token_array=token_array,
                            context_length=context_length,
                            shard_index=shard_idx,
                            total_shards=total_shards,
                            vocab_size=vocab_size,
                        )
                        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                        dataloader_iter = iter(dataloader)
                        if val_token_array is not None and len(val_token_array) > context_length:
                            try:
                                val_dataset = ShardedTokenDataset(
                                    token_array=val_token_array,
                                    context_length=context_length,
                                    shard_index=shard_idx,
                                    total_shards=total_shards,
                                    vocab_size=vocab_size,
                                )
                                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                            except Exception:
                                pass
                except Exception:
                    pass

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
                    val_loss = None
                    if val_loader is not None:
                        try:
                            val_loss = self.evaluate_validation(model, val_loader, max_batches=25)
                        except Exception as exc:
                            self.log(f"Validation evaluation failed in round {current_round + 1}: {exc}", level="WARNING")

                    round_metrics = getattr(self, "_last_round_metrics", {})
                    tokens_per_sec = round_metrics.get("tokens_per_sec", 0.0)
                    tokens_processed = round_metrics.get("tokens_processed", 0)

                    val_str = f", Val loss: {val_loss:.4f}" if val_loss is not None else ""
                    self.log(
                        f"Round {current_round + 1}/{max_rounds} completed. "
                        f"Avg loss: {avg_loss:.4f}{val_str}, Speed: {tokens_per_sec:,.0f} tok/s. Depositing weights & telemetry..."
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
                        "val_loss": round(val_loss, 4) if val_loss is not None else None,
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
        except BaseException as exc:
            import traceback
            tb = traceback.format_exc()
            self.log(f"CRITICAL: Job execution crashed with unhandled exception:\n{tb}", level="ERROR")
            self._current_status = "ERROR"
            raise
        finally:
            # Terminate and reap auxiliary worker subprocesses (primary worker only)
            if self._child_worker_procs:
                self.log(f"Cleaning up {len(self._child_worker_procs)} auxiliary worker process(es)...")
                for proc, aux_wid in self._child_worker_procs:
                    try:
                        if proc.poll() is None:
                            proc.terminate()
                            try:
                                proc.wait(timeout=3.0)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                    except Exception:
                        pass
                    try:
                        self.bus.heartbeat(aux_wid, status="OFFLINE", current_job_id=None)
                    except Exception:
                        pass
                self._child_worker_procs.clear()

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

    def run_hardware_preflight(self) -> tuple[bool, list[str]]:
        """Execute comprehensive hardware, storage, and runtime preflight validation.

        Returns:
            (all_passed: bool, report_lines: list[str])
        """
        report: list[str] = []
        is_healthy = True

        report.append("=" * 72)
        report.append(f"[PREFLIGHT] Node Preflight Hardware Verification (Worker: {self.worker_id})")
        report.append("-" * 72)

        # 1. Device & Compute Capability
        is_cuda = self.device_str.startswith("cuda") and torch.cuda.is_available()
        if is_cuda:
            try:
                device_idx = 0
                if ":" in self.device_str:
                    try:
                        device_idx = int(self.device_str.split(":")[1])
                    except ValueError:
                        device_idx = 0
                props = torch.cuda.get_device_properties(device_idx)
                arch_tag = f"sm_{props.major}{props.minor}"
                arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
                arch_status = "Native SASS verified" if arch_tag in arch_list else "Forward-compatible / PTX"
                report.append(f"[PASS] Compute Device   : {self.device_str} ({props.name}, {self.vram_gb} GB VRAM, {arch_tag})")
                report.append(f"[PASS] PyTorch / CUDA   : {torch.__version__} (Arch: {arch_status})")
            except Exception as exc:
                report.append(f"[WARN] Compute Device   : {self.device_str} ({exc})")
        else:
            report.append(f"[PASS] Compute Device   : {self.device_str} ({self.gpu_name})")

        # 2. Kernel Execution Check
        try:
            test_a = torch.ones((4, 4), device=self.device_str)
            test_b = (test_a * 2.5 + 1.0).sum().item()
            del test_a
            if abs(test_b - 56.0) > 1e-4:
                raise RuntimeError(f"Unexpected tensor math result: {test_b} != 56.0")
            report.append(f"[PASS] Kernel Execution : Tensor arithmetic validated on {self.device_str}")
        except Exception as exc:
            is_healthy = False
            report.append(f"[FAIL] Kernel Execution : {exc}")
            if "no kernel image" in str(exc).lower():
                report.append(f"       Action Required  : PyTorch lacks GPU architecture binary. Install compatible PyTorch (e.g. cu128 nightly).")

        # 3. Mixed Precision (AMP FP16 & BF16) Check
        if is_cuda:
            fp16_ok = False
            bf16_ok = False
            try:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    m1 = torch.randn((8, 8), device=self.device_str)
                    m2 = m1 @ m1
                del m1, m2
                fp16_ok = True
            except Exception as exc:
                report.append(f"[WARN] AMP FP16 Failed  : {exc}")

            bf16_hw = torch.cuda.is_bf16_supported()
            if bf16_hw:
                try:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        m1 = torch.randn((8, 8), device=self.device_str)
                        m2 = m1 @ m1
                    del m1, m2
                    bf16_ok = True
                except Exception as exc:
                    report.append(f"[WARN] AMP BF16 Failed  : {exc}")

            bf16_str = "BF16 (Native)" if bf16_ok else ("BF16 (Emulated)" if not bf16_hw else "BF16 (Unavailable)")
            report.append(f"[PASS] Mixed Precision  : FP16 ({'OK' if fp16_ok else 'FAIL'}), {bf16_str}")

        # 4. Memory Scratch Buffer Allocation & Free
        if is_cuda:
            try:
                scratch = torch.empty((16, 1024, 1024), dtype=torch.float32, device=self.device_str)
                del scratch
                torch.cuda.empty_cache()
                report.append(f"[PASS] Memory Allocation: 64.0 MB scratch buffer allocated & reclaimed")
            except Exception as exc:
                is_healthy = False
                report.append(f"[FAIL] Memory Allocation: Failed to allocate scratch buffer: {exc}")

        # 5. Shared Storage (NFS/SMB) Read & Write Latency
        try:
            shared_dir = self.bus.shared_dir
            if not shared_dir.exists():
                raise FileNotFoundError(f"Shared storage path does not exist: {shared_dir}")
            test_path = shared_dir / f".preflight_{self.worker_id}_{int(time.time())}.tmp"
            t0 = time.time()
            test_path.write_text(f"preflight_{self.worker_id}", encoding="utf-8")
            read_back = test_path.read_text(encoding="utf-8")
            latency_ms = round((time.time() - t0) * 1000, 1)
            test_path.unlink(missing_ok=True)
            if read_back != f"preflight_{self.worker_id}":
                raise IOError("Shared storage read back content mismatch")
            report.append(f"[PASS] Shared Storage   : Read/Write verified (latency: {latency_ms}ms) on {shared_dir}")
        except Exception as exc:
            is_healthy = False
            report.append(f"[FAIL] Shared Storage   : {exc}")

        # 6. Micro-Transformer Mini-Step (Forward + Backward + AdamW)
        try:
            class _PreflightMiniLM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.emb = nn.Embedding(256, 64)
                    self.fc = nn.Linear(64, 256)
                def forward(self, x):
                    return self.fc(self.emb(x))

            mini = _PreflightMiniLM().to(self.device_str)
            opt = torch.optim.AdamW(mini.parameters(), lr=1e-3)
            inp = torch.randint(0, 256, (2, 8), device=self.device_str)
            target = torch.randint(0, 256, (2, 8), device=self.device_str)
            out = mini(inp)
            loss = F.cross_entropy(out.view(-1, 256), target.view(-1))
            loss.backward()
            opt.step()
            del mini, opt, inp, target, out, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            report.append(f"[PASS] Transformer Step : Mini-model forward + backward + AdamW verified")
        except Exception as exc:
            is_healthy = False
            report.append(f"[FAIL] Transformer Step : Mini-model execution failed: {exc}")

        report.append("=" * 72)
        if is_healthy:
            report.append("[PREFLIGHT] System Status: HEALTHY. Node ready to accept cluster jobs.")
        else:
            report.append("[PREFLIGHT] System Status: UNHEALTHY (DEGRADED). Refusing to claim jobs.")
        report.append("=" * 72)

        self._preflight_ok = is_healthy
        return is_healthy, report

    def evaluate_validation(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        max_batches: int = 25,
    ) -> float:
        """Evaluate validation loss on validation data loader using cross entropy.

        Args:
            model: Current PyTorch model.
            val_loader: DataLoader yielding (input_ids, target_ids).
            max_batches: Maximum number of batches to evaluate.

        Returns:
            Average validation loss float.
        """
        model.eval()
        total_loss = 0.0
        batches = 0
        device = torch.device(self.device_str)

        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                if step >= max_batches:
                    break
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0], batch[1]
                elif isinstance(batch, dict):
                    inputs = batch.get("input_ids") or batch.get("inputs")
                    targets = batch.get("target_ids") or batch.get("targets") or batch.get("labels")
                else:
                    continue

                if inputs is None or targets is None:
                    continue

                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                out = model(inputs)
                logits = out[0] if isinstance(out, (tuple, list)) else (out.logits if hasattr(out, "logits") else out)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
                total_loss += float(loss.item())
                batches += 1

        model.train()
        return (total_loss / batches) if batches > 0 else 0.0

    def is_job_compatible(self, job: dict[str, Any]) -> tuple[bool, str]:
        """Check if this worker node and device are compatible with the requested job.

        Returns:
            (True, "Compatible") or (False, "<detailed reason>")
        """
        # 0. Check operator enable/disable status
        if not getattr(self.bus, "is_worker_enabled", lambda wid: True)(self.worker_id):
            return False, f"Worker '{self.worker_id}' is disabled by operator."
        base_id = self.worker_id.split("_slot")[0]
        if base_id != self.worker_id and not getattr(self.bus, "is_worker_enabled", lambda wid: True)(base_id):
            return False, f"Primary worker node '{base_id}' is disabled by operator."

        # 1. Check worker health / preflight status
        if getattr(self, "_current_status", "") == "DEGRADED" or not getattr(self, "_preflight_ok", True):
            return False, "Worker is in DEGRADED status due to failed hardware preflight checks."

        # 2. Check target worker filtering if specified by job
        target_workers = job.get("target_workers")
        if target_workers and isinstance(target_workers, list):
            base_id = self.worker_id.split("_slot")[0]
            if self.worker_id not in target_workers and base_id not in target_workers:
                return False, f"Worker ID '{self.worker_id}' is not in job target_workers list: {target_workers}."

        # 3. Check compute device & PyTorch CUDA kernel execution
        if self.device_str.startswith("cuda"):
            if not torch.cuda.is_available():
                return False, "Job targets CUDA, but CUDA is not available on this host."
            try:
                # Fast kernel test to catch architecture binary incompatibility (e.g. sm_120 on outdated torch)
                dev = torch.device(self.device_str)
                test_t = torch.ones(2, 2, device=dev)
                _ = (test_t + 1.0).sum().item()
                del test_t
            except Exception as exc:
                return False, f"CUDA device '{self.device_str}' failed kernel execution test: {exc}"

        # 4. Check precision compatibility (e.g. bfloat16 hardware support)
        tcfg = job.get("training_config", {})
        if isinstance(tcfg, str):
            try:
                tcfg = json.loads(tcfg)
            except Exception:
                tcfg = {}
        prec = str(tcfg.get("precision", "float32")).lower()
        if prec in {"bfloat16", "bf16"}:
            if self.device_str.startswith("cuda"):
                if not (hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()):
                    return False, f"Device '{self.device_str}' does not natively support bfloat16 precision."

        # 5. Check dataset accessibility
        dataset_path = job.get("dataset_path", "")
        if not dataset_path:
            return False, "Job specifies no dataset_path."
        if not os.path.exists(dataset_path):
            return False, f"Dataset path is not accessible on this node: '{dataset_path}'"

        # 6. Check VRAM capacity against model parameters (if on CUDA)
        if self.device_str.startswith("cuda") and torch.cuda.is_available():
            try:
                model_cfg = job.get("model_config", {})
                if isinstance(model_cfg, str):
                    try:
                        model_cfg = json.loads(model_cfg)
                    except Exception:
                        model_cfg = {}
                vocab_size = int(model_cfg.get("vocab_size") or 32000)
                emb_size = int(model_cfg.get("embedding_size") or 768)
                n_layers = int(model_cfg.get("layer_count") or 12)
                ctx_len = int(model_cfg.get("context_length") or 1024)
                batch_size = int(tcfg.get("batch_size") or 2)

                approx_params = (vocab_size * emb_size * 2) + (n_layers * 12 * emb_size * emb_size)
                approx_bytes = approx_params * 16 + (batch_size * ctx_len * emb_size * n_layers * 4)
                approx_gb = approx_bytes / (1024 ** 3)

                dev_idx = 0
                if ":" in self.device_str:
                    try:
                        dev_idx = int(self.device_str.split(":")[1])
                    except ValueError:
                        dev_idx = 0
                free_b, total_b = torch.cuda.mem_get_info(dev_idx)
                total_gb = total_b / (1024 ** 3)

                if approx_gb > total_gb * 0.95:
                    return False, f"Estimated model memory ({approx_gb:.1f} GB) exceeds total GPU VRAM ({total_gb:.1f} GB)."
            except Exception:
                pass

        return True, "Compatible"

    def run_daemon(self, poll_interval: float = 3.0) -> None:
        """Run persistent background loop polling for jobs and executing them."""
        from .cluster_worker import acquire_singleton_lock, release_singleton_lock, get_device_tag

        device_tag = get_device_tag(self.device_str)
        wid_tag = f"wid_{self.worker_id}"
        if not getattr(self, "_lock_acquired", False):
            if not self.allow_shared_device:
                if not acquire_singleton_lock(device_tag):
                    self.log(f"A worker is already running for device '{self.device_str}' on this machine. Exiting.", level="WARNING")
                    return

            if not acquire_singleton_lock(wid_tag):
                if not self.allow_shared_device:
                    release_singleton_lock(device_tag)
                self.log(f"A worker is already running for worker ID '{self.worker_id}' on this machine. Exiting.", level="WARNING")
                return
            self._lock_acquired = True

        # Run hardware and environment preflight verification
        preflight_ok, report_lines = self.run_hardware_preflight()
        for line in report_lines:
            self.log(line)

        if not preflight_ok:
            self._current_status = "DEGRADED"
            self.log("Worker failed hardware preflight checks. Halting job acquisition until issues are resolved.", level="ERROR")
            try:
                self.bus.heartbeat(self.worker_id, status="DEGRADED", current_job_id=None)
            except Exception:
                pass
            while not self._stop_event.is_set():
                self._stop_event.wait(poll_interval)
            return

        # Dedicated execution loop for ephemeral auxiliary worker processes
        if self.ephemeral_job_id:
            self.log(f"Ephemeral worker slot started for job '{self.ephemeral_job_id}'. Will exit automatically upon completion.")
            try:
                deadline = time.time() + 45.0
                while not self._stop_event.is_set() and time.time() < deadline:
                    active_job = self.bus.get_active_job()
                    if active_job and active_job.get("job_id") == self.ephemeral_job_id and active_job.get("status") in {"RUNNING", "QUEUED"}:
                        self.execute_job(active_job, poll_interval=poll_interval)
                        break
                    time.sleep(poll_interval)
            except BaseException as exc:
                import traceback
                tb = traceback.format_exc()
                self.log(f"CRITICAL: Ephemeral worker crashed with unhandled exception:\n{tb}", level="ERROR")
                print(f"\n[Worker CRITICAL TRACEBACK]\n{tb}", file=sys.stderr, flush=True)
                raise
            finally:
                if not self.allow_shared_device:
                    release_singleton_lock(device_tag)
                release_singleton_lock(wid_tag)
                self._current_status = "OFFLINE"
                self._current_job_id = None
                try:
                    self.bus.heartbeat(self.worker_id, status="OFFLINE", current_job_id=None)
                except Exception:
                    pass
                self.log(f"Ephemeral worker slot {self.worker_id} finished execution. Exiting process.")
                return

        # Persistent daemon loop for primary worker daemon
        self.log(f"Worker daemon started. Listening for jobs on shared drive...")
        try:
            while not self._stop_event.is_set():
                # Check if worker is disabled by operator: remain active and heartbeating, but do not claim jobs
                if not getattr(self.bus, "is_worker_enabled", lambda wid: True)(self.worker_id):
                    self._current_status = "DISABLED"
                    self._current_job_id = None
                    try:
                        self.bus.heartbeat(
                            self.worker_id,
                            status="DISABLED",
                            current_job_id=None,
                            metrics=self._collect_system_metrics(),
                        )
                    except Exception:
                        pass
                    self._stop_event.wait(poll_interval)
                    continue

                self._current_status = "IDLE"
                self._current_job_id = None

                try:
                    active_job = self.bus.get_active_job()
                    if active_job and active_job.get("status") in {"RUNNING", "QUEUED"}:
                        job_id = active_job.get("job_id", "")
                        is_compat, reason = self.is_job_compatible(active_job)
                        if not is_compat:
                            if job_id not in self._incompatible_jobs_reported:
                                self._incompatible_jobs_reported.add(job_id)
                                self.log(f"Worker {self.worker_id} cannot accept job '{job_id}': {reason}. Skipping.", level="WARNING")
                                try:
                                    self.bus.heartbeat(self.worker_id, status="INCOMPATIBLE", current_job_id=None)
                                except Exception:
                                    pass
                        else:
                            self.execute_job(active_job, poll_interval=poll_interval)
                except Exception as exc:
                    import traceback
                    tb = traceback.format_exc()
                    self.log(f"Error in worker daemon:\n{tb}", level="ERROR")
                    time.sleep(poll_interval * 2)

                self._stop_event.wait(poll_interval)
        except BaseException as exc:
            import traceback
            tb = traceback.format_exc()
            self.log(f"CRITICAL: Worker daemon crashed with unhandled exception:\n{tb}", level="ERROR")
            print(f"\n[Worker CRITICAL TRACEBACK]\n{tb}", file=sys.stderr, flush=True)
            raise
        finally:
            if not self.allow_shared_device:
                release_singleton_lock(device_tag)
            release_singleton_lock(wid_tag)
            self._current_status = "OFFLINE"
            self._current_job_id = None
            try:
                self.bus.heartbeat(self.worker_id, status="OFFLINE", current_job_id=None)
            except Exception:
                pass
