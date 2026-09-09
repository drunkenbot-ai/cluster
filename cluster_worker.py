"""Standalone Single-Script Cluster Worker Daemon for Port-Blocked Distributed Training.

This script is 100% self-contained. It can be copied to any remote worker machine
and executed directly without needing the LLM-IDE or engine repository.

Features:
1. Auto-bootstraps missing dependencies (torch, numpy) via pip at first launch.
2. Reads central storage path from `LLM_SHARED_DIR` environment variable or `--shared-dir`.
3. Enforces singleton process per machine (prevents multiple instances on the same host).
4. Reports GPU model, VRAM, and heartbeat to central SQLite WAL database.
5. Claims disjoint token shards and executes local SGD training rounds.
6. Visible as a standalone python process in Task Manager with identifiable title.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import ctypes
import json
import os
import platform
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Generator, Optional


# =============================================================================
# 1. Dependency Auto-Bootstrapping
# =============================================================================

def ensure_dependencies() -> None:
    """Ensure torch and numpy are installed; install via pip if missing."""
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")

    if missing:
        print(f"[ClusterWorker] Missing required packages: {missing}. Installing via pip...")
        try:
            cmd = [sys.executable, "-m", "pip", "install", *missing]
            subprocess.check_call(cmd)
            print("[ClusterWorker] Dependencies installed successfully.")
        except Exception as exc:
            print(f"[ClusterWorker] Failed to install dependencies automatically: {exc}")
            print(f"[ClusterWorker] Please run: pip install {' '.join(missing)}")
            sys.exit(1)


ensure_dependencies()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402


# =============================================================================
# 2. Singleton Process Management
# =============================================================================

LOCK_FILE = Path(tempfile.gettempdir()) / "cluster_worker_process.pid"


def is_pid_running(pid: int) -> bool:
    """Check whether a process with the given PID is currently active."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            process = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not process:
                return False
            exit_code = ctypes.c_ulong()
            success = kernel32.GetExitCodeProcess(process, ctypes.byref(exit_code))
            kernel32.CloseHandle(process)
            # 259 is STILL_ACTIVE
            return bool(success and exit_code.value == 259)
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def get_running_worker_pid() -> Optional[int]:
    """Retrieve the PID of currently running cluster worker if active."""
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            if is_pid_running(pid):
                return pid
        except (ValueError, OSError):
            pass
    return None


def acquire_singleton_lock() -> bool:
    """Ensure only one cluster worker process runs per machine."""
    if LOCK_FILE.exists():
        try:
            old_pid = int(LOCK_FILE.read_text().strip())
            if old_pid != os.getpid() and is_pid_running(old_pid):
                print(f"[ClusterWorker] Another worker process is already running on this machine (PID: {old_pid}). Exiting.")
                return False
        except (ValueError, OSError):
            pass

    try:
        LOCK_FILE.write_text(str(os.getpid()))
        atexit.register(release_singleton_lock)
        return True
    except Exception as exc:
        print(f"[ClusterWorker] Warning: could not write lockfile: {exc}")
        return True


def release_singleton_lock() -> None:
    """Release the process lock file on exit."""
    try:
        if LOCK_FILE.exists():
            old_pid = int(LOCK_FILE.read_text().strip())
            if old_pid == os.getpid():
                LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def stop_running_worker() -> int:
    """Terminate the active cluster worker process identified by the singleton lock."""
    pid = get_running_worker_pid()
    if not pid:
        print("[ClusterWorker] No active cluster worker process is currently running on this machine.")
        if LOCK_FILE.exists():
            LOCK_FILE.unlink(missing_ok=True)
        return 0

    print(f"[ClusterWorker] Stopping running worker process (PID: {pid})...")
    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=False, capture_output=True)
        except Exception as exc:
            print(f"[ClusterWorker] Error invoking taskkill: {exc}")
    else:
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.5)
            if is_pid_running(pid):
                os.kill(pid, signal.SIGKILL)
        except Exception as exc:
            print(f"[ClusterWorker] Error sending termination signal: {exc}")

    time.sleep(0.5)
    if not is_pid_running(pid):
        print(f"[ClusterWorker] Successfully stopped worker process (PID: {pid}).")
        LOCK_FILE.unlink(missing_ok=True)
        return 0
    else:
        print(f"[ClusterWorker] Failed to terminate worker process (PID: {pid}).")
        return 1


def get_worker_executable() -> str:
    """Get or create a named worker executable (cluster_worker.exe) on Windows for Task Manager clarity."""
    if sys.platform != "win32":
        return sys.executable
    try:
        app_data = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "cluster_worker"
        app_data.mkdir(parents=True, exist_ok=True)
        target_exe = app_data / "cluster_worker.exe"
        py_path = Path(sys.executable)
        if not target_exe.exists() or target_exe.stat().st_size != py_path.stat().st_size:
            import shutil
            shutil.copyfile(py_path, target_exe)
        return str(target_exe)
    except Exception:
        return sys.executable


# =============================================================================
# 3. Embedded SQLite WAL Bus Client
# =============================================================================

class StandaloneStorageBus:
    """Self-contained storage bus client operating over central shared network storage."""

    def __init__(self, shared_dir: Path | str) -> None:
        self.shared_dir = Path(shared_dir)
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.shared_dir / "cluster.db"
        self.jobs_dir = self.shared_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextlib.contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level="DEFERRED")
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout = 30000;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    hostname TEXT,
                    gpu_name TEXT,
                    vram_gb REAL,
                    status TEXT DEFAULT 'IDLE',
                    current_job_id TEXT,
                    last_heartbeat REAL,
                    created_at REAL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT DEFAULT 'QUEUED',
                    model_config TEXT,
                    training_config TEXT,
                    dataset_path TEXT,
                    current_round INTEGER DEFAULT 0,
                    max_rounds INTEGER DEFAULT 10,
                    sync_interval_steps INTEGER DEFAULT 250,
                    min_workers INTEGER DEFAULT 1,
                    sync_timeout_seconds REAL DEFAULT 180.0,
                    created_at REAL,
                    updated_at REAL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS job_participants (
                    job_id TEXT,
                    worker_id TEXT,
                    shard_index INTEGER,
                    total_shards INTEGER,
                    last_synced_round INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'ACTIVE',
                    PRIMARY KEY (job_id, worker_id)
                );
            """)

    def register_worker(self, worker_id: str, hostname: str, gpu_name: str, vram_gb: float) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO workers (worker_id, hostname, gpu_name, vram_gb, status, last_heartbeat, created_at)
                VALUES (?, ?, ?, ?, 'IDLE', ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    hostname = excluded.hostname,
                    gpu_name = excluded.gpu_name,
                    vram_gb = excluded.vram_gb,
                    status = 'IDLE',
                    last_heartbeat = excluded.last_heartbeat;
            """, (worker_id, hostname, gpu_name, vram_gb, now, now))

    def heartbeat(self, worker_id: str, status: Optional[str] = None, current_job_id: Optional[str] = None) -> None:
        now = time.time()
        with self._connect() as conn:
            if status is not None:
                conn.execute("""
                    UPDATE workers SET last_heartbeat = ?, status = ?, current_job_id = ?
                    WHERE worker_id = ?;
                """, (now, status, current_job_id, worker_id))
            else:
                conn.execute("UPDATE workers SET last_heartbeat = ? WHERE worker_id = ?;", (now, worker_id))

    def get_active_job(self) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM jobs WHERE status IN ('RUNNING', 'QUEUED', 'PAUSED') ORDER BY created_at ASC LIMIT 1;"
            )
            row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        res["model_config"] = json.loads(res["model_config"])
        res["training_config"] = json.loads(res["training_config"])
        return res

    def claim_job_slot(self, job_id: str, worker_id: str) -> tuple[int, int]:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT shard_index, total_shards FROM job_participants WHERE job_id = ? AND worker_id = ?;",
                (job_id, worker_id),
            )
            existing = cursor.fetchone()
            if existing:
                return int(existing["shard_index"]), int(existing["total_shards"])

            cursor = conn.execute("SELECT COUNT(*) as cnt FROM job_participants WHERE job_id = ?;", (job_id,))
            count = int(cursor.fetchone()["cnt"])
            shard_idx = count
            total = count + 1

            conn.execute("""
                INSERT INTO job_participants (job_id, worker_id, shard_index, total_shards, status)
                VALUES (?, ?, ?, ?, 'ACTIVE');
            """, (job_id, worker_id, shard_idx, total))

            conn.execute("UPDATE job_participants SET total_shards = ? WHERE job_id = ?;", (total, job_id))
            return shard_idx, total

    def is_paused(self, job_id: str) -> bool:
        return (self.jobs_dir / job_id / "signals" / "pause.sig").exists()

    def is_stopped(self, job_id: str) -> bool:
        return (self.jobs_dir / job_id / "signals" / "stop.sig").exists()

    def save_worker_weights(self, job_id: str, round_num: int, worker_id: str, state_dict: dict[str, torch.Tensor]) -> None:
        round_dir = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        target = round_dir / f"{worker_id}.pt"
        tmp_target = round_dir / f"{worker_id}.pt.tmp"
        ready_flag = round_dir / f"{worker_id}.ready"

        torch.save(state_dict, tmp_target)
        tmp_target.replace(target)
        ready_flag.touch()

    def is_global_weights_ready(self, job_id: str, round_num: int) -> bool:
        round_dir = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}"
        return (round_dir / "global_model.ready").exists() and (round_dir / "global_model.pt").exists()

    def load_global_weights(self, job_id: str, round_num: int, device: str = "cpu") -> dict[str, torch.Tensor]:
        weight_path = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}" / "global_model.pt"
        return torch.load(weight_path, map_location=device)


# =============================================================================
# 4. Embedded Data Sharding & Dataset
# =============================================================================

def compute_shard_boundaries(total_tokens: int, shard_index: int, total_shards: int, context_length: int = 512) -> tuple[int, int]:
    if total_shards <= 1:
        return 0, total_tokens
    total_windows = total_tokens // context_length
    windows_per_shard = total_windows // total_shards
    if windows_per_shard == 0:
        return 0, total_tokens

    start_token = shard_index * windows_per_shard * context_length
    end_token = total_tokens if shard_index == total_shards - 1 else (shard_index + 1) * windows_per_shard * context_length
    return start_token, end_token


class StandaloneTokenDataset(Dataset):
    def __init__(self, token_array: np.ndarray, context_length: int, shard_index: int = 0, total_shards: int = 1) -> None:
        self.context_length = context_length
        total_tokens = len(token_array)
        start_idx, end_idx = compute_shard_boundaries(total_tokens, shard_index, total_shards, context_length)
        self.tokens = token_array[start_idx:end_idx]
        usable = len(self.tokens) - self.context_length
        self.sample_count = max(0, (usable // self.context_length) + 1) if usable >= 0 else 0

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.context_length
        end = start + self.context_length
        x = torch.from_numpy(self.tokens[start:end].astype(np.int64))
        if end < len(self.tokens):
            y = torch.from_numpy(self.tokens[start + 1 : end + 1].astype(np.int64))
        else:
            y = x.clone()
        return x, y


# =============================================================================
# 5. Model Architecture Factory
# =============================================================================

def build_worker_model(model_config: dict[str, Any], device: str) -> nn.Module:
    """Build model using engine if available, or lightweight PyTorch Transformer fallback."""
    try:
        from engine.config import ModelConfig
        from engine.model_transformer import TransformerModel
        valid_keys = ModelConfig.__dataclass_fields__.keys()
        filtered = {k: v for k, v in model_config.items() if k in valid_keys}
        return TransformerModel(ModelConfig(**filtered)).to(device)
    except Exception:
        vocab_size = int(model_config.get("vocab_size", 1000))
        embed_dim = int(model_config.get("embedding_size", 128))
        num_heads = int(model_config.get("head_count", 4))
        num_layers = int(model_config.get("layer_count", 2))
        dropout = float(model_config.get("dropout", 0.1))

        class WorkerCausalTransformer(nn.Module):
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
                mask = torch.triu(torch.full((s, s), float("-inf"), device=x.device), diagonal=1)
                h = self.tok_embed(x)
                out = self.transformer(h, mask=mask, is_causal=True)
                return self.lm_head(out)

        return WorkerCausalTransformer().to(device)


# =============================================================================
# 6. Worker Daemon Engine
# =============================================================================

class StandaloneWorker:
    def __init__(self, shared_dir: Path | str, worker_id: Optional[str] = None, device: Optional[str] = None) -> None:
        self.bus = StandaloneStorageBus(shared_dir)
        self.hostname = socket.gethostname()
        self.worker_id = worker_id or f"{self.hostname}_{os.getpid()}"

        if device:
            self.device_str = device
        elif torch.cuda.is_available():
            self.device_str = "cuda:0"
        else:
            self.device_str = "cpu"

        if self.device_str.startswith("cuda") and torch.cuda.is_available():
            dev_idx = 0
            if ":" in self.device_str:
                try:
                    dev_idx = int(self.device_str.split(":")[1])
                except ValueError:
                    dev_idx = 0
            self.gpu_name = torch.cuda.get_device_name(dev_idx)
            props = torch.cuda.get_device_properties(dev_idx)
            self.vram_gb = round(props.total_memory / (1024 ** 3), 2)
        else:
            self.gpu_name = platform.processor() or "CPU"
            self.vram_gb = 0.0

        self.bus.register_worker(self.worker_id, self.hostname, self.gpu_name, self.vram_gb)

    def run(self, poll_interval: float = 3.0) -> None:
        """Main worker loop: registers heartbeat, claims jobs, and trains across rounds."""
        print(f"[ClusterWorker] Node: {self.worker_id}")
        print(f"[ClusterWorker] Host: {self.hostname} | Device: {self.device_str} ({self.gpu_name}, {self.vram_gb} GB VRAM)")
        print(f"[ClusterWorker] Central Storage: {self.bus.shared_dir.resolve()}")
        print(f"[ClusterWorker] Running as process PID: {os.getpid()} (Visible in Task Manager)")
        print("[ClusterWorker] Waiting for cluster jobs...")

        while True:
            self.bus.heartbeat(self.worker_id, status="IDLE", current_job_id=None)
            active_job = self.bus.get_active_job()

            if active_job and active_job.get("status") in {"RUNNING", "QUEUED"}:
                self._execute_job(active_job, poll_interval=poll_interval)

            time.sleep(poll_interval)

    def _execute_job(self, job: dict[str, Any], poll_interval: float = 2.0) -> None:
        job_id = job["job_id"]
        print(f"[ClusterWorker] >>> Claiming job: {job_id}")

        shard_idx, total_shards = self.bus.claim_job_slot(job_id, self.worker_id)
        print(f"[ClusterWorker] Assigned data shard: {shard_idx} of {total_shards} total nodes")

        dataset_path = job.get("dataset_path", "")
        if not os.path.exists(dataset_path):
            print(f"[ClusterWorker] Dataset not found at: {dataset_path}")
            return

        token_array = np.load(dataset_path, mmap_mode="r")
        context_length = int(job.get("model_config", {}).get("context_length", 512))
        dataset = StandaloneTokenDataset(token_array, context_length, shard_idx, total_shards)
        batch_size = int(job.get("training_config", {}).get("batch_size", 4))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloader_iter = iter(dataloader)

        model = build_worker_model(job.get("model_config", {}), self.device_str)
        lr = float(job.get("training_config", {}).get("learning_rate", 3e-4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        max_rounds = int(job.get("max_rounds", 10))
        sync_steps = int(job.get("sync_interval_steps", 250))
        cur_round = int(job.get("current_round", 0))

        while cur_round < max_rounds:
            if self.bus.is_stopped(job_id):
                print(f"[ClusterWorker] Job {job_id} stopped.")
                break

            # Handle cooperative pause
            while self.bus.is_paused(job_id):
                self.bus.heartbeat(self.worker_id, status="PAUSED", current_job_id=job_id)
                time.sleep(1.0)
                if self.bus.is_stopped(job_id):
                    break

            # Sync from global model if round > 0
            if cur_round > 0:
                print(f"[ClusterWorker] Waiting for global averaged model round {cur_round - 1}...")
                while not self.bus.is_global_weights_ready(job_id, cur_round - 1):
                    if self.bus.is_stopped(job_id):
                        return
                    self.bus.heartbeat(self.worker_id, status="SYNC_WAIT", current_job_id=job_id)
                    time.sleep(poll_interval)

                global_weights = self.bus.load_global_weights(job_id, cur_round - 1, device=self.device_str)
                model.load_state_dict(global_weights)

            # Local training for K steps
            self.bus.heartbeat(self.worker_id, status="TRAINING", current_job_id=job_id)
            model.train()
            print(f"[ClusterWorker] Starting round {cur_round}: executing {sync_steps} local SGD steps...")
            step_losses = []

            for s in range(sync_steps):
                if self.bus.is_stopped(job_id):
                    break
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)

                x, y = batch
                x = x.to(self.device_str, non_blocking=True)
                y = y.to(self.device_str, non_blocking=True)

                optimizer.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
                step_losses.append(float(loss.item()))

            avg_loss = sum(step_losses) / max(len(step_losses), 1)
            print(f"[ClusterWorker] Finished round {cur_round} (avg loss: {avg_loss:.4f}). Depositing weights...")

            # Save local weights
            self.bus.save_worker_weights(
                job_id=job_id,
                round_num=cur_round,
                worker_id=self.worker_id,
                state_dict={k: v.cpu() for k, v in model.state_dict().items()},
            )

            # Wait for coordinator to publish global model
            print(f"[ClusterWorker] Deposited weights for round {cur_round}. Waiting for coordinator synchronization...")
            while not self.bus.is_global_weights_ready(job_id, cur_round):
                if self.bus.is_stopped(job_id):
                    return
                self.bus.heartbeat(self.worker_id, status="SYNC_WAIT", current_job_id=job_id)
                time.sleep(poll_interval)

            cur_round += 1

        print(f"[ClusterWorker] Job {job_id} finished. Returning to IDLE.")
        self.bus.heartbeat(self.worker_id, status="IDLE", current_job_id=None)


# =============================================================================
# 7. Main Entry Point
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone Cluster Worker Daemon for Local SGD")
    parser.add_argument("--shared-dir", default=None, help="Path to central shared storage directory (or set LLM_SHARED_DIR)")
    parser.add_argument("--worker-id", default=None, help="Explicit worker node identifier")
    parser.add_argument("--device", default=None, help="Compute device (e.g. 'cuda:0', 'cpu')")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Polling interval in seconds")
    parser.add_argument("--stop", action="store_true", help="Stop any active background worker process on this machine")
    parser.add_argument("--status", action="store_true", help="Check status of background worker on this machine")
    args = parser.parse_args()

    if args.stop:
        return stop_running_worker()

    if args.status:
        active_pid = get_running_worker_pid()
        if active_pid:
            print(f"[ClusterWorker] Worker is RUNNING on this machine (PID: {active_pid}).")
            return 0
        else:
            print("[ClusterWorker] No active worker process is currently running on this machine.")
            return 0

    shared_dir = args.shared_dir or os.environ.get("LLM_SHARED_DIR")
    if not shared_dir:
        print("[ClusterWorker] ERROR: Central shared directory not specified!")
        print("[ClusterWorker] Please either:")
        print("  1. Set the LLM_SHARED_DIR environment variable: e.g. set LLM_SHARED_DIR=Z:\\llm_cluster")
        print("  2. Pass the argument: python cluster_worker.py --shared-dir Z:\\llm_cluster")
        return 1

    # Enforce singleton
    if not acquire_singleton_lock():
        return 0

    # Set process console title on Windows
    if sys.platform == "win32":
        try:
            worker_label = args.worker_id or socket.gethostname()
            ctypes.windll.kernel32.SetConsoleTitleW(f"ClusterWorker [{worker_label}] - PID {os.getpid()}")
        except Exception:
            pass

    # Graceful termination handler
    worker: Optional[StandaloneWorker] = None

    def _sig_handler(signum: int, frame: Any) -> None:
        print("\n[ClusterWorker] Received termination signal. Shutting down gracefully...")
        if worker:
            try:
                worker.bus.heartbeat(worker.worker_id, status="OFFLINE", current_job_id=None)
            except Exception:
                pass
        release_singleton_lock()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    worker = StandaloneWorker(shared_dir=shared_dir, worker_id=args.worker_id, device=args.device)
    try:
        worker.run(poll_interval=args.poll_interval)
    except KeyboardInterrupt:
        _sig_handler(signal.SIGINT, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
