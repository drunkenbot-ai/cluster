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
import shutil
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
# 1. Hardware Detection & Dependency Auto-Bootstrapping
# =============================================================================

def detect_all_gpus() -> list[dict[str, Any]]:
    """Detect all compute GPUs available on the host machine."""
    gpus: list[dict[str, Any]] = []

    # 1. Check PyTorch CUDA if available
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "device": f"cuda:{i}",
                    "name": torch.cuda.get_device_name(i),
                    "vram_gb": round(props.total_memory / (1024 ** 3), 2),
                })
            return gpus
    except Exception:
        pass

    # 2. Check nvidia-smi command directly
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    idx = int(parts[0])
                    name = parts[1]
                    vram_mb = float(parts[2])
                    gpus.append({
                        "index": idx,
                        "device": f"cuda:{idx}",
                        "name": name,
                        "vram_gb": round(vram_mb / 1024.0, 2),
                    })
            if gpus:
                return gpus
        except Exception:
            pass

    # Fallback to CPU
    return [{
        "index": -1,
        "device": "cpu",
        "name": platform.processor() or "CPU",
        "vram_gb": 0.0,
    }]


def ensure_dependencies() -> None:
    """Ensure torch and numpy are installed with CUDA support if an NVIDIA GPU is present."""
    missing = []
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")

    has_nvidia = bool(shutil.which("nvidia-smi"))
    need_cuda_torch = False

    try:
        import torch
        if has_nvidia and not torch.cuda.is_available():
            need_cuda_torch = True
    except ImportError:
        missing.append("torch")
        if has_nvidia:
            need_cuda_torch = True

    if missing or need_cuda_torch:
        print(f"[ClusterWorker] Preparing environment (missing={missing}, need_cuda={need_cuda_torch})...")
        try:
            if need_cuda_torch:
                print("[ClusterWorker] NVIDIA GPU detected. Installing official PyTorch CUDA 12.4 wheel...")
                cmd = [
                    sys.executable, "-m", "pip", "install",
                    "torch==2.6.0+cu124", "torchvision==0.21.0+cu124", "torchaudio==2.6.0+cu124",
                    "--index-url", "https://download.pytorch.org/whl/cu124",
                ]
                if "numpy" in missing:
                    cmd.append("numpy")
                subprocess.check_call(cmd)
            elif missing:
                cmd = [sys.executable, "-m", "pip", "install", *missing]
                subprocess.check_call(cmd)
            print("[ClusterWorker] Dependencies verified successfully.")
        except Exception as exc:
            print(f"[ClusterWorker] Note: automatic dependency installation failed: {exc}")


ensure_dependencies()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402


# =============================================================================
# 2. Per-Device Singleton Process Management
# =============================================================================

def get_device_tag(device: str) -> str:
    """Normalize device string to a clean alphanumeric tag for filenames and worker IDs."""
    return device.lower().replace(":", "_").replace("-", "_")


def get_lock_file(device_tag: str) -> Path:
    return Path(tempfile.gettempdir()) / f"cluster_worker_{device_tag}.pid"


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


def get_running_worker_pid(device_tag: str = "default") -> Optional[int]:
    """Retrieve the PID of currently running cluster worker for a specific device."""
    lock_path = get_lock_file(device_tag)
    if lock_path.exists():
        try:
            pid = int(lock_path.read_text().strip())
            if is_pid_running(pid):
                return pid
        except (ValueError, OSError):
            pass
    return None


def get_all_running_worker_pids() -> dict[str, int]:
    """Find all running cluster worker PIDs across all devices on this host."""
    res = {}
    temp_dir = Path(tempfile.gettempdir())
    for p in temp_dir.glob("cluster_worker_*.pid"):
        try:
            tag = p.stem.replace("cluster_worker_", "")
            pid = int(p.read_text().strip())
            if is_pid_running(pid):
                res[tag] = pid
            else:
                p.unlink(missing_ok=True)
        except Exception:
            pass
    return res


def acquire_singleton_lock(device_tag: str = "default") -> bool:
    """Ensure only one cluster worker process runs per device on this host."""
    lock_path = get_lock_file(device_tag)
    if lock_path.exists():
        try:
            old_pid = int(lock_path.read_text().strip())
            if is_pid_running(old_pid):
                print(f"[ClusterWorker] A worker is already running for device '{device_tag}' (PID: {old_pid}). Exiting.")
                return False
        except (ValueError, OSError):
            pass

    try:
        lock_path.write_text(str(os.getpid()))
        atexit.register(lambda: release_singleton_lock(device_tag))
        return True
    except Exception as exc:
        print(f"[ClusterWorker] Warning: could not write lockfile: {exc}")
        return True


def release_singleton_lock(device_tag: str = "default") -> None:
    """Release the device process lock file on exit."""
    try:
        lock_path = get_lock_file(device_tag)
        if lock_path.exists():
            old_pid = int(lock_path.read_text().strip())
            if old_pid == os.getpid():
                lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def stop_running_worker(device_tag: Optional[str] = None) -> int:
    """Terminate running worker process(es) on this machine."""
    if device_tag:
        pid = get_running_worker_pid(device_tag)
        pids = {device_tag: pid} if pid else {}
    else:
        pids = get_all_running_worker_pids()

    if not pids:
        print("[ClusterWorker] No active cluster worker process found running on this machine.")
        return 0

    for tag, pid in pids.items():
        print(f"[ClusterWorker] Stopping worker for device '{tag}' (PID: {pid})...")
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

        time.sleep(0.3)
        lock_path = get_lock_file(tag)
        lock_path.unlink(missing_ok=True)
        print(f"[ClusterWorker] Stopped worker process for '{tag}' (PID: {pid}).")

    return 0


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
                    command TEXT,
                    last_heartbeat REAL,
                    created_at REAL
                );
            """)
            try:
                conn.execute("ALTER TABLE workers ADD COLUMN command TEXT;")
            except Exception:
                pass
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

    def set_worker_command(self, worker_id: str, command: Optional[str]) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE workers SET command = ? WHERE worker_id = ?;", (command, worker_id))

    def get_worker_command(self, worker_id: str) -> Optional[str]:
        with self._connect() as conn:
            cursor = conn.execute("SELECT command FROM workers WHERE worker_id = ?;", (worker_id,))
            row = cursor.fetchone()
            return row["command"] if row and row["command"] else None

    def delete_worker(self, worker_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM workers WHERE worker_id = ?;", (worker_id,))
            return cursor.rowcount > 0

    def delete_offline_workers(self, stale_threshold_seconds: float = 60.0) -> int:
        now = time.time()
        cutoff = now - stale_threshold_seconds
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM workers WHERE status = 'OFFLINE' OR last_heartbeat < ?;",
                (cutoff,),
            )
            return cursor.rowcount

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

        # Hardware & device resolution
        all_gpus = detect_all_gpus()
        if device:
            self.device_str = device
        elif all_gpus and all_gpus[0]["device"] != "cpu":
            self.device_str = all_gpus[0]["device"]
        else:
            self.device_str = "cpu"

        self.device_tag = get_device_tag(self.device_str)
        self.worker_id = worker_id or f"{self.hostname}_{self.device_tag}"

        matched = next((g for g in all_gpus if g["device"] == self.device_str), None)
        if matched:
            self.gpu_name = matched["name"]
            self.vram_gb = matched["vram_gb"]
        elif self.device_str.startswith("cuda") and torch.cuda.is_available():
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
        print(f"[ClusterWorker] Running as process PID: {os.getpid()} (Device Tag: {self.device_tag})")
        print("[ClusterWorker] Waiting for cluster jobs...")

        while True:
            cmd = self.bus.get_worker_command(self.worker_id)
            if cmd == "STOP":
                print(f"[ClusterWorker] Received STOP command for {self.worker_id}. Shutting down...")
                self.bus.set_worker_command(self.worker_id, None)
                self.bus.heartbeat(self.worker_id, status="OFFLINE", current_job_id=None)
                release_singleton_lock(self.device_tag)
                sys.exit(0)
            elif cmd == "RESTART":
                print(f"[ClusterWorker] Received RESTART command for {self.worker_id}. Resetting...")
                self.bus.set_worker_command(self.worker_id, None)
                self.bus.heartbeat(self.worker_id, status="IDLE", current_job_id=None)
                time.sleep(1.0)

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

            cmd = self.bus.get_worker_command(self.worker_id)
            if cmd == "STOP":
                print(f"[ClusterWorker] Received STOP command during job {job_id}.")
                self.bus.set_worker_command(self.worker_id, None)
                self.bus.heartbeat(self.worker_id, status="OFFLINE", current_job_id=None)
                release_singleton_lock(self.device_tag)
                sys.exit(0)

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

                global_state = self.bus.load_global_weights(job_id, cur_round - 1, device=self.device_str)
                model.load_state_dict(global_state)
                print(f"[ClusterWorker] Loaded round {cur_round - 1} global weights into {self.device_str}.")

            # Local SGD training steps
            print(f"[ClusterWorker] Training round {cur_round}/{max_rounds} ({sync_steps} local steps on {self.device_str})...")
            model.train()
            self.bus.heartbeat(self.worker_id, status="TRAINING", current_job_id=job_id)

            step_losses = []
            for _ in range(sync_steps):
                if self.bus.is_stopped(job_id):
                    return
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
    parser.add_argument("--device", default=None, help="Compute device (e.g. 'cuda:0', 'cuda:1', 'cpu')")
    parser.add_argument("--all-gpus", action="store_true", help="Launch a dedicated background worker process for each detected GPU")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Polling interval in seconds")
    parser.add_argument("--stop", action="store_true", help="Stop active background worker process(es) on this machine")
    parser.add_argument("--status", action="store_true", help="Check status of background worker(s) on this machine")
    args = parser.parse_args()

    shared_dir = args.shared_dir or os.environ.get("LLM_SHARED_DIR")

    if args.stop:
        device_tag = get_device_tag(args.device) if args.device else None
        return stop_running_worker(device_tag)

    if args.status:
        pids = get_all_running_worker_pids()
        if pids:
            print(f"[ClusterWorker] Active workers on this host ({len(pids)}):")
            for dev_tag, pid in pids.items():
                print(f"  - Device [{dev_tag}]: PID {pid}")
        else:
            print("[ClusterWorker] No active worker processes running on this host.")
        return 0

    if not shared_dir:
        print("[ClusterWorker] ERROR: Central shared directory not specified!")
        print("[ClusterWorker] Please either:")
        print("  1. Set the LLM_SHARED_DIR environment variable: e.g. set LLM_SHARED_DIR=Z:\\llm_cluster")
        print("  2. Pass the argument: python cluster_worker.py --shared-dir Z:\\llm_cluster")
        return 1

    # Handle multi-GPU launch
    if args.all_gpus:
        all_gpus = detect_all_gpus()
        devices = [g["device"] for g in all_gpus] if all_gpus else ["cpu"]
        print(f"[ClusterWorker] Multi-GPU mode: spawning worker process for each device: {devices}")
        worker_exe = get_worker_executable()
        script_path = str(Path(__file__).resolve())
        env = os.environ.copy()
        if shared_dir:
            env["LLM_SHARED_DIR"] = shared_dir
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        for dev in devices:
            cmd = [worker_exe, script_path, "--device", dev, "--shared-dir", shared_dir]
            proc = subprocess.Popen(cmd, env=env, creationflags=flags)
            print(f"[ClusterWorker] Started worker for device '{dev}' (PID: {proc.pid})")
        return 0

    # Single device worker
    resolved_device = args.device
    if not resolved_device:
        all_gpus = detect_all_gpus()
        resolved_device = all_gpus[0]["device"] if (all_gpus and all_gpus[0]["device"] != "cpu") else "cpu"

    device_tag = get_device_tag(resolved_device)

    # Enforce per-device singleton
    if not acquire_singleton_lock(device_tag):
        return 0

    # Set process console title on Windows
    if sys.platform == "win32":
        try:
            worker_label = args.worker_id or f"{socket.gethostname()}_{device_tag}"
            ctypes.windll.kernel32.SetConsoleTitleW(f"ClusterWorker [{worker_label}] - PID {os.getpid()}")
        except Exception:
            pass

    worker: Optional[StandaloneWorker] = None

    def _sig_handler(signum: int, frame: Any) -> None:
        print("\n[ClusterWorker] Received termination signal. Shutting down gracefully...")
        if worker:
            try:
                worker.bus.heartbeat(worker.worker_id, status="OFFLINE", current_job_id=None)
            except Exception:
                pass
        release_singleton_lock(device_tag)
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    worker = StandaloneWorker(shared_dir=shared_dir, worker_id=args.worker_id, device=resolved_device)
    try:
        worker.run(poll_interval=args.poll_interval)
    except KeyboardInterrupt:
        _sig_handler(signal.SIGINT, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
