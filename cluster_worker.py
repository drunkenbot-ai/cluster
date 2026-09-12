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
import math
import os
import platform
import random
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Generator, Optional


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

        # 1. Check for offline wheels cache in shared directory
        shared_dir_env = os.environ.get("LLM_SHARED_PATH") or os.environ.get("LLM_SHARED_DIR", "")
        shared_wheels = Path(shared_dir_env) / "wheels" if shared_dir_env else None
        installed_offline = False

        if shared_wheels and shared_wheels.is_dir():
            whls = list(shared_wheels.glob("*.whl"))
            if whls:
                print(f"[ClusterWorker] Offline wheel cache detected with {len(whls)} wheel(s) at: {shared_wheels}")
                try:
                    cmd = [
                        sys.executable, "-m", "pip", "install",
                        "--no-index", f"--find-links={shared_wheels}",
                        "torch", "numpy",
                    ]
                    subprocess.check_call(cmd)
                    installed_offline = True
                    print("[ClusterWorker] Offline dependencies installed successfully from shared storage.")
                except Exception as exc:
                    print(f"[ClusterWorker] Note: offline wheel install failed ({exc}), falling back to online install...")

        # 2. Online install fallback if offline was not used or failed
        if not installed_offline:
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
        conn = None
        for attempt in range(5):
            try:
                conn = sqlite3.connect(str(self.db_path), timeout=60.0, isolation_level="DEFERRED")
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA busy_timeout = 60000;")
                break
            except (sqlite3.OperationalError, sqlite3.DatabaseError):
                if attempt == 4:
                    raise
                time.sleep(0.05 * (2 ** attempt) + random.uniform(0.02, 0.08))
        try:
            yield conn
            conn.commit()
        except Exception:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def _run_with_retry(
        self,
        fn: Callable[[sqlite3.Connection], Any],
        max_retries: int = 5,
        default_on_error: Any = None,
        silent: bool = False,
    ) -> Any:
        """Execute a database function with automatic retry on SQLite operational/locking errors."""
        last_err = None
        for attempt in range(max_retries):
            try:
                with self._connect() as conn:
                    return fn(conn)
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
                last_err = exc
                if attempt == max_retries - 1:
                    if silent:
                        return default_on_error
                    raise
                time.sleep(0.05 * (2 ** attempt) + random.uniform(0.02, 0.08))
        return default_on_error

    def _init_db(self) -> None:
        def _init(conn: sqlite3.Connection) -> None:
            try:
                conn.execute("PRAGMA journal_mode = TRUNCATE;")
                conn.execute("PRAGMA synchronous = NORMAL;")
            except Exception:
                pass
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    hostname TEXT,
                    gpu_name TEXT,
                    vram_gb REAL,
                    status TEXT DEFAULT 'IDLE',
                    current_job_id TEXT,
                    command TEXT,
                    cpu_percent REAL DEFAULT 0.0,
                    ram_used_gb REAL DEFAULT 0.0,
                    ram_total_gb REAL DEFAULT 0.0,
                    vram_used_gb REAL DEFAULT 0.0,
                    metrics_json TEXT,
                    last_heartbeat REAL,
                    created_at REAL
                );
            """)
            for col_def in [
                ("command", "TEXT"),
                ("cpu_percent", "REAL DEFAULT 0.0"),
                ("ram_used_gb", "REAL DEFAULT 0.0"),
                ("ram_total_gb", "REAL DEFAULT 0.0"),
                ("vram_used_gb", "REAL DEFAULT 0.0"),
                ("metrics_json", "TEXT"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE workers ADD COLUMN {col_def[0]} {col_def[1]};")
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
                    sync_timeout_seconds REAL DEFAULT 1800.0,
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS worker_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    worker_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    level TEXT DEFAULT 'INFO',
                    message TEXT NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_worker_logs_wid ON worker_logs(worker_id, timestamp);")
        self._run_with_retry(_init)

    def register_worker(self, worker_id: str, hostname: str, gpu_name: str, vram_gb: float) -> None:
        now = time.time()
        def _op(conn: sqlite3.Connection) -> None:
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
        self._run_with_retry(_op)

    def heartbeat(
        self,
        worker_id: str,
        status: Optional[str] = None,
        current_job_id: Optional[str] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        now = time.time()
        cpu_pct = float(metrics.get("cpu_percent", 0.0)) if metrics and "cpu_percent" in metrics else None
        ram_used = float(metrics.get("ram_used_gb", 0.0)) if metrics and "ram_used_gb" in metrics else None
        ram_tot = float(metrics.get("ram_total_gb", 0.0)) if metrics and "ram_total_gb" in metrics else None
        vram_used = float(metrics.get("vram_used_gb", 0.0)) if metrics and "vram_used_gb" in metrics else None
        metrics_str = json.dumps(metrics, default=str) if metrics else None

        def _op(conn: sqlite3.Connection) -> None:
            if status is not None and metrics is not None:
                conn.execute("""
                    UPDATE workers
                    SET last_heartbeat = ?, status = ?, current_job_id = ?,
                        cpu_percent = COALESCE(?, cpu_percent),
                        ram_used_gb = COALESCE(?, ram_used_gb),
                        ram_total_gb = COALESCE(?, ram_total_gb),
                        vram_used_gb = COALESCE(?, vram_used_gb),
                        metrics_json = COALESCE(?, metrics_json)
                    WHERE worker_id = ?;
                """, (now, status, current_job_id, cpu_pct, ram_used, ram_tot, vram_used, metrics_str, worker_id))
            elif status is not None:
                conn.execute("""
                    UPDATE workers SET last_heartbeat = ?, status = ?, current_job_id = ?
                    WHERE worker_id = ?;
                """, (now, status, current_job_id, worker_id))
            elif metrics is not None:
                conn.execute("""
                    UPDATE workers
                    SET last_heartbeat = ?,
                        cpu_percent = COALESCE(?, cpu_percent),
                        ram_used_gb = COALESCE(?, ram_used_gb),
                        ram_total_gb = COALESCE(?, ram_total_gb),
                        vram_used_gb = COALESCE(?, vram_used_gb),
                        metrics_json = COALESCE(?, metrics_json)
                    WHERE worker_id = ?;
                """, (now, cpu_pct, ram_used, ram_tot, vram_used, metrics_str, worker_id))
            else:
                conn.execute("UPDATE workers SET last_heartbeat = ? WHERE worker_id = ?;", (now, worker_id))
        self._run_with_retry(_op, silent=True)

    def set_worker_command(self, worker_id: str, command: Optional[str]) -> None:
        def _op(conn: sqlite3.Connection) -> None:
            conn.execute("UPDATE workers SET command = ? WHERE worker_id = ?;", (command, worker_id))
        self._run_with_retry(_op, silent=True)

    def get_worker_command(self, worker_id: str) -> Optional[str]:
        def _op(conn: sqlite3.Connection) -> Optional[str]:
            cursor = conn.execute("SELECT command FROM workers WHERE worker_id = ?;", (worker_id,))
            row = cursor.fetchone()
            return row["command"] if row and row["command"] else None
        return self._run_with_retry(_op, default_on_error=None, silent=True)

    def delete_worker(self, worker_id: str) -> bool:
        def _op(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute("DELETE FROM workers WHERE worker_id = ?;", (worker_id,))
            return cursor.rowcount > 0
        return bool(self._run_with_retry(_op, default_on_error=False))

    def delete_offline_workers(self, stale_threshold_seconds: float = 60.0) -> int:
        now = time.time()
        cutoff = now - stale_threshold_seconds
        def _op(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                "DELETE FROM workers WHERE status = 'OFFLINE' OR last_heartbeat < ?;",
                (cutoff,),
            )
            return cursor.rowcount
        return int(self._run_with_retry(_op, default_on_error=0))

    def touch_job(self, job_id: str) -> None:
        """Update job updated_at timestamp to signal active coordinator heartbeat."""
        now = time.time()
        def _op(conn: sqlite3.Connection) -> None:
            conn.execute("UPDATE jobs SET updated_at = ? WHERE job_id = ?;", (now, job_id))
        self._run_with_retry(_op, silent=True)

    def set_worker_status(self, worker_id: str, status: str) -> None:
        """Explicitly set a worker's status (e.g. 'OFFLINE', 'IDLE')."""
        self.heartbeat(worker_id, status=status, current_job_id=None)

    def list_workers(self, active_within_seconds: float = 60.0) -> list[dict[str, Any]]:
        """List all workers, calculating online status based on heartbeat freshness."""
        now = time.time()
        def _op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            cursor = conn.execute("SELECT * FROM workers ORDER BY created_at ASC;")
            return [dict(r) for r in cursor.fetchall()]
        rows = self._run_with_retry(_op, default_on_error=[])

        results = []
        for r in rows:
            data = dict(r)
            last_hb = float(data.get("last_heartbeat") or 0)
            is_fresh = (now - last_hb) < active_within_seconds
            stored_status = str(data.get("status") or "OFFLINE").upper()
            if not is_fresh or stored_status in {"OFFLINE", "STOPPED"}:
                data["is_online"] = False
                data["status"] = "OFFLINE"
            else:
                data["is_online"] = True
            results.append(data)
        return results

    def get_active_job(self, max_stale_seconds: float = 3600.0) -> Optional[dict[str, Any]]:
        now = time.time()
        def _op(conn: sqlite3.Connection) -> Optional[dict[str, Any]]:
            if max_stale_seconds > 0:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'STOPPED', updated_at = ?
                    WHERE status IN ('RUNNING', 'QUEUED')
                      AND (? - updated_at) > ?;
                    """,
                    (now, now, max_stale_seconds),
                )
            cursor = conn.execute(
                "SELECT * FROM jobs WHERE status IN ('RUNNING', 'QUEUED', 'PAUSED') ORDER BY created_at DESC LIMIT 1;"
            )
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            res["model_config"] = json.loads(res["model_config"])
            res["training_config"] = json.loads(res["training_config"])
            return res
        return self._run_with_retry(_op, default_on_error=None, silent=True)

    def claim_job_slot(self, job_id: str, worker_id: str) -> tuple[int, int]:
        return self.get_worker_shard_assignment(job_id, worker_id, round_num=0)

    def get_worker_shard_assignment(self, job_id: str, worker_id: str, round_num: int = 0) -> tuple[int, int]:
        def _op(conn: sqlite3.Connection) -> tuple[int, int]:
            cursor = conn.execute(
                "SELECT status FROM job_participants WHERE job_id = ? AND worker_id = ?;",
                (job_id, worker_id),
            )
            row = cursor.fetchone()
            if not row:
                conn.execute("""
                    INSERT INTO job_participants (job_id, worker_id, shard_index, total_shards, status)
                    VALUES (?, ?, 0, 1, 'ACTIVE');
                """, (job_id, worker_id))
            elif row["status"] != "ACTIVE":
                conn.execute(
                    "UPDATE job_participants SET status = 'ACTIVE' WHERE job_id = ? AND worker_id = ?;",
                    (job_id, worker_id),
                )

            cursor = conn.execute(
                "SELECT worker_id FROM job_participants WHERE job_id = ? AND status = 'ACTIVE' ORDER BY worker_id ASC;",
                (job_id,),
            )
            active_ids = [r["worker_id"] for r in cursor.fetchall()]
            total_active = max(len(active_ids), 1)

            try:
                shard_idx = active_ids.index(worker_id)
            except ValueError:
                shard_idx = 0

            for idx, wid in enumerate(active_ids):
                if wid == worker_id:
                    conn.execute(
                        "UPDATE job_participants SET shard_index = ?, total_shards = ?, last_synced_round = ? WHERE job_id = ? AND worker_id = ?;",
                        (idx, total_active, round_num, job_id, wid),
                    )
                else:
                    conn.execute(
                        "UPDATE job_participants SET shard_index = ?, total_shards = ? WHERE job_id = ? AND worker_id = ?;",
                        (idx, total_active, job_id, wid),
                    )
            return shard_idx, total_active
        return self._run_with_retry(_op)

    def mark_worker_dropped(self, job_id: str, worker_id: str, reason: str = "timeout") -> None:
        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE job_participants SET status = 'DROPPED' WHERE job_id = ? AND worker_id = ?;",
                (job_id, worker_id),
            )
            cursor = conn.execute(
                "SELECT worker_id FROM job_participants WHERE job_id = ? AND status = 'ACTIVE' ORDER BY worker_id ASC;",
                (job_id,),
            )
            active_ids = [r["worker_id"] for r in cursor.fetchall()]
            total_active = max(len(active_ids), 1)
            for idx, wid in enumerate(active_ids):
                conn.execute(
                    "UPDATE job_participants SET shard_index = ?, total_shards = ? WHERE job_id = ? AND worker_id = ?;",
                    (idx, total_active, job_id, wid),
                )
        self._run_with_retry(_op, silent=True)

    def save_worker_telemetry(self, job_id: str, round_num: int, worker_id: str, telemetry: dict[str, Any]) -> None:
        round_dir = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        target = round_dir / f"{worker_id}_telemetry.json"
        target.write_text(json.dumps(telemetry, indent=2, default=str), encoding="utf-8")

    def load_worker_telemetry(self, job_id: str, round_num: int, worker_id: str) -> Optional[dict[str, Any]]:
        target = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}" / f"{worker_id}_telemetry.json"
        if not target.exists():
            return None
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return None

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

    def get_checkpoints_dir(self, job_id: str) -> Path:
        ckpt_dir = self.jobs_dir / job_id / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir

    def save_checkpoint(
        self,
        job_id: str,
        step: int,
        state_dict: dict[str, torch.Tensor],
        is_final: bool = False,
    ) -> Path:
        ckpt_dir = self.get_checkpoints_dir(job_id)
        target = ckpt_dir / f"checkpoint_step_{step:06d}.pt"
        tmp_target = ckpt_dir / f"checkpoint_step_{step:06d}.pt.tmp"
        torch.save(state_dict, tmp_target)
        tmp_target.replace(target)

        latest_target = ckpt_dir / "latest_checkpoint.pt"
        latest_tmp = ckpt_dir / "latest_checkpoint.pt.tmp"
        torch.save(state_dict, latest_tmp)
        latest_tmp.replace(latest_target)

        if is_final:
            final_target = ckpt_dir / "final_model.pt"
            final_tmp = ckpt_dir / "final_model.pt.tmp"
            torch.save(state_dict, final_tmp)
            final_tmp.replace(final_target)

        return target

    def load_latest_checkpoint(self, job_id: str, device: str = "cpu") -> Optional[dict[str, torch.Tensor]]:
        ckpt_dir = self.get_checkpoints_dir(job_id)
        latest = ckpt_dir / "latest_checkpoint.pt"
        if latest.exists():
            return torch.load(latest, map_location=device)
        final = ckpt_dir / "final_model.pt"
        if final.exists():
            return torch.load(final, map_location=device)
        return None

    def write_worker_logs(self, entries: list[tuple[str, float, str, str]]) -> None:
        if not entries:
            return
        def _op(conn: sqlite3.Connection) -> None:
            conn.executemany(
                "INSERT INTO worker_logs (worker_id, timestamp, level, message) VALUES (?, ?, ?, ?);",
                entries,
            )
        self._run_with_retry(_op, silent=True)

    def write_worker_log(self, worker_id: str, message: str, level: str = "INFO") -> None:
        self.write_worker_logs([(worker_id, time.time(), level, message)])

    def get_worker_logs(self, worker_id: str, limit: int = 200) -> list[dict[str, Any]]:
        def _op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            cursor = conn.execute(
                "SELECT id, worker_id, timestamp, level, message FROM worker_logs WHERE worker_id = ? ORDER BY id DESC LIMIT ?;",
                (worker_id, limit),
            )
            rows = cursor.fetchall()
            return [dict(r) for r in reversed(rows)]
        return self._run_with_retry(_op, default_on_error=[])


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
    def __init__(self, token_array: np.ndarray, context_length: int, shard_index: int = 0, total_shards: int = 1, vocab_size: Optional[int] = None) -> None:
        self.context_length = context_length
        self.vocab_size = vocab_size
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
        if self.vocab_size is not None and self.vocab_size > 0:
            x = torch.clamp(x, 0, self.vocab_size - 1)
            y = torch.clamp(y, 0, self.vocab_size - 1)
        return x, y


# =============================================================================
# 5. Model Architecture Factory (MicroGPT Parity)
# =============================================================================

class StandaloneRotaryEmbedding(nn.Module):
    def __init__(self, head_size: int, context_length: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_size, 2).float() / head_size))
        positions = torch.arange(context_length, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, start_pos: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        token_count = query.size(-2)
        cos = self.cos[:, :, start_pos : start_pos + token_count, :]
        sin = self.sin[:, :, start_pos : start_pos + token_count, :]
        q_rot = (query * cos) + (self._rotate_half(query) * sin)
        k_rot = (key * cos) + (self._rotate_half(key) * sin)
        return q_rot, k_rot

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)


class StandaloneRMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class StandaloneLayerNorm(nn.Module):
    def __init__(self, size: int, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


def make_standalone_norm(norm_type: str, dim: int, bias: bool = False) -> nn.Module:
    if norm_type == "rmsnorm":
        return StandaloneRMSNorm(dim)
    return StandaloneLayerNorm(dim, bias=bias)


class StandaloneCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, kv_heads: int, context_length: int, dropout: float = 0.0, bias: bool = False, pos_enc: str = "rope") -> None:
        super().__init__()
        self.head_count = n_heads
        self.kv_head_count = kv_heads or n_heads
        self.embedding_size = d_model
        self.head_size = d_model // n_heads
        self.kv_embedding_size = self.kv_head_count * self.head_size
        self.c_attn = nn.Linear(d_model, d_model + (2 * self.kv_embedding_size), bias=bias)
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.rotary = StandaloneRotaryEmbedding(self.head_size, context_length) if pos_enc == "rope" else None
        self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length, dtype=torch.bool)).view(1, 1, context_length, context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split((self.embedding_size, self.kv_embedding_size, self.kv_embedding_size), dim=2)
        q = q.view(b, t, self.head_count, self.head_size).transpose(1, 2)
        k = k.view(b, t, self.kv_head_count, self.head_size).transpose(1, 2)
        v = v.view(b, t, self.kv_head_count, self.head_size).transpose(1, 2)
        if self.rotary is not None:
            q, k = self.rotary(q, k)
        if self.kv_head_count != self.head_count:
            rep = self.head_count // self.kv_head_count
            k = k[:, :, None, :, :].expand(b, self.kv_head_count, rep, t, self.head_size).reshape(b, self.head_count, t, self.head_size)
            v = v[:, :, None, :, :].expand(b, self.kv_head_count, rep, t, self.head_size).reshape(b, self.head_count, t, self.head_size)

        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
            att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = self.attn_dropout(att) @ v

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))


class StandaloneMLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, mlp_type: str = "swiglu", dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.mlp_type = mlp_type
        if mlp_type == "swiglu":
            self.w1 = nn.Linear(d_model, hidden_dim, bias=bias)
            self.w2 = nn.Linear(hidden_dim, d_model, bias=bias)
            self.w3 = nn.Linear(d_model, hidden_dim, bias=bias)
            self.dropout = nn.Dropout(dropout)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim, bias=bias),
                nn.GELU(),
                nn.Linear(hidden_dim, d_model, bias=bias),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "swiglu":
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        return self.net(x)


class StandaloneBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, kv_heads: int, hidden_dim: int, context_length: int, norm_type: str = "rmsnorm", mlp_type: str = "swiglu", dropout: float = 0.0, bias: bool = False, pos_enc: str = "rope") -> None:
        super().__init__()
        self.ln_1 = make_standalone_norm(norm_type, d_model, bias=bias)
        self.attn = StandaloneCausalSelfAttention(d_model, n_heads, kv_heads, context_length, dropout, bias, pos_enc)
        self.ln_2 = make_standalone_norm(norm_type, d_model, bias=bias)
        self.mlp = StandaloneMLP(d_model, hidden_dim, mlp_type, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class StandaloneMicroGPT(nn.Module):
    """Zero-dependency PyTorch implementation of MicroGPT, 100% state-dict compatible with engine."""
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        vocab_size = int(config.get("vocab_size", 1000))
        d_model = int(config.get("embedding_size", 128))
        context_length = int(config.get("context_length", 256))
        n_heads = int(config.get("head_count", 4))
        kv_heads = int(config.get("kv_head_count") or n_heads)
        n_layers = int(config.get("layer_count", 2))
        hidden_dim = int(config.get("intermediate_size") or (d_model * 4))
        norm_type = str(config.get("norm_type", "rmsnorm"))
        mlp_type = str(config.get("mlp_type", "swiglu"))
        dropout = float(config.get("dropout", 0.0))
        bias = bool(config.get("bias", False))
        pos_enc = str(config.get("position_encoding", "rope"))

        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model) if pos_enc == "learned" else None
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            StandaloneBlock(d_model, n_heads, kv_heads, hidden_dim, context_length, norm_type, mlp_type, dropout, bias, pos_enc)
            for _ in range(n_layers)
        ])
        self.ln_f = make_standalone_norm(norm_type, d_model, bias=bias)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.size()
        if hasattr(self, "token_embedding") and hasattr(self.token_embedding, "num_embeddings"):
            idx = torch.clamp(idx, 0, self.token_embedding.num_embeddings - 1)
        x = self.token_embedding(idx)
        if self.position_embedding is not None:
            if t > self.context_length:
                t = self.context_length
                x = x[:, :t, :]
            positions = torch.arange(0, t, dtype=torch.long, device=idx.device)
            x = x + self.position_embedding(positions)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def build_worker_model(model_config: dict[str, Any], device: str) -> nn.Module:
    """Build model using engine if available, or standalone exact MicroGPT implementation."""
    try:
        from engine.config import ModelConfig
        from engine.model import MicroGPT
        valid_keys = ModelConfig.__dataclass_fields__.keys()
        filtered = {k: v for k, v in model_config.items() if k in valid_keys}
        return MicroGPT(ModelConfig(**filtered)).to(device)
    except Exception:
        return StandaloneMicroGPT(model_config).to(device)


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

    def log(self, message: str, level: str = "INFO") -> None:
        """Write diagnostic log to stdout and central SQLite worker_logs table."""
        timestamp_str = time.strftime("%H:%M:%S")
        try:
            print(f"[{timestamp_str}] [{level}] [Worker {self.worker_id}] {message}", flush=True)
        except Exception:
            pass
        try:
            self.bus.write_worker_log(self.worker_id, message, level=level)
        except Exception:
            pass

    def _heartbeat(self, status: Optional[str] = None, current_job_id: Optional[str] = None) -> None:
        metrics = collect_system_metrics(self.device_str, self.vram_gb)
        self.bus.heartbeat(self.worker_id, status=status, current_job_id=current_job_id, metrics=metrics)

    def _restart_process(self) -> None:
        """Cleanly respawn this worker process and exit."""
        self.log(f"Respawning worker process for {self.worker_id}...")
        self.bus.set_worker_command(self.worker_id, None)
        if hasattr(self, "_cleanup_func"):
            import atexit
            try:
                atexit.unregister(self._cleanup_func)
            except Exception:
                pass
        self._heartbeat(status="OFFLINE", current_job_id=None)
        release_singleton_lock(self.device_tag)
        time.sleep(0.3)
        try:
            flags = 0
            if sys.platform == "win32":
                DETACHED_PROCESS = 0x00000008
                CREATE_NEW_PROCESS_GROUP = 0x00000200
                CREATE_NO_WINDOW = 0x08000000
                flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
            log_path = Path(tempfile.gettempdir()) / "cluster_worker_local.log"
            log_file = open(log_path, "a", encoding="utf-8")
            subprocess.Popen(
                [sys.executable] + sys.argv,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=flags,
                close_fds=True,
            )
        except Exception as exc:
            self.log(f"Failed to respawn worker process: {exc}", level="ERROR")
        sys.exit(0)

    def run(self, poll_interval: float = 3.0) -> None:
        """Main worker loop: registers heartbeat, claims jobs, and trains across rounds."""
        self.log(f"Node online: {self.worker_id}")
        self.log(f"Host: {self.hostname} | Device: {self.device_str} ({self.gpu_name}, {self.vram_gb} GB VRAM)")
        self.log(f"Central Storage: {self.bus.shared_dir.resolve()} | PID: {os.getpid()} ({self.device_tag})")
        self.log("Waiting for cluster jobs...")

        def _cleanup():
            try:
                self.bus.heartbeat(self.worker_id, status="OFFLINE", current_job_id=None)
                release_singleton_lock(self.device_tag)
            except Exception:
                pass

        import atexit, signal
        self._cleanup_func = _cleanup
        atexit.register(_cleanup)
        try:
            signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
            signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
            if hasattr(signal, "SIGBREAK"):
                signal.signal(signal.SIGBREAK, lambda s, f: sys.exit(0))
        except Exception:
            pass

        while True:
            try:
                cmd = self.bus.get_worker_command(self.worker_id)
                if cmd == "STOP":
                    self.log(f"Received STOP command. Shutting down...")
                    self.bus.set_worker_command(self.worker_id, None)
                    self._heartbeat(status="OFFLINE", current_job_id=None)
                    release_singleton_lock(self.device_tag)
                    sys.exit(0)
                elif cmd == "RESTART":
                    self.log(f"Received RESTART command. Respawning process...")
                    self._restart_process()

                self._heartbeat(status="IDLE", current_job_id=None)
                active_job = self.bus.get_active_job()

                if active_job:
                    status = active_job.get("status")
                    jid = active_job.get("job_id", "")
                    if status == "RUNNING" and not self.bus.is_stopped(jid):
                        self._execute_job(active_job, poll_interval=poll_interval)
                    elif status == "QUEUED":
                        self._heartbeat(status="READY", current_job_id=jid)
            except Exception as exc:
                self.log(f"Transient error in worker poll loop: {exc}", level="WARNING")

            time.sleep(poll_interval)

    def _execute_job(self, job: dict[str, Any], poll_interval: float = 2.0) -> None:
        job_id = job["job_id"]
        self.log(f">>> Claimed job: {job_id}")
        self._heartbeat(status="PREPARING", current_job_id=job_id)

        model = None
        optimizer = None
        dataloader = None
        dataloader_iter = None
        dataset = None
        token_array = None
        global_weights = None
        batch = None
        x = None
        y = None
        logits = None
        loss = None

        try:
            shard_idx, total_shards = self.bus.claim_job_slot(job_id, self.worker_id)
            self.log(f"Assigned data shard slot {shard_idx + 1} of {total_shards} total nodes")

            dataset_path = job.get("dataset_path", "")
            if not os.path.exists(dataset_path):
                err = f"Dataset not found at: {dataset_path}"
                self.log(err, level="ERROR")
                self._heartbeat(status="ERROR", current_job_id=job_id)
                return

            token_array = np.load(dataset_path, mmap_mode="r")
            model_cfg = job.get("model_config", {})
            training_cfg = job.get("training_config", {})

            # Auto-detect and guard against invalid vocab_size <= 1 or vocab_size < max token ID in dataset
            vocab_size = int(model_cfg.get("vocab_size", 0) or 0)

            # 1. Inspect metadata files if present on shared storage or dataset dir
            detected_vocab = 0
            for sdir in [Path(dataset_path).parent, self.bus.shared_dir]:
                summary_file = sdir / "dataset_summary.json"
                if summary_file.exists():
                    try:
                        meta = json.loads(summary_file.read_text(encoding="utf-8"))
                        detected_vocab = int(meta.get("tokenizer_vocab_size", 0) or 0)
                        if detected_vocab > 0:
                            break
                    except Exception:
                        pass
                tok_file = sdir / "tokenizer.json"
                if tok_file.exists() and detected_vocab <= 0:
                    try:
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
                # Align with clean multiple of 256 for Tensor Core efficiency
                min_safe_vocab = ((max_token_id + 1 + 255) // 256) * 256
                # Standard LLaMA / Mistral 32k tokenizer alignment
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
                model_cfg["vocab_size"] = vocab_size

            context_length = int(model_cfg.get("context_length", 512))
            batch_size = int(training_cfg.get("batch_size", 4))
            lr = float(training_cfg.get("learning_rate", 3e-4))

            dataset = StandaloneTokenDataset(token_array, context_length, shard_idx, total_shards, vocab_size=vocab_size)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            dataloader_iter = iter(dataloader)

            model = build_worker_model(model_cfg, self.device_str)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            self.log(f"Initialized model for {self.worker_id} on {self.device_str}. Ready to train.")
            self._heartbeat(status="READY", current_job_id=job_id)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self.log(f"Preparation failed for job {job_id}:\n{tb}", level="ERROR")
            self._heartbeat(status="ERROR", current_job_id=job_id)
            return

        max_rounds = int(job.get("max_rounds", 10))
        sync_steps = int(job.get("sync_interval_steps", 250))
        cur_round = int(job.get("current_round", 0))

        # Mixed precision (AMP) configuration
        is_cuda = self.device_str.startswith("cuda") and torch.cuda.is_available()
        precision = str(training_cfg.get("precision", "fp16")).lower()
        use_amp = bool(training_cfg.get("use_amp", True))
        amp_enabled = is_cuda and use_amp and (precision in ("fp16", "bf16"))
        amp_dtype = torch.bfloat16 if (precision == "bf16" and torch.cuda.is_bf16_supported()) else torch.float16
        scaler = None
        if amp_enabled and amp_dtype == torch.float16:
            if not hasattr(self, "_scaler") or self._scaler is None:
                self._scaler = torch.amp.GradScaler('cuda')
            scaler = self._scaler

        try:
            while cur_round < max_rounds:
                if self.bus.is_stopped(job_id):
                    self.log(f"Job {job_id} stopped.")
                    break

                cmd = self.bus.get_worker_command(self.worker_id)
                if cmd == "STOP":
                    self.log(f"Received STOP command during job {job_id}.")
                    self.bus.set_worker_command(self.worker_id, None)
                    self._heartbeat(status="OFFLINE", current_job_id=None)
                    release_singleton_lock(self.device_tag)
                    sys.exit(0)
                elif cmd == "RESTART":
                    self.log(f"Received RESTART command during job {job_id}.")
                    self._restart_process()

                # Handle cooperative pause
                while self.bus.is_paused(job_id):
                    self._heartbeat(status="PAUSED", current_job_id=job_id)
                    time.sleep(1.0)
                    if self.bus.is_stopped(job_id):
                        break

                # Dynamic shard reassignment: verify active worker pool and take over dropped worker slots
                new_shard_idx, new_total_shards = self.bus.get_worker_shard_assignment(job_id, self.worker_id, cur_round)
                if new_shard_idx != shard_idx or new_total_shards != total_shards:
                    self.log(f"Active workers changed. Reallocating shard {new_shard_idx + 1}/{new_total_shards} for round {cur_round}.")
                    shard_idx, total_shards = new_shard_idx, new_total_shards
                    dataset = StandaloneTokenDataset(token_array, context_length, shard_idx, total_shards, vocab_size=vocab_size)
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                    dataloader_iter = iter(dataloader)

                # Sync from global model if round > 0
                if cur_round > 0:
                    self.log(f"Waiting for global averaged model round {cur_round - 1}...")
                    while not self.bus.is_global_weights_ready(job_id, cur_round - 1):
                        if self.bus.is_stopped(job_id):
                            return
                        self._heartbeat(status="SYNC_WAIT", current_job_id=job_id)
                        time.sleep(poll_interval)

                    global_state = self.bus.load_global_weights(job_id, cur_round - 1, device=self.device_str)
                    model.load_state_dict(global_state)
                    self.log(f"Loaded round {cur_round - 1} global weights into {self.device_str}.")

                # Local SGD training steps
                self.log(f"Training round {cur_round}/{max_rounds} ({sync_steps} local steps on {self.device_str}, AMP: {amp_enabled})...")
                model.train()
                self._heartbeat(status="TRAINING", current_job_id=job_id)

                start_time = time.time()
                tokens_processed = 0
                step_losses = []
                for step_idx in range(sync_steps):
                    if step_idx % 5 == 0:
                        cmd = self.bus.get_worker_command(self.worker_id)
                        if cmd == "STOP":
                            self.log("Received STOP command during training. Shutting down...")
                            self.bus.set_worker_command(self.worker_id, None)
                            self._heartbeat(status="OFFLINE", current_job_id=None)
                            release_singleton_lock(self.device_tag)
                            sys.exit(0)
                        elif cmd == "RESTART":
                            self.log("Received RESTART command during training. Restarting process...")
                            self._restart_process()
                    if step_idx % 20 == 0:
                        self._heartbeat(status="TRAINING", current_job_id=job_id)
                    if self.bus.is_stopped(job_id):
                        self.log(f"Job {job_id} stopped. Aborting training loop.")
                        return
                    try:
                        batch = next(dataloader_iter)
                    except StopIteration:
                        dataloader_iter = iter(dataloader)
                        batch = next(dataloader_iter)

                    x, y = batch
                    tokens_processed += int(x.numel())
                    x = torch.clamp(x, 0, vocab_size - 1)
                    y = torch.clamp(y, 0, vocab_size - 1)
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

                    step_losses.append(float(loss.item()))

                elapsed = max(time.time() - start_time, 1e-4)
                tokens_per_sec = tokens_processed / elapsed
                avg_loss = sum(step_losses) / max(len(step_losses), 1)
                self.log(f"Finished round {cur_round} (avg loss: {avg_loss:.4f}, speed: {tokens_per_sec:.0f} tok/s). Depositing weights & telemetry...")

                # Save local weights
                self._heartbeat(status="DEPOSITING", current_job_id=job_id)
                self.bus.save_worker_weights(
                    job_id=job_id,
                    round_num=cur_round,
                    worker_id=self.worker_id,
                    state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                )

                # Save local telemetry
                self.bus.save_worker_telemetry(
                    job_id=job_id,
                    round_num=cur_round,
                    worker_id=self.worker_id,
                    telemetry={
                        "worker_id": self.worker_id,
                        "round": cur_round,
                        "steps_completed": sync_steps,
                        "avg_loss": round(avg_loss, 4),
                        "tokens_processed": tokens_processed,
                        "tokens_per_sec": round(tokens_per_sec, 1),
                        "timestamp": time.time(),
                    },
                )

                # Wait for coordinator to publish global model
                self.log(f"Deposited weights for round {cur_round}. Waiting for coordinator synchronization...")
                while not self.bus.is_global_weights_ready(job_id, cur_round):
                    if self.bus.is_stopped(job_id):
                        return
                    self._heartbeat(status="SYNC_WAIT", current_job_id=job_id)
                    time.sleep(poll_interval)

                # Load synchronized global model weights into local model for next round
                global_weights = self.bus.load_global_weights(job_id, cur_round, device=self.device_str)
                model.load_state_dict(global_weights)
                self.log(f"Successfully loaded averaged global weights for round {cur_round}.")

                cur_round += 1

            self.log(f"Job {job_id} finished. Returning to IDLE.")
            self._heartbeat(status="IDLE", current_job_id=None)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self.log(f"Training failed for job {job_id}:\n{tb}", level="ERROR")
            self._heartbeat(status="ERROR", current_job_id=job_id)
            return
        finally:
            # Explicitly free model weights, optimizer states, and CUDA memory
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
            global_weights = None
            batch = None
            x = None
            y = None
            logits = None
            loss = None

            import gc
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

            try:
                cmd = self.bus.get_worker_command(self.worker_id)
                if cmd != "STOP":
                    self._heartbeat(status="IDLE", current_job_id=None)
            except Exception:
                pass


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
    parser.add_argument("--detach", "--background", action="store_true", help="Launch worker in the background detached from this terminal")
    args = parser.parse_args()

    shared_dir = args.shared_dir or os.environ.get("LLM_SHARED_PATH") or os.environ.get("LLM_SHARED_DIR")

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

    # Handle detached background launch
    if getattr(args, "detach", False):
        flags = 0
        if sys.platform == "win32":
            flags = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            )
        worker_exe = get_worker_executable()
        script_path = str(Path(__file__).resolve())
        cmd = [worker_exe, script_path] + [arg for arg in sys.argv[1:] if arg not in ("--detach", "--background")]
        log_dir = Path(shared_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        wid_label = args.worker_id or f"{socket.gethostname()}_{get_device_tag(args.device or 'auto')}"
        log_file_path = log_dir / f"{wid_label}.log"
        log_out = open(log_file_path, "a", encoding="utf-8")

        proc = subprocess.Popen(
            cmd,
            stdout=log_out,
            stderr=log_out,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
            close_fds=True,
        )
        print(f"[ClusterWorker] Started detached background worker '{wid_label}' (PID: {proc.pid})")
        print(f"[ClusterWorker] Output is being logged to: {log_file_path.resolve()}")
        print("[ClusterWorker] You can now safely close this PowerShell window.")
        return 0

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
