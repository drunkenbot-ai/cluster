"""Centralized storage bus and SQLite coordination for port-blocked clusters.

Operates over shared network filesystems (SMB, NFS, NAS) using:
1. SQLite in network-compatible TRUNCATE journal mode with busy timeouts and ephemeral connections.
2. Atomic filesystem renames for binary tensor checkpoints.
3. Signal file detection for sub-second cooperative pause and stop actions.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Generator, Optional

import torch


def safe_torch_load(
    path: Path | str,
    device: str = "cpu",
    max_retries: int = 5,
    retry_delay: float = 0.5,
) -> Any:
    """Safely load PyTorch checkpoint with retry and local temporary caching to prevent Windows SMB Errno 22.

    Windows SMB network redirectors fail with ERROR_INVALID_PARAMETER (Errno 22) when PyTorch's C++
    zip reader attempts random seeks across network mapped drives. Copying to a local temp file
    first completely resolves this issue and shields against mid-flush SMB race conditions.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    is_net = False
    p_str = str(p)
    if p_str.startswith(("\\\\", "//")):
        is_net = True
    elif len(p_str) >= 2 and p_str[1] == ":" and sys.platform == "win32":
        try:
            import ctypes
            drive_type = ctypes.windll.kernel32.GetDriveTypeW(f"{p_str[:2].upper()}\\")
            is_net = (drive_type == 4)  # DRIVE_REMOTE
        except Exception:
            is_net = False

    last_err = None
    for attempt in range(max_retries):
        try:
            if is_net:
                temp_fd, temp_path = tempfile.mkstemp(suffix=".pt", prefix=f".llm_load_{os.getpid()}_")
                os.close(temp_fd)
                try:
                    shutil.copyfile(str(p), temp_path)
                    data = torch.load(temp_path, map_location=device)
                    return data
                finally:
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
            else:
                return torch.load(str(p), map_location=device)
        except (OSError, RuntimeError) as exc:
            last_err = exc
            time.sleep(retry_delay * (attempt + 1))
        except Exception as exc:
            last_err = exc
            time.sleep(retry_delay * (attempt + 1))

    if is_net:
        return torch.load(str(p), map_location=device)
    if last_err:
        raise last_err
    return torch.load(str(p), map_location=device)


_DEFAULT_NOT_SET = object()


class ClusterStorageBus:
    """Manages SQLite state and file-based message exchange on central shared storage."""

    def __init__(self, shared_dir: Path | str) -> None:
        """Initialize the storage bus.

        Args:
            shared_dir: Path to the central shared network directory.
        """
        self.shared_dir = Path(shared_dir)
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.shared_dir / "cluster.db"
        self.jobs_dir = self.shared_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextlib.contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Ephemeral connection context manager with retries to handle network storage locking."""
        conn = None
        for attempt in range(8):
            try:
                conn = sqlite3.connect(
                    str(self.db_path),
                    timeout=30.0,
                    isolation_level="DEFERRED",
                )
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA busy_timeout = 30000;")
                conn.execute("PRAGMA synchronous = NORMAL;")
                conn.execute("PRAGMA temp_store = MEMORY;")
                break
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
                err_msg = str(exc).lower()
                if "malformed" in err_msg or "file is not a database" in err_msg or "corrupt" in err_msg:
                    try:
                        self._recover_malformed_db(exc)
                        continue
                    except Exception:
                        pass
                if attempt == 7:
                    raise
                time.sleep(min(2.0, 0.05 * (1.6 ** attempt) + random.uniform(0.03, 0.15)))
        try:
            yield conn
            if conn and conn.in_transaction:
                conn.commit()
        except Exception:
            if conn:
                try:
                    if conn.in_transaction:
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

    def _recover_malformed_db(self, exc: Exception) -> bool:
        """Automatically recover when SQLite reports a malformed database disk image on network storage.

        Backs up the corrupted database, salvages any readable rows from intact tables,
        cleans up stale journal/WAL files, and initializes a clean new database.
        """
        print(f"[ClusterStorageBus] WARNING: Malformed SQLite database detected ({exc}). Attempting automatic self-healing recovery...")
        ts = int(time.time())
        corrupt_backup = self.shared_dir / f"cluster.db.corrupt_{ts}"
        salvaged_data: dict[str, list[dict[str, Any]]] = {}

        # 1. Attempt to salvage rows from undamaged tables via a read-only connection
        conn_old = None
        try:
            conn_old = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0)
            conn_old.row_factory = sqlite3.Row
            for tbl in ["workers", "jobs", "job_participants", "round_history"]:
                try:
                    cursor = conn_old.execute(f"SELECT * FROM {tbl};")
                    salvaged_data[tbl] = [dict(r) for r in cursor.fetchall()]
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            if conn_old is not None:
                try:
                    conn_old.close()
                except Exception:
                    pass
                del conn_old
            import gc
            gc.collect()

        # 2. Rename the corrupted database file out of the way
        try:
            if self.db_path.exists():
                moved = False
                for _ in range(5):
                    try:
                        shutil.move(str(self.db_path), str(corrupt_backup))
                        moved = True
                        break
                    except Exception:
                        import gc
                        gc.collect()
                        time.sleep(0.1)
                if not moved:
                    try:
                        shutil.copy2(str(self.db_path), str(corrupt_backup))
                        self.db_path.unlink(missing_ok=True)
                    except Exception as fallback_err:
                        print(f"[ClusterStorageBus] Could not archive corrupt database: {fallback_err}")
                        return False
            for ext in ["-journal", "-wal", "-shm"]:
                j_file = self.shared_dir / f"cluster.db{ext}"
                if j_file.exists():
                    try:
                        j_file.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception as move_err:
            print(f"[ClusterStorageBus] Could not archive corrupt database: {move_err}")
            return False

        # 3. Create a fresh clean database and restore schema
        try:
            self._init_db()
        except Exception as init_err:
            print(f"[ClusterStorageBus] Error initializing fresh database: {init_err}")
            return False

        # 4. Insert salvaged rows back into the clean database
        try:
            with self._connect() as conn_new:
                for tbl, rows in salvaged_data.items():
                    if not rows:
                        continue
                    cols = list(rows[0].keys())
                    placeholders = ", ".join(["?"] * len(cols))
                    col_names = ", ".join(cols)
                    stmt = f"INSERT OR REPLACE INTO {tbl} ({col_names}) VALUES ({placeholders});"
                    for row in rows:
                        try:
                            conn_new.execute(stmt, [row.get(c) for c in cols])
                        except Exception:
                            pass
                conn_new.commit()
        except Exception as restore_err:
            print(f"[ClusterStorageBus] Warning: Could not restore all salvaged rows: {restore_err}")

        print(f"[ClusterStorageBus] Database self-healing complete. Intact records restored; corrupt file saved to {corrupt_backup.name}.")
        return True

    def _run_with_retry(
        self,
        fn: Callable[[sqlite3.Connection], Any],
        max_retries: int = 10,
        default_on_error: Any = _DEFAULT_NOT_SET,
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
                err_msg = str(exc).lower()
                if "malformed" in err_msg or "file is not a database" in err_msg or "corrupt" in err_msg:
                    try:
                        if self._recover_malformed_db(exc):
                            with self._connect() as conn:
                                return fn(conn)
                    except Exception as rec_err:
                        print(f"[ClusterStorageBus] Database self-healing retry failed: {rec_err}")
                        if default_on_error is not _DEFAULT_NOT_SET:
                            return default_on_error
                        if silent:
                            return None
                        raise exc
                if attempt == max_retries - 1:
                    if default_on_error is not _DEFAULT_NOT_SET:
                        return default_on_error
                    if silent:
                        return None
                    raise
                backoff = min(2.5, 0.1 * (1.6 ** attempt) + random.uniform(0.05, 0.25))
                time.sleep(backoff)
        return default_on_error if default_on_error is not _DEFAULT_NOT_SET else None

    def _init_db(self) -> None:
        """Initialize database tables with network-compatible journal mode."""
        def _init(conn: sqlite3.Connection) -> None:
            try:
                conn.execute("PRAGMA journal_mode = TRUNCATE;")
                conn.execute("PRAGMA synchronous = NORMAL;")
                conn.execute("PRAGMA temp_store = MEMORY;")
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
                ("enabled", "INTEGER DEFAULT 1"),
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
                    sync_timeout_seconds REAL DEFAULT 180.0,
                    job_type TEXT DEFAULT 'pretrain',
                    base_checkpoint_path TEXT,
                    peft_method TEXT DEFAULT 'none',
                    lora_config TEXT,
                    created_at REAL,
                    updated_at REAL
                );
            """)
            for col_def in [
                ("job_type", "TEXT DEFAULT 'pretrain'"),
                ("base_checkpoint_path", "TEXT"),
                ("peft_method", "TEXT DEFAULT 'none'"),
                ("lora_config", "TEXT"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {col_def[0]} {col_def[1]};")
                except Exception:
                    pass
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
                CREATE TABLE IF NOT EXISTS round_history (
                    job_id TEXT,
                    round_number INTEGER,
                    participating_workers TEXT,
                    averaged_at REAL,
                    avg_loss REAL,
                    metrics TEXT,
                    PRIMARY KEY (job_id, round_number)
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_worker_logs_wid_id ON worker_logs(worker_id, id DESC);")
        self._run_with_retry(_init)

    # -------------------------------------------------------------------------
    # Worker Lifecycle & Heartbeats
    # -------------------------------------------------------------------------

    def register_worker(
        self,
        worker_id: str,
        hostname: str,
        gpu_name: str,
        vram_gb: float,
    ) -> None:
        """Register or update a compute worker node."""
        now = time.time()
        def _op(conn: sqlite3.Connection) -> None:
            conn.execute("""
                INSERT INTO workers (worker_id, hostname, gpu_name, vram_gb, status, last_heartbeat, created_at)
                VALUES (?, ?, ?, ?, 'INITIALIZING', ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    hostname = excluded.hostname,
                    gpu_name = excluded.gpu_name,
                    vram_gb = excluded.vram_gb,
                    status = CASE WHEN status = 'RESTARTING' THEN 'INITIALIZING' ELSE status END,
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
        """Update a worker's last heartbeat timestamp, status, and resource usage metrics."""
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
                    UPDATE workers
                    SET last_heartbeat = ?, status = ?, current_job_id = ?
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
                conn.execute("""
                    UPDATE workers
                    SET last_heartbeat = ?
                    WHERE worker_id = ?;
                """, (now, worker_id))
        self._run_with_retry(_op, silent=True)

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
            time_diff = now - last_hb
            stored_status = str(data.get("status") or "OFFLINE").upper()
            # Allow extended grace period (180s) for workers booting, restarting, or running hardware preflight
            thresh = 180.0 if stored_status in {"STARTING", "INITIALIZING", "RESTARTING", "PREFLIGHT"} else active_within_seconds
            # Allow bidirectional clock skew across networked workstations (up to thresh behind, or up to 5 min in future)
            is_fresh = (time_diff < thresh) and (time_diff > -300.0)
            data["enabled"] = bool(data.get("enabled", 1)) if data.get("enabled") is not None else True
            if not is_fresh or stored_status in {"OFFLINE", "STOPPED"}:
                data["is_online"] = False
                data["status"] = "OFFLINE"
            else:
                data["is_online"] = True
                if not data["enabled"]:
                    data["status"] = "DISABLED"
            results.append(data)
        return results

    def set_worker_status(self, worker_id: str, status: str) -> None:
        """Explicitly set a worker's status (e.g. 'OFFLINE', 'IDLE')."""
        self.heartbeat(worker_id, status=status, current_job_id=None)

    def set_worker_command(self, worker_id: str, command: Optional[str]) -> None:
        """Send a cooperative command signal (e.g. 'STOP', 'RESTART') to a worker."""
        def _op(conn: sqlite3.Connection) -> None:
            conn.execute("UPDATE workers SET command = ? WHERE worker_id = ?;", (command, worker_id))
        self._run_with_retry(_op, silent=True)

    def get_worker_command(self, worker_id: str) -> Optional[str]:
        """Check for pending cooperative commands for this worker."""
        def _op(conn: sqlite3.Connection) -> Optional[str]:
            cursor = conn.execute("SELECT command FROM workers WHERE worker_id = ?;", (worker_id,))
            row = cursor.fetchone()
            return row["command"] if row and row["command"] else None
        return self._run_with_retry(_op, default_on_error=None, silent=True)

    def set_worker_enabled(self, worker_id: str, enabled: bool) -> None:
        """Enable or disable a worker from claiming cluster jobs."""
        val = 1 if enabled else 0
        def _op(conn: sqlite3.Connection) -> None:
            try:
                conn.execute("ALTER TABLE workers ADD COLUMN enabled INTEGER DEFAULT 1;")
            except Exception:
                pass
            conn.execute(
                "UPDATE workers SET enabled = ?, status = CASE WHEN ? = 0 THEN 'DISABLED' ELSE 'IDLE' END WHERE worker_id = ?;",
                (val, val, worker_id),
            )
            if not enabled:
                conn.execute(
                    "UPDATE job_participants SET status = 'DROPPED' WHERE worker_id = ?;",
                    (worker_id,),
                )
        try:
            self._run_with_retry(_op, silent=True)
        except Exception:
            pass

    def is_worker_enabled(self, worker_id: str) -> bool:
        """Check whether a worker is enabled to claim cluster jobs."""
        def _op(conn: sqlite3.Connection) -> bool:
            try:
                cursor = conn.execute("SELECT enabled FROM workers WHERE worker_id = ?;", (worker_id,))
                row = cursor.fetchone()
                if row is not None and row[0] is not None:
                    return bool(row[0])
            except sqlite3.OperationalError as oe:
                if "no such column" in str(oe).lower():
                    return True
                raise
            return True
        try:
            return self._run_with_retry(_op, default_on_error=True, silent=True)
        except Exception:
            return True

    def delete_worker(self, worker_id: str) -> bool:
        """Delete a worker record from the database."""
        def _op(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute("DELETE FROM workers WHERE worker_id = ?;", (worker_id,))
            conn.execute("UPDATE job_participants SET status = 'DROPPED' WHERE worker_id = ?;", (worker_id,))
            return cursor.rowcount > 0
        return bool(self._run_with_retry(_op, default_on_error=False))

    def delete_offline_workers(self, stale_threshold_seconds: float = 60.0) -> int:
        """Delete all stale/offline workers from the database."""
        now = time.time()
        cutoff = now - stale_threshold_seconds
        def _op(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                "DELETE FROM workers WHERE status = 'OFFLINE' OR last_heartbeat < ?;",
                (cutoff,),
            )
            conn.execute("UPDATE job_participants SET status = 'DROPPED' WHERE worker_id NOT IN (SELECT worker_id FROM workers);")
            return cursor.rowcount
        return int(self._run_with_retry(_op, default_on_error=0))

    # -------------------------------------------------------------------------
    # Job Management & Scheduling
    # -------------------------------------------------------------------------

    def create_job(
        self,
        job_id: str,
        model_config: dict[str, Any],
        training_config: dict[str, Any],
        dataset_path: str,
        max_rounds: int = 10,
        sync_interval_steps: int = 250,
        min_workers: int = 1,
        sync_timeout_seconds: float = 1800.0,
        job_type: str = "pretrain",
        base_checkpoint_path: Optional[str] = None,
        peft_method: str = "none",
        lora_config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Create a new distributed training job."""
        now = time.time()
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "signals").mkdir(parents=True, exist_ok=True)
        (job_dir / "rounds").mkdir(parents=True, exist_ok=True)

        staged_base_path = None
        if base_checkpoint_path and os.path.exists(base_checkpoint_path):
            staged_base = job_dir / "base_model.pt"
            try:
                if not staged_base.exists() or staged_base.stat().st_size != Path(base_checkpoint_path).stat().st_size:
                    shutil.copyfile(base_checkpoint_path, staged_base)
                staged_base_path = str(staged_base)
                # Copy lineage metadata alongside base model if present
                base_dir = Path(base_checkpoint_path).parent
                for meta_name in ("model_lineage.json", "training_summary.json"):
                    src_m = base_dir / meta_name
                    dst_m = job_dir / meta_name
                    if src_m.exists() and (not dst_m.exists() or dst_m.stat().st_size != src_m.stat().st_size):
                        try:
                            shutil.copyfile(src_m, dst_m)
                        except Exception:
                            pass
            except Exception:
                pass

        # Guarantee tokenizer.json is copied to job_dir and checkpoints/ for both pretrain and fine_tune
        ckpt_dir = self.get_checkpoints_dir(job_id)
        tok_src = None
        if base_checkpoint_path and os.path.exists(base_checkpoint_path):
            cand_tok = Path(base_checkpoint_path).parent / "tokenizer.json"
            if cand_tok.exists():
                tok_src = cand_tok
        if not tok_src and dataset_path:
            ds_p = Path(dataset_path)
            cand_tok = ds_p.parent / "tokenizer.json" if ds_p.is_file() else ds_p / "tokenizer.json"
            if cand_tok.exists():
                tok_src = cand_tok
            elif (self.shared_dir / "tokenizer.json").exists():
                tok_src = self.shared_dir / "tokenizer.json"

        if tok_src and tok_src.exists():
            for dst_t in (job_dir / "tokenizer.json", ckpt_dir / "tokenizer.json"):
                try:
                    if not dst_t.exists() or dst_t.stat().st_size != tok_src.stat().st_size:
                        shutil.copyfile(tok_src, dst_t)
                except Exception:
                    pass

        def _op(conn: sqlite3.Connection) -> None:
            conn.execute("""
                INSERT INTO jobs (
                    job_id, status, model_config, training_config, dataset_path,
                    current_round, max_rounds, sync_interval_steps, min_workers,
                    sync_timeout_seconds, job_type, base_checkpoint_path,
                    peft_method, lora_config, created_at, updated_at
                )
                VALUES (?, 'QUEUED', ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                job_id,
                json.dumps(model_config, default=str),
                json.dumps(training_config, default=str),
                str(dataset_path),
                max_rounds,
                sync_interval_steps,
                min_workers,
                sync_timeout_seconds,
                job_type,
                staged_base_path or (str(base_checkpoint_path) if base_checkpoint_path else None),
                peft_method,
                json.dumps(lora_config, default=str) if lora_config else None,
                now,
                now,
            ))
        self._run_with_retry(_op)

    def load_base_model_weights(
        self,
        job_id: str,
        device: str = "cpu",
    ) -> Optional[dict[str, torch.Tensor]]:
        """Load staged base model weights for fine-tuning."""
        candidate = self.jobs_dir / job_id / "base_model.pt"
        if not candidate.exists():
            job = self.get_job(job_id)
            if job and job.get("base_checkpoint_path"):
                alt = Path(job["base_checkpoint_path"])
                if alt.exists():
                    candidate = alt
        if candidate.exists():
            obj = safe_torch_load(candidate, device=device)
            if isinstance(obj, dict):
                for k in ("model_state_dict", "state_dict", "model"):
                    if k in obj and isinstance(obj[k], dict):
                        return obj[k]
            return obj
        return None

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Retrieve details for a specific job."""
        def _op(conn: sqlite3.Connection) -> Optional[dict[str, Any]]:
            cursor = conn.execute("SELECT * FROM jobs WHERE job_id = ?;", (job_id,))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            res["model_config"] = json.loads(res["model_config"])
            res["training_config"] = json.loads(res["training_config"])
            if res.get("lora_config"):
                try:
                    res["lora_config"] = json.loads(res["lora_config"])
                except Exception:
                    pass
            return res
        return self._run_with_retry(_op, default_on_error=None, silent=True)

    def touch_job(self, job_id: str) -> None:
        """Update job updated_at timestamp to signal active coordinator heartbeat."""
        now = time.time()
        def _op(conn: sqlite3.Connection) -> None:
            conn.execute("UPDATE jobs SET updated_at = ? WHERE job_id = ?;", (now, job_id))
        self._run_with_retry(_op, silent=True)

    def get_active_job(self, max_stale_seconds: float = 3600.0) -> Optional[dict[str, Any]]:
        """Get the currently active or queued job, automatically sweeping stale uncoordinated jobs."""
        now = time.time()
        def _op(conn: sqlite3.Connection) -> Optional[dict[str, Any]]:
            # Auto-sweep stale uncoordinated jobs whose updated_at has lapsed
            if max_stale_seconds > 0:
                cursor = conn.execute(
                    """
                    SELECT job_id FROM jobs
                    WHERE status IN ('RUNNING', 'QUEUED')
                      AND (? - updated_at) > ?;
                    """,
                    (now, max_stale_seconds),
                )
                swept_rows = cursor.fetchall()
                if swept_rows:
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'STOPPED', updated_at = ?
                        WHERE status IN ('RUNNING', 'QUEUED')
                          AND (? - updated_at) > ?;
                        """,
                        (now, now, max_stale_seconds),
                    )
                    for sr in swept_rows:
                        try:
                            s_dir = self.jobs_dir / sr["job_id"] / "signals"
                            s_dir.mkdir(parents=True, exist_ok=True)
                            (s_dir / "stop.sig").touch()
                        except Exception:
                            pass
            cursor = conn.execute(
                "SELECT * FROM jobs WHERE status IN ('RUNNING', 'QUEUED', 'PAUSED') ORDER BY created_at DESC LIMIT 1;"
            )
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            res["model_config"] = json.loads(res["model_config"])
            res["training_config"] = json.loads(res["training_config"])
            if res.get("lora_config") and isinstance(res["lora_config"], str):
                try:
                    res["lora_config"] = json.loads(res["lora_config"])
                except Exception:
                    pass
            return res
        return self._run_with_retry(_op, default_on_error=None, silent=True)

    def set_job_status(self, job_id: str, status: str) -> None:
        """Update job status and manage signal files."""
        now = time.time()
        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?;",
                (status, now, job_id),
            )
        self._run_with_retry(_op)

        job_dir = self.jobs_dir / job_id
        signals_dir = job_dir / "signals"
        signals_dir.mkdir(parents=True, exist_ok=True)

        if status == "PAUSED":
            (signals_dir / "pause.sig").touch()
            stop_sig = signals_dir / "stop.sig"
            if stop_sig.exists():
                stop_sig.unlink(missing_ok=True)
        elif status == "RUNNING":
            pause_sig = signals_dir / "pause.sig"
            if pause_sig.exists():
                pause_sig.unlink(missing_ok=True)
            stop_sig = signals_dir / "stop.sig"
            if stop_sig.exists():
                stop_sig.unlink(missing_ok=True)
        elif status in {"STOPPING", "STOPPED", "FAILED", "COMPLETED"}:
            (signals_dir / "stop.sig").touch()

    def advance_job_round(self, job_id: str, new_round: int) -> None:
        """Advance job to the next synchronization round."""
        now = time.time()
        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE jobs SET current_round = ?, updated_at = ? WHERE job_id = ?;",
                (new_round, now, job_id),
            )
        self._run_with_retry(_op)

    def list_all_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve all jobs ordered by created_at DESC."""
        def _op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            cursor = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?;",
                (limit,),
            )
            rows = []
            for r in cursor.fetchall():
                d = dict(r)
                try:
                    d["model_config"] = json.loads(d["model_config"])
                except Exception:
                    d["model_config"] = {}
                try:
                    d["training_config"] = json.loads(d["training_config"])
                except Exception:
                    d["training_config"] = {}
                rows.append(d)
            return rows
        return self._run_with_retry(_op, default_on_error=[])

    def get_job_tasks(self, job_id: str) -> list[dict[str, Any]]:
        """Retrieve detailed tasks/shards and worker allocations for a job."""
        def _op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            cursor = conn.execute(
                """
                SELECT p.job_id, p.worker_id, p.shard_index, p.total_shards,
                       p.last_synced_round, p.status as participant_status,
                       w.hostname, w.gpu_name, w.status as worker_status,
                       w.vram_used_gb, w.vram_gb, w.cpu_percent
                FROM job_participants p
                LEFT JOIN workers w ON p.worker_id = w.worker_id
                WHERE p.job_id = ?
                ORDER BY p.shard_index ASC;
                """,
                (job_id,),
            )
            return [dict(r) for r in cursor.fetchall()]

        rows = self._run_with_retry(_op, default_on_error=[])
        # Read worker telemetry files from disk outside of SQLite connection to avoid holding lock
        for task in rows:
            round_num = task.get("last_synced_round", 0)
            round_dir = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}"
            telem_file = round_dir / f"{task['worker_id']}_telemetry.json"
            if telem_file.exists():
                try:
                    task["telemetry"] = json.loads(telem_file.read_text(encoding="utf-8"))
                except Exception:
                    task["telemetry"] = {}
            else:
                task["telemetry"] = {}
        return rows

    def requeue_job(self, job_id: str, reset_rounds: bool = False) -> bool:
        """Re-queue an existing job to allow resuming or re-running."""
        now = time.time()
        def _op(conn: sqlite3.Connection) -> bool:
            if reset_rounds:
                conn.execute(
                    "UPDATE jobs SET status = 'QUEUED', current_round = 0, updated_at = ? WHERE job_id = ?;",
                    (now, job_id),
                )
                conn.execute(
                    "UPDATE job_participants SET last_synced_round = 0 WHERE job_id = ?;",
                    (job_id,),
                )
                conn.execute("DELETE FROM round_history WHERE job_id = ?;", (job_id,))
            else:
                conn.execute(
                    "UPDATE jobs SET status = 'QUEUED', updated_at = ? WHERE job_id = ?;",
                    (now, job_id),
                )
            return True
        success = bool(self._run_with_retry(_op, default_on_error=False))
        if success:
            signals_dir = self.jobs_dir / job_id / "signals"
            for sig in ["stop.sig", "pause.sig"]:
                p = signals_dir / sig
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass
            if reset_rounds:
                rounds_dir = self.jobs_dir / job_id / "rounds"
                if rounds_dir.exists():
                    try:
                        import shutil
                        shutil.rmtree(rounds_dir, ignore_errors=True)
                        rounds_dir.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass
        return success

    def delete_job(self, job_id: str) -> bool:
        """Delete a job record and its artifacts from shared storage."""
        def _op(conn: sqlite3.Connection) -> bool:
            conn.execute("DELETE FROM jobs WHERE job_id = ?;", (job_id,))
            conn.execute("DELETE FROM job_participants WHERE job_id = ?;", (job_id,))
            conn.execute("DELETE FROM round_history WHERE job_id = ?;", (job_id,))
            return True
        try:
            success = bool(self._run_with_retry(_op, default_on_error=False, silent=True))
        except Exception:
            success = True
        job_dir = self.jobs_dir / job_id
        if job_dir.exists():
            try:
                import shutil
                shutil.rmtree(job_dir, ignore_errors=True)
            except Exception:
                pass
        return True

    # -------------------------------------------------------------------------
    # Signals (Fast Local Check without DB query)
    # -------------------------------------------------------------------------

    def is_paused(self, job_id: str) -> bool:
        """Check if pause signal file exists or job status is PAUSED in SQLite."""
        if (self.jobs_dir / job_id / "signals" / "pause.sig").exists():
            return True
        def _op(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute("SELECT status FROM jobs WHERE job_id = ?;", (job_id,))
            row = cursor.fetchone()
            return bool(row and str(row["status"]).upper() == "PAUSED")
        return bool(self._run_with_retry(_op, default_on_error=False, silent=True))

    def is_stopped(self, job_id: str) -> bool:
        """Check if stop signal file exists or job status is stopped/completed in SQLite."""
        if (self.jobs_dir / job_id / "signals" / "stop.sig").exists():
            return True
        def _op(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute("SELECT status FROM jobs WHERE job_id = ?;", (job_id,))
            row = cursor.fetchone()
            if row and str(row["status"]).upper() in {"STOPPED", "STOPPING", "FAILED", "COMPLETED"}:
                return True
            return False
        return bool(self._run_with_retry(_op, default_on_error=False, silent=True))

    # -------------------------------------------------------------------------
    # Sharding & Participation
    # -------------------------------------------------------------------------

    def claim_job_slot(self, job_id: str, worker_id: str) -> tuple[int, int]:
        """Claim a shard index for a job. Returns (shard_index, total_shards)."""
        def _op(conn: sqlite3.Connection) -> tuple[int, int]:
            # Guard: reject slot claiming if worker is degraded, incompatible, offline, or disabled
            cursor = conn.execute("SELECT status, enabled FROM workers WHERE worker_id = ?;", (worker_id,))
            w_row = cursor.fetchone()
            if w_row:
                if w_row["enabled"] is not None and not bool(w_row["enabled"]):
                    raise RuntimeError(f"Worker '{worker_id}' is DISABLED and cannot claim a job slot.")
                if str(w_row["status"]).upper() in {"DEGRADED", "INCOMPATIBLE", "OFFLINE", "DISABLED"}:
                    raise RuntimeError(f"Worker '{worker_id}' is in status '{w_row['status']}' and cannot claim a job slot.")

            # Check if worker already has a slot
            cursor = conn.execute(
                "SELECT shard_index, total_shards FROM job_participants WHERE job_id = ? AND worker_id = ?;",
                (job_id, worker_id),
            )
            existing = cursor.fetchone()
            if existing:
                return int(existing["shard_index"]), int(existing["total_shards"])

            # Count existing participants
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM job_participants WHERE job_id = ?;",
                (job_id,),
            )
            count = int(cursor.fetchone()["cnt"])
            shard_idx = count
            total = count + 1

            conn.execute("""
                INSERT INTO job_participants (job_id, worker_id, shard_index, total_shards, status)
                VALUES (?, ?, ?, ?, 'ACTIVE');
            """, (job_id, worker_id, shard_idx, total))

            # Update total_shards for all participants in this job
            conn.execute(
                "UPDATE job_participants SET total_shards = ? WHERE job_id = ?;",
                (total, job_id),
            )
            return shard_idx, total
        return self._run_with_retry(_op)

    def get_job_participants(self, job_id: str) -> list[dict[str, Any]]:
        """Get all active participants for a job."""
        def _op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            cursor = conn.execute(
                "SELECT * FROM job_participants WHERE job_id = ? AND status = 'ACTIVE' ORDER BY worker_id ASC;",
                (job_id,),
            )
            return [dict(r) for r in cursor.fetchall()]
        return self._run_with_retry(_op, default_on_error=[])

    def get_worker_shard_assignment(
        self,
        job_id: str,
        worker_id: str,
        round_num: int = 0,
    ) -> tuple[int, int]:
        """Dynamically return (shard_index, total_shards) based strictly on currently active workers.

        If a worker drops or crashes, the surviving active workers are re-indexed across
        0..M-1 so that 100% of the dataset token range is partitioned among active workers.
        """
        def _op(conn: sqlite3.Connection) -> tuple[int, int]:
            # Ensure this worker is registered as ACTIVE if not already
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

            # Retrieve all currently active participants sorted deterministically
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
        """Mark a worker as DROPPED so its workload is reallocated to remaining active workers."""
        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE job_participants SET status = 'DROPPED' WHERE job_id = ? AND worker_id = ?;",
                (job_id, worker_id),
            )
            # Rebalance total_shards across remaining active workers
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

    # -------------------------------------------------------------------------
    # Telemetry & Metrics Persistence
    # -------------------------------------------------------------------------

    def save_worker_telemetry(
        self,
        job_id: str,
        round_num: int,
        worker_id: str,
        telemetry: dict[str, Any],
    ) -> Path:
        """Save a worker's round telemetry (loss, tokens/sec, steps) to shared storage."""
        round_dir = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        target = round_dir / f"{worker_id}_telemetry.json"
        target.write_text(json.dumps(telemetry, indent=2, default=str), encoding="utf-8")
        return target

    def load_worker_telemetry(
        self,
        job_id: str,
        round_num: int,
        worker_id: str,
    ) -> Optional[dict[str, Any]]:
        """Load a worker's deposited round telemetry."""
        target = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}" / f"{worker_id}_telemetry.json"
        if not target.exists():
            return None
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return None

    def record_round_summary(
        self,
        job_id: str,
        round_num: int,
        participating_workers: list[str],
        avg_loss: float,
        metrics: dict[str, Any],
    ) -> None:
        """Record global round aggregation telemetry into SQLite round_history."""
        now = time.time()
        def _op(conn: sqlite3.Connection) -> None:
            conn.execute("""
                INSERT OR REPLACE INTO round_history (job_id, round_number, participating_workers, averaged_at, avg_loss, metrics)
                VALUES (?, ?, ?, ?, ?, ?);
            """, (
                job_id,
                round_num,
                json.dumps(participating_workers, default=str),
                now,
                avg_loss,
                json.dumps(metrics, default=str),
            ))
        self._run_with_retry(_op, silent=True)

    def get_all_round_history(self, job_id: str) -> list[dict[str, Any]]:
        """Fetch all recorded round history summaries for a job."""
        def _op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            cursor = conn.execute(
                "SELECT * FROM round_history WHERE job_id = ? ORDER BY round_number ASC;",
                (job_id,),
            )
            rows = []
            for r in cursor.fetchall():
                d = dict(r)
                d["participating_workers"] = json.loads(d["participating_workers"]) if d.get("participating_workers") else []
                d["metrics"] = json.loads(d["metrics"]) if d.get("metrics") else {}
                rows.append(d)
            return rows
        return self._run_with_retry(_op, default_on_error=[])

    # Alias for convenience
    get_round_history = get_all_round_history

    def get_latest_rounds_for_all_jobs(self) -> dict[str, dict[str, Any]]:
        """Fetch the latest recorded round history summary for each job in a single fast query."""
        def _op(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
            cursor = conn.execute("""
                SELECT rh.job_id, rh.round_number, rh.avg_loss, rh.metrics
                FROM round_history rh
                INNER JOIN (
                    SELECT job_id, MAX(round_number) AS max_rnd
                    FROM round_history
                    GROUP BY job_id
                ) latest ON rh.job_id = latest.job_id AND rh.round_number = latest.max_rnd;
            """)
            result = {}
            for r in cursor.fetchall():
                d = dict(r)
                try:
                    d["metrics"] = json.loads(d["metrics"]) if d.get("metrics") else {}
                except Exception:
                    d["metrics"] = {}
                result[d["job_id"]] = d
            return result
        return self._run_with_retry(_op, default_on_error={})

    # -------------------------------------------------------------------------
    # Binary Tensor Checkpoint Exchange (Atomic Save / Load)
    # -------------------------------------------------------------------------

    def save_worker_weights(
        self,
        job_id: str,
        round_num: int,
        worker_id: str,
        state_dict: dict[str, torch.Tensor],
    ) -> Path:
        """Atomically save local worker weights for a round."""
        round_dir = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        target = round_dir / f"{worker_id}.pt"
        tmp_target = round_dir / f"{worker_id}.pt.tmp"
        ready_flag = round_dir / f"{worker_id}.ready"

        torch.save(state_dict, tmp_target)
        tmp_target.replace(target)
        ready_flag.touch()
        return target

    def is_worker_weights_ready(self, job_id: str, round_num: int, worker_id: str) -> bool:
        """Check whether a worker has finished depositing weights for a round."""
        round_dir = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}"
        return (round_dir / f"{worker_id}.ready").exists() and (round_dir / f"{worker_id}.pt").exists()

    def get_ready_workers_for_round(self, job_id: str, round_num: int) -> list[str]:
        """List all worker IDs that have deposited weights for a round."""
        round_dir = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}"
        if not round_dir.exists():
            return []
        ready_flags = round_dir.glob("*.ready")
        return [f.stem for f in ready_flags if f.stem != "global_model" and (round_dir / f"{f.stem}.pt").exists()]

    def load_worker_weights(
        self,
        job_id: str,
        round_num: int,
        worker_id: str,
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Load deposited weights from a specific worker."""
        weight_path = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}" / f"{worker_id}.pt"
        return safe_torch_load(weight_path, device=device)

    def save_global_weights(
        self,
        job_id: str,
        round_num: int,
        state_dict: dict[str, torch.Tensor],
    ) -> Path:
        """Atomically save the averaged global weights for a round."""
        round_dir = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        target = round_dir / "global_model.pt"
        tmp_target = round_dir / "global_model.pt.tmp"
        ready_flag = round_dir / "global_model.ready"

        torch.save(state_dict, tmp_target)
        tmp_target.replace(target)
        ready_flag.touch()
        return target

    def is_global_weights_ready(self, job_id: str, round_num: int) -> bool:
        """Check whether the averaged global model has been published."""
        round_dir = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}"
        return (round_dir / "global_model.ready").exists() and (round_dir / "global_model.pt").exists()

    def load_global_weights(
        self,
        job_id: str,
        round_num: int,
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Load the averaged global weights for a round."""
        weight_path = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}" / "global_model.pt"
        return safe_torch_load(weight_path, device=device)





    # -------------------------------------------------------------------------
    # Durable Checkpoints Storage
    # -------------------------------------------------------------------------

    def get_checkpoints_dir(self, job_id: str) -> Path:
        """Return the checkpoints directory on central shared storage for a job."""
        ckpt_dir = self.jobs_dir / job_id / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir

    def purge_round_worker_weights(self, job_id: str, round_num: int) -> int:
        """Immediately delete deposited worker weight files for a completed round.

        Once the coordinator has averaged worker weights into global_model.pt,
        individual worker .pt files are obsolete and should be purged immediately
        to prevent consuming gigabytes/terabytes of storage.
        """
        round_dir = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}"
        if not round_dir.exists():
            return 0
        purged = 0
        for f in round_dir.glob("*.pt"):
            if f.name != "global_model.pt":
                try:
                    f.unlink(missing_ok=True)
                    purged += 1
                except Exception:
                    pass
        for f in round_dir.glob("*.pt.tmp"):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
        for f in round_dir.glob("*.ready"):
            if f.name != "global_model.ready":
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
        return purged

    def purge_stale_rounds(self, job_id: str, keep_last_rounds: int = 2) -> int:
        """Remove historical round folders older than the active rolling window."""
        rounds_dir = self.jobs_dir / job_id / "rounds"
        if not rounds_dir.exists():
            return 0
        round_dirs = sorted([d for d in rounds_dir.glob("round_*") if d.is_dir()])
        if len(round_dirs) <= keep_last_rounds:
            return 0
        purged = 0
        for old_dir in round_dirs[:-keep_last_rounds]:
            try:
                shutil.rmtree(old_dir, ignore_errors=True)
                purged += 1
            except Exception:
                pass
        return purged

    def save_checkpoint(
        self,
        job_id: str,
        step: int,
        state_dict: dict[str, torch.Tensor],
        is_final: bool = False,
        model_config: Optional[dict[str, Any]] = None,
        training_config: Optional[dict[str, Any]] = None,
        round_num: Optional[int] = None,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        peft_method: str = "none",
        lora_config: Optional[dict[str, Any]] = None,
        adapter_state_dict: Optional[dict[str, torch.Tensor]] = None,
        max_keep: int = 2,
    ) -> Path:
        """Save a durable, chat-ready checkpoint on central shared storage."""
        ckpt_dir = self.get_checkpoints_dir(job_id)
        job_dir = self.jobs_dir / job_id

        # Resolve model_config and training_config from database if not supplied
        if model_config is None or training_config is None:
            job = self.get_job(job_id)
            if job:
                if model_config is None:
                    model_config = job.get("model_config") or {}
                if training_config is None:
                    training_config = job.get("training_config") or {}
                if lora_config is None:
                    lora_config = job.get("lora_config")
                if peft_method == "none":
                    peft_method = str(job.get("peft_method") or "none").lower()

        payload: dict[str, Any] = {
            "artifact_type": "inference" if is_final else "resume",
            "model_config": model_config or {},
            "training_config": training_config or {},
            "global_step": step,
            "round": round_num if round_num is not None else 0,
            "train_loss": train_loss if train_loss is not None else 0.0,
            "val_loss": val_loss,
            "model_state_dict": {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in state_dict.items()
            },
        }
        if peft_method == "lora":
            payload["peft_method"] = "lora"
            if lora_config:
                payload["lora_config"] = lora_config
            if adapter_state_dict:
                payload["adapter_state_dict"] = {
                    k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in adapter_state_dict.items()
                }

        target = ckpt_dir / f"checkpoint_step_{step:06d}.pt"
        tmp_target = ckpt_dir / f"checkpoint_step_{step:06d}.pt.tmp"
        torch.save(payload, tmp_target)
        tmp_target.replace(target)

        # Update latest_checkpoint.pt
        latest_target = ckpt_dir / "latest_checkpoint.pt"
        latest_tmp = ckpt_dir / "latest_checkpoint.pt.tmp"
        torch.save(payload, latest_tmp)
        latest_tmp.replace(latest_target)

        if is_final:
            final_target = ckpt_dir / "final_model.pt"
            final_tmp = ckpt_dir / "final_model.pt.tmp"
            torch.save(payload, final_tmp)
            final_tmp.replace(final_target)

            # Also provide aliases for convenience and backward compatibility
            for alias_name, alias_dir in (("final_model.pt", job_dir), ("model.pt", ckpt_dir), ("model.pt", job_dir)):
                try:
                    alias_path = alias_dir / alias_name
                    shutil.copyfile(final_target, alias_path)
                except Exception:
                    pass

        # Prune older step checkpoints if requested
        if max_keep > 0:
            step_files = sorted(ckpt_dir.glob("checkpoint_step_*.pt"))
            if len(step_files) > max_keep:
                for old_f in step_files[:-max_keep]:
                    try:
                        old_f.unlink(missing_ok=True)
                    except Exception:
                        pass

        return target

    def load_latest_checkpoint(
        self,
        job_id: str,
        device: str = "cpu",
    ) -> Optional[dict[str, torch.Tensor]]:
        """Load the latest checkpoint from central shared storage if available."""
        ckpt_dir = self.get_checkpoints_dir(job_id)
        latest = ckpt_dir / "latest_checkpoint.pt"
        if latest.exists():
            obj = safe_torch_load(latest, device=device)
            if isinstance(obj, dict) and "model_state_dict" in obj:
                return obj["model_state_dict"]
            return obj
        final = ckpt_dir / "final_model.pt"
        if final.exists():
            obj = safe_torch_load(final, device=device)
            if isinstance(obj, dict) and "model_state_dict" in obj:
                return obj["model_state_dict"]
            return obj
        return None

    # -------------------------------------------------------------------------
    # Worker Logging to SQLite
    # -------------------------------------------------------------------------

    def write_worker_logs(self, entries: list[tuple[str, float, str, str]]) -> None:
        """Batch write worker log messages into SQLite bus.

        Args:
            entries: List of (worker_id, timestamp, level, message) tuples.
        """
        if not entries:
            return
        def _op(conn: sqlite3.Connection) -> None:
            conn.executemany(
                "INSERT INTO worker_logs (worker_id, timestamp, level, message) VALUES (?, ?, ?, ?);",
                entries,
            )
        self._run_with_retry(_op, silent=True)

    def write_worker_log(self, worker_id: str, message: str, level: str = "INFO") -> None:
        """Write a single worker log message."""
        self.write_worker_logs([(worker_id, time.time(), level, message)])

    def get_worker_logs(self, worker_id: str, limit: int = 200) -> list[dict[str, Any]]:
        """Retrieve recent diagnostic logs for a specific worker node."""
        def _op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            cursor = conn.execute(
                "SELECT id, worker_id, timestamp, level, message FROM worker_logs WHERE worker_id = ? ORDER BY id DESC LIMIT ?;",
                (worker_id, limit),
            )
            rows = cursor.fetchall()
            return [dict(r) for r in reversed(rows)]
        return self._run_with_retry(_op, default_on_error=[])
