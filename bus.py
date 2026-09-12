"""Centralized storage bus and SQLite coordination for port-blocked clusters.

Operates over shared network filesystems (SMB, NFS, NAS) using:
1. SQLite in network-compatible TRUNCATE journal mode with busy timeouts and ephemeral connections.
2. Atomic filesystem renames for binary tensor checkpoints.
3. Signal file detection for sub-second cooperative pause and stop actions.
"""

from __future__ import annotations

import contextlib
import json
import random
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Generator, Optional

import torch


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
        for attempt in range(5):
            try:
                conn = sqlite3.connect(
                    str(self.db_path),
                    timeout=60.0,
                    isolation_level="DEFERRED",
                )
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
        """Initialize database tables with network-compatible journal mode."""
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
            is_fresh = (now - last_hb) < active_within_seconds
            stored_status = str(data.get("status") or "OFFLINE").upper()
            if not is_fresh or stored_status in {"OFFLINE", "STOPPED"}:
                data["is_online"] = False
                data["status"] = "OFFLINE"
            else:
                data["is_online"] = True
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

    def delete_worker(self, worker_id: str) -> bool:
        """Delete a worker record from the database."""
        def _op(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute("DELETE FROM workers WHERE worker_id = ?;", (worker_id,))
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
    ) -> None:
        """Create a new distributed training job."""
        now = time.time()
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "signals").mkdir(parents=True, exist_ok=True)
        (job_dir / "rounds").mkdir(parents=True, exist_ok=True)

        def _op(conn: sqlite3.Connection) -> None:
            conn.execute("""
                INSERT INTO jobs (
                    job_id, status, model_config, training_config, dataset_path,
                    current_round, max_rounds, sync_interval_steps, min_workers,
                    sync_timeout_seconds, created_at, updated_at
                )
                VALUES (?, 'QUEUED', ?, ?, ?, 0, ?, ?, ?, ?, ?, ?);
            """, (
                job_id,
                json.dumps(model_config, default=str),
                json.dumps(training_config, default=str),
                str(dataset_path),
                max_rounds,
                sync_interval_steps,
                min_workers,
                sync_timeout_seconds,
                now,
                now,
            ))
        self._run_with_retry(_op)

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
            rows = []
            for r in cursor.fetchall():
                task = dict(r)
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
                rows.append(task)
            return rows
        return self._run_with_retry(_op, default_on_error=[])

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
        success = bool(self._run_with_retry(_op, default_on_error=False))
        job_dir = self.jobs_dir / job_id
        if job_dir.exists():
            try:
                import shutil
                shutil.rmtree(job_dir, ignore_errors=True)
            except Exception:
                pass
        return success

    # -------------------------------------------------------------------------
    # Signals (Fast Local Check without DB query)
    # -------------------------------------------------------------------------

    def is_paused(self, job_id: str) -> bool:
        """Check if pause signal file exists."""
        return (self.jobs_dir / job_id / "signals" / "pause.sig").exists()

    def is_stopped(self, job_id: str) -> bool:
        """Check if stop signal file exists."""
        return (self.jobs_dir / job_id / "signals" / "stop.sig").exists()

    # -------------------------------------------------------------------------
    # Sharding & Participation
    # -------------------------------------------------------------------------

    def claim_job_slot(self, job_id: str, worker_id: str) -> tuple[int, int]:
        """Claim a shard index for a job. Returns (shard_index, total_shards)."""
        def _op(conn: sqlite3.Connection) -> tuple[int, int]:
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
        return [f.stem for f in ready_flags if (round_dir / f"{f.stem}.pt").exists()]

    def load_worker_weights(
        self,
        job_id: str,
        round_num: int,
        worker_id: str,
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Load deposited weights from a specific worker."""
        weight_path = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}" / f"{worker_id}.pt"
        return torch.load(weight_path, map_location=device)

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
        return torch.load(weight_path, map_location=device)

    # -------------------------------------------------------------------------
    # Durable Checkpoints Storage
    # -------------------------------------------------------------------------

    def get_checkpoints_dir(self, job_id: str) -> Path:
        """Return the checkpoints directory on central shared storage for a job."""
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
        """Save a durable checkpoint on central shared storage."""
        ckpt_dir = self.get_checkpoints_dir(job_id)
        target = ckpt_dir / f"checkpoint_step_{step:06d}.pt"
        tmp_target = ckpt_dir / f"checkpoint_step_{step:06d}.pt.tmp"
        torch.save(state_dict, tmp_target)
        tmp_target.replace(target)

        # Update latest_checkpoint.pt
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

    def load_latest_checkpoint(
        self,
        job_id: str,
        device: str = "cpu",
    ) -> Optional[dict[str, torch.Tensor]]:
        """Load the latest checkpoint from central shared storage if available."""
        ckpt_dir = self.get_checkpoints_dir(job_id)
        latest = ckpt_dir / "latest_checkpoint.pt"
        if latest.exists():
            return torch.load(latest, map_location=device)
        final = ckpt_dir / "final_model.pt"
        if final.exists():
            return torch.load(final, map_location=device)
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
