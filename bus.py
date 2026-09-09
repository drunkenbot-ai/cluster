"""Centralized storage bus and SQLite coordination for port-blocked clusters.

Operates over shared network filesystems (SMB, NFS, NAS) using:
1. SQLite in WAL mode with aggressive busy timeouts and ephemeral connections.
2. Atomic filesystem renames for binary tensor checkpoints.
3. Signal file detection for sub-second cooperative pause and stop actions.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Generator, Optional

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
        """Ephemeral connection context manager to minimize network drive lock duration."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level="DEFERRED",
        )
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
        """Initialize database tables with WAL mode."""
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
            try:
                conn.execute("ALTER TABLE round_history ADD COLUMN metrics TEXT;")
            except Exception:
                pass

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

    def heartbeat(
        self,
        worker_id: str,
        status: Optional[str] = None,
        current_job_id: Optional[str] = None,
    ) -> None:
        """Update a worker's last heartbeat timestamp and status."""
        now = time.time()
        with self._connect() as conn:
            if status is not None:
                conn.execute("""
                    UPDATE workers
                    SET last_heartbeat = ?, status = ?, current_job_id = ?
                    WHERE worker_id = ?;
                """, (now, status, current_job_id, worker_id))
            else:
                conn.execute("""
                    UPDATE workers
                    SET last_heartbeat = ?
                    WHERE worker_id = ?;
                """, (now, worker_id))

    def list_workers(self, active_within_seconds: float = 60.0) -> list[dict[str, Any]]:
        """List all workers, calculating online status based on heartbeat freshness."""
        now = time.time()
        with self._connect() as conn:
            cursor = conn.execute("SELECT * FROM workers ORDER BY created_at ASC;")
            rows = cursor.fetchall()

        results = []
        for r in rows:
            data = dict(r)
            is_online = (now - float(data.get("last_heartbeat") or 0)) < active_within_seconds
            data["is_online"] = is_online
            if not is_online:
                data["status"] = "OFFLINE"
            results.append(data)
        return results

    def set_worker_command(self, worker_id: str, command: Optional[str]) -> None:
        """Send a cooperative command signal (e.g. 'STOP', 'RESTART') to a worker."""
        with self._connect() as conn:
            conn.execute("UPDATE workers SET command = ? WHERE worker_id = ?;", (command, worker_id))

    def get_worker_command(self, worker_id: str) -> Optional[str]:
        """Check for pending cooperative commands for this worker."""
        with self._connect() as conn:
            cursor = conn.execute("SELECT command FROM workers WHERE worker_id = ?;", (worker_id,))
            row = cursor.fetchone()
            return row["command"] if row and row["command"] else None

    def delete_worker(self, worker_id: str) -> bool:
        """Delete a worker record from the database."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM workers WHERE worker_id = ?;", (worker_id,))
            return cursor.rowcount > 0

    def delete_offline_workers(self, stale_threshold_seconds: float = 60.0) -> int:
        """Delete all stale/offline workers from the database."""
        now = time.time()
        cutoff = now - stale_threshold_seconds
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM workers WHERE status = 'OFFLINE' OR last_heartbeat < ?;",
                (cutoff,),
            )
            return cursor.rowcount

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
        sync_timeout_seconds: float = 180.0,
    ) -> None:
        """Create a new distributed training job."""
        now = time.time()
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "signals").mkdir(parents=True, exist_ok=True)
        (job_dir / "rounds").mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.execute("""
                INSERT INTO jobs (
                    job_id, status, model_config, training_config, dataset_path,
                    current_round, max_rounds, sync_interval_steps, min_workers,
                    sync_timeout_seconds, created_at, updated_at
                )
                VALUES (?, 'QUEUED', ?, ?, ?, 0, ?, ?, ?, ?, ?, ?);
            """, (
                job_id,
                json.dumps(model_config),
                json.dumps(training_config),
                dataset_path,
                max_rounds,
                sync_interval_steps,
                min_workers,
                sync_timeout_seconds,
                now,
                now,
            ))

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Retrieve details for a specific job."""
        with self._connect() as conn:
            cursor = conn.execute("SELECT * FROM jobs WHERE job_id = ?;", (job_id,))
            row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        res["model_config"] = json.loads(res["model_config"])
        res["training_config"] = json.loads(res["training_config"])
        return res

    def get_active_job(self) -> Optional[dict[str, Any]]:
        """Get the currently active or queued job."""
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

    def set_job_status(self, job_id: str, status: str) -> None:
        """Update job status and manage signal files."""
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?;",
                (status, now, job_id),
            )

        job_dir = self.jobs_dir / job_id
        signals_dir = job_dir / "signals"
        signals_dir.mkdir(parents=True, exist_ok=True)

        if status == "PAUSED":
            (signals_dir / "pause.sig").touch()
        elif status == "RUNNING":
            pause_sig = signals_dir / "pause.sig"
            if pause_sig.exists():
                pause_sig.unlink()
        elif status in {"STOPPING", "STOPPED", "FAILED", "COMPLETED"}:
            (signals_dir / "stop.sig").touch()

    def advance_job_round(self, job_id: str, new_round: int) -> None:
        """Advance job to the next synchronization round."""
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET current_round = ?, updated_at = ? WHERE job_id = ?;",
                (new_round, now, job_id),
            )

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
        with self._connect() as conn:
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

    def get_job_participants(self, job_id: str) -> list[dict[str, Any]]:
        """Get all active participants for a job."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM job_participants WHERE job_id = ? AND status = 'ACTIVE' ORDER BY worker_id ASC;",
                (job_id,),
            )
            return [dict(r) for r in cursor.fetchall()]

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
        with self._connect() as conn:
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

    def mark_worker_dropped(self, job_id: str, worker_id: str, reason: str = "timeout") -> None:
        """Mark a worker as DROPPED so its workload is reallocated to remaining active workers."""
        with self._connect() as conn:
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
        target.write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
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
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO round_history (job_id, round_number, participating_workers, averaged_at, avg_loss, metrics)
                VALUES (?, ?, ?, ?, ?, ?);
            """, (
                job_id,
                round_num,
                json.dumps(participating_workers),
                now,
                avg_loss,
                json.dumps(metrics),
            ))

    def get_all_round_history(self, job_id: str) -> list[dict[str, Any]]:
        """Fetch all recorded round history summaries for a job."""
        with self._connect() as conn:
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
