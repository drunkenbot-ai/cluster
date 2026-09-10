"""Command Line Interface for Port-Blocked Distributed Training Clusters.

Provides subcommands for running workers, monitoring cluster status, and controlling jobs:
- `python -m cluster.cli worker --shared-dir /path/to/shared --worker-id worker-1`
- `python -m cluster.cli status --shared-dir /path/to/shared`
- `python -m cluster.cli submit --shared-dir /path/to/shared --job-id job-1 ...`
- `python -m cluster.cli stop --shared-dir /path/to/shared --job-id job-1`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .bus import ClusterStorageBus
from .worker import ClusterWorker


def cmd_worker(args: argparse.Namespace) -> int:
    """Run a persistent headless worker daemon."""
    shared_dir = Path(args.shared_dir)

    if getattr(args, "detach", False):
        import subprocess
        flags = 0
        if sys.platform == "win32":
            flags = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            )

        cmd = [sys.executable, "-m", "cluster.cli"] + [arg for arg in sys.argv[1:] if arg not in ("--detach", "--background")]

        log_dir = shared_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        wid_label = args.worker_id or "worker"
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
        print(f"[Worker] Started detached background worker '{wid_label}' (PID: {proc.pid})")
        print(f"[Worker] Output is being logged to: {log_file_path.resolve()}")
        print("[Worker] You can now safely close this PowerShell window.")
        return 0

    bus = ClusterStorageBus(shared_dir)
    worker = ClusterWorker(
        bus=bus,
        worker_id=args.worker_id,
        device=args.device,
        heartbeat_interval=args.heartbeat_interval,
    )
    print(f"[Worker] Started node {worker.worker_id} on {worker.device_str} ({worker.gpu_name}, {worker.vram_gb} GB VRAM)")
    print(f"[Worker] Connected to shared storage: {shared_dir.resolve()}")
    try:
        worker.run_daemon(poll_interval=args.poll_interval)
    except KeyboardInterrupt:
        print("\n[Worker] Stopping worker daemon gracefully...")
    finally:
        worker.stop()
        print("[Worker] Stopped.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Print current cluster worker fleet and active job telemetry."""
    shared_dir = Path(args.shared_dir)
    bus = ClusterStorageBus(shared_dir)

    print("=" * 70)
    print(f"CLUSTER STATUS — Storage: {shared_dir.resolve()}")
    print("=" * 70)

    workers = bus.list_workers(active_within_seconds=args.heartbeat_timeout)
    print(f"\nREGISTERED WORKERS ({len(workers)}):")
    if not workers:
        print("  (No workers registered)")
    else:
        print(f"  {'Worker ID':<22} {'Host':<15} {'Device/GPU':<20} {'Status':<10}")
        print("  " + "-" * 67)
        for w in workers:
            status = w.get("status", "UNKNOWN")
            gpu = f"{w.get('gpu_name', '')[:16]} ({w.get('vram_gb', 0)}G)"
            print(f"  {w['worker_id']:<22} {w.get('hostname', '')[:14]:<15} {gpu:<20} {status:<10}")

    active_job = bus.get_active_job()
    print("\nACTIVE JOB:")
    if not active_job:
        print("  (No active or queued jobs)")
    else:
        jid = active_job["job_id"]
        status = active_job["status"]
        cur_round = active_job.get("current_round", 0)
        max_rounds = active_job.get("max_rounds", 0)
        sync_steps = active_job.get("sync_interval_steps", 250)
        print(f"  Job ID:        {jid}")
        print(f"  Status:        {status}")
        print(f"  Round:         {cur_round} / {max_rounds} (Sync every {sync_steps} steps)")
        print(f"  Dataset:       {active_job.get('dataset_path')}")

        participants = bus.get_job_participants(jid)
        ready_workers = bus.get_ready_workers_for_round(jid, cur_round)
        print(f"  Participants:  {len(participants)} active nodes")
        print(f"  Ready Round {cur_round}: {len(ready_workers)} / {len(participants)} workers deposited weights")

    print("=" * 70)
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    """Submit a new distributed training job."""
    shared_dir = Path(args.shared_dir)
    bus = ClusterStorageBus(shared_dir)

    model_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "embedding_size": args.embedding_size,
        "head_count": args.head_count,
        "layer_count": args.layer_count,
    }
    training_config = {
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
    }

    job_id = args.job_id or f"job_{int(time.time())}"
    bus.create_job(
        job_id=job_id,
        model_config=model_config,
        training_config=training_config,
        dataset_path=str(Path(args.dataset_path).resolve()),
        max_rounds=args.max_rounds,
        sync_interval_steps=args.sync_interval_steps,
        min_workers=args.min_workers,
        sync_timeout_seconds=args.sync_timeout,
    )
    bus.set_job_status(job_id, "RUNNING")
    print(f"Successfully submitted and started job: {job_id}")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Send stop signal to a running job."""
    shared_dir = Path(args.shared_dir)
    bus = ClusterStorageBus(shared_dir)
    bus.set_job_status(args.job_id, "STOPPED")
    print(f"Stopped job: {args.job_id}")
    return 0


def cmd_pause(args: argparse.Namespace) -> int:
    """Send pause signal to a running job."""
    shared_dir = Path(args.shared_dir)
    bus = ClusterStorageBus(shared_dir)
    bus.set_job_status(args.job_id, "PAUSED")
    print(f"Paused job: {args.job_id}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Resume a paused job."""
    shared_dir = Path(args.shared_dir)
    bus = ClusterStorageBus(shared_dir)
    bus.set_job_status(args.job_id, "RUNNING")
    print(f"Resumed job: {args.job_id}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="cluster", description="Distributed Port-Blocked Training Cluster CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Worker subcommand
    p_worker = subparsers.add_parser("worker", help="Run a worker daemon")
    p_worker.add_argument("--shared-dir", required=True, help="Shared network directory")
    p_worker.add_argument("--worker-id", default=None, help="Custom worker ID")
    p_worker.add_argument("--device", default=None, help="Device to use (e.g. 'cuda:0', 'cpu')")
    p_worker.add_argument("--heartbeat-interval", type=float, default=5.0, help="Heartbeat interval in seconds")
    p_worker.add_argument("--poll-interval", type=float, default=2.0, help="Job polling interval in seconds")
    p_worker.add_argument("--detach", "--background", action="store_true", help="Launch detached in background so you can close this terminal")
    p_worker.set_defaults(func=cmd_worker)

    # Status subcommand
    p_status = subparsers.add_parser("status", help="Show cluster status")
    p_status.add_argument("--shared-dir", required=True, help="Shared network directory")
    p_status.add_argument("--heartbeat-timeout", type=float, default=45.0, help="Seconds before marking worker offline")
    p_status.set_defaults(func=cmd_status)

    # Submit subcommand
    p_submit = subparsers.add_parser("submit", help="Submit a new distributed training job")
    p_submit.add_argument("--shared-dir", required=True, help="Shared network directory")
    p_submit.add_argument("--job-id", default=None, help="Unique job identifier")
    p_submit.add_argument("--dataset-path", required=True, help="Path to token array file (.npy)")
    p_submit.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size")
    p_submit.add_argument("--context-length", type=int, default=512, help="Context length")
    p_submit.add_argument("--embedding-size", type=int, default=256, help="Embedding dimension")
    p_submit.add_argument("--head-count", type=int, default=4, help="Attention head count")
    p_submit.add_argument("--layer-count", type=int, default=4, help="Transformer layer count")
    p_submit.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p_submit.add_argument("--batch-size", type=int, default=4, help="Batch size")
    p_submit.add_argument("--max-rounds", type=int, default=10, help="Total synchronization rounds")
    p_submit.add_argument("--sync-interval-steps", type=int, default=250, help="Local SGD steps before sync")
    p_submit.add_argument("--min-workers", type=int, default=1, help="Minimum workers to proceed")
    p_submit.add_argument("--sync-timeout", type=float, default=180.0, help="Straggler timeout in seconds")
    p_submit.set_defaults(func=cmd_submit)

    # Stop subcommand
    p_stop = subparsers.add_parser("stop", help="Stop a distributed job")
    p_stop.add_argument("--shared-dir", required=True, help="Shared network directory")
    p_stop.add_argument("--job-id", required=True, help="Job identifier")
    p_stop.set_defaults(func=cmd_stop)

    # Pause subcommand
    p_pause = subparsers.add_parser("pause", help="Pause a distributed job")
    p_pause.add_argument("--shared-dir", required=True, help="Shared network directory")
    p_pause.add_argument("--job-id", required=True, help="Job identifier")
    p_pause.set_defaults(func=cmd_pause)

    # Resume subcommand
    p_resume = subparsers.add_parser("resume", help="Resume a paused job")
    p_resume.add_argument("--shared-dir", required=True, help="Shared network directory")
    p_resume.add_argument("--job-id", required=True, help="Job identifier")
    p_resume.set_defaults(func=cmd_resume)

    parsed = parser.parse_args(argv)
    return parsed.func(parsed)


if __name__ == "__main__":
    sys.exit(main())
