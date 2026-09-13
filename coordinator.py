"""Cluster Synchronization Coordinator for Local SGD.

Averages model weight checkpoints deposited by distributed workers and coordinates
cross-round progression with straggler timeouts.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable, Optional

import torch

from .bus import ClusterStorageBus


def average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Compute the arithmetic mean across a collection of PyTorch state dicts.

    Floating point weights are averaged; integer tensors/buffers (such as RoPE cached
    indices or position buffers) are preserved from the first state dict.

    Args:
        state_dicts: List of model parameter state dicts from participating workers.

    Returns:
        Averaged global state dict.

    Raises:
        ValueError: If state_dicts is empty.
    """
    if not state_dicts:
        raise ValueError("Cannot average an empty list of state dicts.")

    if len(state_dicts) == 1:
        return {k: v.clone() for k, v in state_dicts[0].items()}

    first = state_dicts[0]
    num_models = len(state_dicts)
    averaged: dict[str, torch.Tensor] = {}

    for key, val in first.items():
        if val.is_floating_point():
            # Initialize with first tensor
            accum = val.clone().float()
            for other in state_dicts[1:]:
                accum += other[key].float()
            averaged[key] = (accum / num_models).to(val.dtype)
        else:
            # For non-floating point buffers (integers, masks), preserve identity
            averaged[key] = val.clone()

    return averaged


class ClusterCoordinator:
    """Orchestrates weight averaging and round advancement for a distributed job."""

    def __init__(self, bus: ClusterStorageBus, job_id: str) -> None:
        """Initialize the coordinator.

        Args:
            bus: Central storage bus instance.
            job_id: Unique job identifier being coordinated.
        """
        self.bus = bus
        self.job_id = job_id
        self._cumulative_tokens: int = 0
        self._dataset_tokens: Optional[int] = None

    def wait_and_average_round(
        self,
        round_num: int,
        poll_interval_seconds: float = 2.0,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> Optional[dict[str, torch.Tensor]]:
        """Wait for worker checkpoints and publish the averaged global model.

        Respects minimum worker participation and straggler timeout.

        Args:
            round_num: Current synchronization round index.
            poll_interval_seconds: Polling sleep interval.
            progress_callback: Optional progress reporter.

        Returns:
            Averaged state dict, or None if the job was stopped.
        """
        job = self.bus.get_job(self.job_id)
        if not job:
            raise ValueError(f"Job {self.job_id} not found in database.")

        min_workers = int(job.get("min_workers") or 1)
        sync_timeout = float(job.get("sync_timeout_seconds") or 180.0)

        started_wait = time.time()
        first_ready_time: Optional[float] = None

        while True:
            # Signal coordinator liveness
            if hasattr(self.bus, "touch_job"):
                self.bus.touch_job(self.job_id)

            # Check for stop signal
            if self.bus.is_stopped(self.job_id):
                return None

            ready_workers = self.bus.get_ready_workers_for_round(self.job_id, round_num)
            participants = self.bus.get_job_participants(self.job_id)
            total_participants = max(len(participants), 1)

            if ready_workers and first_ready_time is None:
                first_ready_time = time.time()

            elapsed_wait = time.time() - started_wait

            if progress_callback:
                progress_callback({
                    "type": "round_waiting",
                    "round": round_num,
                    "ready_workers": len(ready_workers),
                    "total_participants": total_participants,
                    "elapsed_seconds": elapsed_wait,
                })

            # Check completion criteria:
            # 1. All active registered workers deposited weights
            all_ready = len(ready_workers) >= total_participants

            # 2. Straggler timeout expired and at least min_workers deposited weights
            timeout_expired = (
                first_ready_time is not None
                and (time.time() - first_ready_time) > sync_timeout
                and len(ready_workers) >= min_workers
            )

            if (all_ready or timeout_expired) and ready_workers:
                break

            time.sleep(poll_interval_seconds)

        # Fault tolerance: if any worker failed to deposit weights (straggler / crashed),
        # mark it as DROPPED so remaining active workers re-index and take over its shard.
        all_participant_ids = [p["worker_id"] for p in participants]
        for wid in all_participant_ids:
            if wid not in ready_workers:
                print(f"[Coordinator] Worker '{wid}' failed to deposit weights for round {round_num}. Marking DROPPED to reassign workload.")
                self.bus.mark_worker_dropped(self.job_id, wid, reason="timeout")

        # Load weights from all ready workers
        worker_states = []
        worker_telemetries: dict[str, dict[str, Any]] = {}
        for wid in ready_workers:
            state = self.bus.load_worker_weights(self.job_id, round_num, wid, device="cpu")
            worker_states.append(state)
            tel = self.bus.load_worker_telemetry(self.job_id, round_num, wid)
            if not tel:
                # Brief retry to handle network SMB/NFS cache latency
                time.sleep(0.5)
                tel = self.bus.load_worker_telemetry(self.job_id, round_num, wid)
            if tel:
                worker_telemetries[wid] = tel

        # Average weights
        global_state = average_state_dicts(worker_states)

        # Atomically save to central storage
        self.bus.save_global_weights(self.job_id, round_num, global_state)

        # Aggregate telemetry across ready workers
        losses = [float(t["avg_loss"]) for t in worker_telemetries.values() if "avg_loss" in t]
        val_losses = [float(t["val_loss"]) for t in worker_telemetries.values() if t.get("val_loss") is not None]
        throughputs = [float(t.get("tokens_per_sec", 0.0)) for t in worker_telemetries.values()]
        tokens_list = [int(t.get("tokens_processed", 0)) for t in worker_telemetries.values()]

        global_avg_loss = sum(losses) / len(losses) if losses else 0.0
        global_val_loss = round(sum(val_losses) / len(val_losses), 4) if val_losses else None
        aggregate_tokens_sec = sum(throughputs)
        total_tokens_round = sum(tokens_list)

        sync_interval = int(job.get("sync_interval_steps") or 250)
        max_rounds = int(job.get("max_rounds") or 10)
        effective_step = (round_num + 1) * sync_interval

        # Save durable checkpoint to shared storage (checkpoints/checkpoint_step_XXXXXX.pt & latest_checkpoint.pt)
        is_final_round = (round_num + 1 >= max_rounds)
        self.bus.save_checkpoint(self.job_id, effective_step, global_state, is_final=is_final_round)

        if is_final_round:
            ckpt_dir = self.bus.get_checkpoints_dir(self.job_id)
            peft_m = str(job.get("peft_method") or "none").lower()
            if peft_m == "lora":
                adapters = {k: v for k, v in global_state.items() if ".lora_a" in k or ".lora_b" in k}
                if adapters:
                    adapter_target = ckpt_dir / "adapter_model.pt"
                    torch.save({"adapter_state_dict": adapters, "lora_config": job.get("lora_config")}, adapter_target)
                try:
                    m_cfg = job.get("model_config", {})
                    if isinstance(m_cfg, str):
                        m_cfg = json.loads(m_cfg)
                    from cluster.worker import build_model_from_config, apply_lora_adapters, merged_lora_state_dict
                    m_temp = build_model_from_config(m_cfg, "cpu")
                    l_cfg = job.get("lora_config") or {}
                    if isinstance(l_cfg, str):
                        l_cfg = json.loads(l_cfg)
                    apply_lora_adapters(
                        m_temp,
                        rank=int(l_cfg.get("rank", 8)),
                        alpha=float(l_cfg.get("alpha", 16.0)),
                        dropout=float(l_cfg.get("dropout", 0.0)),
                        target_modules=str(l_cfg.get("target_modules", "attention")),
                    )
                    m_temp.load_state_dict(global_state, strict=False)
                    merged = merged_lora_state_dict(m_temp)
                    torch.save({"model_state_dict": merged, "model_config": m_cfg}, ckpt_dir / "final_model_merged.pt")
                except Exception:
                    pass

            try:
                tcfg_raw = job.get("training_config", {})
                tcfg_dict = json.loads(tcfg_raw) if isinstance(tcfg_raw, str) else dict(tcfg_raw or {})
                summary_data = {
                    "job_id": self.job_id,
                    "job_type": job.get("job_type", "pretrain"),
                    "completed_at": time.time(),
                    "total_rounds": round_num + 1,
                    "final_loss": round(global_avg_loss, 4),
                    "final_val_loss": global_val_loss,
                    "model_config": job.get("model_config"),
                    "training_config": tcfg_dict,
                    "peft_method": peft_m,
                    "lora_config": job.get("lora_config"),
                    "base_checkpoint_path": job.get("base_checkpoint_path"),
                }
                (ckpt_dir / "training_summary.json").write_text(json.dumps(summary_data, indent=2, default=str), encoding="utf-8")
                lineage_data = {
                    "base_checkpoint": job.get("base_checkpoint_path"),
                    "training_mode": job.get("job_type", "pretrain"),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                }
                (ckpt_dir / "model_lineage.json").write_text(json.dumps(lineage_data, indent=2, default=str), encoding="utf-8")
            except Exception as exc:
                print(f"[Coordinator summary export failed]: {exc}")

        # Determine total dataset tokens if not yet cached
        if self._dataset_tokens is None:
            ds_path = job.get("dataset_path")
            if ds_path and os.path.exists(ds_path):
                try:
                    import numpy as np
                    arr = np.load(ds_path, mmap_mode="r")
                    self._dataset_tokens = int(len(arr))
                except Exception:
                    self._dataset_tokens = None

        self._cumulative_tokens += total_tokens_round
        epoch_float = None
        if self._dataset_tokens and self._dataset_tokens > 0:
            epoch_float = round(self._cumulative_tokens / self._dataset_tokens, 4)

        tcfg = job.get("training_config", {})
        if isinstance(tcfg, str):
            try:
                tcfg = json.loads(tcfg)
            except Exception:
                tcfg = {}
        target_epochs = int(tcfg.get("epochs") or 1)

        summary_metrics = {
            "job_id": self.job_id,
            "round": round_num,
            "max_rounds": max_rounds,
            "effective_step": effective_step,
            "global_loss": round(global_avg_loss, 4),
            "val_loss": global_val_loss,
            "aggregate_tokens_per_sec": round(aggregate_tokens_sec, 1),
            "total_tokens_round": total_tokens_round,
            "cumulative_tokens": self._cumulative_tokens,
            "dataset_tokens": self._dataset_tokens,
            "epoch": epoch_float,
            "target_epochs": target_epochs,
            "ready_workers_count": len(ready_workers),
            "participating_workers": ready_workers,
            "worker_losses": {wid: round(float(t.get("avg_loss", 0.0)), 4) for wid, t in worker_telemetries.items()},
            "worker_val_losses": {wid: round(float(t["val_loss"]), 4) for wid, t in worker_telemetries.items() if t.get("val_loss") is not None},
        }

        # Persist round summary to SQLite database
        self.bus.record_round_summary(
            job_id=self.job_id,
            round_num=round_num,
            participating_workers=ready_workers,
            avg_loss=global_avg_loss,
            metrics=summary_metrics,
        )

        # Advance round in SQLite bus
        self.bus.advance_job_round(self.job_id, round_num + 1)

        # Report round completion to progress callback
        if progress_callback:
            summary_metrics["type"] = "round_completed"
            progress_callback(summary_metrics)

        return global_state

    # Alias for convenience
    wait_for_round_and_aggregate = wait_and_average_round

    def run_job(
        self,
        poll_interval_seconds: float = 2.0,
        telemetry_callback: Optional[Callable[[dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        """Run all remaining rounds of the job to completion.

        Args:
            poll_interval_seconds: Interval between polling checks.
            telemetry_callback: Callback triggered after each round with aggregate telemetry.
            stop_event: Optional threading.Event to stop early.

        Returns:
            True if all rounds completed successfully, False otherwise.
        """
        job = self.bus.get_job(self.job_id)
        if not job:
            return False

        max_rounds = int(job.get("max_rounds") or 10)
        cur_round = int(job.get("current_round") or 0)

        while cur_round < max_rounds:
            if stop_event and stop_event.is_set():
                break
            if self.bus.is_stopped(self.job_id):
                break
            while self.bus.is_paused(self.job_id):
                if stop_event and stop_event.is_set() or self.bus.is_stopped(self.job_id):
                    return False
                time.sleep(poll_interval_seconds)

            state = self.wait_and_average_round(
                round_num=cur_round,
                poll_interval_seconds=poll_interval_seconds,
                progress_callback=telemetry_callback,
            )
            if state is None:
                return False
            cur_round += 1

        if cur_round >= max_rounds:
            self.bus.set_job_status(self.job_id, "COMPLETED")
            return True
        return False


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for running ClusterCoordinator as a standalone detached daemon."""
    import argparse
    import os
    from pathlib import Path
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

    parser = argparse.ArgumentParser(description="Cluster Synchronization Coordinator for Local SGD")
    parser.add_argument("--shared-dir", required=True, help="Central shared network directory")
    parser.add_argument("--job-id", required=True, help="Unique cluster job ID")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Polling interval in seconds")
    args = parser.parse_args(argv)

    storage_path = Path(args.shared_dir)
    bus = ClusterStorageBus(storage_path)
    coordinator = ClusterCoordinator(bus, args.job_id)

    pid_file = bus.jobs_dir / args.job_id / "coordinator.pid"
    try:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()), encoding="utf-8")
    except Exception:
        pass

    print(f"[COORDINATOR] Starting coordinator daemon for job '{args.job_id}' (PID: {os.getpid()})...", flush=True)
    print(f"[COORDINATOR] Central storage: {storage_path}", flush=True)

    def _on_round_progress(metrics: dict[str, Any]) -> None:
        m_type = metrics.get("type")
        if m_type == "round_completed":
            r_num = int(metrics.get("round", 0)) + 1
            max_r = int(metrics.get("max_rounds", 0))
            loss = float(metrics.get("global_loss", 0.0))
            val_loss = metrics.get("val_loss")
            val_str = f" | Validation Loss {float(val_loss):.4f}" if val_loss is not None else ""
            spd = float(metrics.get("aggregate_tokens_per_sec", 0.0))
            print(f"[COORDINATOR] >>> Round {r_num}/{max_r} Averaged! Global Loss: {loss:.4f}{val_str} | Speed: {spd:,.0f} tok/s", flush=True)
        elif m_type == "round_waiting":
            r_num = int(metrics.get("round", 0)) + 1
            ready = int(metrics.get("ready_workers", 0))
            total = int(metrics.get("total_participants", 1))
            elapsed = float(metrics.get("elapsed_seconds", 0.0))
            if int(elapsed) % 10 == 0:
                print(f"[COORDINATOR] Waiting for round {r_num} weights ({ready}/{total} workers ready, {elapsed:.0f}s)...", flush=True)

    try:
        success = coordinator.run_job(
            poll_interval_seconds=args.poll_interval,
            telemetry_callback=_on_round_progress,
        )
    finally:
        try:
            if pid_file.exists():
                pid_file.unlink(missing_ok=True)
        except Exception:
            pass

    print(f"[COORDINATOR] Coordinator execution finished: success={success}", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
