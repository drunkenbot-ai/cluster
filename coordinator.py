"""Cluster Synchronization Coordinator for Local SGD.

Averages model weight checkpoints deposited by distributed workers and coordinates
cross-round progression with straggler timeouts.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path
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
        self._best_val_loss: Optional[float] = None
        self._best_checkpoint_path: Optional[str] = None

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
        t_agg_start = time.time()
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
        agg_sec = max(time.time() - t_agg_start, 0.0)

        # 1. Immediately delete worker weights to reclaim storage space
        if hasattr(self.bus, "purge_round_worker_weights"):
            self.bus.purge_round_worker_weights(self.job_id, round_num)
        # 2. Prune old round directories older than the rolling window (keep 2)
        if hasattr(self.bus, "purge_stale_rounds"):
            self.bus.purge_stale_rounds(self.job_id, keep_last_rounds=2)

        # Aggregate telemetry across ready workers
        losses = [float(t["avg_loss"]) for t in worker_telemetries.values() if "avg_loss" in t]
        val_losses = [float(t["val_loss"]) for t in worker_telemetries.values() if t.get("val_loss") is not None]
        throughputs = [float(t.get("tokens_per_sec", 0.0)) for t in worker_telemetries.values()]
        tokens_list = [int(t.get("tokens_processed", 0)) for t in worker_telemetries.values()]
        compute_times = [float(t["compute_sec"]) for t in worker_telemetries.values() if "compute_sec" in t]
        avg_compute_sec = round(sum(compute_times) / len(compute_times), 1) if compute_times else None

        compute_str = f", Avg Compute: {avg_compute_sec:.1f}s" if avg_compute_sec is not None else ""
        print(f"[Coordinator] Round {round_num + 1} synchronized in {agg_sec:.1f}s (Aggregated {len(worker_states)} workers{compute_str}).")

        global_avg_loss = sum(losses) / len(losses) if losses else 0.0
        global_val_loss = round(sum(val_losses) / len(val_losses), 4) if val_losses else None
        aggregate_tokens_sec = sum(throughputs)
        total_tokens_round = sum(tokens_list)

        sync_interval = int(job.get("sync_interval_steps") or 250)
        max_rounds = int(job.get("max_rounds") or 10)
        effective_step = (round_num + 1) * sync_interval

        # Track validation loss and save best validation checkpoint
        ckpt_dir = self.bus.get_checkpoints_dir(self.job_id)
        if global_val_loss is not None:
            if self._best_val_loss is None or global_val_loss < self._best_val_loss:
                self._best_val_loss = global_val_loss
                best_path = ckpt_dir / "checkpoint_best_val.pt"
                best_alias = ckpt_dir / "best_checkpoint.pt"
                self.bus.save_checkpoint(
                    self.job_id,
                    effective_step,
                    global_state,
                    is_final=False,
                    round_num=round_num,
                    train_loss=global_avg_loss,
                    val_loss=global_val_loss,
                )
                try:
                    latest_p = ckpt_dir / "latest_checkpoint.pt"
                    if latest_p.exists():
                        shutil.copyfile(latest_p, best_path)
                        shutil.copyfile(latest_p, best_alias)
                except Exception:
                    pass
                self._best_checkpoint_path = str(best_path)

        # Save durable checkpoint to shared storage (checkpoints/checkpoint_step_XXXXXX.pt & latest_checkpoint.pt)
        is_final_round = (round_num + 1 >= max_rounds)
        self.bus.save_checkpoint(
            self.job_id,
            effective_step,
            global_state,
            is_final=is_final_round,
            round_num=round_num,
            train_loss=global_avg_loss,
            val_loss=global_val_loss,
        )

        if is_final_round:
            self.finalize_job_artifacts(
                global_state,
                round_num,
                is_final=True,
                stopped=False,
                global_avg_loss=global_avg_loss,
                global_val_loss=global_val_loss,
            )

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
            "avg_compute_sec": avg_compute_sec,
            "coordinator_agg_sec": round(agg_sec, 2),
            "worker_compute_times": {wid: round(float(t["compute_sec"]), 2) for wid, t in worker_telemetries.items() if "compute_sec" in t},
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

    def finalize_job_artifacts(
        self,
        global_state: dict[str, torch.Tensor],
        round_num: int,
        is_final: bool = True,
        stopped: bool = False,
        global_avg_loss: float = 0.0,
        global_val_loss: Optional[float] = None,
    ) -> None:
        """Finalize job artifacts, export chat-loadable models and summaries, and ensure tokenizer availability."""
        job = self.bus.get_job(self.job_id) or {}
        ckpt_dir = self.bus.get_checkpoints_dir(self.job_id)
        job_dir = self.bus.jobs_dir / self.job_id
        peft_m = str(job.get("peft_method") or "none").lower()
        sync_interval = int(job.get("sync_interval_steps") or 250)
        effective_step = (round_num + 1) * sync_interval

        # Ensure tokenizer.json is copied to job and checkpoint dirs
        tok_src = None
        base_path = job.get("base_checkpoint_path")
        if base_path and os.path.exists(base_path):
            c_tok = Path(base_path).parent / "tokenizer.json"
            if c_tok.exists():
                tok_src = c_tok
        if not tok_src:
            ds_p = job.get("dataset_path")
            if ds_p:
                p = Path(ds_p)
                c_tok = p.parent / "tokenizer.json" if p.is_file() else p / "tokenizer.json"
                if c_tok.exists():
                    tok_src = c_tok
                elif (self.bus.shared_dir / "tokenizer.json").exists():
                    tok_src = self.bus.shared_dir / "tokenizer.json"
        if tok_src and tok_src.exists():
            for dst in (job_dir / "tokenizer.json", ckpt_dir / "tokenizer.json"):
                try:
                    if not dst.exists() or dst.stat().st_size != tok_src.stat().st_size:
                        shutil.copyfile(tok_src, dst)
                except Exception:
                    pass

        # Handle LoRA fine-tuning vs standard pretraining
        if peft_m == "lora":
            adapters = {k: v for k, v in global_state.items() if ".lora_a" in k or ".lora_b" in k}
            if adapters:
                for a_name in ("adapter_model.pt", "final_adapter.pt"):
                    torch.save({"adapter_state_dict": adapters, "lora_config": job.get("lora_config")}, ckpt_dir / a_name)
                    try:
                        shutil.copyfile(ckpt_dir / a_name, job_dir / a_name)
                    except Exception:
                        pass
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

                merged_payload = {
                    "artifact_type": "inference",
                    "model_config": m_cfg,
                    "model_state_dict": merged,
                    "global_step": effective_step,
                    "round": round_num,
                    "train_loss": global_avg_loss,
                    "val_loss": global_val_loss,
                }
                # Save both final_model_merged.pt and final_model.pt so Chat & Export can load directly
                for m_target in (ckpt_dir / "final_model_merged.pt", ckpt_dir / "final_model.pt", job_dir / "final_model.pt", ckpt_dir / "model.pt", job_dir / "model.pt"):
                    torch.save(merged_payload, m_target)
            except Exception as exc:
                print(f"[Coordinator LoRA merge notice]: {exc}")
        else:
            # Full pre-training or standard fine-tuning
            self.bus.save_checkpoint(
                self.job_id,
                effective_step,
                global_state,
                is_final=True,
                round_num=round_num,
                train_loss=global_avg_loss,
                val_loss=global_val_loss,
            )

        try:
            tcfg_raw = job.get("training_config", {})
            tcfg_dict = json.loads(tcfg_raw) if isinstance(tcfg_raw, str) else dict(tcfg_raw or {})
            summary_data = {
                "job_id": self.job_id,
                "job_type": job.get("job_type", "pretrain"),
                "completed_at": time.time(),
                "stopped": stopped,
                "total_rounds": round_num + 1,
                "final_loss": round(global_avg_loss, 4),
                "final_val_loss": global_val_loss,
                "best_val_loss": self._best_val_loss,
                "best_checkpoint_path": self._best_checkpoint_path,
                "recommended_checkpoint_path": self._best_checkpoint_path or str(ckpt_dir / "final_model.pt"),
                "model_config": job.get("model_config"),
                "training_config": tcfg_dict,
                "peft_method": peft_m,
                "lora_config": job.get("lora_config"),
                "base_checkpoint_path": job.get("base_checkpoint_path"),
            }
            summary_text = json.dumps(summary_data, indent=2, default=str)
            (ckpt_dir / "training_summary.json").write_text(summary_text, encoding="utf-8")
            (job_dir / "training_summary.json").write_text(summary_text, encoding="utf-8")
            lineage_data = {
                "base_checkpoint": job.get("base_checkpoint_path"),
                "training_mode": job.get("job_type", "pretrain"),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            }
            lineage_text = json.dumps(lineage_data, indent=2, default=str)
            (ckpt_dir / "model_lineage.json").write_text(lineage_text, encoding="utf-8")
            (job_dir / "model_lineage.json").write_text(lineage_text, encoding="utf-8")
        except Exception as exc:
            print(f"[Coordinator summary export notice]: {exc}")

    def run_all_rounds(
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
        last_state = None
        last_round = cur_round

        while cur_round < max_rounds:
            if stop_event and stop_event.is_set():
                break
            if self.bus.is_stopped(self.job_id):
                break
            while self.bus.is_paused(self.job_id):
                if stop_event and stop_event.is_set() or self.bus.is_stopped(self.job_id):
                    break
                time.sleep(poll_interval_seconds)
            if stop_event and stop_event.is_set() or self.bus.is_stopped(self.job_id):
                break

            state = self.wait_and_average_round(
                round_num=cur_round,
                poll_interval_seconds=poll_interval_seconds,
                progress_callback=telemetry_callback,
            )
            if state is None:
                break
            last_state = state
            last_round = cur_round
            cur_round += 1

        is_completed = (cur_round >= max_rounds)
        if last_state is not None:
            self.finalize_job_artifacts(
                last_state,
                last_round,
                is_final=is_completed,
                stopped=not is_completed,
            )
        elif not is_completed:
            latest_weights = self.bus.load_latest_checkpoint(self.job_id)
            if latest_weights:
                self.finalize_job_artifacts(
                    latest_weights,
                    max(cur_round - 1, 0),
                    is_final=False,
                    stopped=True,
                )

        if is_completed:
            self.bus.set_job_status(self.job_id, "COMPLETED")
            return True
        return False

    # Alias
    run_job = run_all_rounds


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
