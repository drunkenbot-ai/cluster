"""Cluster Synchronization Coordinator for Local SGD.

Averages model weight checkpoints deposited by distributed workers and coordinates
cross-round progression with straggler timeouts.
"""

from __future__ import annotations

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

        # Load weights from all ready workers
        worker_states = []
        for wid in ready_workers:
            state = self.bus.load_worker_weights(self.job_id, round_num, wid, device="cpu")
            worker_states.append(state)

        # Average weights
        global_state = average_state_dicts(worker_states)

        # Atomically save to central storage
        self.bus.save_global_weights(self.job_id, round_num, global_state)

        # Advance round in SQLite bus
        self.bus.advance_job_round(self.job_id, round_num + 1)

        return global_state
