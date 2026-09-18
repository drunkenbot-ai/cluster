"""Drunkenbot Cluster: Distributed Local SGD Training via Centralized Storage.

Enables multi-node parallel model training across port-blocked heterogeneous machines
over standard 10Gbps networks using a central shared storage bus (SMB/NFS/NAS).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ClusterStorageBus",
    "ClusterCoordinator",
    "average_state_dicts",
    "ClusterWorker",
    "build_model_from_config",
    "get_hardware_info",
    "ShardedTokenDataset",
    "compute_shard_boundaries",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name == "ClusterStorageBus":
        from .bus import ClusterStorageBus
        return ClusterStorageBus
    if name in ("ClusterCoordinator", "average_state_dicts"):
        from .coordinator import ClusterCoordinator, average_state_dicts
        return locals()[name]
    if name in ("ClusterWorker", "build_model_from_config", "get_hardware_info"):
        from .worker import ClusterWorker, build_model_from_config, get_hardware_info
        return locals()[name]
    if name in ("ShardedTokenDataset", "compute_shard_boundaries"):
        from .sharding import ShardedTokenDataset, compute_shard_boundaries
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

