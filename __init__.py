"""Drunkenbot Cluster: Distributed Local SGD Training via Centralized Storage.

Enables multi-node parallel model training across port-blocked heterogeneous machines
over standard 10Gbps networks using a central shared storage bus (SMB/NFS/NAS).
"""

from __future__ import annotations

from .bus import ClusterStorageBus
from .coordinator import ClusterCoordinator, average_state_dicts
from .sharding import ShardedTokenDataset, compute_shard_boundaries
from .worker import ClusterWorker, build_model_from_config, get_hardware_info

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
