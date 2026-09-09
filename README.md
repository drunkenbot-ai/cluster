# DrunkenBot Cluster (`drunkenbot-cluster`)

Lightweight, port-blocked distributed LLM training via centralized storage using **Local SGD / Periodic Weight Averaging**.

## Overview

When compute nodes are separated by firewalls, restricted subnets, or blocked communication ports, standard distributed frameworks (such as PyTorch DDP with NCCL) cannot function.

`drunkenbot-cluster` resolves this by eliminating direct peer-to-peer network socket requirements. All distributed training coordination operates through a **centralized shared network drive** (SMB, NFS, NAS, or cloud mount):

1. **Central SQLite WAL Bus**: Manages worker discovery, registration, heartbeats, and atomic job status tracking without socket listeners.
2. **Local SGD / Weight Averaging**: Each worker trains independently on disjoint data shards for $K$ steps (e.g. 200–500 steps), dumps weights to the central drive, and the coordinator synchronizes by arithmetic averaging.
3. **Straggler & Dropout Resilience**: The coordinator dynamically averages weights when a threshold of workers completes, preventing slow or hung workers from stalling the entire cluster.
4. **Featherweight Worker Node**: Worker machines only need `python` and `torch`. No GUI, no PySide6, no heavy dependencies.

---

## Directory Architecture on Shared Network Storage

```text
CENTRAL SHARED DRIVE (e.g. Z:\llm_cluster or /mnt/nas/cluster)
├── cluster.db                        # SQLite in WAL mode (workers, jobs, rounds)
└── jobs/
    └── <job_id>/
        ├── signals/                  # Control signals: pause.sig, stop.sig
        └── rounds/
            ├── round_0000/
            │   ├── worker_01.pt
            │   ├── worker_01.ready
            │   ├── worker_02.pt
            │   ├── worker_02.ready
            │   ├── global_model.pt
            │   └── global_model.ready
            └── round_0001/ ...
```

---

## Quickstart for Worker Nodes

Run a worker daemon on any machine connected to the shared network storage:

```bash
# Secondary PC or GPU Server
pip install torch
python -m cluster.cli worker --shared-dir /mnt/shared/llm_cluster --worker-id node-rtx4090
```

The worker registers in the central database, reports its GPU capabilities, and enters `IDLE` state waiting for jobs dispatched from the primary node, LLM-IDE, or CLI.

---

## CLI Management Commands

```bash
# Check cluster worker fleet and active job status
python -m cluster.cli status --shared-dir Z:\llm_cluster

# Submit a distributed training job across all active workers
python -m cluster.cli submit \
    --shared-dir Z:\llm_cluster \
    --dataset-path Z:\llm_cluster\train_tokens.npy \
    --max-rounds 20 \
    --sync-interval-steps 250 \
    --min-workers 2

# Pause an active job cooperatively across all nodes
python -m cluster.cli pause --shared-dir Z:\llm_cluster --job-id <job_id>

# Resume a paused job
python -m cluster.cli resume --shared-dir Z:\llm_cluster --job-id <job_id>

# Stop an active job
python -m cluster.cli stop --shared-dir Z:\llm_cluster --job-id <job_id>
```

---

## Python API

```python
from cluster import ClusterStorageBus, ClusterCoordinator, ClusterWorker

# 1. Access the shared bus
bus = ClusterStorageBus("Z:/llm_cluster")

# 2. Worker node instantiation
worker = ClusterWorker(bus=bus, worker_id="node-rtx4090")
worker.run_daemon()

# 3. Master coordinator orchestration
coordinator = ClusterCoordinator(bus=bus, job_id="job_001")
coordinator.wait_and_average_round(round_num=0)
```
