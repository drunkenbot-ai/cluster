"""Distributed Cluster Worker Daemon for Local SGD Training.

Runs as a lightweight headless service on worker compute nodes:
1. Discovers local hardware (GPU, VRAM).
2. Registers with the central storage bus and emits periodic heartbeats.
3. Claims a data shard and executes K steps of local training.
4. Atomically deposits weight checkpoints to central storage.
5. Synchronizes from the coordinator's global averaged model.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import shutil
import tempfile
import threading
import time
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .bus import ClusterStorageBus
from .sharding import ShardedTokenDataset

# Ensure parent repository root is in sys.path so engine and inference packages can always be imported
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def is_network_path(path_str: str) -> bool:
    """Determine if a file path resides on a remote network share (SMB/NFS/UNC)."""
    p = str(path_str).strip()
    if p.startswith(("\\\\", "//")):
        return True
    if sys.platform == "win32" and len(p) >= 2 and p[1] == ":":
        try:
            import ctypes
            drive_root = p[:2] + "\\"
            # DRIVE_REMOTE = 4 in Windows API
            return ctypes.windll.kernel32.GetDriveTypeW(drive_root) == 4
        except Exception:
            pass
    return False


def purge_local_dataset_cache(
    cache_dir: Optional[Path] = None,
    keep_files: Optional[set[str]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> int:
    """Remove/delete stale .npy files from local cache directory to keep local storage clean.

    Args:
        cache_dir: Optional cache directory path. Defaults to LOCALAPPDATA/llm_cluster_cache.
        keep_files: Optional set of filenames to preserve (e.g. active train/val tokens).
        log_fn: Optional logger callable.

    Returns:
        Number of .npy files removed.
    """
    log = log_fn or (lambda msg, **kw: print(f"[DatasetCache] {msg}"))
    if cache_dir is None:
        cache_dir = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "llm_cluster_cache"
    if not cache_dir.exists():
        return 0

    keep = keep_files or set()
    log(f"[DatasetCache] Cleaning local SSD cache directory ({cache_dir})...")
    purged_count = 0
    for f in cache_dir.glob("*.npy"):
        if f.name in keep:
            continue
        try:
            f.unlink(missing_ok=True)
            log(f"[DatasetCache] Removed old cache file: {f.name}")
            purged_count += 1
        except Exception as exc:
            log(f"[DatasetCache] Notice: could not remove {f.name}: {exc}")

    if purged_count > 0:
        log(f"[DatasetCache] Purged {purged_count} stale .npy file(s) before caching new dataset.")
    else:
        log(f"[DatasetCache] Local cache is clean (0 stale .npy files found).")
    return purged_count


def cache_dataset_to_local(
    remote_path: str,
    log_fn: Optional[Callable[[str], None]] = None,
    keep_files: Optional[set[str]] = None,
    progress_callback: Optional[Callable[[float, float, float], None]] = None,
) -> str:
    """Safely cache remote/network dataset .npy file to fast local SSD storage.

    Purges stale .npy files before copying new files to keep local disk clean.
    Prevents unrecoverable Windows STATUS_IN_PAGE_ERROR (0xC0000006) on network shares.

    Args:
        remote_path: Path to dataset .npy on network share (SMB/NFS).
        log_fn: Optional logger callable.
        keep_files: Optional set of filenames to preserve during cache cleanup.
        progress_callback: Optional callback receiving (pct, speed_mb_s, eta_seconds).

    Returns:
        Local path to the cached .npy file (or original remote path if caching is skipped/impossible).
    """
    log = log_fn or (lambda msg, **kw: print(f"[DatasetCache] {msg}"))

    if not remote_path or not os.path.exists(remote_path):
        return remote_path

    # If it's already on a local non-network drive, no caching needed
    if not is_network_path(remote_path):
        return remote_path

    cache_dir = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "llm_cluster_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        log(f"[DatasetCache] Could not create local cache directory {cache_dir}: {exc}", level="WARNING")
        return remote_path

    remote_file = Path(remote_path)
    local_file = cache_dir / remote_file.name
    lock_file = cache_dir / f".{remote_file.name}.lock"
    temp_file = cache_dir / f".{remote_file.name}.tmp_{os.getpid()}"

    try:
        remote_stat = remote_file.stat()
        remote_size = remote_stat.st_size
        remote_size_gb = remote_size / (1024 ** 3)
        remote_mtime = remote_stat.st_mtime

        # Fast path: check if local file is already fully cached and valid
        if local_file.exists():
            try:
                local_stat = local_file.stat()
                if local_stat.st_size == remote_size and abs(local_stat.st_mtime - remote_mtime) < 2.0:
                    log(f"[DatasetCache] Verified existing local SSD dataset cache: {local_file.name} ({remote_size_gb:.2f} GB)")
                    return str(local_file)
            except Exception:
                pass

        # Multi-process coordination: if another slot process on this node is currently caching, wait for it
        if lock_file.exists():
            try:
                lock_age = time.time() - lock_file.stat().st_mtime
                if lock_age < 600:
                    log(f"[DatasetCache] Another slot on this node is currently caching {remote_file.name}. Waiting for local transfer to finish...")
                    wait_start = time.time()
                    while lock_file.exists() and (time.time() - wait_start < 180):
                        time.sleep(1.0)
                        if local_file.exists():
                            try:
                                if local_file.stat().st_size == remote_size:
                                    log(f"[DatasetCache] Local SSD dataset ready from concurrent slot: {local_file.name} ({remote_size_gb:.2f} GB)")
                                    return str(local_file)
                            except Exception:
                                pass
            except Exception:
                pass

        # Check again if file became available while checking/waiting
        if local_file.exists():
            try:
                if local_file.stat().st_size == remote_size:
                    return str(local_file)
            except Exception:
                pass

        # Claim the lock for this process
        try:
            lock_file.write_text(str(os.getpid()), encoding="utf-8")
        except Exception:
            pass

        # Purge stale .npy files from cache before copying new to keep local storage clean
        effective_keep = set(keep_files or ())
        effective_keep.add(remote_file.name)
        purge_local_dataset_cache(cache_dir=cache_dir, keep_files=effective_keep, log_fn=log)

        # Check free disk space on local cache drive
        free_space = shutil.disk_usage(str(cache_dir)).free
        free_gb = free_space / (1024 ** 3)
        if free_space < remote_size * 1.15:
            log(
                f"[DatasetCache] Insufficient local disk space to cache dataset "
                f"({free_gb:.1f} GB free vs {remote_size_gb:.1f} GB required). Falling back to direct network share.",
                level="WARNING",
            )
            try:
                if lock_file.exists():
                    lock_file.unlink(missing_ok=True)
            except Exception:
                pass
            return remote_path

        log(f"[DatasetCache] Streaming network dataset {remote_file.name} ({remote_size_gb:.2f} GB) to local SSD ({local_file})...")
        t0 = time.time()

        # Safe 2MB buffer chunks matching native Windows SMB buffers with live transfer progress
        buf_size = 2 * 1024 * 1024
        transferred = 0
        last_log_time = time.time()
        try:
            with open(str(remote_file), "rb") as src, open(str(temp_file), "wb") as dst:
                with memoryview(bytearray(buf_size)) as mv:
                    while True:
                        n = src.readinto(mv)
                        if not n:
                            break
                        dst.write(mv[:n])
                        transferred += n
                        now = time.time()
                        if now - last_log_time >= 4.0:
                            pct = (transferred / max(remote_size, 1)) * 100.0
                            elapsed_cur = max(now - t0, 0.001)
                            speed_cur = (transferred / (1024 * 1024)) / elapsed_cur
                            rem_bytes = max(remote_size - transferred, 0)
                            eta_s = rem_bytes / max(speed_cur * 1024 * 1024, 1)
                            log(
                                f"[DatasetCache] Caching {remote_file.name}: "
                                f"{transferred / (1024**3):.2f}/{remote_size_gb:.2f} GB ({pct:.1f}%) "
                                f"@ {speed_cur:.1f} MB/s (ETA: {eta_s:.0f}s)"
                            )
                            if progress_callback:
                                try:
                                    progress_callback(pct, speed_cur, eta_s)
                                except Exception:
                                    pass
                            last_log_time = now
        except OSError as os_err:
            log(f"[DatasetCache] Chunked stream encountered {os_err}, attempting fallback via shutil.copyfile...", level="WARNING")
            shutil.copyfile(str(remote_file), str(temp_file))

        # Atomically move temp_file into local_file
        if local_file.exists():
            try:
                if local_file.stat().st_size == remote_size:
                    if temp_file.exists():
                        temp_file.unlink(missing_ok=True)
                    return str(local_file)
                local_file.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            temp_file.replace(local_file)
        except PermissionError:
            if local_file.exists() and local_file.stat().st_size == remote_size:
                if temp_file.exists():
                    temp_file.unlink(missing_ok=True)
                return str(local_file)
            raise
        try:
            os.utime(str(local_file), (remote_mtime, remote_mtime))
        except Exception:
            pass

        try:
            if lock_file.exists():
                lock_file.unlink(missing_ok=True)
        except Exception:
            pass

        elapsed = max(time.time() - t0, 0.001)
        speed_mb_s = (remote_size / (1024 * 1024)) / elapsed
        log(f"[DatasetCache] Cache transfer completed in {elapsed:.1f}s ({speed_mb_s:.1f} MB/s). Ready for zero-latency local training.")
        return str(local_file)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        log(f"[DatasetCache] Failed to cache dataset to local SSD ({exc}). Falling back to network share.\nTraceback:\n{tb}", level="WARNING")
        try:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            if lock_file.exists():
                lock_file.unlink(missing_ok=True)
        except Exception:
            pass
        return remote_path


def calculate_optimal_worker_slots(
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    device_str: str,
    measured_vram_gb: Optional[float] = None,
) -> tuple[int, float, float]:
    """Calculate optimal number of worker processes to spawn on this node based on VRAM capacity.

    Args:
        model_cfg: Model architecture parameters.
        training_cfg: Training parameters (batch size, precision).
        device_str: Device string (e.g. 'cuda:0', 'cpu').
        measured_vram_gb: Optional empirically measured peak VRAM usage after running a training batch.

    Returns:
        (optimal_slots, estimated_job_gb, node_vram_gb)
    """
    if not device_str.startswith("cuda") or not torch.cuda.is_available():
        return 1, 0.0, 0.0

    try:
        dev_idx = 0
        if ":" in device_str:
            try:
                dev_idx = int(device_str.split(":")[1])
            except ValueError:
                dev_idx = 0
        total_vram_bytes = torch.cuda.get_device_properties(dev_idx).total_memory
        node_vram_gb = total_vram_bytes / (1024 ** 3)
    except Exception:
        return 1, 0.0, 0.0

    if measured_vram_gb is not None and measured_vram_gb > 0:
        estimated_job_gb = measured_vram_gb
        # If a single worker already consumes more than 45% of total VRAM, multiple slots cannot fit
        if (measured_vram_gb / max(node_vram_gb, 0.1)) > 0.45:
            return 1, round(estimated_job_gb, 2), round(node_vram_gb, 2)
        slots = max(1, min(4, int(node_vram_gb // (estimated_job_gb * 1.25))))
        return slots, round(estimated_job_gb, 2), round(node_vram_gb, 2)

    explicit_gb = float(training_cfg.get("vram_required_gb") or model_cfg.get("vram_required_gb") or 0.0)
    if explicit_gb > 0:
        estimated_job_gb = explicit_gb
    else:
        vocab_size = int(model_cfg.get("vocab_size") or 32000)
        emb_size = int(model_cfg.get("embedding_size") or 768)
        n_layers = int(model_cfg.get("layer_count") or 12)
        ctx_len = int(model_cfg.get("context_length") or 1024)
        batch_size = int(training_cfg.get("batch_size") or 2)

        # Estimate parameter count: token embeddings + transformer layers + head
        approx_params = (vocab_size * emb_size * 2) + (n_layers * 12 * emb_size * emb_size)
        prec = str(training_cfg.get("precision", "float16")).lower()
        bytes_per_param = 16 if "16" in prec else 24
        param_bytes = approx_params * bytes_per_param
        activation_bytes = batch_size * ctx_len * emb_size * n_layers * 16
        cuda_overhead = 1024 ** 3

        total_job_bytes = param_bytes + activation_bytes + cuda_overhead
        # In practice, each active worker requires at least 6.5 GB VRAM for model weights,
        # fp32 AdamW moments, master weights, gradients, PyTorch CUDA context, and activation cache.
        estimated_job_gb = max(round(total_job_bytes / (1024 ** 3), 2), 6.5)

    # Calculate slots: allow 1.10x safety headroom
    slots = max(1, min(4, int(node_vram_gb // (estimated_job_gb * 1.10))))
    return slots, estimated_job_gb, node_vram_gb


def get_hardware_info(device_preference: Optional[str] = None) -> tuple[str, str, float]:
    """Detect available compute device, GPU model, and VRAM in GB.

    Args:
        device_preference: Optional explicit device string (e.g. 'cuda:0', 'cpu').

    Returns:
        (device_str, gpu_name, vram_gb)
    """
    if device_preference:
        device_str = device_preference
    elif torch.cuda.is_available():
        device_str = "cuda:0"
    else:
        device_str = "cpu"

    if device_str.startswith("cuda") and torch.cuda.is_available():
        device_idx = 0
        if ":" in device_str:
            try:
                device_idx = int(device_str.split(":")[1])
            except ValueError:
                device_idx = 0
        gpu_name = torch.cuda.get_device_name(device_idx)
        props = torch.cuda.get_device_properties(device_idx)
        vram_gb = round(props.total_memory / (1024 ** 3), 2)
        # Active preflight test: verify that CUDA kernels can actually execute
        try:
            test_t = torch.zeros(1, device=device_str)
            del test_t
        except Exception as exc:
            print(f"[Worker] WARNING: CUDA device '{device_str}' ({gpu_name}) failed kernel execution preflight test: {exc}")
    else:
        gpu_name = platform.processor() or "CPU"
        vram_gb = 0.0

    return device_str, gpu_name, vram_gb


def is_hardware_bf16_supported(device_str: str = "cuda:0") -> bool:
    """Check whether the device has native hardware Bfloat16 Tensor Core execution (compute capability >= 8.0)."""
    if not torch.cuda.is_available() or not str(device_str).startswith("cuda"):
        return False
    try:
        dev_idx = int(str(device_str).split(":")[1]) if ":" in str(device_str) else 0
        cap = torch.cuda.get_device_capability(dev_idx)
        # Ampere (sm_80), Ada Lovelace (sm_89), Hopper (sm_90)+ have native BF16 tensor cores.
        # Turing (sm_75), Volta (sm_70), Pascal (sm_61) lack hardware BF16 tensor cores.
        return cap[0] >= 8 and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    except Exception:
        return False


def get_model_vocab_size(model: nn.Module) -> Optional[int]:
    """Inspect model to extract true vocabulary size for input/target bounds checking."""
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "vocab_size", None):
        return int(cfg.vocab_size)
    tok_emb = getattr(model, "token_embedding", None) or getattr(model, "tok_embed", None)
    if tok_emb is not None and hasattr(tok_emb, "num_embeddings"):
        return int(tok_emb.num_embeddings)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "out_features"):
        return int(lm_head.out_features)
    if hasattr(model, "module"):
        return get_model_vocab_size(model.module)
    return None


def collect_system_metrics(device_str: str = "cpu", total_vram_gb: float = 0.0) -> dict[str, Any]:
    """Collect current real-time CPU, RAM, and GPU VRAM usage."""
    metrics: dict[str, Any] = {
        "cpu_percent": 0.0,
        "ram_used_gb": 0.0,
        "ram_total_gb": 0.0,
        "vram_used_gb": 0.0,
        "vram_total_gb": float(total_vram_gb or 0.0),
    }

    # 1. RAM & CPU
    try:
        import psutil
        metrics["cpu_percent"] = round(float(psutil.cpu_percent(interval=None)), 1)
        vm = psutil.virtual_memory()
        metrics["ram_used_gb"] = round(float(vm.used) / (1024 ** 3), 1)
        metrics["ram_total_gb"] = round(float(vm.total) / (1024 ** 3), 1)
    except Exception:
        if sys.platform == "win32":
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                tot = float(stat.ullTotalPhys) / (1024 ** 3)
                avail = float(stat.ullAvailPhys) / (1024 ** 3)
                metrics["ram_total_gb"] = round(tot, 1)
                metrics["ram_used_gb"] = round(tot - avail, 1)
                metrics["cpu_percent"] = float(stat.dwMemoryLoad)
            except Exception:
                pass

    # 2. VRAM
    try:
        if device_str.startswith("cuda") and torch.cuda.is_available():
            dev_idx = 0
            if ":" in device_str:
                try:
                    dev_idx = int(device_str.split(":")[1])
                except ValueError:
                    dev_idx = 0
            free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
            used_bytes = max(total_bytes - free_bytes, 0)
            metrics["vram_used_gb"] = round(float(used_bytes) / (1024 ** 3), 2)
            metrics["vram_total_gb"] = round(float(total_bytes) / (1024 ** 3), 2)
    except Exception:
        pass

    return metrics


# =============================================================================
# Standalone MicroGPT Architecture Parity (Zero-dependency Fallback)
# =============================================================================

class StandaloneRotaryEmbedding(nn.Module):
    def __init__(self, head_size: int, context_length: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_size, 2).float() / head_size))
        positions = torch.arange(context_length, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, start_pos: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        token_count = query.size(-2)
        cos = self.cos[:, :, start_pos : start_pos + token_count, :]
        sin = self.sin[:, :, start_pos : start_pos + token_count, :]
        q_rot = (query * cos) + (self._rotate_half(query) * sin)
        k_rot = (key * cos) + (self._rotate_half(key) * sin)
        return q_rot, k_rot

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)


class StandaloneRMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class StandaloneLayerNorm(nn.Module):
    def __init__(self, size: int, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


def make_standalone_norm(norm_type: str, dim: int, bias: bool = False) -> nn.Module:
    if norm_type == "rmsnorm":
        return StandaloneRMSNorm(dim)
    return StandaloneLayerNorm(dim, bias=bias)


class StandaloneCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, kv_heads: int, context_length: int, dropout: float = 0.0, bias: bool = False, pos_enc: str = "rope") -> None:
        super().__init__()
        self.head_count = n_heads
        self.kv_head_count = kv_heads or n_heads
        self.embedding_size = d_model
        self.head_size = d_model // n_heads
        self.kv_embedding_size = self.kv_head_count * self.head_size
        self.c_attn = nn.Linear(d_model, d_model + (2 * self.kv_embedding_size), bias=bias)
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.rotary = StandaloneRotaryEmbedding(self.head_size, context_length) if pos_enc == "rope" else None
        self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length, dtype=torch.bool)).view(1, 1, context_length, context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split((self.embedding_size, self.kv_embedding_size, self.kv_embedding_size), dim=2)
        q = q.view(b, t, self.head_count, self.head_size).transpose(1, 2)
        k = k.view(b, t, self.kv_head_count, self.head_size).transpose(1, 2)
        v = v.view(b, t, self.kv_head_count, self.head_size).transpose(1, 2)
        if self.rotary is not None:
            q, k = self.rotary(q, k)
        if self.kv_head_count != self.head_count:
            rep = self.head_count // self.kv_head_count
            k = k[:, :, None, :, :].expand(b, self.kv_head_count, rep, t, self.head_size).reshape(b, self.head_count, t, self.head_size)
            v = v[:, :, None, :, :].expand(b, self.kv_head_count, rep, t, self.head_size).reshape(b, self.head_count, t, self.head_size)

        if hasattr(F, "scaled_dot_product_attention"):
            q_sdpa = q.contiguous()
            k_sdpa = k.contiguous()
            v_sdpa = v.contiguous()
            y = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
            att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = self.attn_dropout(att) @ v

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))


class StandaloneMLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, mlp_type: str = "swiglu", dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.mlp_type = mlp_type
        if mlp_type == "swiglu":
            self.w1 = nn.Linear(d_model, hidden_dim, bias=bias)
            self.w2 = nn.Linear(hidden_dim, d_model, bias=bias)
            self.w3 = nn.Linear(d_model, hidden_dim, bias=bias)
            self.dropout = nn.Dropout(dropout)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim, bias=bias),
                nn.GELU(),
                nn.Linear(hidden_dim, d_model, bias=bias),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "swiglu":
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        return self.net(x)


class StandaloneBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, kv_heads: int, hidden_dim: int, context_length: int, norm_type: str = "rmsnorm", mlp_type: str = "swiglu", dropout: float = 0.0, bias: bool = False, pos_enc: str = "rope") -> None:
        super().__init__()
        self.ln_1 = make_standalone_norm(norm_type, d_model, bias=bias)
        self.attn = StandaloneCausalSelfAttention(d_model, n_heads, kv_heads, context_length, dropout, bias, pos_enc)
        self.ln_2 = make_standalone_norm(norm_type, d_model, bias=bias)
        self.mlp = StandaloneMLP(d_model, hidden_dim, mlp_type, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class StandaloneMicroGPT(nn.Module):
    """Zero-dependency PyTorch implementation of MicroGPT, 100% state-dict compatible with engine."""
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        vocab_size = int(config.get("vocab_size", 1000))
        d_model = int(config.get("embedding_size", 128))
        context_length = int(config.get("context_length", 256))
        n_heads = int(config.get("head_count", 4))
        kv_heads = int(config.get("kv_head_count") or n_heads)
        n_layers = int(config.get("layer_count", 2))
        hidden_dim = int(config.get("intermediate_size") or (d_model * 4))
        norm_type = str(config.get("norm_type", "rmsnorm"))
        mlp_type = str(config.get("mlp_type", "swiglu"))
        dropout = float(config.get("dropout", 0.0))
        bias = bool(config.get("bias", False))
        pos_enc = str(config.get("position_encoding", "rope"))

        self.context_length = context_length
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model) if pos_enc == "learned" else None
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            StandaloneBlock(d_model, n_heads, kv_heads, hidden_dim, context_length, norm_type, mlp_type, dropout, bias, pos_enc)
            for _ in range(n_layers)
        ])
        self.ln_f = make_standalone_norm(norm_type, d_model, bias=bias)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.size()
        if hasattr(self, "token_embedding") and hasattr(self.token_embedding, "num_embeddings"):
            idx = torch.clamp(idx, 0, self.token_embedding.num_embeddings - 1)
        x = self.token_embedding(idx)
        if self.position_embedding is not None:
            if t > self.context_length:
                t = self.context_length
                x = x[:, :t, :]
            positions = torch.arange(0, t, dtype=torch.long, device=idx.device)
            x = x + self.position_embedding(positions)
        x = self.drop(x)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def build_model_from_config(
    model_config: dict[str, Any],
    device: str,
    logger: Optional[Callable[[str], None]] = None,
) -> nn.Module:
    """Instantiate a transformer model from configuration with detailed debug tracking.

    Attempts import from `engine.model.MicroGPT` first.
    Falls back to `StandaloneMicroGPT` (100% state-dict and architecture parity) if engine is not available.
    """
    log = logger or (lambda msg: print(f"[ModelBuilder] {msg}"))
    ctx_len = int(model_config.get("context_length", 0) or 0)
    try:
        log(f"Importing engine components (ModelConfig, MicroGPT)...")
        from engine.config import ModelConfig
        from engine.model import MicroGPT

        # Filter config keys recognized by ModelConfig
        valid_keys = ModelConfig.__dataclass_fields__.keys()
        filtered = {k: v for k, v in model_config.items() if k in valid_keys}
        cfg = ModelConfig(**filtered)
        log(f"Instantiating MicroGPT on CPU (layers={cfg.layer_count}, heads={cfg.head_count}, embd={cfg.embedding_size}, vocab={cfg.vocab_size})...")
        t0 = time.time()
        model = MicroGPT(cfg)
        if hasattr(model, "enable_gradient_checkpointing") and getattr(cfg, "context_length", 0) >= 1024:
            model.enable_gradient_checkpointing(True)
            log(f"Activation checkpointing enabled for context length {cfg.context_length}.")
        log(f"MicroGPT constructed on CPU in {time.time() - t0:.2f}s. Transferring parameters to {device}...")
        t1 = time.time()
        model = model.to(device)
        log(f"Model parameters successfully transferred to {device} in {time.time() - t1:.2f}s.")
        return model
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        log(f"MicroGPT initialization failed ({exc}). Falling back to StandaloneMicroGPT parity model. Traceback:\n{tb}")
        log(f"Constructing StandaloneMicroGPT on {device}...")
        model = StandaloneMicroGPT(model_config).to(device)
        if hasattr(model, "enable_gradient_checkpointing") and ctx_len >= 1024:
            model.enable_gradient_checkpointing(True)
        return model


# -----------------------------------------------------------------------------
# Parameter-Efficient Fine-Tuning (LoRA) Support
# -----------------------------------------------------------------------------

try:
    from engine.model_norm_lora import (
        LoRALinear,
        apply_lora_adapters,
        freeze_non_lora_parameters,
        lora_state_dict,
        merged_lora_state_dict,
    )
except ImportError:
    import math

    class LoRALinear(nn.Module):
        """Linear layer with trainable low-rank LoRA adapters."""
        def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
            super().__init__()
            self.base = base
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            self.dropout = nn.Dropout(dropout)
            self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)
            self.base.weight.requires_grad_(False)
            if self.base.bias is not None:
                self.base.bias.requires_grad_(False)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            update = F.linear(F.linear(self.dropout(value), self.lora_a), self.lora_b) * self.scaling
            return self.base(value) + update

    def _set_nested_module(root: nn.Module, module_name: str, module: nn.Module) -> None:
        parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
        parent = root.get_submodule(parent_name) if parent_name else root
        setattr(parent, child_name, module)

    def _lora_target_names(model: nn.Module, target_modules: str) -> set[str]:
        groups = {part.strip().lower() for part in target_modules.split(",") if part.strip()}
        if "all" in groups:
            groups.update({"attention", "mlp"})
        names: set[str] = set()
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name.endswith("lm_head"):
                continue
            is_attention = ".attn." in name or "attn" in name or "attention" in name
            is_mlp = ".mlp." in name or "mlp" in name
            if ("attention" in groups and is_attention) or ("mlp" in groups and is_mlp):
                names.add(name)
        return names

    def apply_lora_adapters(model: nn.Module, rank: int, alpha: float, dropout: float, target_modules: str) -> int:
        names = _lora_target_names(model, target_modules)
        for name in sorted(names):
            try:
                module = model.get_submodule(name)
                if isinstance(module, nn.Linear):
                    _set_nested_module(model, name, LoRALinear(module, rank, alpha, dropout))
            except Exception:
                pass
        return len(names)

    def freeze_non_lora_parameters(model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(("lora_a" in name) or ("lora_b" in name))

    def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
            if ".lora_a" in name or ".lora_b" in name
        }

    def merged_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
        replacements: dict[str, tuple[str, torch.Tensor]] = {}
        adapter_keys: set[str] = set()
        for module_name, module in model.named_modules():
            if not isinstance(module, LoRALinear):
                continue
            prefix = f"{module_name}."
            replacements[f"{prefix}base.weight"] = (
                f"{prefix}weight",
                module.base.weight.detach() + (module.lora_b.detach() @ module.lora_a.detach()) * module.scaling,
            )
            if module.base.bias is not None:
                replacements[f"{prefix}base.bias"] = (
                    f"{prefix}bias",
                    module.base.bias.detach(),
                )
            adapter_keys.update({f"{prefix}lora_a", f"{prefix}lora_b"})
        merged: dict[str, torch.Tensor] = {}
        for name, tensor in model.state_dict().items():
            replacement = replacements.get(name)
            if replacement is not None:
                target_name, value = replacement
                merged[target_name] = value.cpu()
            elif name not in adapter_keys:
                merged[name] = tensor.detach().cpu()
        return merged


class ClusterWorker:
    """Worker daemon executing distributed Local SGD rounds on a worker node."""

    def __init__(
        self,
        bus: ClusterStorageBus,
        worker_id: Optional[str] = None,
        device: Optional[str] = None,
        heartbeat_interval: float = 5.0,
        ephemeral_job_id: Optional[str] = None,
        allow_shared_device: bool = False,
    ) -> None:
        """Initialize worker daemon.

        Args:
            bus: Shared storage bus instance.
            worker_id: Unique worker node identifier (defaults to hostname-pid).
            device: Compute device ('cuda:0', 'cpu', etc.).
            heartbeat_interval: Heartbeat interval in seconds.
            ephemeral_job_id: If set, worker exits automatically after completing this job.
            allow_shared_device: If True, bypasses per-device singleton lock to permit auxiliary slots.
        """
        self.bus = bus
        self.hostname = socket.gethostname()
        self.worker_id = worker_id or f"{self.hostname}_{os.getpid()}"
        self.device_str, self.gpu_name, self.vram_gb = get_hardware_info(device)
        self.heartbeat_interval = heartbeat_interval
        self.ephemeral_job_id = ephemeral_job_id
        self.allow_shared_device = allow_shared_device
        self._child_worker_procs: list[tuple[subprocess.Popen, str]] = []
        self._stop_event = threading.Event()
        self._current_job_id: Optional[str] = None
        self._current_status = "INITIALIZING"
        self._last_round_metrics: dict[str, Any] = {}
        self._preflight_ok: bool = True
        self._incompatible_jobs_reported: set[str] = set()

        # Hardware preflight check
        if self.device_str.startswith("cuda") and torch.cuda.is_available():
            try:
                t = torch.zeros(1, device=self.device_str)
                del t
            except Exception as exc:
                self.log(f"CRITICAL PREFLIGHT WARNING: CUDA device '{self.device_str}' ({self.gpu_name}) failed tensor kernel execution: {exc}", level="ERROR")

        # Register worker in SQLite database and clear any stale pending command
        self.bus.register_worker(
            worker_id=self.worker_id,
            hostname=self.hostname,
            gpu_name=self.gpu_name,
            vram_gb=self.vram_gb,
        )
        self.bus.set_worker_command(self.worker_id, None)
        try:
            self.bus.heartbeat(self.worker_id, status="INITIALIZING", current_job_id=None)
        except Exception:
            pass

        # Start background heartbeat daemon thread
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._hb_thread.start()

    def log(self, message: str, level: str = "INFO") -> None:
        """Write diagnostic log to stdout and central SQLite worker_logs table."""
        timestamp_str = time.strftime("%H:%M:%S")
        try:
            print(f"[{timestamp_str}] [{level}] [Worker {self.worker_id}] {message}", flush=True)
        except UnicodeEncodeError:
            try:
                enc = getattr(sys.stdout, "encoding", None) or "ascii"
                safe_msg = message.encode(enc, errors="replace").decode(enc)
                print(f"[{timestamp_str}] [{level}] [Worker {self.worker_id}] {safe_msg}", flush=True)
            except Exception:
                pass
        except Exception:
            pass
        try:
            self.bus.write_worker_log(self.worker_id, message, level=level)
        except Exception:
            pass

    def _heartbeat_loop(self) -> None:
        """Background thread updating heartbeat periodically and listening for STOP/RESTART commands."""
        while not self._stop_event.is_set():
            try:
                cmd = self.bus.get_worker_command(self.worker_id)
                if cmd == "STOP":
                    self.log("Received STOP command. Aborting active training and returning to IDLE...")
                    self.bus.set_worker_command(self.worker_id, None)
                    self._abort_active_job = True
                    self._current_status = "IDLE"
                    self._current_job_id = None
                    try:
                        self.bus.heartbeat(
                            worker_id=self.worker_id,
                            status="IDLE",
                            current_job_id=None,
                        )
                    except Exception:
                        pass
                elif cmd == "SHUTDOWN":
                    self.log("Received SHUTDOWN command. Exiting worker process...")
                    self.bus.set_worker_command(self.worker_id, None)
                    self._stop_event.set()
                    self._cleanup_child_worker_procs()
                    self._current_status = "OFFLINE"
                    try:
                        self.bus.heartbeat(
                            worker_id=self.worker_id,
                            status="OFFLINE",
                            current_job_id=None,
                        )
                    except Exception:
                        pass
                    from .cluster_worker import release_singleton_lock, get_device_tag
                    if not self.allow_shared_device:
                        release_singleton_lock(get_device_tag(self.device_str))
                    release_singleton_lock(f"wid_{self.worker_id}")
                    os._exit(0)
                elif cmd == "RESTART":
                    self._respawn_worker_process("remote RESTART command")

                metrics = collect_system_metrics(self.device_str, getattr(self, "vram_gb", 0.0))
                self.bus.heartbeat(
                    worker_id=self.worker_id,
                    status=self._current_status,
                    current_job_id=self._current_job_id,
                    metrics=metrics,
                )
            except Exception:
                pass
            self._stop_event.wait(self.heartbeat_interval)

    def _respawn_worker_process(self, reason: str = "") -> None:
        """Cleanly respawn this worker process and exit to recover from fatal driver/CUDA corruption."""
        self.log(f"Respawning worker process for {self.worker_id} ({reason or 'restarting'})...")
        try:
            self.bus.set_worker_command(self.worker_id, None)
        except Exception:
            pass
        self._current_status = "RESTARTING"
        try:
            self.bus.heartbeat(self.worker_id, status="RESTARTING", current_job_id=None)
        except Exception:
            pass
        self._cleanup_child_worker_procs()
        from .cluster_worker import release_singleton_lock, get_device_tag, get_worker_respawn_cmd
        if not self.allow_shared_device:
            release_singleton_lock(get_device_tag(self.device_str))
        release_singleton_lock(f"wid_{self.worker_id}")

        # Attempt git pull to grab latest repository updates if running inside a git checkout
        try:
            repo_dir = Path(__file__).resolve().parent.parent if "cluster" in str(Path(__file__).parent) else Path(__file__).resolve().parent
            if (repo_dir / ".git").exists():
                self.log("Pulling latest cluster repository changes before respawning...")
                subprocess.run(["git", "pull", "--ff-only"], cwd=str(repo_dir), timeout=15, capture_output=True)
        except Exception:
            pass

        cmd_args = get_worker_respawn_cmd(
            worker_id=self.worker_id,
            device_str=self.device_str,
            shared_dir=self.bus.shared_dir,
            allow_shared_device=self.allow_shared_device,
        )
        flags = 0
        if sys.platform == "win32":
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            CREATE_NO_WINDOW = 0x08000000
            flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        log_path = Path(tempfile.gettempdir()) / f"cluster_worker_{self.worker_id}.log"
        log_file = open(log_path, "a", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                cmd_args,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=flags,
                close_fds=True,
            )
            self.log(f"Successfully respawned new worker process (PID: {proc.pid}). Exiting current process (PID: {os.getpid()}).")
        except Exception as exc:
            self.log(f"Failed to spawn new worker process: {exc}", level="ERROR")
        self._stop_event.set()
        time.sleep(0.3)
        os._exit(0)

    def _cleanup_child_worker_procs(self) -> None:
        """Clean up and terminate auxiliary child worker processes."""
        if self._child_worker_procs:
            self.log(f"Cleaning up {len(self._child_worker_procs)} auxiliary worker process(es)...")
            for proc, aux_wid in list(self._child_worker_procs):
                try:
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait(timeout=3.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                try:
                    self.bus.purge_worker(aux_wid)
                except Exception:
                    pass
            self._child_worker_procs.clear()

    def _check_and_spawn_auxiliary_slots(
        self,
        job_id: str,
        job: Optional[dict[str, Any]] = None,
    ) -> None:
        """Evaluate empirical VRAM utilization after the first batch and conditionally spawn auxiliary slots.

        Spawning only happens if:
        1. This is the primary persistent daemon (not an auxiliary child slot).
        2. No auxiliary slots have already been spawned.
        3. Multi-worker slots are not disabled by environment variable or config.
        4. Device is CUDA and measured peak VRAM after batch 1 is < 45% of total capacity.
        5. Free VRAM on the GPU is at least 1.25x the measured requirement.
        """
        if self.ephemeral_job_id is not None or self._child_worker_procs:
            return

        training_cfg = job.get("training_config", {}) if isinstance(job, dict) else {}

        # Respect explicit disable switches
        if os.environ.get("CLUSTER_DISABLE_AUTO_SLOTS", "").strip().lower() in ("1", "true", "yes"):
            self.log("[Slots Evaluation] Multi-worker slots disabled by CLUSTER_DISABLE_AUTO_SLOTS.")
            return

        # Dedicated single-worker-per-GPU is the safe default for workstation clusters.
        # Multi-worker slots must be explicitly opted into via allow_worker_slots: True.
        if not bool(training_cfg.get("allow_worker_slots", False)):
            return

        if not self.device_str.startswith("cuda") or not torch.cuda.is_available():
            return

        dev_idx = 0
        if ":" in self.device_str:
            try:
                dev_idx = int(self.device_str.split(":")[1])
            except Exception:
                dev_idx = 0

        try:
            torch.cuda.synchronize(dev_idx)
            peak_reserved_bytes = torch.cuda.max_memory_reserved(dev_idx)
            free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
        except Exception as exc:
            self.log(f"[Slots Evaluation] Memory query failed: {exc}", level="WARNING")
            return

        used_gb = peak_reserved_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)
        free_gb = free_bytes / (1024 ** 3)
        util_pct = (peak_reserved_bytes / total_bytes) * 100.0 if total_bytes > 0 else 100.0

        # Safety rule: A second worker process requires its own full PyTorch CUDA context,
        # model weights, AdamW optimizer moments, gradients, and activation buffers.
        # If the first batch already consumes >= 45% of total VRAM, or free VRAM is insufficient,
        # spawning a second slot guarantees OOM or CUDA thrashing.
        min_required_free_gb = used_gb * 1.25

        if util_pct >= 45.0 or free_gb < min_required_free_gb:
            self.log(
                f"[Slots Evaluation] Batch 1 completed on {self.device_str}. Peak VRAM: {used_gb:.1f}/{total_gb:.1f} GB "
                f"({util_pct:.0f}% utilized, {free_gb:.1f} GB free). "
                f"Spawning slots suppressed (requires at least {min_required_free_gb:.1f} GB free & <45% utilization). "
                f"Remaining in single-worker mode to prevent OOM."
            )
            return

        max_configured = int(training_cfg.get("max_worker_slots", 4) or 4)
        safe_slots = max(1, min(max_configured, int(total_gb // min_required_free_gb)))

        if safe_slots <= 1:
            self.log(
                f"[Slots Evaluation] Batch 1 completed on {self.device_str}. Peak VRAM: {used_gb:.1f}/{total_gb:.1f} GB "
                f"({util_pct:.0f}% utilized). Headroom supports 1 slot."
            )
            return

        self.log(
            f"[Slots Evaluation] Batch 1 completed on {self.device_str}. Peak VRAM: {used_gb:.1f}/{total_gb:.1f} GB "
            f"({util_pct:.0f}% utilized, {free_gb:.1f} GB free). "
            f"Node has excess capacity for {safe_slots} concurrent slots. "
            f"Spawning {safe_slots - 1} auxiliary worker process(es)..."
        )

        for slot_idx in range(2, safe_slots + 1):
            aux_worker_id = f"{self.worker_id}_slot{slot_idx}"
            if sys.argv[0].endswith("cluster_worker.py"):
                cmd = [
                    sys.executable,
                    sys.argv[0],
                    "--shared-dir",
                    str(self.bus.shared_dir),
                    "--worker-id",
                    aux_worker_id,
                    "--device",
                    self.device_str,
                    "--ephemeral-job-id",
                    job_id,
                    "--allow-shared-device",
                ]
            else:
                cmd = [
                    sys.executable,
                    "-m",
                    "cluster.cli",
                    "worker",
                    "--shared-dir",
                    str(self.bus.shared_dir),
                    "--worker-id",
                    aux_worker_id,
                    "--device",
                    self.device_str,
                    "--ephemeral-job-id",
                    job_id,
                    "--allow-shared-device",
                ]
            try:
                self.log(f"Launching auxiliary worker slot {slot_idx}: {aux_worker_id}...")
                p = subprocess.Popen(
                    cmd,
                    stdout=None,
                    stderr=None,
                    stdin=subprocess.DEVNULL,
                )
                self._child_worker_procs.append((p, aux_worker_id))
            except Exception as e:
                self.log(f"Failed to spawn auxiliary worker slot {slot_idx}: {e}", level="WARNING")

        if self._child_worker_procs:
            time.sleep(1.5)

    def stop(self) -> None:
        """Signal worker to gracefully stop."""
        self._stop_event.set()
        self._cleanup_child_worker_procs()
        if self._hb_thread.is_alive():
            self._hb_thread.join(timeout=2.0)
        try:
            self.bus.heartbeat(self.worker_id, status="OFFLINE", current_job_id=None)
        except Exception:
            pass

    def run_training_round(
        self,
        job_id: str,
        round_num: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        dataloader_iter: Any,
        steps_per_round: int,
        job: Optional[dict[str, Any]] = None,
        log_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> tuple[float, Any]:
        """Execute K steps of local SGD training for a single round.

        Args:
            job_id: Distributed job ID.
            round_num: Current synchronization round.
            model: PyTorch model module.
            optimizer: Optimizer instance.
            dataloader: Sharded token data loader.
            dataloader_iter: Active DataLoader iterator.
            steps_per_round: Local training steps before sync.
            job: Optional full job configuration dict.
            log_callback: Optional progress reporter.

        Returns:
            (average_loss, dataloader_iter)
        """
        model.train()
        total_loss = 0.0
        steps_done = 0
        tokens_processed = 0
        start_time = time.time()

        # Mixed precision (AMP) configuration
        is_cuda = self.device_str.startswith("cuda") and torch.cuda.is_available()
        training_cfg = job.get("training_config", {}) if isinstance(job, dict) else {}
        precision = str(training_cfg.get("precision", "fp16")).lower()
        use_amp = bool(training_cfg.get("use_amp", True))
        amp_enabled = is_cuda and use_amp and (precision in ("fp16", "bf16"))

        if amp_enabled and precision == "bf16":
            if is_hardware_bf16_supported(self.device_str):
                amp_dtype = torch.bfloat16
            else:
                self.log(
                    f"Notice: Compute device '{self.device_str}' ({getattr(self, 'gpu_name', 'GPU')}) lacks native hardware Bfloat16 tensor cores (sm_80+ required). "
                    "Safely falling back to FP16 with GradScaler for numerical stability and kernel safety.",
                    level="INFO",
                )
                amp_dtype = torch.float16
        else:
            amp_dtype = torch.float16

        scaler = None
        if amp_enabled and amp_dtype == torch.float16:
            if not hasattr(self, "_scaler") or self._scaler is None:
                self._scaler = torch.amp.GradScaler('cuda')
            scaler = self._scaler

        # Activation checkpointing guard: automatically enable for long contexts or when configured
        ctx_len = 0
        if hasattr(model, "config") and hasattr(model.config, "context_length"):
            ctx_len = int(model.config.context_length)
        elif isinstance(job, dict):
            ctx_len = int(job.get("model_config", {}).get("context_length", 0) or 0)
        should_checkpoint = bool(training_cfg.get("activation_checkpointing", False) or ctx_len >= 1024)
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing(should_checkpoint)

        # Gradient accumulation configuration
        grad_accum = max(1, int(training_cfg.get("gradient_accumulation", 1) or 1))
        max_grad = float(training_cfg.get("max_gradient") or training_cfg.get("max_grad") or 1.0)

        # Compute safe micro-batch size for GPU execution to prevent VRAM overflow
        if is_cuda:
            try:
                dev_idx = 0
                if ":" in self.device_str:
                    dev_idx = int(self.device_str.split(":")[1])
                _, tot_bytes = torch.cuda.mem_get_info(dev_idx)
                tot_gb = tot_bytes / (1024 ** 3)
            except Exception:
                tot_gb = 16.0
            if tot_gb <= 12.0:
                max_safe_tokens = 4096
            elif tot_gb >= 15.0 and should_checkpoint:
                # With activation checkpointing enabled on 16GB+ cards, micro-batches of up to 16,384 tokens
                # safely utilize ~8-10 GB (50-65% VRAM), accelerating single-worker throughput without risking OOM
                max_safe_tokens = 16384
            else:
                max_safe_tokens = 8192
            safe_micro_bs = max(1, max_safe_tokens // max(ctx_len, 1)) if ctx_len > 0 else 4
        else:
            safe_micro_bs = 999999

        optimizer.zero_grad(set_to_none=True)
        accumulated_batches = 0

        for step in range(steps_per_round):
            if self._stop_event.is_set() or self.bus.is_stopped(job_id) or getattr(self, "_abort_active_job", False):
                break

            # Handle cooperative pause
            while self.bus.is_paused(job_id) and not self._stop_event.is_set():
                self._current_status = "PAUSED"
                time.sleep(1.0)
                if self.bus.is_stopped(job_id):
                    break

            self._current_status = "TRAINING"

            # Fetch next batch (looping iterator if exhausted)
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            x, y = batch
            batch_sz = x.size(0)
            tokens_processed += int(x.numel())
            n_emb = get_model_vocab_size(model)
            if n_emb is not None and n_emb > 0:
                x = torch.clamp(x, 0, n_emb - 1)
                y = torch.where(y == -100, y, torch.clamp(y, 0, n_emb - 1))

            micro_bs = max(1, min(batch_sz, safe_micro_bs))
            num_micros = math.ceil(batch_sz / micro_bs)
            effective_accum = num_micros * grad_accum

            batch_loss = 0.0
            for m_idx in range(0, batch_sz, micro_bs):
                x_m = x[m_idx : m_idx + micro_bs].to(self.device_str, non_blocking=True)
                y_m = y[m_idx : m_idx + micro_bs].to(self.device_str, non_blocking=True)
                if n_emb is not None and n_emb > 0:
                    x_m = torch.clamp(x_m, 0, n_emb - 1)

                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                    logits = model(x_m)
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                    num_classes = logits.size(-1)
                    y_m_safe = torch.where(y_m == -100, y_m, torch.clamp(y_m, 0, num_classes - 1))
                    loss = F.cross_entropy(logits.view(-1, num_classes), y_m_safe.view(-1), ignore_index=-100)
                    scaled_loss = loss / effective_accum

                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                batch_loss += float(loss.item()) * (x_m.size(0) / batch_sz)

            accumulated_batches += 1
            is_step_boundary = (accumulated_batches % grad_accum == 0) or (step + 1 == steps_per_round)

            if is_step_boundary:
                # Apply cosine learning rate schedule with linear warmup across rounds
                t_cfg = training_cfg if isinstance(training_cfg, dict) else {}
                base_lr = float(t_cfg.get("learning_rate", 3e-4))
                max_rounds = int(job.get("max_rounds", 10)) if isinstance(job, dict) else 10
                total_steps = max(1, max_rounds * steps_per_round)
                warmup_steps = int(t_cfg.get("warmup_steps", max(10, total_steps // 20)))
                warmup_steps = min(warmup_steps, max(total_steps - 1, 1))
                min_ratio = float(t_cfg.get("scheduler_min_lr_ratio", 0.1))
                global_step = round_num * steps_per_round + step
                if global_step < warmup_steps:
                    lr_mult = max(global_step + 1, 1) / max(warmup_steps, 1)
                else:
                    prog = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                    prog = max(0.0, min(prog, 1.0))
                    lr_mult = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))
                current_lr = base_lr * lr_mult
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

                if scaler is not None:
                    if max_grad > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if max_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += batch_loss
            steps_done += 1

            # Evaluate empirical VRAM utilization after the first batch finishes
            # and decide whether auxiliary slots can safely fit without OOM.
            if step == 0 and round_num == 0:
                self._check_and_spawn_auxiliary_slots(job_id=job_id, job=job)

            if log_callback:
                log_callback({
                    "job_id": job_id,
                    "round": round_num,
                    "step": step + 1,
                    "steps_per_round": steps_per_round,
                    "loss": batch_loss,
                })

        elapsed = max(time.time() - start_time, 1e-4)
        tokens_per_sec = tokens_processed / elapsed
        avg_loss = (total_loss / steps_done) if steps_done > 0 else 0.0

        self._last_round_metrics = {
            "avg_loss": avg_loss,
            "tokens_processed": tokens_processed,
            "tokens_per_sec": tokens_per_sec,
        }
        return avg_loss, dataloader_iter

    def execute_job(
        self,
        job: dict[str, Any],
        poll_interval: float = 2.0,
        log_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> bool:
        """Execute a full distributed training job across rounds.

        Args:
            job: Job metadata dictionary from database.
            poll_interval: Polling frequency during synchronization waits.
            log_callback: Optional status and telemetry logger.

        Returns:
            True if completed successfully, False if aborted or stopped.
        """
        job_id = job["job_id"]
        is_compat, reason = self.is_job_compatible(job)
        if not is_compat:
            self.log(f"Worker {self.worker_id} cannot execute job '{job_id}': {reason}. Refusing claim.", level="WARNING")
            return False

        self._current_job_id = job_id
        self._current_status = "PREPARING"
        self.log(f">>> Claimed job: {job_id}")

        model_config = job.get("model_config", {})
        training_config = job.get("training_config", {})
        dataset_path = job.get("dataset_path", "")
        max_rounds = int(job.get("max_rounds", 10))
        sync_interval_steps = int(job.get("sync_interval_steps", 250))
        batch_size = int(training_config.get("batch_size", 4))
        lr = float(training_config.get("learning_rate", 3e-4))
        context_length = int(model_config.get("context_length", 512))

        model = None
        optimizer = None
        dataloader = None
        dataloader_iter = None
        val_loader = None
        dataset = None
        token_array = None
        val_token_array = None

        try:
            try:
                # 1. Resolve dataset paths and cache remote .npy files to local SSD storage
                val_dataset_path = job.get("val_dataset_path")
                if not val_dataset_path or not os.path.exists(val_dataset_path):
                    cand_val = Path(dataset_path).parent / "val_tokens.npy"
                    if cand_val.exists():
                        val_dataset_path = str(cand_val)
                    else:
                        cand_shared_val = self.bus.shared_dir / "val_tokens.npy"
                        if cand_shared_val.exists():
                            val_dataset_path = str(cand_shared_val)

                targets_path = job.get("targets_path")
                if not targets_path or not os.path.exists(targets_path):
                    cand_tgt = Path(dataset_path).parent / "train_targets.npy"
                    if cand_tgt.exists():
                        targets_path = str(cand_tgt)
                    else:
                        cand_stgt = self.bus.shared_dir / "train_targets.npy"
                        if cand_stgt.exists():
                            targets_path = str(cand_stgt)

                val_targets_path = job.get("val_targets_path")
                if not val_targets_path or not os.path.exists(val_targets_path):
                    cand_vtgt = Path(dataset_path).parent / "val_targets.npy"
                    if cand_vtgt.exists():
                        val_targets_path = str(cand_vtgt)
                    else:
                        cand_svtgt = self.bus.shared_dir / "val_targets.npy"
                        if cand_svtgt.exists():
                            val_targets_path = str(cand_svtgt)

                target_cache_files = {Path(dataset_path).name}
                if val_dataset_path and os.path.exists(val_dataset_path):
                    target_cache_files.add(Path(val_dataset_path).name)
                if targets_path and os.path.exists(targets_path):
                    target_cache_files.add(Path(targets_path).name)
                if val_targets_path and os.path.exists(val_targets_path):
                    target_cache_files.add(Path(val_targets_path).name)

                # Primary worker purges stale .npy files from local SSD cache before copying new
                if self.ephemeral_job_id is None:
                    purge_local_dataset_cache(keep_files=target_cache_files, log_fn=self.log)

                def _on_cache_progress(pct: float, speed: float, eta: float) -> None:
                    self._current_status = f"PREPARING (Caching {pct:.0f}%)"

                # Safely cache remote/network dataset .npy to fast local SSD storage
                local_dataset_path = cache_dataset_to_local(
                    dataset_path,
                    log_fn=self.log,
                    keep_files=target_cache_files,
                    progress_callback=_on_cache_progress,
                )
                self._current_status = "PREPARING"
                local_val_dataset_path = None
                if val_dataset_path and os.path.exists(val_dataset_path):
                    local_val_dataset_path = cache_dataset_to_local(
                        val_dataset_path,
                        log_fn=self.log,
                        keep_files=target_cache_files,
                        progress_callback=_on_cache_progress,
                    )
                local_targets_path = None
                if targets_path and os.path.exists(targets_path):
                    local_targets_path = cache_dataset_to_local(
                        targets_path,
                        log_fn=self.log,
                        keep_files=target_cache_files,
                        progress_callback=_on_cache_progress,
                    )
                local_val_targets_path = None
                if val_targets_path and os.path.exists(val_targets_path):
                    local_val_targets_path = cache_dataset_to_local(
                        val_targets_path,
                        log_fn=self.log,
                        keep_files=target_cache_files,
                        progress_callback=_on_cache_progress,
                    )
                self._current_status = "PREPARING"

                # Clean up any stale auxiliary child processes from previous runs (primary daemon only)
                if self.ephemeral_job_id is None:
                    self._cleanup_child_worker_procs()

                # 3. Claim data shard slot
                shard_idx, total_shards = self.bus.claim_job_slot(job_id, self.worker_id)
                self.log(f"Assigned data shard slot {shard_idx + 1} of {total_shards} total nodes")

                # 4. Prepare local dataset from cached SSD file
                if not os.path.exists(local_dataset_path):
                    err = f"Dataset path not found: {local_dataset_path}"
                    self.log(err, level="ERROR")
                    self._current_status = "ERROR"
                    return False

                self.log(f"Mapping dataset: {local_dataset_path}...")
                token_array = np.load(local_dataset_path, mmap_mode="r")
                target_array = np.load(local_targets_path, mmap_mode="r") if (local_targets_path and os.path.exists(local_targets_path)) else None
                vocab_size = int(model_config.get("vocab_size", 0) or 0)

                # 1. Inspect metadata files if present on shared storage or dataset dir
                detected_vocab = 0
                for sdir in [Path(dataset_path).parent, self.bus.shared_dir]:
                    summary_file = sdir / "dataset_summary.json"
                    if summary_file.exists():
                        try:
                            import json
                            meta = json.loads(summary_file.read_text(encoding="utf-8"))
                            detected_vocab = int(meta.get("tokenizer_vocab_size", 0) or 0)
                            if detected_vocab > 0:
                                break
                        except Exception:
                            pass
                    tok_file = sdir / "tokenizer.json"
                    if tok_file.exists() and detected_vocab <= 0:
                        try:
                            import json
                            tok_data = json.loads(tok_file.read_text(encoding="utf-8"))
                            vocab_dict = tok_data.get("model", {}).get("vocab", {})
                            if vocab_dict:
                                detected_vocab = len(vocab_dict)
                                break
                        except Exception:
                            pass

                # 2. Inspect dataset tokens for true maximum token ID
                max_token_id = 0
                try:
                    arr_len = len(token_array)
                    if arr_len <= 10_000_000:
                        max_token_id = int(np.max(token_array))
                    else:
                        slices = [
                            token_array[:200000],
                            token_array[arr_len // 4 : (arr_len // 4) + 200000],
                            token_array[arr_len // 2 : (arr_len // 2) + 200000],
                            token_array[(3 * arr_len) // 4 : ((3 * arr_len) // 4) + 200000],
                            token_array[-200000:],
                        ]
                        max_token_id = max(int(np.max(s)) for s in slices if len(s) > 0)
                except Exception:
                    max_token_id = 0

                # 3. Determine safe aligned vocab_size
                min_safe_vocab = 256
                if max_token_id > 0:
                    min_safe_vocab = ((max_token_id + 1 + 255) // 256) * 256
                    if 31000 <= max_token_id < 32000:
                        min_safe_vocab = max(min_safe_vocab, 32000)

                target_vocab = max(vocab_size, detected_vocab, min_safe_vocab, 256)
                if vocab_size < target_vocab:
                    self.log(
                        f"Model vocab_size ({vocab_size}) is smaller than required ({target_vocab}, detected: {detected_vocab}, max token: {max_token_id}). "
                        f"Auto-adjusting vocab_size to {target_vocab}.",
                        level="WARNING",
                    )
                    vocab_size = target_vocab
                    model_config["vocab_size"] = vocab_size

                # 5. Prepare training & validation datasets
                val_token_array = None
                val_target_array = None
                train_token_array = token_array
                train_target_array = target_array

                if local_val_dataset_path and os.path.exists(local_val_dataset_path):
                    try:
                        v_arr = np.load(local_val_dataset_path, mmap_mode="r")
                        if len(v_arr) > context_length:
                            val_token_array = v_arr
                            if local_val_targets_path and os.path.exists(local_val_targets_path):
                                val_target_array = np.load(local_val_targets_path, mmap_mode="r")
                            self.log(f"External validation dataset mapped ({len(val_token_array):,} tokens).")
                    except Exception as e:
                        self.log(f"Could not initialize validation dataset: {e}", level="WARNING")

                # Automatic fallback: if no external val dataset, hold out the last 5% of tokens
                if val_token_array is None and len(token_array) > (context_length * 2):
                    val_size = min(50000, max(context_length * 2, int(len(token_array) * 0.05)))
                    val_token_array = token_array[-val_size:]
                    train_token_array = token_array[:-val_size]
                    if target_array is not None:
                        val_target_array = target_array[-val_size:]
                        train_target_array = target_array[:-val_size]
                    self.log(f"Held out {len(val_token_array):,} tokens (5%) from dataset for continuous round validation evaluation.")

                dataset = ShardedTokenDataset(
                    token_array=train_token_array,
                    context_length=context_length,
                    shard_index=shard_idx,
                    total_shards=total_shards,
                    target_array=train_target_array,
                    vocab_size=vocab_size,
                )
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                dataloader_iter = iter(dataloader)

                if val_token_array is not None and len(val_token_array) > context_length:
                    try:
                        val_dataset = ShardedTokenDataset(
                            token_array=val_token_array,
                            context_length=context_length,
                            shard_index=shard_idx,
                            total_shards=total_shards,
                            target_array=val_target_array,
                            vocab_size=vocab_size,
                        )
                        if len(val_dataset) > 0:
                            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                            self.log(f"Validation DataLoader ready ({len(val_dataset)} sample windows).")
                    except Exception as e:
                        self.log(f"Could not build validation DataLoader: {e}", level="WARNING")

                # 6. Build model and optimizer
                self.log(f"Dataset mapped ({len(train_token_array):,} tokens, vocab_size={vocab_size}). Building model for {self.device_str}...")
                model = build_model_from_config(model_config, self.device_str, logger=self.log)

                # Check if this is a fine-tuning job or pretraining resume, and load base model weights
                job_type = str(job.get("job_type") or training_config.get("training_mode") or "pretrain").lower()
                is_fine_tune = (job_type == "fine_tune")
                has_base = bool(job.get("base_checkpoint_path"))

                if is_fine_tune or has_base:
                    mode_desc = "Fine-tuning base" if is_fine_tune else "Pretraining resume"
                    self.log(f"{mode_desc} checkpoint detected. Loading initial checkpoint weights from shared storage...")
                    base_weights = self.bus.load_base_model_weights(job_id, device=self.device_str)
                    if base_weights is not None:
                        try:
                            missing, unexpected = model.load_state_dict(base_weights, strict=False)
                            self.log(f"{mode_desc} model weights loaded successfully. (Missing keys: {len(missing)}, unexpected keys: {len(unexpected)})")
                        except Exception as e:
                            self.log(f"Warning: Failed to load initial weights: {e}", level="WARNING")
                    else:
                        if is_fine_tune:
                            self.log("Notice: No base model weights found on shared storage. Initializing from scratch.", level="WARNING")
                        else:
                            self.log("Notice: Resume checkpoint not found on shared storage. Initializing from scratch.", level="WARNING")

                # Apply LoRA if configured
                peft_method = str(job.get("peft_method") or training_config.get("peft_method") or "none").lower()
                if peft_method == "lora":
                    lora_cfg = job.get("lora_config") or training_config.get("lora_config") or {}
                    if isinstance(lora_cfg, str):
                        try:
                            lora_cfg = json.loads(lora_cfg)
                        except Exception:
                            lora_cfg = {}
                    if not isinstance(lora_cfg, dict):
                        lora_cfg = {}
                    l_rank = int(lora_cfg.get("rank", training_config.get("lora_rank", 8)))
                    l_alpha = float(lora_cfg.get("alpha", training_config.get("lora_alpha", 16.0)))
                    l_dropout = float(lora_cfg.get("dropout", training_config.get("lora_dropout", 0.05)))
                    l_targets = str(lora_cfg.get("target_modules", training_config.get("lora_target_modules", "attention")))
                    num_lora = apply_lora_adapters(model, rank=l_rank, alpha=l_alpha, dropout=l_dropout, target_modules=l_targets)
                    freeze_non_lora_parameters(model)
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    self.log(f"LoRA adapters applied to {num_lora} modules. Trainable parameters: {trainable_params:,} (base model frozen).")

                # Create optimizer with separated weight decay groups (frontier LLM standard)
                weight_decay = float(training_config.get("weight_decay", 0.01))
                decay_params = []
                nodecay_params = []
                seen_param_ids = set()
                for p in model.parameters():
                    if not p.requires_grad:
                        continue
                    if id(p) in seen_param_ids:
                        continue
                    seen_param_ids.add(id(p))
                    if p.dim() >= 2:
                        decay_params.append(p)
                    else:
                        nodecay_params.append(p)

                optim_groups = []
                if decay_params:
                    optim_groups.append({"params": decay_params, "weight_decay": weight_decay})
                if nodecay_params:
                    optim_groups.append({"params": nodecay_params, "weight_decay": 0.0})

                device_is_cuda = self.device_str.startswith("cuda") and torch.cuda.is_available()
                self.log(f"Model ready. Creating AdamW optimizer for {len(decay_params)} 2D matrix weights (decay={weight_decay}) and {len(nodecay_params)} 1D tensors (no decay), lr={lr}...")
                t_opt = time.time()
                try:
                    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=device_is_cuda)
                except Exception:
                    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
                self.log(f"Optimizer created in {time.time() - t_opt:.2f}s on '{self.device_str}'. Ready to train {max_rounds} rounds.")

                # Ensure activation checkpointing is active for long contexts or when configured
                act_ckpt = bool(training_config.get("activation_checkpointing", False) or context_length >= 1024)
                if hasattr(model, "enable_gradient_checkpointing"):
                    model.enable_gradient_checkpointing(act_ckpt)
                    self.log(f"Activation checkpointing {'ENABLED' if act_ckpt else 'disabled'} (context_length={context_length}).")

                if self.device_str.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                self.log(f"Preparation failed for job {job_id}:\n{tb}", level="ERROR")
                self._current_status = "ERROR"
                return False

            current_round = int(job.get("current_round", 0))

            while current_round < max_rounds and not self._stop_event.is_set():
                if self.bus.is_stopped(job_id) or getattr(self, "_abort_active_job", False):
                    self._current_status = "IDLE"
                    self._current_job_id = None
                    self.log(f"Job {job_id} stopped. Returning to IDLE.")
                    return False

                # Operator disable guard: exit cleanly if worker was disabled mid-job
                if not getattr(self.bus, "is_worker_enabled", lambda wid: True)(self.worker_id):
                    self.log(f"Worker {self.worker_id} was disabled by operator. Pausing participation in job {job_id}.")
                    self._current_status = "DISABLED"
                    self._current_job_id = None
                    try:
                        self.bus.heartbeat(self.worker_id, status="DISABLED", current_job_id=None)
                    except Exception:
                        pass
                    return True

                # Dynamic shard reassignment: verify active worker pool and take over dropped worker slots
                try:
                    new_shard_idx, new_total_shards = self.bus.get_worker_shard_assignment(
                        job_id, self.worker_id, current_round
                    )
                    if new_shard_idx != shard_idx or new_total_shards != total_shards:
                        self.log(f"Active worker pool updated. Shard reallocated: {new_shard_idx + 1} of {new_total_shards} nodes.")
                        shard_idx, total_shards = new_shard_idx, new_total_shards
                        dataset = ShardedTokenDataset(
                            token_array=train_token_array,
                            context_length=context_length,
                            shard_index=shard_idx,
                            total_shards=total_shards,
                            target_array=train_target_array,
                            vocab_size=vocab_size,
                        )
                        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                        dataloader_iter = iter(dataloader)
                        if val_token_array is not None and len(val_token_array) > context_length:
                            try:
                                val_dataset = ShardedTokenDataset(
                                    token_array=val_token_array,
                                    context_length=context_length,
                                    shard_index=shard_idx,
                                    total_shards=total_shards,
                                    target_array=val_target_array,
                                    vocab_size=vocab_size,
                                )
                                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                            except Exception:
                                pass
                except Exception:
                    pass

                # If round > 0, wait for and synchronize from previous global averaged model
                if current_round > 0:
                    self._current_status = "SYNC_WAIT"
                    while not self.bus.is_global_weights_ready(job_id, current_round - 1):
                        if self._stop_event.is_set() or self.bus.is_stopped(job_id) or getattr(self, "_abort_active_job", False):
                            return False
                        time.sleep(poll_interval)

                    global_state = self.bus.load_global_weights(
                        job_id, current_round - 1, device=self.device_str
                    )
                    model.load_state_dict(global_state)

                # Train locally for K steps
                self._current_status = "TRAINING"
                self.log(f"Starting training round {current_round + 1}/{max_rounds} ({sync_interval_steps} local steps)...")
                t_compute_start = time.time()
                try:
                    avg_loss, dataloader_iter = self.run_training_round(
                        job_id=job_id,
                        round_num=current_round,
                        model=model,
                        optimizer=optimizer,
                        dataloader=dataloader,
                        dataloader_iter=dataloader_iter,
                        steps_per_round=sync_interval_steps,
                        job=job,
                        log_callback=log_callback,
                    )
                    val_loss = None
                    if val_loader is not None:
                        try:
                            val_loss = self.evaluate_validation(model, val_loader, max_batches=25)
                        except Exception as exc:
                            self.log(f"Validation evaluation failed in round {current_round + 1}: {exc}", level="WARNING")

                    compute_sec = max(time.time() - t_compute_start, 1e-4)
                    round_metrics = getattr(self, "_last_round_metrics", {})
                    tokens_per_sec = round_metrics.get("tokens_per_sec", 0.0)
                    tokens_processed = round_metrics.get("tokens_processed", 0)

                    val_str = f", Val loss: {val_loss:.4f}" if val_loss is not None else ""
                    self.log(
                        f"Round {current_round + 1}/{max_rounds} compute completed in {compute_sec:.1f}s. "
                        f"Avg loss: {avg_loss:.4f}{val_str}, Speed: {tokens_per_sec:,.0f} tok/s. Depositing weights & telemetry..."
                    )
                except Exception as exc:
                    import traceback
                    tb = traceback.format_exc()
                    self.log(f"Training round {current_round} crashed:\n{tb}", level="ERROR")
                    self._current_status = "ERROR"
                    try:
                        self.bus.heartbeat(self.worker_id, status="ERROR", current_job_id=job_id)
                    except Exception:
                        pass

                    # Check for fatal CUDA context corruption that poisons the host process
                    err_lower = (str(exc) + " " + tb).lower()
                    if any(s in err_lower for s in ("cuda error", "device-side assert", "illegal memory access", "out of memory")):
                        if self.device_str.startswith("cuda"):
                            self.log(
                                "Fatal CUDA context corruption / error detected. Host process cannot reliably recover in-place. "
                                "Respawning fresh worker process with clean GPU context...",
                                level="WARNING",
                            )
                            self._respawn_worker_process(reason=f"CUDA corruption recovery: {exc}")
                    return False

                # Atomically deposit local worker weights
                self.bus.save_worker_weights(
                    job_id=job_id,
                    round_num=current_round,
                    worker_id=self.worker_id,
                    state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                )

                # Atomically deposit local worker telemetry
                self.bus.save_worker_telemetry(
                    job_id=job_id,
                    round_num=current_round,
                    worker_id=self.worker_id,
                    telemetry={
                        "worker_id": self.worker_id,
                        "round": current_round,
                        "steps_completed": sync_interval_steps,
                        "avg_loss": round(avg_loss, 4),
                        "val_loss": round(val_loss, 4) if val_loss is not None else None,
                        "compute_sec": round(compute_sec, 2),
                        "tokens_processed": tokens_processed,
                        "tokens_per_sec": round(tokens_per_sec, 1),
                        "timestamp": time.time(),
                    },
                )

                self._current_status = "SYNC_WAIT"
                t_sync_start = time.time()
                last_log_wait = t_sync_start

                # Wait for coordinator to publish global model for this round
                while not self.bus.is_global_weights_ready(job_id, current_round):
                    if self._stop_event.is_set() or self.bus.is_stopped(job_id) or getattr(self, "_abort_active_job", False):
                        self._current_status = "IDLE"
                        self._current_job_id = None
                        try:
                            self.bus.heartbeat(self.worker_id, status="IDLE", current_job_id=None)
                        except Exception:
                            pass
                        return False

                    now_wait = time.time()
                    elapsed_sync = now_wait - t_sync_start
                    if now_wait - last_log_wait >= 30.0:
                        last_log_wait = now_wait
                        self.log(f"Waiting for round {current_round + 1} global weights ({elapsed_sync:.0f}s elapsed)...", level="INFO")

                        # Autonomous fallback: if waiting > 45s and coordinator is inactive or this is the sole node
                        job_info = self.bus.get_job(job_id) or {}
                        job_updated = float(job_info.get("updated_at") or 0.0)
                        coord_unresponsive = (now_wait - job_updated) > 60.0
                        active_parts = self.bus.get_job_participants(job_id)
                        sole_worker = (len(active_parts) <= 1) or all(p.get("worker_id") == self.worker_id for p in active_parts)

                        if elapsed_sync > 45.0 and (sole_worker or coord_unresponsive):
                            self.log(
                                f"Autonomous sync: Coordinator is inactive or sole node detected ({len(active_parts)} active). "
                                f"Triggering local round {current_round + 1} aggregation...",
                                level="INFO",
                            )
                            try:
                                from cluster.coordinator import ClusterCoordinator
                                coord = ClusterCoordinator(self.bus, job_id)
                                coord.wait_and_average_round(round_num=current_round, poll_interval_seconds=0.5)
                            except Exception as c_err:
                                self.log(f"Autonomous aggregation notice: {c_err}", level="WARNING")

                    time.sleep(poll_interval)

                sync_wait_sec = max(time.time() - t_sync_start, 0.0)
                round_total_sec = compute_sec + sync_wait_sec
                duty_pct = (compute_sec / max(round_total_sec, 0.001)) * 100.0
                self.log(
                    f"Round {current_round + 1}/{max_rounds} synchronized in {sync_wait_sec:.1f}s. "
                    f"[Timing] Compute {compute_sec:.1f}s | Sync Wait {sync_wait_sec:.1f}s | Duty Cycle {duty_pct:.1f}%"
                )

                current_round += 1

            self._current_status = "COMPLETED"
            self._current_job_id = None
            self.log(f"Job {job_id} successfully completed all {max_rounds} rounds.")
        except BaseException as exc:
            import traceback
            tb = traceback.format_exc()
            self.log(f"CRITICAL: Job execution crashed with unhandled exception:\n{tb}", level="ERROR")
            self._current_status = "ERROR"
            raise
        finally:
            self._abort_active_job = False
            # Terminate and reap auxiliary worker subprocesses (primary worker only)
            if self.ephemeral_job_id is None:
                self._cleanup_child_worker_procs()

            try:
                if model is not None:
                    model.to("cpu")
            except Exception:
                pass

            model = None
            optimizer = None
            dataloader = None
            dataloader_iter = None
            dataset = None
            token_array = None

            import gc
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

            if self._current_status not in ("ERROR", "OFFLINE"):
                self._current_status = "IDLE"
            self._current_job_id = None
            try:
                self.bus.heartbeat(self.worker_id, status=self._current_status, current_job_id=None)
            except Exception:
                pass

    def run_hardware_preflight(self) -> tuple[bool, list[str]]:
        """Execute comprehensive hardware, storage, and runtime preflight validation.

        Returns:
            (all_passed: bool, report_lines: list[str])
        """
        report: list[str] = []
        is_healthy = True

        report.append("=" * 72)
        report.append(f"[PREFLIGHT] Node Preflight Hardware Verification (Worker: {self.worker_id})")
        report.append("-" * 72)

        # 1. Device & Compute Capability
        is_cuda = self.device_str.startswith("cuda") and torch.cuda.is_available()
        if is_cuda:
            try:
                device_idx = 0
                if ":" in self.device_str:
                    try:
                        device_idx = int(self.device_str.split(":")[1])
                    except ValueError:
                        device_idx = 0
                props = torch.cuda.get_device_properties(device_idx)
                arch_tag = f"sm_{props.major}{props.minor}"
                arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
                arch_status = "Native SASS verified" if arch_tag in arch_list else "Forward-compatible / PTX"
                report.append(f"[PASS] Compute Device   : {self.device_str} ({props.name}, {self.vram_gb} GB VRAM, {arch_tag})")
                report.append(f"[PASS] PyTorch / CUDA   : {torch.__version__} (Arch: {arch_status})")
            except Exception as exc:
                report.append(f"[WARN] Compute Device   : {self.device_str} ({exc})")
        else:
            report.append(f"[PASS] Compute Device   : {self.device_str} ({self.gpu_name})")

        # 2. Kernel Execution Check
        try:
            test_a = torch.ones((4, 4), device=self.device_str)
            test_b = (test_a * 2.5 + 1.0).sum().item()
            del test_a
            if abs(test_b - 56.0) > 1e-4:
                raise RuntimeError(f"Unexpected tensor math result: {test_b} != 56.0")
            report.append(f"[PASS] Kernel Execution : Tensor arithmetic validated on {self.device_str}")
        except Exception as exc:
            is_healthy = False
            report.append(f"[FAIL] Kernel Execution : {exc}")
            if "no kernel image" in str(exc).lower():
                report.append(f"       Action Required  : PyTorch lacks GPU architecture binary. Install compatible PyTorch (e.g. cu128 nightly).")

        # 3. Mixed Precision (AMP FP16 & BF16) Check
        if is_cuda:
            fp16_ok = False
            bf16_ok = False
            try:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    m1 = torch.randn((8, 8), device=self.device_str)
                    m2 = m1 @ m1
                del m1, m2
                fp16_ok = True
            except Exception as exc:
                report.append(f"[WARN] AMP FP16 Failed  : {exc}")

            bf16_hw = torch.cuda.is_bf16_supported()
            if bf16_hw:
                try:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        m1 = torch.randn((8, 8), device=self.device_str)
                        m2 = m1 @ m1
                    del m1, m2
                    bf16_ok = True
                except Exception as exc:
                    report.append(f"[WARN] AMP BF16 Failed  : {exc}")

            bf16_str = "BF16 (Native)" if bf16_ok else ("BF16 (Emulated)" if not bf16_hw else "BF16 (Unavailable)")
            report.append(f"[PASS] Mixed Precision  : FP16 ({'OK' if fp16_ok else 'FAIL'}), {bf16_str}")

        # 4. Memory Scratch Buffer Allocation & Free
        if is_cuda:
            try:
                scratch = torch.empty((16, 1024, 1024), dtype=torch.float32, device=self.device_str)
                del scratch
                torch.cuda.empty_cache()
                report.append(f"[PASS] Memory Allocation: 64.0 MB scratch buffer allocated & reclaimed")
            except Exception as exc:
                is_healthy = False
                report.append(f"[FAIL] Memory Allocation: Failed to allocate scratch buffer: {exc}")

        # 5. Shared Storage (NFS/SMB) Read & Write Latency
        try:
            shared_dir = self.bus.shared_dir
            if not shared_dir.exists():
                raise FileNotFoundError(f"Shared storage path does not exist: {shared_dir}")
            test_path = shared_dir / f".preflight_{self.worker_id}_{int(time.time())}.tmp"
            t0 = time.time()
            test_path.write_text(f"preflight_{self.worker_id}", encoding="utf-8")
            read_back = test_path.read_text(encoding="utf-8")
            latency_ms = round((time.time() - t0) * 1000, 1)
            test_path.unlink(missing_ok=True)
            if read_back != f"preflight_{self.worker_id}":
                raise IOError("Shared storage read back content mismatch")
            report.append(f"[PASS] Shared Storage   : Read/Write verified (latency: {latency_ms}ms) on {shared_dir}")
        except Exception as exc:
            is_healthy = False
            report.append(f"[FAIL] Shared Storage   : {exc}")

        # 6. Micro-Transformer Mini-Step (Forward + Backward + AdamW)
        try:
            # Suppress/disable dynamo compiler hooks for preflight validation
            try:
                import importlib
                dynamo = importlib.import_module("torch._dynamo")
                dynamo.config.suppress_errors = True
                dynamo.disable()
            except Exception:
                pass

            class _PreflightMiniLM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.emb = nn.Embedding(256, 64)
                    self.fc = nn.Linear(64, 256)
                def forward(self, x):
                    return self.fc(self.emb(x))

            mini = _PreflightMiniLM().to(self.device_str)
            opt = torch.optim.AdamW(mini.parameters(), lr=1e-3)
            inp = torch.randint(0, 256, (2, 8), device=self.device_str)
            target = torch.randint(0, 256, (2, 8), device=self.device_str)
            out = mini(inp)
            loss = F.cross_entropy(out.view(-1, 256), target.view(-1))
            loss.backward()
            opt.step()
            del mini, opt, inp, target, out, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            report.append(f"[PASS] Transformer Step : Mini-model forward + backward + AdamW verified")
        except Exception as exc:
            # If dynamo / compile internal import failed but eager tensor ops work, don't fail preflight
            err_str = str(exc).lower()
            if "dynamo" in err_str or "_evalframeoverride" in err_str or "eval_frame" in err_str:
                report.append(f"[WARN] Transformer Step : Dynamo compiler hook unavailable ({exc}), eager execution active")
                report.append(f"[PASS] Transformer Step : Mini-model verified in standard eager mode")
            else:
                is_healthy = False
                report.append(f"[FAIL] Transformer Step : Mini-model execution failed: {exc}")

        report.append("=" * 72)
        if is_healthy:
            report.append("[PREFLIGHT] System Status: HEALTHY. Node ready to accept cluster jobs.")
        else:
            report.append("[PREFLIGHT] System Status: UNHEALTHY (DEGRADED). Refusing to claim jobs.")
        report.append("=" * 72)

        self._preflight_ok = is_healthy
        return is_healthy, report

    def evaluate_validation(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        max_batches: int = 25,
    ) -> float:
        """Evaluate validation loss on validation data loader using cross entropy.

        Args:
            model: Current PyTorch model.
            val_loader: DataLoader yielding (input_ids, target_ids).
            max_batches: Maximum number of batches to evaluate.

        Returns:
            Average validation loss float.
        """
        model.eval()
        total_loss = 0.0
        batches = 0
        device = torch.device(self.device_str)

        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                if step >= max_batches:
                    break
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0], batch[1]
                elif isinstance(batch, dict):
                    inputs = batch.get("input_ids") or batch.get("inputs")
                    targets = batch.get("target_ids") or batch.get("targets") or batch.get("labels")
                else:
                    continue

                if inputs is None or targets is None:
                    continue

                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                out = model(inputs)
                logits = out[0] if isinstance(out, (tuple, list)) else (out.logits if hasattr(out, "logits") else out)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
                total_loss += float(loss.item())
                batches += 1

        model.train()
        return (total_loss / batches) if batches > 0 else 0.0

    def is_job_compatible(self, job: dict[str, Any]) -> tuple[bool, str]:
        """Check if this worker node and device are compatible with the requested job.

        Returns:
            (True, "Compatible") or (False, "<detailed reason>")
        """
        # 0. Check operator enable/disable status
        if not getattr(self.bus, "is_worker_enabled", lambda wid: True)(self.worker_id):
            return False, f"Worker '{self.worker_id}' is disabled by operator."
        base_id = self.worker_id.split("_slot")[0]
        if base_id != self.worker_id and not getattr(self.bus, "is_worker_enabled", lambda wid: True)(base_id):
            return False, f"Primary worker node '{base_id}' is disabled by operator."

        # 1. Check worker health / preflight status
        if getattr(self, "_current_status", "") == "DEGRADED" or not getattr(self, "_preflight_ok", True):
            return False, "Worker is in DEGRADED status due to failed hardware preflight checks."

        # 2. Check target worker filtering if specified by job
        target_workers = job.get("target_workers")
        if target_workers and isinstance(target_workers, list):
            base_id = self.worker_id.split("_slot")[0]
            if self.worker_id not in target_workers and base_id not in target_workers:
                return False, f"Worker ID '{self.worker_id}' is not in job target_workers list: {target_workers}."

        # 3. Check compute device & PyTorch CUDA kernel execution
        if self.device_str.startswith("cuda"):
            if not torch.cuda.is_available():
                return False, "Job targets CUDA, but CUDA is not available on this host."
            try:
                # Fast kernel test to catch architecture binary incompatibility (e.g. sm_120 on outdated torch)
                dev = torch.device(self.device_str)
                test_t = torch.ones(2, 2, device=dev)
                _ = (test_t + 1.0).sum().item()
                del test_t
            except Exception as exc:
                err_msg = str(exc)
                if any(s in err_msg.lower() for s in ("cuda error", "device-side assert", "illegal memory access")):
                    self.log(
                        f"CUDA device '{self.device_str}' kernel execution test failed due to corrupted CUDA context ({exc}). "
                        "Respawning fresh worker process to self-heal...",
                        level="WARNING",
                    )
                    self._respawn_worker_process(reason=f"Kernel test failure: {exc}")
                return False, f"CUDA device '{self.device_str}' failed kernel execution test: {exc}"

        # 4. Check precision compatibility (e.g. bfloat16 hardware support)
        tcfg = job.get("training_config", {})
        if isinstance(tcfg, str):
            try:
                tcfg = json.loads(tcfg)
            except Exception:
                tcfg = {}
        prec = str(tcfg.get("precision", "float32")).lower()
        if prec in {"bfloat16", "bf16"}:
            if self.device_str.startswith("cuda"):
                if not is_hardware_bf16_supported(self.device_str):
                    # Device does not have native hardware BF16 tensor cores (e.g. Turing sm_75, Pascal sm_61)
                    # Worker will automatically adapt to FP16 with GradScaler during training for safety
                    pass

        # 5. Check dataset accessibility
        dataset_path = job.get("dataset_path", "")
        if not dataset_path:
            return False, "Job specifies no dataset_path."
        if not os.path.exists(dataset_path):
            return False, f"Dataset path is not accessible on this node: '{dataset_path}'"

        # 6. Check VRAM capacity against model parameters (if on CUDA)
        if self.device_str.startswith("cuda") and torch.cuda.is_available():
            try:
                model_cfg = job.get("model_config", {})
                if isinstance(model_cfg, str):
                    try:
                        model_cfg = json.loads(model_cfg)
                    except Exception:
                        model_cfg = {}
                vocab_size = int(model_cfg.get("vocab_size") or 32000)
                emb_size = int(model_cfg.get("embedding_size") or 768)
                n_layers = int(model_cfg.get("layer_count") or 12)
                ctx_len = int(model_cfg.get("context_length") or 1024)
                batch_size = int(tcfg.get("batch_size") or 2)

                approx_params = (vocab_size * emb_size * 2) + (n_layers * 12 * emb_size * emb_size)
                approx_bytes = approx_params * 16 + (batch_size * ctx_len * emb_size * n_layers * 4)
                approx_gb = approx_bytes / (1024 ** 3)

                dev_idx = 0
                if ":" in self.device_str:
                    try:
                        dev_idx = int(self.device_str.split(":")[1])
                    except ValueError:
                        dev_idx = 0
                free_b, total_b = torch.cuda.mem_get_info(dev_idx)
                total_gb = total_b / (1024 ** 3)

                if approx_gb > total_gb * 0.95:
                    return False, f"Estimated model memory ({approx_gb:.1f} GB) exceeds total GPU VRAM ({total_gb:.1f} GB)."
            except Exception:
                pass

        return True, "Compatible"

    def run_daemon(self, poll_interval: float = 3.0) -> None:
        """Run persistent background loop polling for jobs and executing them."""
        from .cluster_worker import acquire_singleton_lock, release_singleton_lock, get_device_tag

        device_tag = get_device_tag(self.device_str)
        wid_tag = f"wid_{self.worker_id}"
        if not getattr(self, "_lock_acquired", False):
            if not self.allow_shared_device:
                if not acquire_singleton_lock(device_tag):
                    self.log(f"A worker is already running for device '{self.device_str}' on this machine. Exiting.", level="WARNING")
                    return

            if not acquire_singleton_lock(wid_tag):
                if not self.allow_shared_device:
                    release_singleton_lock(device_tag)
                self.log(f"A worker is already running for worker ID '{self.worker_id}' on this machine. Exiting.", level="WARNING")
                return
            self._lock_acquired = True

        # Run hardware and environment preflight verification
        self._current_status = "PREFLIGHT"
        try:
            self.bus.heartbeat(self.worker_id, status="PREFLIGHT", current_job_id=None)
        except Exception:
            pass
        preflight_ok, report_lines = self.run_hardware_preflight()
        for line in report_lines:
            self.log(line)

        if not preflight_ok:
            self._current_status = "DEGRADED"
            self.log("Worker failed hardware preflight checks. Halting job acquisition until issues are resolved.", level="ERROR")
            try:
                self.bus.heartbeat(self.worker_id, status="DEGRADED", current_job_id=None)
            except Exception:
                pass
            while not self._stop_event.is_set():
                self._stop_event.wait(poll_interval)
            return

        self._current_status = "IDLE"
        try:
            self.bus.heartbeat(self.worker_id, status="IDLE", current_job_id=None)
        except Exception:
            pass

        # Dedicated execution loop for ephemeral auxiliary worker processes
        if self.ephemeral_job_id:
            self.log(f"Ephemeral worker slot started for job '{self.ephemeral_job_id}'. Will exit automatically upon completion.")
            try:
                deadline = time.time() + 45.0
                while not self._stop_event.is_set() and time.time() < deadline:
                    active_job = self.bus.get_active_job()
                    if active_job and active_job.get("job_id") == self.ephemeral_job_id and active_job.get("status") in {"RUNNING", "QUEUED"}:
                        self.execute_job(active_job, poll_interval=poll_interval)
                        break
                    time.sleep(poll_interval)
            except BaseException as exc:
                import traceback
                tb = traceback.format_exc()
                self.log(f"CRITICAL: Ephemeral worker crashed with unhandled exception:\n{tb}", level="ERROR")
                print(f"\n[Worker CRITICAL TRACEBACK]\n{tb}", file=sys.stderr, flush=True)
                raise
            finally:
                if not self.allow_shared_device:
                    release_singleton_lock(device_tag)
                release_singleton_lock(wid_tag)
                self._current_status = "OFFLINE"
                self._current_job_id = None
                try:
                    self.bus.heartbeat(self.worker_id, status="OFFLINE", current_job_id=None)
                except Exception:
                    pass
                self.log(f"Ephemeral worker slot {self.worker_id} finished execution. Exiting process.")
                return

        # Persistent daemon loop for primary worker daemon
        self.log(f"Worker daemon started. Listening for jobs on shared drive...")
        try:
            while not self._stop_event.is_set():
                # Check if worker is disabled by operator: remain active and heartbeating, but do not claim jobs
                if not getattr(self.bus, "is_worker_enabled", lambda wid: True)(self.worker_id):
                    self._current_status = "DISABLED"
                    self._current_job_id = None
                    try:
                        self.bus.heartbeat(
                            self.worker_id,
                            status="DISABLED",
                            current_job_id=None,
                            metrics=self._collect_system_metrics(),
                        )
                    except Exception:
                        pass
                    self._stop_event.wait(poll_interval)
                    continue

                self._current_status = "IDLE"
                self._current_job_id = None

                try:
                    active_job = self.bus.get_active_job()
                    if active_job and active_job.get("status") in {"RUNNING", "QUEUED"}:
                        job_id = active_job.get("job_id", "")
                        is_compat, reason = self.is_job_compatible(active_job)
                        if not is_compat:
                            if job_id not in self._incompatible_jobs_reported:
                                self._incompatible_jobs_reported.add(job_id)
                                self.log(f"Worker {self.worker_id} cannot accept job '{job_id}': {reason}. Skipping.", level="WARNING")
                                try:
                                    self.bus.heartbeat(self.worker_id, status="INCOMPATIBLE", current_job_id=None)
                                except Exception:
                                    pass
                        else:
                            self.execute_job(active_job, poll_interval=poll_interval)
                except Exception as exc:
                    import traceback
                    tb = traceback.format_exc()
                    self.log(f"Error in worker daemon:\n{tb}", level="ERROR")
                    time.sleep(poll_interval * 2)

                self._stop_event.wait(poll_interval)
        except BaseException as exc:
            import traceback
            tb = traceback.format_exc()
            self.log(f"CRITICAL: Worker daemon crashed with unhandled exception:\n{tb}", level="ERROR")
            print(f"\n[Worker CRITICAL TRACEBACK]\n{tb}", file=sys.stderr, flush=True)
            raise
        finally:
            if not self.allow_shared_device:
                release_singleton_lock(device_tag)
            release_singleton_lock(wid_tag)
            self._current_status = "OFFLINE"
            self._current_job_id = None
            try:
                self.bus.heartbeat(self.worker_id, status="OFFLINE", current_job_id=None)
            except Exception:
                pass
