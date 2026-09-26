"""Standalone Single-Script Cluster Worker Daemon for Port-Blocked Distributed Training.

This script is 100% self-contained. It can be copied to any remote worker machine
and executed directly without needing the LLM-IDE or engine repository.

Features:
1. Auto-bootstraps missing dependencies (torch, numpy) via pip at first launch.
2. Reads central storage path from `LLM_SHARED_DIR` environment variable or `--shared-dir`.
3. Enforces singleton process per machine (prevents multiple instances on the same host).
4. Reports GPU model, VRAM, and heartbeat to central SQLite WAL database.
5. Claims disjoint token shards and executes local SGD training rounds.
6. Visible as a standalone python process in Task Manager with identifiable title.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import ctypes
import json
import math
import os
import platform
import random
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Generator, Optional

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


# =============================================================================
# 1. Hardware Detection & Dependency Auto-Bootstrapping
# =============================================================================

# Dictionary mapping RTX and modern NVIDIA GPU models to their required CUDA wheel version.
# Keys are normalized (uppercase, single-spaced) GPU model identifiers.
RTX_CUDA_MAP: dict[str, str] = {
    # -------------------------------------------------------------------------
    # GeForce RTX 50 Series (Blackwell Architecture - Requires CUDA 12.8+)
    # -------------------------------------------------------------------------
    "RTX 5090": "cu128",
    "RTX 5090 D": "cu128",
    "RTX 5080": "cu128",
    "RTX 5070 TI": "cu128",
    "RTX 5070": "cu128",
    "RTX 5060 TI": "cu128",
    "RTX 5060": "cu128",
    "RTX 5050": "cu128",

    # -------------------------------------------------------------------------
    # GeForce RTX 40 Series (Ada Lovelace Architecture - Optimal on CUDA 12.8)
    # -------------------------------------------------------------------------
    "RTX 4090": "cu128",
    "RTX 4090 D": "cu128",
    "RTX 4080 SUPER": "cu128",
    "RTX 4080": "cu128",
    "RTX 4070 TI SUPER": "cu128",
    "RTX 4070 TI": "cu128",
    "RTX 4070 SUPER": "cu128",
    "RTX 4070": "cu128",
    "RTX 4060 TI": "cu128",
    "RTX 4060": "cu128",
    "RTX 4050": "cu128",

    # -------------------------------------------------------------------------
    # RTX Workstation & Professional (Ada Lovelace Architecture)
    # -------------------------------------------------------------------------
    "RTX 6000 ADA": "cu128",
    "RTX 5000 ADA": "cu128",
    "RTX 4500 ADA": "cu128",
    "RTX 4000 ADA": "cu128",
    "RTX 4000 SFF ADA": "cu128",
    "RTX 3500 ADA": "cu128",
    "RTX 3000 ADA": "cu128",
    "RTX 2000 ADA": "cu128",
    "RTX 1000 ADA": "cu128",
    "RTX 500 ADA": "cu128",

    # -------------------------------------------------------------------------
    # GeForce RTX 30 Series (Ampere Architecture - Optimal on CUDA 12.8)
    # -------------------------------------------------------------------------
    "RTX 3090 TI": "cu128",
    "RTX 3090": "cu128",
    "RTX 3080 TI": "cu128",
    "RTX 3080": "cu128",
    "RTX 3070 TI": "cu128",
    "RTX 3070": "cu128",
    "RTX 3060 TI": "cu128",
    "RTX 3060": "cu128",
    "RTX 3050": "cu128",

    # -------------------------------------------------------------------------
    # RTX Workstation & Professional (Ampere Architecture)
    # -------------------------------------------------------------------------
    "RTX A6000": "cu128",
    "RTX A5500": "cu128",
    "RTX A5000": "cu128",
    "RTX A4500": "cu128",
    "RTX A4000": "cu128",
    "RTX A3000": "cu128",
    "RTX A2000": "cu128",
    "RTX A1000": "cu128",
    "RTX A500": "cu128",
    "RTX A400": "cu128",

    # -------------------------------------------------------------------------
    # GeForce RTX 20 Series & Titan (Turing Architecture - CUDA 12.8)
    # -------------------------------------------------------------------------
    "RTX 2080 TI": "cu128",
    "RTX 2080 SUPER": "cu128",
    "RTX 2080": "cu128",
    "RTX 2070 SUPER": "cu128",
    "RTX 2070": "cu128",
    "RTX 2060 SUPER": "cu128",
    "RTX 2060": "cu128",
    "TITAN RTX": "cu128",

    # -------------------------------------------------------------------------
    # Quadro RTX Series
    # -------------------------------------------------------------------------
    "QUADRO RTX 8000": "cu128",
    "QUADRO RTX 6000": "cu128",
    "QUADRO RTX 5000": "cu128",
    "QUADRO RTX 4000": "cu128",
    "QUADRO RTX 3000": "cu128",

    # -------------------------------------------------------------------------
    # Data Center & Cloud Accelerators
    # -------------------------------------------------------------------------
    "B200": "cu128",
    "B100": "cu128",
    "GB200": "cu128",
    "H200": "cu128",
    "H100": "cu128",
    "L40S": "cu128",
    "L40": "cu128",
    "L4": "cu128",
    "A100": "cu128",
    "A40": "cu128",
    "A30": "cu128",
    "A16": "cu128",
    "A10": "cu128",
    "A2": "cu128",
    "V100": "cu128",
    "T4": "cu128",

    # -------------------------------------------------------------------------
    # Legacy NVIDIA Architecture (Pascal - CUDA 12.4)
    # -------------------------------------------------------------------------
    "GTX 1080 TI": "cu124",
    "GTX 1080": "cu124",
    "GTX 1070 TI": "cu124",
    "GTX 1070": "cu124",
    "GTX 1060": "cu124",
}


def _find_nvidia_smi() -> Optional[str]:
    """Find the path to the nvidia-smi executable on Windows or Linux."""
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    if sys.platform == "win32":
        # Check standard Windows paths if nvidia-smi is not in user PATH
        candidates = [
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
        ]
        for c in candidates:
            if c.is_file():
                return str(c)
    return None


def resolve_cuda_version_for_gpu(gpu_name: str) -> Optional[str]:
    """Determine the optimal CUDA wheel version ('cu128', 'cu124', etc.) for a given GPU name.

    Returns None if the device is CPU or an unrecognized non-CUDA device.
    """
    if not gpu_name:
        return None

    forced = os.environ.get("LLM_FORCE_CUDA_VERSION") or os.environ.get("LLM_CUDA_VERSION")
    if forced:
        forced = forced.strip().lower()
        if not forced.startswith("cu") and forced != "cpu":
            forced = f"cu{forced.replace('.', '')}"
        return forced

    raw = gpu_name.upper().replace("-", " ").replace("_", " ")
    for noise in ["NVIDIA", "GEFORCE", "GRAPHICS", "LAPTOP GPU", "GENERATION", "(R)", "(TM)", "WITH MAX-Q DESIGN"]:
        raw = raw.replace(noise, " ")
    cleaned = " ".join(raw.split())

    # 1. Exact match in dictionary
    if cleaned in RTX_CUDA_MAP:
        return RTX_CUDA_MAP[cleaned]

    # 2. Key contains match (longer keys first to match "RTX 4070 TI" before "RTX 4070")
    for key, cuda_ver in sorted(RTX_CUDA_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if key in cleaned:
            return cuda_ver

    # 3. Any RTX card fallback -> cu128
    if "RTX" in cleaned:
        return "cu128"

    # 4. Pascal legacy card fallback -> cu124
    if "GTX 10" in cleaned:
        return "cu124"

    # 5. Generic NVIDIA GPU fallback
    if "NVIDIA" in gpu_name.upper() or "GTX" in cleaned or "QUADRO" in cleaned or "TESLA" in cleaned:
        return "cu128"

    return None


def detect_all_gpus() -> list[dict[str, Any]]:
    """Detect all compute GPUs available on the host machine without prematurely importing torch."""
    gpus: list[dict[str, Any]] = []

    # 1. Prefer querying nvidia-smi directly (fast, accurate, doesn't lock torch DLLs)
    smi = _find_nvidia_smi()
    if smi:
        try:
            out = subprocess.check_output(
                [smi, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    idx = int(parts[0])
                    name = parts[1]
                    vram_mb = float(parts[2])
                    gpus.append({
                        "index": idx,
                        "device": f"cuda:{idx}",
                        "name": name,
                        "vram_gb": round(vram_mb / 1024.0, 2),
                    })
            if gpus:
                return gpus
        except Exception:
            pass

    # 2. Check Windows WMI if nvidia-smi is not available
    if sys.platform == "win32":
        try:
            wmi_out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            gpu_idx = 0
            for line in wmi_out.strip().splitlines():
                line = line.strip()
                if "NVIDIA" in line.upper():
                    gpus.append({
                        "index": gpu_idx,
                        "device": f"cuda:{gpu_idx}",
                        "name": line,
                        "vram_gb": 0.0,
                    })
                    gpu_idx += 1
            if gpus:
                return gpus
        except Exception:
            pass

    # 3. Fallback to PyTorch CUDA if torch is already imported or available
    if "torch" in sys.modules:
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append({
                        "index": i,
                        "device": f"cuda:{i}",
                        "name": torch.cuda.get_device_name(i),
                        "vram_gb": round(props.total_memory / (1024 ** 3), 2),
                    })
                return gpus
        except Exception:
            pass

    # Fallback to CPU
    return [{
        "index": -1,
        "device": "cpu",
        "name": platform.processor() or "CPU",
        "vram_gb": 0.0,
    }]


def get_installed_torch_status() -> tuple[Optional[str], Optional[str], bool]:
    """Inspect current torch installation without importing torch into the current process.

    Returns:
        (torch_version_str, cuda_tag, is_cuda_available)
        e.g. ("2.6.0+cu124", "cu124", True) or (None, None, False) if not installed.
    """
    version_str: Optional[str] = None
    try:
        import importlib.metadata
        version_str = importlib.metadata.version("torch")
    except Exception:
        version_str = None

    if not version_str:
        return None, None, False

    cuda_tag: Optional[str] = None
    if "+cu" in version_str:
        cuda_tag = "cu" + version_str.split("+cu")[-1].split(".")[0].split("+")[0]
    elif "+cpu" in version_str:
        cuda_tag = "cpu"

    # Verify runtime CUDA availability using a lightweight subprocess probe
    # to avoid loading torch DLLs into the worker process if uninstallation is needed.
    is_cuda_avail = False
    try:
        probe_code = (
            "import warnings; warnings.filterwarnings('ignore'); "
            "import torch; "
            "print('CUDA_AVAIL=' + str(bool(torch.cuda.is_available())) + "
            "';CUDA_VER=' + str(getattr(torch.version, 'cuda', '') or ''))"
        )
        out = subprocess.check_output(
            [sys.executable, "-W", "ignore", "-c", probe_code],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=45,
        )
        if "CUDA_AVAIL=True" in out:
            is_cuda_avail = True
        elif "CUDA_AVAIL=False" in out:
            is_cuda_avail = False
        if "CUDA_VER=" in out:
            for line in out.splitlines():
                for part in line.split(";"):
                    if part.startswith("CUDA_VER="):
                        sub_ver = part.split("=")[1].strip()
                        if sub_ver and not cuda_tag:
                            cuda_tag = "cu" + sub_ver.replace(".", "")
    except Exception:
        pass

    return version_str, cuda_tag, is_cuda_avail


def _cleanup_orphaned_site_packages() -> None:
    """Remove lingering ~orch, ~-rch or ~* directories left behind by Windows pip uninstalls."""
    try:
        import site
        site_dirs: list[str] = []
        if hasattr(site, "getsitepackages"):
            site_dirs.extend(site.getsitepackages())
        if hasattr(site, "getusersitepackages"):
            site_dirs.append(site.getusersitepackages())
        for sdir in site_dirs:
            sp = Path(sdir)
            if sp.is_dir():
                for orphan in sp.glob("~*"):
                    try:
                        if orphan.is_dir():
                            shutil.rmtree(orphan, ignore_errors=True)
                        elif orphan.is_file():
                            orphan.unlink(missing_ok=True)
                    except Exception:
                        pass
    except Exception:
        pass


def ensure_dependencies() -> None:
    """Ensure torch and numpy are installed with the exact CUDA wheel matching the host's GPU."""
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("LLM_TEST_BOOTSTRAP_EXECUTE"):
        return
    if os.environ.get("LLM_SKIP_BOOTSTRAP"):
        return

    # 1. Check for numpy
    missing_numpy = False
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing_numpy = True

    # 2. Discover physical GPUs
    detected_gpus = detect_all_gpus()
    has_nvidia = any(g.get("device", "").startswith("cuda") for g in detected_gpus)

    # 3. Determine target CUDA version for the detected hardware
    target_cuda: Optional[str] = None
    target_gpu_name = ""
    is_rtx_detected = False
    if has_nvidia:
        for g in detected_gpus:
            gname = g.get("name", "")
            cuda_ver = resolve_cuda_version_for_gpu(gname)
            if "RTX" in gname.upper():
                is_rtx_detected = True
            if cuda_ver:
                target_cuda = cuda_ver
                target_gpu_name = gname
                break
        if not target_cuda:
            target_cuda = "cu128"
            target_gpu_name = detected_gpus[0].get("name", "NVIDIA GPU")

    # 4. Inspect current installed PyTorch
    installed_ver, installed_cuda_tag, is_cuda_avail = get_installed_torch_status()

    need_uninstall = False
    need_install = False
    reason = ""

    if target_cuda:
        if installed_ver is None:
            need_install = True
            reason = f"PyTorch is not installed (target: {target_cuda} for {target_gpu_name})"
        elif not is_cuda_avail:
            # If the wheel already matches target_cuda (e.g. cu128 for RTX 5060 Ti), do not repeatedly uninstall
            if installed_cuda_tag and installed_cuda_tag == target_cuda:
                need_uninstall = False
                need_install = False
            else:
                need_uninstall = True
                need_install = True
                reason = f"Installed PyTorch ({installed_ver}) lacks working CUDA for {target_gpu_name} (needs {target_cuda})"
        elif is_rtx_detected and installed_cuda_tag != target_cuda:
            need_uninstall = True
            need_install = True
            reason = f"GPU {target_gpu_name} requires {target_cuda}, but installed PyTorch has {installed_cuda_tag} ({installed_ver})"
        elif os.environ.get("LLM_FORCE_CUDA_VERSION") and installed_cuda_tag != target_cuda:
            need_uninstall = True
            need_install = True
            reason = f"Forced CUDA {target_cuda} requested, but installed PyTorch has {installed_cuda_tag}"
    else:
        if installed_ver is None:
            need_install = True
            reason = "PyTorch is not installed (CPU mode)"

    if not need_uninstall and not need_install and not missing_numpy:
        return

    print(f"[ClusterWorker] Checking dependencies: {reason or 'verifying packages'}...")
    _cleanup_orphaned_site_packages()

    # Step A: Uninstall incompatible PyTorch if needed
    if need_uninstall:
        print(f"[ClusterWorker] Mismatch detected: {reason}")
        print("[ClusterWorker] Uninstalling existing PyTorch modules (torch, torchvision, torchaudio)...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "uninstall",
                "torch", "torchvision", "torchaudio", "-y",
            ])
            print("[ClusterWorker] Old PyTorch packages uninstalled cleanly.")
        except Exception as exc:
            print(f"[ClusterWorker] Warning during pip uninstall: {exc}")
        _cleanup_orphaned_site_packages()

    # Step B: Install target PyTorch packages
    if need_install or missing_numpy:
        # Check offline wheel cache in shared directory first
        shared_dir_env = os.environ.get("LLM_SHARED_PATH") or os.environ.get("LLM_SHARED_DIR", "")
        shared_wheels = Path(shared_dir_env) / "wheels" if shared_dir_env else None
        installed_offline = False

        if shared_wheels and shared_wheels.is_dir():
            whls = list(shared_wheels.glob("*.whl"))
            if whls:
                print(f"[ClusterWorker] Offline wheel cache detected with {len(whls)} wheel(s) at: {shared_wheels}")
                try:
                    cmd = [
                        sys.executable, "-m", "pip", "install",
                        "--no-index", f"--find-links={shared_wheels}",
                        "torch", "torchvision", "torchaudio",
                    ]
                    if missing_numpy:
                        cmd.append("numpy")
                    subprocess.check_call(cmd)
                    installed_offline = True
                    print("[ClusterWorker] Offline dependencies installed successfully from shared storage.")
                except Exception as exc:
                    print(f"[ClusterWorker] Offline wheel install failed ({exc}), falling back to online install...")

        if not installed_offline:
            try:
                if target_cuda:
                    index_url = f"https://download.pytorch.org/whl/{target_cuda}"
                    print(f"[ClusterWorker] Installing official PyTorch for {target_gpu_name} ({target_cuda}) from {index_url}...")
                    cmd = [
                        sys.executable, "-m", "pip", "install",
                        "torch", "torchvision", "torchaudio",
                        "--index-url", index_url,
                    ]
                    if missing_numpy:
                        cmd.append("numpy")
                    subprocess.check_call(cmd)
                else:
                    print("[ClusterWorker] Installing standard PyTorch...")
                    cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
                    if missing_numpy:
                        cmd.append("numpy")
                    subprocess.check_call(cmd)
            except Exception as exc:
                print(f"[ClusterWorker] Error installing dependencies: {exc}")

        if need_uninstall or need_install:
            _cleanup_orphaned_site_packages()
            print("[ClusterWorker] PyTorch dependencies installed and verified successfully.")
            # Re-launch in clean interpreter so new CUDA runtime and C-extensions load cleanly
            if not os.environ.get("PYTEST_CURRENT_TEST"):
                print("[ClusterWorker] Re-launching worker with fresh interpreter environment...")
                ret = subprocess.call([sys.executable] + sys.argv)
                sys.exit(ret)
        else:
            print("[ClusterWorker] PyTorch dependencies installed and verified successfully.")


ensure_dependencies()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def is_hardware_bf16_supported(device_str: str = "cuda:0") -> bool:
    """Check whether the device has native hardware Bfloat16 Tensor Core execution (compute capability >= 8.0)."""
    if not torch.cuda.is_available() or not str(device_str).startswith("cuda"):
        return False
    try:
        dev_idx = int(str(device_str).split(":")[1]) if ":" in str(device_str) else 0
        cap = torch.cuda.get_device_capability(dev_idx)
        return cap[0] >= 8 and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    except Exception:
        return False


# =============================================================================
# 2. Per-Device Singleton Process Management
# =============================================================================

def get_device_tag(device: str) -> str:
    """Normalize device string to a clean alphanumeric tag for filenames and worker IDs."""
    return device.lower().replace(":", "_").replace("-", "_")


def get_lock_file(device_tag: str) -> Path:
    return Path(tempfile.gettempdir()) / f"cluster_worker_{device_tag}.pid"


def is_pid_running(pid: int) -> bool:
    """Check whether a process with the given PID is currently active."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            process = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not process:
                # If access is denied (ERROR_ACCESS_DENIED = 5), the process definitely exists and is running
                return bool(kernel32.GetLastError() == 5)
            exit_code = ctypes.c_ulong()
            success = kernel32.GetExitCodeProcess(process, ctypes.byref(exit_code))
            kernel32.CloseHandle(process)
            # 259 is STILL_ACTIVE
            return bool(success and exit_code.value == 259)
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def get_running_worker_pid(device_tag: str = "default") -> Optional[int]:
    """Retrieve the PID of currently running cluster worker for a specific device."""
    lock_path = get_lock_file(device_tag)
    if lock_path.exists():
        try:
            pid = int(lock_path.read_text().strip())
            if is_pid_running(pid):
                return pid
        except (ValueError, OSError):
            pass
    return None


def get_all_running_worker_pids() -> dict[str, int]:
    """Find all running cluster worker PIDs across all devices on this host."""
    res = {}
    temp_dir = Path(tempfile.gettempdir())
    for p in temp_dir.glob("cluster_worker_*.pid"):
        try:
            tag = p.stem.replace("cluster_worker_", "")
            pid = int(p.read_text().strip())
            if is_pid_running(pid):
                res[tag] = pid
            else:
                p.unlink(missing_ok=True)
        except Exception:
            pass
    return res


def acquire_singleton_lock(device_tag: str = "default", timeout_seconds: float = 0.0) -> bool:
    """Ensure only one cluster worker process runs per device or worker ID on this host."""
    lock_path = get_lock_file(device_tag)
    deadline = time.time() + max(timeout_seconds, 0.0)
    while True:
        if lock_path.exists():
            try:
                old_pid = int(lock_path.read_text().strip())
                if is_pid_running(old_pid):
                    if time.time() < deadline:
                        time.sleep(0.1)
                        continue
                    tag_label = f"worker ID '{device_tag[4:]}'" if device_tag.startswith("wid_") else f"device '{device_tag}'"
                    print(f"[ClusterWorker] A worker is already running for {tag_label} (PID: {old_pid}). Exiting.")
                    return False
                else:
                    lock_path.unlink(missing_ok=True)
            except (ValueError, OSError):
                pass
        break

    try:
        lock_path.write_text(str(os.getpid()))
        atexit.register(lambda: release_singleton_lock(device_tag))
        return True
    except Exception as exc:
        print(f"[ClusterWorker] Warning: could not write lockfile: {exc}")
        return True


def get_worker_respawn_cmd(
    worker_id: str,
    device_str: str,
    shared_dir: Union[str, Path],
    allow_shared_device: bool = False,
) -> list[str]:
    """Construct a clean, robust command to respawn a worker process."""
    # 1. Check sys.orig_argv (Python 3.10+)
    if hasattr(sys, "orig_argv") and sys.orig_argv:
        cmd = list(sys.orig_argv)
        cmd[0] = sys.executable
        if len(cmd) > 1 and not cmd[1].startswith("-"):
            p = Path(cmd[1])
            if p.exists() or Path(p.resolve()).exists():
                cmd[1] = str(p.resolve())
        # Ensure critical identity and location arguments are strictly preserved across respawns
        if "--worker-id" not in cmd and worker_id:
            cmd.extend(["--worker-id", worker_id])
        if "--shared-dir" not in cmd and shared_dir:
            cmd.extend(["--shared-dir", str(shared_dir)])
        if "--device" not in cmd and device_str:
            cmd.extend(["--device", device_str])
        if allow_shared_device and "--allow-shared-device" not in cmd:
            cmd.append("--allow-shared-device")
        return cmd

    # 2. Check if invoked via module
    script = sys.argv[0] if sys.argv else ""
    if script.endswith("cli.py") or "cluster.cli" in script or (len(sys.argv) > 1 and sys.argv[1] == "worker"):
        cmd = [
            sys.executable,
            "-m",
            "cluster.cli",
            "worker",
            "--shared-dir",
            str(shared_dir),
            "--worker-id",
            worker_id,
            "--device",
            device_str,
        ]
        if allow_shared_device:
            cmd.append("--allow-shared-device")
        return cmd

    # 3. Direct script fallback (e.g. cluster_worker.py)
    script_path = Path(script).resolve() if script else Path.cwd() / "cluster_worker.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--shared-dir",
        str(shared_dir),
        "--worker-id",
        worker_id,
        "--device",
        device_str,
    ]
    if allow_shared_device:
        cmd.append("--allow-shared-device")
    return cmd


def release_singleton_lock(device_tag: str = "default") -> None:
    """Release the device process lock file on exit."""
    try:
        lock_path = get_lock_file(device_tag)
        if lock_path.exists():
            old_pid = int(lock_path.read_text().strip())
            if old_pid == os.getpid():
                lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def stop_running_worker(device_tag: Optional[str] = None) -> int:
    """Terminate running worker process(es) on this machine."""
    if device_tag:
        pid = get_running_worker_pid(device_tag)
        pids = {device_tag: pid} if pid else {}
    else:
        pids = get_all_running_worker_pids()

    if not pids:
        print("[ClusterWorker] No active cluster worker process found running on this machine.")
        return 0

    for tag, pid in pids.items():
        print(f"[ClusterWorker] Stopping worker for device '{tag}' (PID: {pid})...")
        if sys.platform == "win32":
            try:
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=False, capture_output=True)
            except Exception as exc:
                print(f"[ClusterWorker] Error invoking taskkill: {exc}")
        else:
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                if is_pid_running(pid):
                    os.kill(pid, signal.SIGKILL)
            except Exception as exc:
                print(f"[ClusterWorker] Error sending termination signal: {exc}")

        time.sleep(0.3)
        lock_path = get_lock_file(tag)
        lock_path.unlink(missing_ok=True)
        print(f"[ClusterWorker] Stopped worker process for '{tag}' (PID: {pid}).")

    return 0


def is_network_path(path_str: str) -> bool:
    """Determine if a file path resides on a remote network share (SMB/NFS/UNC)."""
    p = str(path_str).strip()
    if p.startswith(("\\\\", "//")):
        return True
    if sys.platform == "win32" and len(p) >= 2 and p[1] == ":":
        try:
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
    """Remove/delete stale .npy files from local cache directory to keep local storage clean."""
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
    """Safely cache remote/network dataset .npy file to fast local SSD storage."""
    log = log_fn or (lambda msg, **kw: print(f"[DatasetCache] {msg}"))

    if not remote_path or not os.path.exists(remote_path):
        return remote_path

    if not is_network_path(remote_path):
        return remote_path

    cache_dir = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "llm_cluster_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        log(f"[DatasetCache] Could not create local cache directory {cache_dir}: {exc}")
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

        effective_keep = set(keep_files or ())
        effective_keep.add(remote_file.name)
        purge_local_dataset_cache(cache_dir=cache_dir, keep_files=effective_keep, log_fn=log)

        # Check local disk space before attempting cache copy (require 1.15x buffer)
        free_space = shutil.disk_usage(cache_dir).free
        free_gb = free_space / (1024 ** 3)
        if free_space < remote_size * 1.15:
            log(
                f"[DatasetCache] Insufficient local disk space to cache dataset "
                f"({free_gb:.1f} GB free vs {remote_size_gb:.1f} GB required). Falling back to network share.",
            )
            try:
                if lock_file.exists():
                    lock_file.unlink(missing_ok=True)
            except Exception:
                pass
            return remote_path

        log(f"[DatasetCache] Streaming network dataset {remote_file.name} ({remote_size_gb:.2f} GB) to local SSD ({local_file})...")
        t0 = time.time()
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
        log(f"[DatasetCache] Failed to cache dataset to local SSD ({exc}). Falling back to network share.\nTraceback:\n{tb}")
        try:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            if lock_file.exists():
                lock_file.unlink(missing_ok=True)
        except Exception:
            pass
        return remote_path


def get_worker_executable() -> str:
    """Get or create a named worker executable (cluster_worker.exe) on Windows for Task Manager clarity."""
    if sys.platform != "win32":
        return sys.executable
    try:
        app_data = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "cluster_worker"
        app_data.mkdir(parents=True, exist_ok=True)
        target_exe = app_data / "cluster_worker.exe"
        py_path = Path(sys.executable)
        if not target_exe.exists() or target_exe.stat().st_size != py_path.stat().st_size:
            shutil.copyfile(py_path, target_exe)
        return str(target_exe)
    except Exception:
        return sys.executable


# =============================================================================
# 3. Embedded SQLite WAL Bus Client
# =============================================================================

def safe_torch_load(
    path: Path | str,
    device: str = "cpu",
    max_retries: int = 5,
    retry_delay: float = 0.5,
) -> Any:
    """Safely load PyTorch checkpoint with retry and local temporary caching to prevent Windows SMB Errno 22."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    is_net = False
    p_str = str(p)
    if p_str.startswith(("\\\\", "//")):
        is_net = True
    elif len(p_str) >= 2 and p_str[1] == ":" and sys.platform == "win32":
        try:
            import ctypes
            drive_type = ctypes.windll.kernel32.GetDriveTypeW(f"{p_str[:2].upper()}\\")
            is_net = (drive_type == 4)  # DRIVE_REMOTE
        except Exception:
            is_net = False

    last_err = None
    for attempt in range(max_retries):
        try:
            if is_net:
                temp_fd, temp_path = tempfile.mkstemp(suffix=".pt", prefix=f".llm_load_{os.getpid()}_")
                os.close(temp_fd)
                try:
                    shutil.copyfile(str(p), temp_path)
                    data = torch.load(temp_path, map_location=device)
                    return data
                finally:
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
            else:
                return torch.load(str(p), map_location=device)
        except (OSError, RuntimeError) as exc:
            last_err = exc
            time.sleep(retry_delay * (attempt + 1))
        except Exception as exc:
            last_err = exc
            time.sleep(retry_delay * (attempt + 1))

    if is_net:
        return torch.load(str(p), map_location=device)
    if last_err:
        raise last_err
    return torch.load(str(p), map_location=device)


_DEFAULT_NOT_SET = object()


class StandaloneStorageBus:
    """Self-contained storage bus client operating over central shared network storage."""

    def __init__(self, shared_dir: Path | str) -> None:
        self.shared_dir = Path(shared_dir)
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.shared_dir / "cluster.db"
        self.jobs_dir = self.shared_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextlib.contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = None
        for attempt in range(8):
            try:
                conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level="DEFERRED")
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA busy_timeout = 30000;")
                conn.execute("PRAGMA synchronous = NORMAL;")
                conn.execute("PRAGMA temp_store = MEMORY;")
                break
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
                err_msg = str(exc).lower()
                if "malformed" in err_msg or "file is not a database" in err_msg or "corrupt" in err_msg:
                    try:
                        self._recover_malformed_db(exc)
                        continue
                    except Exception:
                        pass
                if attempt == 7:
                    raise
                time.sleep(min(2.0, 0.05 * (1.6 ** attempt) + random.uniform(0.03, 0.15)))
        try:
            yield conn
            if conn and conn.in_transaction:
                conn.commit()
        except Exception:
            if conn:
                try:
                    if conn.in_transaction:
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

    def _recover_malformed_db(self, exc: Exception) -> bool:
        """Automatically recover when SQLite reports a malformed database disk image on network storage."""
        print(f"[StandaloneStorageBus] WARNING: Corrupted/malformed SQLite database detected ({exc}). Attempting automatic self-healing recovery...")
        ts = int(time.time())
        corrupt_backup = self.shared_dir / f"cluster.db.corrupt_{ts}"
        salvaged_data: dict[str, list[dict[str, Any]]] = {}

        conn_old = None
        try:
            conn_old = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0)
            conn_old.row_factory = sqlite3.Row
            for tbl in ["workers", "jobs", "job_participants", "round_history"]:
                try:
                    cursor = conn_old.execute(f"SELECT * FROM {tbl};")
                    salvaged_data[tbl] = [dict(r) for r in cursor.fetchall()]
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            if conn_old is not None:
                try:
                    conn_old.close()
                except Exception:
                    pass
                del conn_old
            import gc
            gc.collect()

        try:
            if self.db_path.exists():
                moved = False
                for _ in range(5):
                    try:
                        shutil.move(str(self.db_path), str(corrupt_backup))
                        moved = True
                        break
                    except Exception:
                        import gc
                        gc.collect()
                        time.sleep(0.1)
                if not moved:
                    try:
                        shutil.copy2(str(self.db_path), str(corrupt_backup))
                        self.db_path.unlink(missing_ok=True)
                    except Exception as fallback_err:
                        print(f"[StandaloneStorageBus] Could not archive corrupt database: {fallback_err}")
                        return False
            for ext in ["-journal", "-wal", "-shm"]:
                j_file = self.shared_dir / f"cluster.db{ext}"
                if j_file.exists():
                    try:
                        j_file.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception as move_err:
            print(f"[StandaloneStorageBus] Could not archive corrupt database: {move_err}")
            return False

        try:
            self._init_db()
        except Exception as init_err:
            print(f"[StandaloneStorageBus] Error initializing fresh database: {init_err}")
            return False

        try:
            with self._connect() as conn_new:
                for tbl, rows in salvaged_data.items():
                    if not rows:
                        continue
                    cols = list(rows[0].keys())
                    placeholders = ", ".join(["?"] * len(cols))
                    col_names = ", ".join(cols)
                    stmt = f"INSERT OR REPLACE INTO {tbl} ({col_names}) VALUES ({placeholders});"
                    for row in rows:
                        try:
                            conn_new.execute(stmt, [row.get(c) for c in cols])
                        except Exception:
                            pass
                conn_new.commit()
        except Exception as restore_err:
            print(f"[StandaloneStorageBus] Warning: Could not restore all salvaged rows: {restore_err}")

        print(f"[StandaloneStorageBus] Database self-healing complete. Intact records restored; corrupt file saved to {corrupt_backup.name}.")
        return True

    def _run_with_retry(
        self,
        fn: Callable[[sqlite3.Connection], Any],
        max_retries: int = 10,
        default_on_error: Any = _DEFAULT_NOT_SET,
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
                err_msg = str(exc).lower()
                if "malformed" in err_msg or "file is not a database" in err_msg or "corrupt" in err_msg:
                    try:
                        if self._recover_malformed_db(exc):
                            with self._connect() as conn:
                                return fn(conn)
                    except Exception as rec_err:
                        print(f"[StandaloneStorageBus] Database self-healing retry failed: {rec_err}")
                        if default_on_error is not _DEFAULT_NOT_SET:
                            return default_on_error
                        if silent:
                            return None
                        raise exc
                if attempt == max_retries - 1:
                    if default_on_error is not _DEFAULT_NOT_SET:
                        return default_on_error
                    if silent:
                        return None
                    raise
                backoff = min(2.5, 0.1 * (1.6 ** attempt) + random.uniform(0.05, 0.25))
                time.sleep(backoff)
        return default_on_error if default_on_error is not _DEFAULT_NOT_SET else None

    def _init_db(self) -> None:
        def _init(conn: sqlite3.Connection) -> None:
            try:
                conn.execute("PRAGMA journal_mode = TRUNCATE;")
                conn.execute("PRAGMA synchronous = NORMAL;")
                conn.execute("PRAGMA temp_store = MEMORY;")
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
                ("enabled", "INTEGER DEFAULT 1"),
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
                    sync_timeout_seconds REAL DEFAULT 1800.0,
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

    def register_worker(self, worker_id: str, hostname: str, gpu_name: str, vram_gb: float) -> None:
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
                    UPDATE workers SET last_heartbeat = ?, status = ?, current_job_id = ?
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
                conn.execute("UPDATE workers SET last_heartbeat = ? WHERE worker_id = ?;", (now, worker_id))
        self._run_with_retry(_op, silent=True)

    def set_worker_command(self, worker_id: str, command: Optional[str]) -> None:
        def _op(conn: sqlite3.Connection) -> None:
            conn.execute("UPDATE workers SET command = ? WHERE worker_id = ?;", (command, worker_id))
        self._run_with_retry(_op, silent=True)

    def get_worker_command(self, worker_id: str) -> Optional[str]:
        def _op(conn: sqlite3.Connection) -> Optional[str]:
            cursor = conn.execute("SELECT command FROM workers WHERE worker_id = ?;", (worker_id,))
            row = cursor.fetchone()
            return row["command"] if row and row["command"] else None
        return self._run_with_retry(_op, default_on_error=None, silent=True)

    def delete_worker(self, worker_id: str) -> bool:
        def _op(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute("DELETE FROM workers WHERE worker_id = ?;", (worker_id,))
            conn.execute("UPDATE job_participants SET status = 'DROPPED' WHERE worker_id = ?;", (worker_id,))
            return cursor.rowcount > 0
        return bool(self._run_with_retry(_op, default_on_error=False))

    def delete_offline_workers(self, stale_threshold_seconds: float = 60.0) -> int:
        now = time.time()
        cutoff = now - stale_threshold_seconds
        def _op(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                "DELETE FROM workers WHERE status = 'OFFLINE' OR last_heartbeat < ?;",
                (cutoff,),
            )
            conn.execute("UPDATE job_participants SET status = 'DROPPED' WHERE worker_id NOT IN (SELECT worker_id FROM workers);")
            return cursor.rowcount
        return int(self._run_with_retry(_op, default_on_error=0))

    def touch_job(self, job_id: str) -> None:
        """Update job updated_at timestamp to signal active coordinator heartbeat."""
        now = time.time()
        def _op(conn: sqlite3.Connection) -> None:
            conn.execute("UPDATE jobs SET updated_at = ? WHERE job_id = ?;", (now, job_id))
        self._run_with_retry(_op, silent=True)

    def set_worker_status(self, worker_id: str, status: str) -> None:
        """Explicitly set a worker's status (e.g. 'OFFLINE', 'IDLE')."""
        self.heartbeat(worker_id, status=status, current_job_id=None)

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
            time_diff = now - last_hb
            stored_status = str(data.get("status") or "OFFLINE").upper()
            thresh = 180.0 if stored_status in {"STARTING", "INITIALIZING", "RESTARTING", "PREFLIGHT"} else active_within_seconds
            is_fresh = (time_diff < thresh) and (time_diff > -300.0)
            data["enabled"] = bool(data.get("enabled", 1)) if data.get("enabled") is not None else True
            if not is_fresh or stored_status in {"OFFLINE", "STOPPED"}:
                data["is_online"] = False
                data["status"] = "OFFLINE"
            else:
                data["is_online"] = True
                if not data["enabled"]:
                    data["status"] = "DISABLED"
            results.append(data)
        return results

    def get_active_job(self, max_stale_seconds: float = 3600.0) -> Optional[dict[str, Any]]:
        now = time.time()
        def _op(conn: sqlite3.Connection) -> Optional[dict[str, Any]]:
            if max_stale_seconds > 0:
                cursor = conn.execute(
                    """
                    SELECT job_id FROM jobs
                    WHERE status IN ('RUNNING', 'QUEUED')
                      AND (? - updated_at) > ?;
                    """,
                    (now, max_stale_seconds),
                )
                swept_rows = cursor.fetchall()
                if swept_rows:
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'STOPPED', updated_at = ?
                        WHERE status IN ('RUNNING', 'QUEUED')
                          AND (? - updated_at) > ?;
                        """,
                        (now, now, max_stale_seconds),
                    )
                    for sr in swept_rows:
                        try:
                            s_dir = self.jobs_dir / sr["job_id"] / "signals"
                            s_dir.mkdir(parents=True, exist_ok=True)
                            (s_dir / "stop.sig").touch()
                        except Exception:
                            pass
            cursor = conn.execute(
                "SELECT * FROM jobs WHERE status IN ('RUNNING', 'QUEUED', 'PAUSED') ORDER BY created_at DESC LIMIT 1;"
            )
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            res["model_config"] = json.loads(res["model_config"])
            res["training_config"] = json.loads(res["training_config"])
            if res.get("lora_config") and isinstance(res["lora_config"], str):
                try:
                    res["lora_config"] = json.loads(res["lora_config"])
                except Exception:
                    pass
            return res
        return self._run_with_retry(_op, default_on_error=None, silent=True)

    def claim_job_slot(self, job_id: str, worker_id: str) -> tuple[int, int]:
        if not self.is_worker_enabled(worker_id):
            raise RuntimeError(f"Worker '{worker_id}' is DISABLED and cannot claim a job slot.")
        return self.get_worker_shard_assignment(job_id, worker_id, round_num=0)

    def set_worker_enabled(self, worker_id: str, enabled: bool) -> None:
        """Enable or disable a worker from claiming cluster jobs."""
        val = 1 if enabled else 0
        def _op(conn: sqlite3.Connection) -> None:
            try:
                conn.execute("ALTER TABLE workers ADD COLUMN enabled INTEGER DEFAULT 1;")
            except Exception:
                pass
            conn.execute(
                "UPDATE workers SET enabled = ?, status = CASE WHEN ? = 0 THEN 'DISABLED' ELSE 'IDLE' END WHERE worker_id = ?;",
                (val, val, worker_id),
            )
            if not enabled:
                conn.execute(
                    "UPDATE job_participants SET status = 'DROPPED' WHERE worker_id = ?;",
                    (worker_id,),
                )
        try:
            self._run_with_retry(_op, silent=True)
        except Exception:
            pass

    def is_worker_enabled(self, worker_id: str) -> bool:
        """Check whether a worker is enabled to claim cluster jobs."""
        def _op(conn: sqlite3.Connection) -> bool:
            try:
                cursor = conn.execute("SELECT enabled FROM workers WHERE worker_id = ?;", (worker_id,))
                row = cursor.fetchone()
                if row is not None and row[0] is not None:
                    return bool(row[0])
            except sqlite3.OperationalError as oe:
                if "no such column" in str(oe).lower():
                    return True
                raise
            return True
        try:
            return self._run_with_retry(_op, default_on_error=True, silent=True)
        except Exception:
            return True

    def get_worker_shard_assignment(self, job_id: str, worker_id: str, round_num: int = 0) -> tuple[int, int]:
        def _op(conn: sqlite3.Connection) -> tuple[int, int]:
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
        def _op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE job_participants SET status = 'DROPPED' WHERE job_id = ? AND worker_id = ?;",
                (job_id, worker_id),
            )
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

    def save_worker_telemetry(self, job_id: str, round_num: int, worker_id: str, telemetry: dict[str, Any]) -> None:
        round_dir = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        target = round_dir / f"{worker_id}_telemetry.json"
        target.write_text(json.dumps(telemetry, indent=2, default=str), encoding="utf-8")

    def load_worker_telemetry(self, job_id: str, round_num: int, worker_id: str) -> Optional[dict[str, Any]]:
        target = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}" / f"{worker_id}_telemetry.json"
        if not target.exists():
            return None
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return None

    def is_paused(self, job_id: str) -> bool:
        """Check if pause signal file exists or job status is PAUSED in SQLite."""
        if (self.jobs_dir / job_id / "signals" / "pause.sig").exists():
            return True
        def _op(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute("SELECT status FROM jobs WHERE job_id = ?;", (job_id,))
            row = cursor.fetchone()
            return bool(row and str(row["status"]).upper() == "PAUSED")
        return bool(self._run_with_retry(_op, default_on_error=False, silent=True))

    def is_stopped(self, job_id: str) -> bool:
        """Check if stop signal file exists or job status is stopped/completed in SQLite."""
        if (self.jobs_dir / job_id / "signals" / "stop.sig").exists():
            return True
        def _op(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute("SELECT status FROM jobs WHERE job_id = ?;", (job_id,))
            row = cursor.fetchone()
            if row and str(row["status"]).upper() in {"STOPPED", "STOPPING", "FAILED", "COMPLETED"}:
                return True
            return False
        return bool(self._run_with_retry(_op, default_on_error=False, silent=True))

    def save_worker_weights(self, job_id: str, round_num: int, worker_id: str, state_dict: dict[str, torch.Tensor]) -> None:
        round_dir = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        target = round_dir / f"{worker_id}.pt"
        tmp_target = round_dir / f"{worker_id}.pt.tmp"
        ready_flag = round_dir / f"{worker_id}.ready"

        torch.save(state_dict, tmp_target)
        tmp_target.replace(target)
        ready_flag.touch()

    def is_global_weights_ready(self, job_id: str, round_num: int) -> bool:
        round_dir = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}"
        return (round_dir / "global_model.ready").exists() and (round_dir / "global_model.pt").exists()

    def load_global_weights(self, job_id: str, round_num: int, device: str = "cpu") -> dict[str, torch.Tensor]:
        weight_path = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}" / "global_model.pt"
        return safe_torch_load(weight_path, device=device)

    def load_base_model_weights(self, job_id: str, device: str = "cpu") -> Optional[dict[str, torch.Tensor]]:
        """Load staged base model weights for fine-tuning."""
        candidate = self.jobs_dir / job_id / "base_model.pt"
        if not candidate.exists():
            job = self.get_active_job()
            if job and job.get("job_id") == job_id and job.get("base_checkpoint_path"):
                alt = Path(job["base_checkpoint_path"])
                if alt.exists():
                    candidate = alt
        if candidate.exists():
            obj = safe_torch_load(candidate, device=device)
            if isinstance(obj, dict):
                for k in ("model_state_dict", "state_dict", "model"):
                    if k in obj and isinstance(obj[k], dict):
                        return obj[k]
            return obj
        return None





    def get_checkpoints_dir(self, job_id: str) -> Path:
        ckpt_dir = self.jobs_dir / job_id / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir

    def purge_round_worker_weights(self, job_id: str, round_num: int) -> int:
        round_dir = self.jobs_dir / job_id / "rounds" / f"round_{round_num:04d}"
        if not round_dir.exists():
            return 0
        purged = 0
        for f in round_dir.glob("*.pt"):
            if f.name != "global_model.pt":
                try:
                    f.unlink(missing_ok=True)
                    purged += 1
                except Exception:
                    pass
        for f in round_dir.glob("*.pt.tmp"):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
        for f in round_dir.glob("*.ready"):
            if f.name != "global_model.ready":
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
        return purged

    def purge_stale_rounds(self, job_id: str, keep_last_rounds: int = 2) -> int:
        rounds_dir = self.jobs_dir / job_id / "rounds"
        if not rounds_dir.exists():
            return 0
        round_dirs = sorted([d for d in rounds_dir.glob("round_*") if d.is_dir()])
        if len(round_dirs) <= keep_last_rounds:
            return 0
        purged = 0
        for old_dir in round_dirs[:-keep_last_rounds]:
            try:
                shutil.rmtree(old_dir, ignore_errors=True)
                purged += 1
            except Exception:
                pass
        return purged

    def save_checkpoint(
        self,
        job_id: str,
        step: int,
        state_dict: dict[str, torch.Tensor],
        is_final: bool = False,
        model_config: Optional[dict[str, Any]] = None,
        training_config: Optional[dict[str, Any]] = None,
        round_num: Optional[int] = None,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        peft_method: str = "none",
        lora_config: Optional[dict[str, Any]] = None,
        adapter_state_dict: Optional[dict[str, torch.Tensor]] = None,
        max_keep: int = 2,
    ) -> Path:
        ckpt_dir = self.get_checkpoints_dir(job_id)
        job_dir = self.jobs_dir / job_id

        if model_config is None or training_config is None:
            job = self.get_job(job_id)
            if job:
                if model_config is None:
                    model_config = job.get("model_config") or {}
                if training_config is None:
                    training_config = job.get("training_config") or {}
                if lora_config is None:
                    lora_config = job.get("lora_config")
                if peft_method == "none":
                    peft_method = str(job.get("peft_method") or "none").lower()

        payload: dict[str, Any] = {
            "artifact_type": "inference" if is_final else "resume",
            "model_config": model_config or {},
            "training_config": training_config or {},
            "global_step": step,
            "round": round_num if round_num is not None else 0,
            "train_loss": train_loss if train_loss is not None else 0.0,
            "val_loss": val_loss,
            "model_state_dict": {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in state_dict.items()
            },
        }
        if peft_method == "lora":
            payload["peft_method"] = "lora"
            if lora_config:
                payload["lora_config"] = lora_config
            if adapter_state_dict:
                payload["adapter_state_dict"] = {
                    k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in adapter_state_dict.items()
                }

        target = ckpt_dir / f"checkpoint_step_{step:06d}.pt"
        tmp_target = ckpt_dir / f"checkpoint_step_{step:06d}.pt.tmp"
        torch.save(payload, tmp_target)
        tmp_target.replace(target)

        latest_target = ckpt_dir / "latest_checkpoint.pt"
        latest_tmp = ckpt_dir / "latest_checkpoint.pt.tmp"
        torch.save(payload, latest_tmp)
        latest_tmp.replace(latest_target)

        if is_final:
            final_target = ckpt_dir / "final_model.pt"
            final_tmp = ckpt_dir / "final_model.pt.tmp"
            torch.save(payload, final_tmp)
            final_tmp.replace(final_target)

            for alias_name, alias_dir in (("final_model.pt", job_dir), ("model.pt", ckpt_dir), ("model.pt", job_dir)):
                try:
                    shutil.copyfile(final_target, alias_dir / alias_name)
                except Exception:
                    pass

        if max_keep > 0:
            step_files = sorted(ckpt_dir.glob("checkpoint_step_*.pt"))
            if len(step_files) > max_keep:
                for old_f in step_files[:-max_keep]:
                    try:
                        old_f.unlink(missing_ok=True)
                    except Exception:
                        pass

        return target

    def load_latest_checkpoint(self, job_id: str, device: str = "cpu") -> Optional[dict[str, torch.Tensor]]:
        ckpt_dir = self.get_checkpoints_dir(job_id)
        latest = ckpt_dir / "latest_checkpoint.pt"
        if latest.exists():
            obj = safe_torch_load(latest, device=device)
            if isinstance(obj, dict) and "model_state_dict" in obj:
                return obj["model_state_dict"]
            return obj
        final = ckpt_dir / "final_model.pt"
        if final.exists():
            obj = safe_torch_load(final, device=device)
            if isinstance(obj, dict) and "model_state_dict" in obj:
                return obj["model_state_dict"]
            return obj
        return None

    def write_worker_logs(self, entries: list[tuple[str, float, str, str]]) -> None:
        if not entries:
            return
        def _op(conn: sqlite3.Connection) -> None:
            conn.executemany(
                "INSERT INTO worker_logs (worker_id, timestamp, level, message) VALUES (?, ?, ?, ?);",
                entries,
            )
        self._run_with_retry(_op, silent=True)

    def write_worker_log(self, worker_id: str, message: str, level: str = "INFO") -> None:
        self.write_worker_logs([(worker_id, time.time(), level, message)])

    def get_worker_logs(self, worker_id: str, limit: int = 200) -> list[dict[str, Any]]:
        def _op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            cursor = conn.execute(
                "SELECT id, worker_id, timestamp, level, message FROM worker_logs WHERE worker_id = ? ORDER BY id DESC LIMIT ?;",
                (worker_id, limit),
            )
            rows = cursor.fetchall()
            return [dict(r) for r in reversed(rows)]
        return self._run_with_retry(_op, default_on_error=[])


# =============================================================================
# 4. Embedded Data Sharding & Dataset
# =============================================================================

def compute_shard_boundaries(total_tokens: int, shard_index: int, total_shards: int, context_length: int = 512) -> tuple[int, int]:
    if total_shards <= 1:
        return 0, total_tokens
    total_windows = total_tokens // context_length
    windows_per_shard = total_windows // total_shards
    if windows_per_shard == 0:
        return 0, total_tokens

    start_token = shard_index * windows_per_shard * context_length
    end_token = total_tokens if shard_index == total_shards - 1 else (shard_index + 1) * windows_per_shard * context_length
    return start_token, end_token


class StandaloneTokenDataset(Dataset):
    def __init__(self, token_array: np.ndarray, context_length: int, shard_index: int = 0, total_shards: int = 1, vocab_size: Optional[int] = None, target_array: Optional[np.ndarray] = None) -> None:
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.has_targets = target_array is not None
        total_tokens = len(token_array)
        start_idx, end_idx = compute_shard_boundaries(total_tokens, shard_index, total_shards, context_length)
        self.tokens = token_array[start_idx:end_idx]
        self.targets = target_array[start_idx:end_idx] if self.has_targets else None
        usable = len(self.tokens) - self.context_length
        self.sample_count = max(0, (usable // self.context_length) + 1) if usable >= 0 else 0

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.context_length
        end = start + self.context_length
        x = torch.from_numpy(self.tokens[start:end].astype(np.int64))
        if self.has_targets and self.targets is not None:
            y = torch.from_numpy(self.targets[start:end].astype(np.int64))
        else:
            if end < len(self.tokens):
                y = torch.from_numpy(self.tokens[start + 1 : end + 1].astype(np.int64))
            else:
                y = x.clone()
        if self.vocab_size is not None and self.vocab_size > 0:
            x = torch.clamp(x, 0, self.vocab_size - 1)
            y = torch.where(y == -100, y, torch.clamp(y, 0, self.vocab_size - 1))
        return x, y


# =============================================================================
# 5. Model Architecture Factory (MicroGPT Parity)
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


def build_worker_model(model_config: dict[str, Any], device: str) -> nn.Module:
    """Build model using engine if available, or standalone exact MicroGPT implementation."""
    ctx_len = int(model_config.get("context_length", 0) or 0)
    try:
        from engine.config import ModelConfig
        from engine.model import MicroGPT
        valid_keys = ModelConfig.__dataclass_fields__.keys()
        filtered = {k: v for k, v in model_config.items() if k in valid_keys}
        m = MicroGPT(ModelConfig(**filtered)).to(device)
    except Exception:
        m = StandaloneMicroGPT(model_config).to(device)

    if hasattr(m, "enable_gradient_checkpointing") and ctx_len >= 1024:
        m.enable_gradient_checkpointing(True)
    return m


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
# 6. Worker Daemon Engine
# =============================================================================

class StandaloneWorker:
    def __init__(self, shared_dir: Path | str, worker_id: Optional[str] = None, device: Optional[str] = None) -> None:
        self.bus = StandaloneStorageBus(shared_dir)
        self.hostname = socket.gethostname()

        # Hardware & device resolution
        all_gpus = detect_all_gpus()
        if device:
            self.device_str = device
        elif all_gpus and all_gpus[0]["device"] != "cpu":
            self.device_str = all_gpus[0]["device"]
        else:
            self.device_str = "cpu"

        self.device_tag = get_device_tag(self.device_str)
        self.worker_id = worker_id or f"{self.hostname}_{self.device_tag}"

        matched = next((g for g in all_gpus if g["device"] == self.device_str), None)
        if matched:
            self.gpu_name = matched["name"]
            self.vram_gb = matched["vram_gb"]
        elif self.device_str.startswith("cuda") and torch.cuda.is_available():
            dev_idx = 0
            if ":" in self.device_str:
                try:
                    dev_idx = int(self.device_str.split(":")[1])
                except ValueError:
                    dev_idx = 0
            self.gpu_name = torch.cuda.get_device_name(dev_idx)
            props = torch.cuda.get_device_properties(dev_idx)
            self.vram_gb = round(props.total_memory / (1024 ** 3), 2)
        else:
            self.gpu_name = platform.processor() or "CPU"
            self.vram_gb = 0.0

        self.bus.register_worker(self.worker_id, self.hostname, self.gpu_name, self.vram_gb)

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

    def _heartbeat(self, status: Optional[str] = None, current_job_id: Optional[str] = None) -> None:
        try:
            metrics = collect_system_metrics(self.device_str, self.vram_gb)
            self.bus.heartbeat(self.worker_id, status=status, current_job_id=current_job_id, metrics=metrics)
        except Exception:
            pass

    def _restart_process(self) -> None:
        """Cleanly respawn this worker process and exit."""
        self.log(f"Respawning worker process for {self.worker_id}...")
        self.bus.set_worker_command(self.worker_id, None)
        try:
            self._heartbeat(status="RESTARTING", current_job_id=None)
        except Exception:
            pass
        if hasattr(self, "_cleanup_func"):
            import atexit
            try:
                atexit.unregister(self._cleanup_func)
            except Exception:
                pass
        release_singleton_lock(self.device_tag)
        release_singleton_lock(f"wid_{self.worker_id}")

        # Attempt git pull to grab latest repository updates if running inside a git checkout
        try:
            root_dir = Path(__file__).resolve().parent.parent if "cluster" in str(Path(__file__).parent) else Path(__file__).resolve().parent
            if (root_dir / ".git").exists():
                self.log("Pulling latest cluster repository changes before respawning...")
                pull_flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
                subprocess.run(["git", "pull", "--ff-only"], cwd=str(root_dir), timeout=15, capture_output=True, creationflags=pull_flags)
        except Exception:
            pass

        cmd_args = get_worker_respawn_cmd(
            worker_id=self.worker_id,
            device_str=self.device_str,
            shared_dir=self.bus.shared_dir,
            allow_shared_device=self.allow_shared_device,
        )
        try:
            flags = 0
            startupinfo = None
            if sys.platform == "win32":
                DETACHED_PROCESS = 0x00000008
                CREATE_NEW_PROCESS_GROUP = 0x00000200
                CREATE_NO_WINDOW = 0x08000000
                flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0  # SW_HIDE (prevents Windows Terminal / conhost flashing)
            log_dir = Path(self.bus.shared_dir) / "logs"
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{self.worker_id}.log"
            except Exception:
                log_path = Path(tempfile.gettempdir()) / f"cluster_worker_{self.worker_id}.log"
            log_file = open(log_path, "a", encoding="utf-8")
            root_dir = str(Path(__file__).resolve().parent.parent if "cluster" in str(Path(__file__).parent) else Path(__file__).resolve().parent)
            proc = subprocess.Popen(
                cmd_args,
                cwd=root_dir,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=flags,
                startupinfo=startupinfo,
                close_fds=True,
            )
            self.log(f"Successfully spawned new worker process (PID: {proc.pid}). Exiting current process (PID: {os.getpid()}).")
        except Exception as exc:
            self.log(f"Failed to respawn worker process: {exc}", level="ERROR")
        time.sleep(0.3)
        os._exit(0)

    def run(self, poll_interval: float = 3.0) -> None:
        """Main worker loop: registers heartbeat, claims jobs, and trains across rounds."""
        self.log(f"Node online: {self.worker_id}")
        self.log(f"Host: {self.hostname} | Device: {self.device_str} ({self.gpu_name}, {self.vram_gb} GB VRAM)")
        self.log(f"Central Storage: {self.bus.shared_dir.resolve()} | PID: {os.getpid()} ({self.device_tag})")
        self.log("Waiting for cluster jobs...")

        def _cleanup():
            try:
                self.bus.heartbeat(self.worker_id, status="OFFLINE", current_job_id=None)
                release_singleton_lock(self.device_tag)
                release_singleton_lock(f"wid_{self.worker_id}")
            except Exception:
                pass

        import atexit, signal
        self._cleanup_func = _cleanup
        atexit.register(_cleanup)
        try:
            signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
            signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
            if hasattr(signal, "SIGBREAK"):
                signal.signal(signal.SIGBREAK, lambda s, f: sys.exit(0))
        except Exception:
            pass

        while True:
            try:
                try:
                    cmd = self.bus.get_worker_command(self.worker_id)
                except Exception:
                    cmd = None
                if cmd == "STOP":
                    self.log("Received STOP command. Setting status to IDLE and waiting for job...")
                    try:
                        self.bus.set_worker_command(self.worker_id, None)
                    except Exception:
                        pass
                    self._abort_active_job = True
                    self._heartbeat(status="IDLE", current_job_id=None)
                elif cmd == "SHUTDOWN":
                    self.log("Received SHUTDOWN command. Shutting down worker process...")
                    try:
                        self.bus.set_worker_command(self.worker_id, None)
                    except Exception:
                        pass
                    self._heartbeat(status="OFFLINE", current_job_id=None)
                    release_singleton_lock(self.device_tag)
                    release_singleton_lock(f"wid_{self.worker_id}")
                    os._exit(0)
                elif cmd == "RESTART":
                    self.log("Received RESTART command. Respawning process...")
                    self._restart_process()

                # Check if worker is disabled by operator: remain active and heartbeating, but do not claim jobs
                try:
                    is_enabled = getattr(self.bus, "is_worker_enabled", lambda wid: True)(self.worker_id)
                except Exception:
                    is_enabled = True

                if not is_enabled:
                    self._heartbeat(status="DISABLED", current_job_id=None)
                    time.sleep(poll_interval)
                    continue

                self._heartbeat(status="IDLE", current_job_id=None)
                try:
                    active_job = self.bus.get_active_job()
                except Exception:
                    active_job = None

                if active_job:
                    status = active_job.get("status")
                    jid = active_job.get("job_id", "")
                    try:
                        is_job_stopped = self.bus.is_stopped(jid)
                    except Exception:
                        is_job_stopped = False
                    if status == "RUNNING" and not is_job_stopped:
                        is_compat, reason = self.is_job_compatible(active_job)
                        if not is_compat:
                            if not hasattr(self, "_incompat_reported"):
                                self._incompat_reported = set()
                            if jid not in self._incompat_reported:
                                self._incompat_reported.add(jid)
                                self.log(f"Worker {self.worker_id} cannot accept job '{jid}': {reason}. Skipping.", level="WARNING")
                                self._heartbeat(status="INCOMPATIBLE", current_job_id=None)
                        else:
                            self._execute_job(active_job, poll_interval=poll_interval)
                    elif status == "QUEUED":
                        self._heartbeat(status="READY", current_job_id=jid)
            except Exception as exc:
                self.log(f"Transient error in worker poll loop: {exc}", level="WARNING")

            time.sleep(poll_interval)

    def is_job_compatible(self, job: dict[str, Any]) -> tuple[bool, str]:
        """Check if this worker node and device are compatible with the requested job."""
        try:
            is_enabled = getattr(self.bus, "is_worker_enabled", lambda wid: True)(self.worker_id)
        except Exception:
            is_enabled = True
        if not is_enabled:
            return False, f"Worker '{self.worker_id}' is disabled by operator."

        if getattr(self, "_current_status", "") == "DEGRADED":
            return False, "Worker is in DEGRADED status due to failed hardware preflight checks."

        target_workers = job.get("target_workers")
        if target_workers and isinstance(target_workers, list):
            if self.worker_id not in target_workers:
                return False, f"Worker ID '{self.worker_id}' is not in job target_workers list: {target_workers}."

        if self.device_str.startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                return False, "Job targets CUDA, but CUDA is not available on this host."
            try:
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
                    self._restart_process()
                return False, f"CUDA device '{self.device_str}' failed kernel execution test: {exc}"

        tcfg = job.get("training_config", {})
        if isinstance(tcfg, str):
            try:
                tcfg = json.loads(tcfg)
            except Exception:
                tcfg = {}
        prec = str(tcfg.get("precision", "float32")).lower()
        if prec in {"bfloat16", "bf16"} and self.device_str.startswith("cuda"):
            if not is_hardware_bf16_supported(self.device_str):
                # Device lacks native hardware BF16 tensor cores; training will safely adapt to FP16 with GradScaler
                pass

        dataset_path = job.get("dataset_path", "")
        if not dataset_path:
            return False, "Job specifies no dataset_path."
        if not os.path.exists(dataset_path):
            return False, f"Dataset path is not accessible on this node: '{dataset_path}'"

        return True, "Compatible"

    def _execute_job(self, job: dict[str, Any], poll_interval: float = 2.0) -> None:
        job_id = job["job_id"]
        is_compat, reason = self.is_job_compatible(job)
        if not is_compat:
            self.log(f"Worker {self.worker_id} cannot execute job '{job_id}': {reason}. Refusing claim.", level="WARNING")
            self._heartbeat(status="INCOMPATIBLE", current_job_id=None)
            return

        self.log(f">>> Claimed job: {job_id}")
        self._heartbeat(status="PREPARING", current_job_id=job_id)

        model = None
        optimizer = None
        dataloader = None
        dataloader_iter = None
        val_loader = None
        dataset = None
        token_array = None
        val_token_array = None
        global_weights = None
        batch = None
        x = None
        y = None
        logits = None
        loss = None

        try:
            shard_idx, total_shards = self.bus.claim_job_slot(job_id, self.worker_id)
            self.log(f"Assigned data shard slot {shard_idx + 1} of {total_shards} total nodes")

            dataset_path = job.get("dataset_path", "")
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

            purge_local_dataset_cache(keep_files=target_cache_files, log_fn=self.log)
            local_dataset_path = cache_dataset_to_local(dataset_path, log_fn=self.log, keep_files=target_cache_files)
            local_targets_path = None
            if targets_path and os.path.exists(targets_path):
                local_targets_path = cache_dataset_to_local(targets_path, log_fn=self.log, keep_files=target_cache_files)
            local_val_targets_path = None
            if val_targets_path and os.path.exists(val_targets_path):
                local_val_targets_path = cache_dataset_to_local(val_targets_path, log_fn=self.log, keep_files=target_cache_files)

            if not os.path.exists(local_dataset_path):
                err = f"Dataset not found at: {local_dataset_path}"
                self.log(err, level="ERROR")
                self._heartbeat(status="ERROR", current_job_id=job_id)
                return

            token_array = np.load(local_dataset_path, mmap_mode="r")
            target_array = np.load(local_targets_path, mmap_mode="r") if (local_targets_path and os.path.exists(local_targets_path)) else None
            model_cfg = job.get("model_config", {})
            training_cfg = job.get("training_config", {})

            # Auto-detect and guard against invalid vocab_size <= 1 or vocab_size < max token ID in dataset
            vocab_size = int(model_cfg.get("vocab_size", 0) or 0)

            # 1. Inspect metadata files if present on shared storage or dataset dir
            detected_vocab = 0
            for sdir in [Path(dataset_path).parent, self.bus.shared_dir]:
                summary_file = sdir / "dataset_summary.json"
                if summary_file.exists():
                    try:
                        meta = json.loads(summary_file.read_text(encoding="utf-8"))
                        detected_vocab = int(meta.get("tokenizer_vocab_size", 0) or 0)
                        if detected_vocab > 0:
                            break
                    except Exception:
                        pass
                tok_file = sdir / "tokenizer.json"
                if tok_file.exists() and detected_vocab <= 0:
                    try:
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
                # Align with clean multiple of 256 for Tensor Core efficiency
                min_safe_vocab = ((max_token_id + 1 + 255) // 256) * 256
                # Standard LLaMA / Mistral 32k tokenizer alignment
                if 31000 <= max_token_id < 32000:
                    min_safe_vocab = max(min_safe_vocab, 32000)

            target_vocab = max(vocab_size, detected_vocab, min_safe_vocab)
            if vocab_size < target_vocab:
                self.log(
                    f"Model vocab_size ({vocab_size}) is smaller than required ({target_vocab}, detected: {detected_vocab}, max token: {max_token_id}). "
                    f"Auto-adjusting vocab_size to {target_vocab}.",
                    level="WARNING",
                )
                vocab_size = target_vocab
                model_cfg["vocab_size"] = vocab_size

            context_length = int(model_cfg.get("context_length", 512))
            batch_size = int(training_cfg.get("batch_size", 4))
            lr = float(training_cfg.get("learning_rate", 3e-4))

            val_token_array = None
            val_target_array = None
            train_token_array = token_array
            train_target_array = target_array

            # Prepare validation dataset if available
            if val_dataset_path and os.path.exists(val_dataset_path):
                try:
                    local_val_path = cache_dataset_to_local(val_dataset_path, log_fn=self.log, keep_files=target_cache_files)
                    v_arr = np.load(local_val_path, mmap_mode="r")
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

            dataset = StandaloneTokenDataset(
                train_token_array, context_length, shard_idx, total_shards, vocab_size=vocab_size, target_array=train_target_array
            )
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            dataloader_iter = iter(dataloader)

            if val_token_array is not None and len(val_token_array) > context_length:
                try:
                    val_ds = StandaloneTokenDataset(
                        val_token_array, context_length, shard_idx, total_shards, vocab_size=vocab_size, target_array=val_target_array
                    )
                    if len(val_ds) > 0:
                        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
                        self.log(f"Validation DataLoader ready ({len(val_ds)} sample windows).")
                except Exception as e:
                    self.log(f"Could not initialize validation dataset: {e}", level="WARNING")

            model = build_worker_model(model_cfg, self.device_str)

            # Check if this is a fine-tuning job or pretraining resume, and load base model weights
            job_type = str(job.get("job_type") or training_cfg.get("training_mode") or "pretrain").lower()
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
            peft_method = str(job.get("peft_method") or training_cfg.get("peft_method") or "none").lower()
            if peft_method == "lora":
                lora_cfg = job.get("lora_config") or training_cfg.get("lora_config") or {}
                if isinstance(lora_cfg, str):
                    try:
                        lora_cfg = json.loads(lora_cfg)
                    except Exception:
                        lora_cfg = {}
                if not isinstance(lora_cfg, dict):
                    lora_cfg = {}
                l_rank = int(lora_cfg.get("rank", training_cfg.get("lora_rank", 8)))
                l_alpha = float(lora_cfg.get("alpha", training_cfg.get("lora_alpha", 16.0)))
                l_dropout = float(lora_cfg.get("dropout", training_cfg.get("lora_dropout", 0.05)))
                l_targets = str(lora_cfg.get("target_modules", training_cfg.get("lora_target_modules", "attention")))
                num_lora = apply_lora_adapters(model, rank=l_rank, alpha=l_alpha, dropout=l_dropout, target_modules=l_targets)
                freeze_non_lora_parameters(model)
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                self.log(f"LoRA adapters applied to {num_lora} modules. Trainable parameters: {trainable_params:,} (base model frozen).")

            # Create optimizer with separated weight decay groups (frontier LLM standard)
            weight_decay = float(training_cfg.get("weight_decay", 0.01))
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
            try:
                optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=device_is_cuda)
            except Exception:
                optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
            self.log(f"Initialized model for {self.worker_id} on {self.device_str} ({len(decay_params)} 2D matrix weights, {len(nodecay_params)} 1D tensors). Ready to train.")

            # Ensure activation checkpointing is active for long contexts or when configured
            act_ckpt = bool(training_cfg.get("activation_checkpointing", False) or context_length >= 1024)
            if hasattr(model, "enable_gradient_checkpointing"):
                model.enable_gradient_checkpointing(act_ckpt)
                self.log(f"Activation checkpointing {'ENABLED' if act_ckpt else 'disabled'} (context_length={context_length}).")

            if self.device_str.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._heartbeat(status="READY", current_job_id=job_id)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self.log(f"Preparation failed for job {job_id}:\n{tb}", level="ERROR")
            self._heartbeat(status="ERROR", current_job_id=job_id)
            return

        max_rounds = int(job.get("max_rounds", 10))
        sync_steps = int(job.get("sync_interval_steps", 250))
        cur_round = int(job.get("current_round", 0))

        # Mixed precision (AMP) configuration
        is_cuda = self.device_str.startswith("cuda") and torch.cuda.is_available()
        precision = str(training_cfg.get("precision", "fp16")).lower()
        use_amp = bool(training_cfg.get("use_amp", True))
        amp_enabled = is_cuda and use_amp and (precision in ("fp16", "bf16"))
        if amp_enabled and precision == "bf16":
            if is_hardware_bf16_supported(self.device_str):
                amp_dtype = torch.bfloat16
            else:
                self.log(
                    f"Notice: Compute device '{self.device_str}' ({getattr(self, 'gpu_name', 'GPU')}) lacks native hardware Bfloat16 tensor cores. "
                    "Safely using FP16 with GradScaler for numerical stability and kernel safety.",
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

        try:
            while cur_round < max_rounds:
                try:
                    stopped = self.bus.is_stopped(job_id)
                except Exception:
                    stopped = False
                if stopped or getattr(self, "_abort_active_job", False):
                    self.log(f"Job {job_id} stopped. Returning to IDLE.")
                    break

                try:
                    cmd = self.bus.get_worker_command(self.worker_id)
                except Exception:
                    cmd = None
                if cmd == "STOP":
                    self.log(f"Received STOP command during job {job_id}. Aborting job and returning to IDLE...")
                    try:
                        self.bus.set_worker_command(self.worker_id, None)
                    except Exception:
                        pass
                    self._abort_active_job = True
                    break
                elif cmd == "SHUTDOWN":
                    self.log(f"Received SHUTDOWN command during job {job_id}. Shutting down worker...")
                    try:
                        self.bus.set_worker_command(self.worker_id, None)
                    except Exception:
                        pass
                    self._heartbeat(status="OFFLINE", current_job_id=None)
                    release_singleton_lock(self.device_tag)
                    release_singleton_lock(f"wid_{self.worker_id}")
                    os._exit(0)
                elif cmd == "RESTART":
                    self.log(f"Received RESTART command during job {job_id}.")
                    self._restart_process()

                # Handle cooperative pause
                while True:
                    try:
                        paused = self.bus.is_paused(job_id)
                    except Exception:
                        paused = False
                    if not paused:
                        break
                    self._heartbeat(status="PAUSED", current_job_id=job_id)
                    time.sleep(1.0)
                    try:
                        if self.bus.is_stopped(job_id):
                            break
                    except Exception:
                        pass

                # Dynamic shard reassignment: verify active worker pool and take over dropped worker slots
                try:
                    new_shard_idx, new_total_shards = self.bus.get_worker_shard_assignment(job_id, self.worker_id, cur_round)
                    if new_shard_idx != shard_idx or new_total_shards != total_shards:
                        self.log(f"Active workers changed. Reallocating shard {new_shard_idx + 1}/{new_total_shards} for round {cur_round}.")
                        shard_idx, total_shards = new_shard_idx, new_total_shards
                        dataset = StandaloneTokenDataset(
                            train_token_array, context_length, shard_idx, total_shards, vocab_size=vocab_size, target_array=train_target_array
                        )
                        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                        dataloader_iter = iter(dataloader)
                        if val_token_array is not None and len(val_token_array) > context_length:
                            try:
                                val_ds = StandaloneTokenDataset(
                                    val_token_array, context_length, shard_idx, total_shards, vocab_size=vocab_size, target_array=val_target_array
                                )
                                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
                            except Exception:
                                pass
                except Exception as shard_err:
                    self.log(f"Notice: Shard check deferred due to busy database ({shard_err}). Retaining shard {shard_idx + 1}/{total_shards}.", level="WARNING")

                # Sync from global model if round > 0
                if cur_round > 0:
                    self.log(f"Waiting for global averaged model round {cur_round - 1}...")
                    while not self.bus.is_global_weights_ready(job_id, cur_round - 1):
                        try:
                            if self.bus.is_stopped(job_id):
                                return
                        except Exception:
                            pass
                        self._heartbeat(status="SYNC_WAIT", current_job_id=job_id)
                        time.sleep(poll_interval)

                    global_state = self.bus.load_global_weights(job_id, cur_round - 1, device=self.device_str)
                    model.load_state_dict(global_state)
                    self.log(f"Loaded round {cur_round - 1} global weights into {self.device_str}.")

                # Local SGD training steps
                self.log(f"Training round {cur_round}/{max_rounds} ({sync_steps} local steps on {self.device_str}, AMP: {amp_enabled})...")
                model.train()
                self._heartbeat(status="TRAINING", current_job_id=job_id)

                grad_accum = max(1, int(training_cfg.get("gradient_accumulation", 1) or 1))
                should_checkpoint = bool(training_cfg.get("activation_checkpointing", False) or context_length >= 1024)
                if is_cuda:
                    try:
                        dev_idx = 0
                        if ":" in self.device_str:
                            dev_idx = int(self.device_str.split(":")[1])
                        _, tot_bytes = torch.cuda.mem_get_info(dev_idx)
                        tot_gb = tot_bytes / (1024 ** 3)
                    except Exception:
                        tot_gb = 16.0

                    # Scale max_safe_tokens to target 85-90% VRAM utilization while capping
                    # per-kernel execution to prevent Windows WDDM driver watchdog (TDR) timeouts
                    if tot_gb < 8.5:
                        max_safe_tokens = 16384 if should_checkpoint else 8192
                    elif tot_gb <= 13.0:
                        max_safe_tokens = 32768 if should_checkpoint else 16384
                    elif tot_gb <= 18.0:
                        max_safe_tokens = 49152 if should_checkpoint else 24576
                    elif tot_gb <= 26.0:
                        max_safe_tokens = 65536 if should_checkpoint else 32768
                    else:
                        max_safe_tokens = 98304 if should_checkpoint else 49152

                    safe_micro_bs = max(1, max_safe_tokens // max(context_length, 1)) if context_length > 0 else 4
                else:
                    safe_micro_bs = 999999

                optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0

                start_time = time.time()
                tokens_processed = 0
                step_losses = []
                for step_idx in range(sync_steps):
                    if step_idx % 5 == 0:
                        try:
                            cmd = self.bus.get_worker_command(self.worker_id)
                            if cmd == "STOP":
                                self.log("Received STOP command during training. Aborting job and returning to IDLE...")
                                try:
                                    self.bus.set_worker_command(self.worker_id, None)
                                except Exception:
                                    pass
                                self._abort_active_job = True
                                return
                            elif cmd == "SHUTDOWN":
                                self.log("Received SHUTDOWN command during training. Shutting down...")
                                try:
                                    self.bus.set_worker_command(self.worker_id, None)
                                except Exception:
                                    pass
                                self._heartbeat(status="OFFLINE", current_job_id=None)
                                release_singleton_lock(self.device_tag)
                                release_singleton_lock(f"wid_{self.worker_id}")
                                os._exit(0)
                            elif cmd == "RESTART":
                                self.log("Received RESTART command during training. Restarting process...")
                                self._restart_process()
                        except Exception:
                            pass
                    if step_idx % 20 == 0:
                        self._heartbeat(status="TRAINING", current_job_id=job_id)
                    try:
                        stopped = self.bus.is_stopped(job_id)
                    except Exception:
                        stopped = False
                    if stopped or getattr(self, "_abort_active_job", False):
                        self.log(f"Job {job_id} stopped. Aborting training loop.")
                        return
                    try:
                        batch = next(dataloader_iter)
                    except StopIteration:
                        dataloader_iter = iter(dataloader)
                        batch = next(dataloader_iter)

                    x, y = batch
                    batch_sz = x.size(0)
                    tokens_processed += int(x.numel())
                    x = torch.clamp(x, 0, vocab_size - 1)
                    y = torch.where(y == -100, y, torch.clamp(y, 0, vocab_size - 1))

                    micro_bs = max(1, min(batch_sz, safe_micro_bs))
                    num_micros = math.ceil(batch_sz / micro_bs)
                    effective_accum = num_micros * grad_accum

                    batch_loss = 0.0
                    for m_idx in range(0, batch_sz, micro_bs):
                        x_m = x[m_idx : m_idx + micro_bs].to(self.device_str, non_blocking=True)
                        y_m = y[m_idx : m_idx + micro_bs].to(self.device_str, non_blocking=True)
                        x_m = torch.clamp(x_m, 0, vocab_size - 1)

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
                    is_step_boundary = (accumulated_batches % grad_accum == 0) or (step_idx + 1 == sync_steps)

                    max_grad = float(training_cfg.get("max_gradient") or training_cfg.get("max_grad") or 1.0)
                    if is_step_boundary:
                        # Apply cosine learning rate schedule with linear warmup across rounds
                        t_cfg = training_cfg if isinstance(training_cfg, dict) else {}
                        base_lr = float(t_cfg.get("learning_rate", 3e-4))
                        total_steps = max(1, max_rounds * sync_steps)
                        warmup_steps = int(t_cfg.get("warmup_steps", max(10, total_steps // 20)))
                        warmup_steps = min(warmup_steps, max(total_steps - 1, 1))
                        min_ratio = float(t_cfg.get("scheduler_min_lr_ratio", 0.1))
                        annealing_steps = int(t_cfg.get("annealing_steps", 0))
                        global_step = cur_round * sync_steps + step_idx
                        if global_step < warmup_steps:
                            lr_mult = max(global_step + 1, 1) / max(warmup_steps, 1)
                        elif annealing_steps > 0 and global_step >= (total_steps - annealing_steps):
                            # Frontier curriculum annealing cooldown
                            anneal_prog = (global_step - (total_steps - annealing_steps)) / max(annealing_steps, 1)
                            anneal_prog = max(0.0, min(anneal_prog, 1.0))
                            start_anneal_step = total_steps - annealing_steps
                            base_prog = (start_anneal_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                            base_prog = max(0.0, min(base_prog, 1.0))
                            start_lr_mult = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * base_prog))
                            lr_mult = min_ratio + (start_lr_mult - min_ratio) * (1.0 - anneal_prog)
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

                    step_losses.append(batch_loss)

                elapsed = max(time.time() - start_time, 1e-4)
                tokens_per_sec = tokens_processed / elapsed
                avg_loss = sum(step_losses) / max(len(step_losses), 1)

                val_loss = None
                if val_loader is not None:
                    try:
                        model.eval()
                        v_loss_sum = 0.0
                        v_batches = 0
                        with torch.no_grad():
                            for v_step, (vx, vy) in enumerate(val_loader):
                                if v_step >= 25:
                                    break
                                vx = vx.to(self.device_str, non_blocking=True)
                                vy = vy.to(self.device_str, non_blocking=True)
                                v_out = model(vx)
                                v_logits = v_out[0] if isinstance(v_out, (tuple, list)) else (v_out.logits if hasattr(v_out, "logits") else v_out)
                                v_loss = F.cross_entropy(v_logits.view(-1, v_logits.size(-1)), vy.view(-1), ignore_index=-100)
                                v_loss_sum += float(v_loss.item())
                                v_batches += 1
                        val_loss = v_loss_sum / max(v_batches, 1)
                        model.train()
                    except Exception as exc:
                        self.log(f"Validation evaluation failed: {exc}", level="WARNING")

                val_str = f", val loss: {val_loss:.4f}" if val_loss is not None else ""
                self.log(f"Finished round {cur_round} in {elapsed:.1f}s (avg loss: {avg_loss:.4f}{val_str}, speed: {tokens_per_sec:.0f} tok/s). Depositing weights & telemetry...")

                # Save local weights
                self._heartbeat(status="DEPOSITING", current_job_id=job_id)
                self.bus.save_worker_weights(
                    job_id=job_id,
                    round_num=cur_round,
                    worker_id=self.worker_id,
                    state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                )

                # Save local telemetry
                try:
                    self.bus.save_worker_telemetry(
                        job_id=job_id,
                        round_num=cur_round,
                        worker_id=self.worker_id,
                        telemetry={
                            "worker_id": self.worker_id,
                            "round": cur_round,
                            "steps_completed": sync_steps,
                            "avg_loss": round(avg_loss, 4),
                            "val_loss": round(val_loss, 4) if val_loss is not None else None,
                            "compute_sec": round(elapsed, 2),
                            "tokens_processed": tokens_processed,
                            "tokens_per_sec": round(tokens_per_sec, 1),
                            "timestamp": time.time(),
                        },
                    )
                except Exception as t_err:
                    self.log(f"Notice: Failed to save round telemetry ({t_err}).", level="WARNING")

                # Wait for coordinator to publish global model
                t_sync_start = time.time()
                last_log_wait = t_sync_start
                self.log(f"Deposited weights for round {cur_round}. Waiting for coordinator synchronization...")
                while not self.bus.is_global_weights_ready(job_id, cur_round):
                    try:
                        if self.bus.is_stopped(job_id):
                            return
                    except Exception:
                        pass
                    self._heartbeat(status="SYNC_WAIT", current_job_id=job_id)

                    now_wait = time.time()
                    elapsed_sync = now_wait - t_sync_start
                    if now_wait - last_log_wait >= 30.0:
                        last_log_wait = now_wait
                        self.log(f"Waiting for round {cur_round} global weights ({elapsed_sync:.0f}s elapsed)...")

                        try:
                            # Autonomous fallback: if waiting > 45s and coordinator is inactive or this is the sole node
                            job_info = self.bus.get_job(job_id) or {}
                            job_updated = float(job_info.get("updated_at") or 0.0)
                            coord_unresponsive = (now_wait - job_updated) > 60.0
                            active_parts = self.bus.get_job_participants(job_id)
                            sole_worker = (len(active_parts) <= 1) or all(p.get("worker_id") == self.worker_id for p in active_parts)

                            if elapsed_sync > 45.0 and (sole_worker or coord_unresponsive):
                                self.log(
                                    f"Autonomous sync: Coordinator is inactive or sole node detected ({len(active_parts)} active). "
                                    f"Triggering local round {cur_round} aggregation..."
                                )
                                from cluster.coordinator import ClusterCoordinator
                                coord = ClusterCoordinator(self.bus, job_id)
                                coord.wait_and_average_round(round_num=cur_round, poll_interval_seconds=0.5)
                        except Exception as c_err:
                            self.log(f"Autonomous aggregation notice: {c_err}")

                    time.sleep(poll_interval)

                sync_wait_sec = max(time.time() - t_sync_start, 0.0)
                round_total_sec = elapsed + sync_wait_sec
                duty_pct = (elapsed / max(round_total_sec, 0.001)) * 100.0

                # Load synchronized global model weights into local model for next round
                global_weights = self.bus.load_global_weights(job_id, cur_round, device=self.device_str)
                model.load_state_dict(global_weights)
                self.log(
                    f"Successfully loaded averaged global weights for round {cur_round} in {sync_wait_sec:.1f}s. "
                    f"[Timing] Compute {elapsed:.1f}s | Sync Wait {sync_wait_sec:.1f}s | Duty Cycle {duty_pct:.1f}%"
                )

                cur_round += 1

            self.log(f"Job {job_id} finished. Returning to IDLE.")
            self._heartbeat(status="IDLE", current_job_id=None)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self.log(f"Training failed for job {job_id}:\n{tb}", level="ERROR")
            self._heartbeat(status="ERROR", current_job_id=job_id)
            err_lower = (str(exc) + " " + tb).lower()
            if any(s in err_lower for s in ("cuda error", "device-side assert", "illegal memory access", "out of memory")):
                if self.device_str.startswith("cuda"):
                    self.log(
                        "Fatal CUDA context corruption / error detected. Respawning fresh worker process with clean GPU context...",
                        level="WARNING",
                    )
                    self._restart_process()
            return
        finally:
            # Explicitly free model weights, optimizer states, and CUDA memory
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
            global_weights = None
            batch = None
            x = None
            y = None
            logits = None
            loss = None

            import gc
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

            try:
                self._abort_active_job = False
                self._heartbeat(status="IDLE", current_job_id=None)
            except Exception:
                pass


# =============================================================================
# 7. Main Entry Point
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone Cluster Worker Daemon for Local SGD")
    parser.add_argument("--shared-dir", default=None, help="Path to central shared storage directory (or set LLM_SHARED_DIR)")
    parser.add_argument("--worker-id", default=None, help="Explicit worker node identifier")
    parser.add_argument("--device", default=None, help="Compute device (e.g. 'cuda:0', 'cuda:1', 'cpu')")
    parser.add_argument("--all-gpus", action="store_true", help="Launch a dedicated background worker process for each detected GPU")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Polling interval in seconds")
    parser.add_argument("--stop", action="store_true", help="Stop active background worker process(es) on this machine")
    parser.add_argument("--status", action="store_true", help="Check status of background worker(s) on this machine")
    parser.add_argument("--detach", "--background", action="store_true", help="Launch worker in the background detached from this terminal")
    args = parser.parse_args()

    shared_dir = args.shared_dir or os.environ.get("LLM_SHARED_PATH") or os.environ.get("LLM_SHARED_DIR")

    if args.stop:
        device_tag = get_device_tag(args.device) if args.device else None
        return stop_running_worker(device_tag)

    if args.status:
        pids = get_all_running_worker_pids()
        if pids:
            print(f"[ClusterWorker] Active workers on this host ({len(pids)}):")
            for dev_tag, pid in pids.items():
                print(f"  - Device [{dev_tag}]: PID {pid}")
        else:
            print("[ClusterWorker] No active worker processes running on this host.")
        return 0

    if not shared_dir:
        print("[ClusterWorker] ERROR: Central shared directory not specified!")
        print("[ClusterWorker] Please either:")
        print("  1. Set the LLM_SHARED_DIR environment variable: e.g. set LLM_SHARED_DIR=Z:\\llm_cluster")
        print("  2. Pass the argument: python cluster_worker.py --shared-dir Z:\\llm_cluster")
        return 1

    # Handle detached background launch
    if getattr(args, "detach", False):
        flags = 0
        if sys.platform == "win32":
            flags = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            )
        worker_exe = get_worker_executable()
        script_path = str(Path(__file__).resolve())
        cmd = [worker_exe, script_path] + [arg for arg in sys.argv[1:] if arg not in ("--detach", "--background")]
        log_dir = Path(shared_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        wid_label = args.worker_id or f"{socket.gethostname()}_{get_device_tag(args.device or 'auto')}"
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
        print(f"[ClusterWorker] Started detached background worker '{wid_label}' (PID: {proc.pid})")
        print(f"[ClusterWorker] Output is being logged to: {log_file_path.resolve()}")
        print("[ClusterWorker] You can now safely close this PowerShell window.")
        return 0

    # Handle multi-GPU launch
    if args.all_gpus:
        all_gpus = detect_all_gpus()
        devices = [g["device"] for g in all_gpus] if all_gpus else ["cpu"]
        print(f"[ClusterWorker] Multi-GPU mode: spawning worker process for each device: {devices}")
        worker_exe = get_worker_executable()
        script_path = str(Path(__file__).resolve())
        env = os.environ.copy()
        if shared_dir:
            env["LLM_SHARED_DIR"] = shared_dir
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        for dev in devices:
            cmd = [worker_exe, script_path, "--device", dev, "--shared-dir", shared_dir]
            proc = subprocess.Popen(cmd, env=env, creationflags=flags)
            print(f"[ClusterWorker] Started worker for device '{dev}' (PID: {proc.pid})")
        return 0

    # Single device worker
    resolved_device = args.device
    if not resolved_device:
        all_gpus = detect_all_gpus()
        resolved_device = all_gpus[0]["device"] if (all_gpus and all_gpus[0]["device"] != "cpu") else "cpu"

    device_tag = get_device_tag(resolved_device)

    # Enforce per-device singleton
    if not acquire_singleton_lock(device_tag, timeout_seconds=3.0):
        return 0

    wid_tag = f"wid_{args.worker_id}" if args.worker_id else None
    if wid_tag and not acquire_singleton_lock(wid_tag, timeout_seconds=3.0):
        release_singleton_lock(device_tag)
        return 0

    # Set process console title on Windows
    if sys.platform == "win32":
        try:
            worker_label = args.worker_id or f"{socket.gethostname()}_{device_tag}"
            ctypes.windll.kernel32.SetConsoleTitleW(f"ClusterWorker [{worker_label}] - PID {os.getpid()}")
        except Exception:
            pass

    worker: Optional[StandaloneWorker] = None

    def _sig_handler(signum: int, frame: Any) -> None:
        print("\n[ClusterWorker] Received termination signal. Shutting down gracefully...")
        if worker:
            try:
                worker.bus.heartbeat(worker.worker_id, status="OFFLINE", current_job_id=None)
            except Exception:
                pass
        release_singleton_lock(device_tag)
        if wid_tag:
            release_singleton_lock(wid_tag)
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    worker = StandaloneWorker(shared_dir=shared_dir, worker_id=args.worker_id, device=resolved_device)
    try:
        worker.run(poll_interval=args.poll_interval)
    except KeyboardInterrupt:
        _sig_handler(signal.SIGINT, None)
    finally:
        release_singleton_lock(device_tag)
        if wid_tag:
            release_singleton_lock(wid_tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
