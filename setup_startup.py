"""Windows Startup Installer for LLM Cluster Workers.

Installs a persistent background worker daemon into the current Windows user's
Startup folder. Operates entirely in the user's profile (%APPDATA% / %LOCALAPPDATA%),
requiring ZERO Administrator privileges.

Usage:
    python setup_worker_startup.py                  # Install to Startup
    python setup_worker_startup.py --run-now        # Install and start immediately
    python setup_worker_startup.py --uninstall      # Remove from Startup
    python setup_worker_startup.py --status         # Check current installation status
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

# Default network configuration (can be overridden via CLI args)
DEFAULT_SHARED_DIR = r"J:\GPWRG\REF\mumbai\users\ncj\ai\SHARED"
DEFAULT_WORKING_DIR = r"J:\GPWRG\REF\mumbai\users\ncj\ai\LLM-IDE"


def get_startup_dir() -> Path:
    """Return current user's Windows Startup directory (no admin required)."""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise RuntimeError("Environment variable %APPDATA% is not set.")
    startup_dir = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    startup_dir.mkdir(parents=True, exist_ok=True)
    return startup_dir


def get_local_worker_dir() -> Path:
    """Return local user directory for persistent worker scripts."""
    localappdata = os.environ.get("LOCALAPPDATA")
    if not localappdata:
        localappdata = str(Path.home() / "AppData" / "Local")
    worker_dir = Path(localappdata) / "LLM_Cluster"
    worker_dir.mkdir(parents=True, exist_ok=True)
    return worker_dir


def generate_worker_script_content(shared_dir: str, working_dir: str, python_exe: str) -> str:
    """Generate python script executed on Windows boot/login."""
    escaped_shared = shared_dir.replace("\\", "\\\\")
    escaped_working = working_dir.replace("\\", "\\\\")
    escaped_python = python_exe.replace("\\", "\\\\")

    return f'''# Auto-generated LLM Cluster Worker Bootstrapper
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

SHARED_DIR = r"{escaped_shared}"
WORKING_DIR = r"{escaped_working}"
PYTHON_EXE = r"{escaped_python}"


def wait_for_network(timeout_s: int = 90) -> bool:
    """Wait for network drive / shared storage to become accessible after boot."""
    start = time.time()
    while time.time() - start < timeout_s:
        if os.path.exists(SHARED_DIR) and os.path.exists(WORKING_DIR):
            return True
        time.sleep(3)
    return False


def is_worker_already_running(worker_id: str) -> bool:
    """Check if worker process is already running on this machine."""
    try:
        import psutil
        my_pid = os.getpid()
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if proc.pid == my_pid:
                    continue
                cmdline = " ".join(proc.info.get("cmdline") or [])
                if "cluster.cli" in cmdline and "worker" in cmdline and worker_id in cmdline:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return False
    except Exception:
        return False


def main():
    worker_id = platform.node()

    # 1. Wait for mapped network drive (J:) or UNC path
    if not wait_for_network(timeout_s=90):
        # Write local diagnostic if network is unreachable
        err_file = Path(os.environ.get("TEMP", ".")) / f"worker_{{worker_id}}_net_error.log"
        err_file.write_text(f"Timed out waiting for network share: {{SHARED_DIR}}", encoding="utf-8")
        return

    # 2. Prevent duplicate workers
    if is_worker_already_running(worker_id):
        return

    # 3. Prepare logging
    log_dir = Path(SHARED_DIR) / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(log_dir / f"{{worker_id}}.log", "a", encoding="utf-8")
    except Exception:
        log_file = subprocess.DEVNULL

    # 4. Launch detached background worker daemon
    cmd = [
        PYTHON_EXE,
        "-m",
        "cluster.cli",
        "worker",
        "--shared-dir",
        SHARED_DIR,
        "--worker-id",
        worker_id,
    ]

    subprocess.Popen(
        cmd,
        cwd=WORKING_DIR,
        creationflags=(
            0x08000000  # CREATE_NO_WINDOW
            | 0x00000008  # DETACHED_PROCESS
            | 0x00000200  # CREATE_NEW_PROCESS_GROUP
        ),
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=log_file,
        close_fds=True,
    )


if __name__ == "__main__":
    main()
'''


def install_startup(shared_dir: str, working_dir: str, python_exe: Optional[str] = None) -> None:
    """Install the worker daemon launcher into Windows Startup folder."""
    py_exe = python_exe or sys.executable
    # Locate pythonw.exe if available (to prevent any console window)
    pyw_candidate = Path(py_exe).parent / "pythonw.exe"
    pyw_exe = str(pyw_candidate) if pyw_candidate.exists() else py_exe

    worker_dir = get_local_worker_dir()
    worker_py = worker_dir / "start_worker.py"
    startup_dir = get_startup_dir()
    vbs_launcher = startup_dir / "launch_llm_worker.vbs"

    # 1. Write the worker bootstrap script
    content = generate_worker_script_content(shared_dir, working_dir, py_exe)
    worker_py.write_text(content, encoding="utf-8")
    print(f"[OK] Generated worker runner: {worker_py}")

    # 2. Write the silent VBScript launcher into Startup
    # The 0 parameter hides the console window completely; False means asynchronous/non-blocking
    vbs_content = (
        'Set WshShell = CreateObject("WScript.Shell")\n'
        f'WshShell.Run """{pyw_exe}"" ""{worker_py}""", 0, False\n'
    )
    vbs_launcher.write_text(vbs_content, encoding="utf-8")
    print(f"[OK] Created Startup launcher: {vbs_launcher}")

    print("\n" + "=" * 70)
    print(f"SUCCESS: LLM Cluster Worker successfully installed to Windows Startup!")
    print(f"Target User      : {os.environ.get('USERNAME', 'Current User')}")
    print(f"Admin Required   : NO (Installed in user profile)")
    print(f"Node / Worker ID : {platform.node()}")
    print(f"Python Runtime   : {py_exe}")
    print(f"Shared Storage   : {shared_dir}")
    print(f"Working Dir      : {working_dir}")
    print("=" * 70)
    print("The worker will now launch automatically in the background on every login.")


def uninstall_startup() -> None:
    """Remove the worker launcher from Windows Startup folder."""
    startup_dir = get_startup_dir()
    vbs_launcher = startup_dir / "launch_llm_worker.vbs"
    removed = False
    if vbs_launcher.exists():
        vbs_launcher.unlink()
        print(f"[OK] Removed startup launcher: {vbs_launcher}")
        removed = True

    # Also clean legacy .bat or .py if present
    for legacy in ["launch_llm_worker.bat", "launch_llm_worker.cmd", "start_worker.py"]:
        p = startup_dir / legacy
        if p.exists():
            p.unlink()
            print(f"[OK] Removed legacy startup file: {p}")
            removed = True

    if removed:
        print("[SUCCESS] Cluster worker uninstalled from Windows Startup.")
    else:
        print("[INFO] No cluster worker startup launcher found.")


def check_status() -> None:
    """Display current startup installation and worker status."""
    startup_dir = get_startup_dir()
    vbs_launcher = startup_dir / "launch_llm_worker.vbs"
    worker_dir = get_local_worker_dir()
    worker_py = worker_dir / "start_worker.py"

    print("=" * 60)
    print("LLM Cluster Worker Startup Status")
    print("=" * 60)
    print(f"Node Hostname    : {platform.node()}")
    print(f"Startup Launcher : {'INSTALLED' if vbs_launcher.exists() else 'NOT INSTALLED'}")
    if vbs_launcher.exists():
        print(f"Launcher Path    : {vbs_launcher}")
    print(f"Worker Script    : {'EXISTS' if worker_py.exists() else 'NOT FOUND'}")
    if worker_py.exists():
        print(f"Script Path      : {worker_py}")

    # Check if currently running
    worker_id = platform.node()
    running_pids = []
    try:
        import psutil
        my_pid = os.getpid()
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if proc.pid == my_pid:
                    continue
                cmdline = " ".join(proc.info.get("cmdline") or [])
                if "cluster.cli" in cmdline and "worker" in cmdline and worker_id in cmdline:
                    running_pids.append((proc.pid, cmdline))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:
        pass

    if running_pids:
        print(f"Active Worker    : RUNNING (PID {', '.join(str(p[0]) for p in running_pids)})")
        for pid, cmd in running_pids:
            print(f"  -> PID {pid}: {cmd[:100]}...")
    else:
        print("Active Worker    : NOT RUNNING")
    print("=" * 60)


def run_now() -> None:
    """Execute the local worker script immediately."""
    worker_dir = get_local_worker_dir()
    worker_py = worker_dir / "start_worker.py"
    if not worker_py.exists():
        print(f"[ERROR] Worker script not found at {worker_py}. Run install first.")
        return
    pyw_candidate = Path(sys.executable).parent / "pythonw.exe"
    pyw_exe = str(pyw_candidate) if pyw_candidate.exists() else sys.executable
    subprocess.Popen([pyw_exe, str(worker_py)])
    print(f"[OK] Launched worker in background via {pyw_exe} {worker_py}")


def main():
    parser = argparse.ArgumentParser(description="Install LLM Cluster Worker into Windows Startup (No Admin Required)")
    parser.add_argument("--shared-dir", default=DEFAULT_SHARED_DIR, help=f"Shared storage path (default: {DEFAULT_SHARED_DIR})")
    parser.add_argument("--working-dir", default=DEFAULT_WORKING_DIR, help=f"LLM-IDE code directory (default: {DEFAULT_WORKING_DIR})")
    parser.add_argument("--python", default=None, help="Custom python.exe path (defaults to current environment python)")
    parser.add_argument("--uninstall", action="store_true", help="Remove worker launcher from Startup")
    parser.add_argument("--status", action="store_true", help="Check current installation and running status")
    parser.add_argument("--run-now", action="store_true", help="Trigger worker start immediately after installation")

    args = parser.parse_args()

    if args.uninstall:
        uninstall_startup()
    elif args.status:
        check_status()
    else:
        install_startup(args.shared_dir, args.working_dir, args.python)
        if args.run_now:
            run_now()


if __name__ == "__main__":
    main()
