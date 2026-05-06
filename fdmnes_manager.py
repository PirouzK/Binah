"""
fdmnes_manager.py — Optional FDMNES setup helpers for Binah.

FDMNES (https://fdmnes.neel.cnrs.fr/) is a finite-difference DFT XANES code
that handles bound-state pre-edge transitions properly — complementary to
FEFF, which is better for above-edge XANES + EXAFS but cannot reproduce the
discrete 1s→3d pre-edge for transition-metal complexes.

FDMNES is distributed as a single Windows binary (fdmnes_win64.exe) but the
download page is registration-walled, so we cannot auto-download it. The
"setup" flow here is therefore manual:

  1. User downloads fdmnes_win64.exe from neel.cnrs.fr
  2. Binah's "FDMNES Setup..." dialog: user clicks Browse, picks the .exe
  3. We copy (or just record) the path under ~/.binah_tools/fdmnes/
  4. A smoke test runs the binary to confirm it actually executes

Public API mirrors feff_manager.py so the calling code in
simulation_studio_tab.py can treat both engines uniformly.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable


FDMNES_DOWNLOAD_URL = "https://fdmnes.neel.cnrs.fr/"
PATH_CANDIDATES = ("fdmnes_win64.exe", "fdmnes.exe", "fdmnes")

# Default search path for the FDMNES Linux/parallel bundle that ships with
# the website's "fdmnes_mpi_linux64.zip" archive. Users typically extract it
# to C:\FDMNES\parallel_fdmnes — we look there by default.
PARALLEL_LAUNCHER_CANDIDATES = (
    r"C:\FDMNES\parallel_fdmnes\mpirun_fdmnes",
    r"C:\FDMNES\parallel\mpirun_fdmnes",
    str(Path.home() / "FDMNES" / "parallel_fdmnes" / "mpirun_fdmnes"),
)


def _default_install_dir() -> str:
    return str(Path.home() / ".binah_tools" / "fdmnes")


def _default_state() -> dict:
    return {
        "install_dir":      _default_install_dir(),
        "exe_path":         "",
        "version":          "",
        "last_verified":    "",
        "last_status":      "",
        "last_error":       "",
        "notes":            "",
        # Parallel (WSL) settings — independent of the serial Windows binary.
        "parallel_launcher":   "",  # path to mpirun_fdmnes (Linux bash script)
        "parallel_n_procs":    4,
        "parallel_last_status": "",
    }


def _read_config(cfg_path: str) -> dict:
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _write_config(cfg_path: str, cfg: dict) -> None:
    path = Path(cfg_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)


def load_fdmnes_setup_state(cfg_path: str) -> dict:
    cfg = _read_config(cfg_path)
    state = dict(_default_state())
    state.update(cfg.get("fdmnes_setup", {}))
    return state


def update_fdmnes_setup_state(cfg_path: str, updates: dict) -> dict:
    cfg = _read_config(cfg_path)
    state = dict(_default_state())
    state.update(cfg.get("fdmnes_setup", {}))
    state.update(updates)
    cfg["fdmnes_setup"] = state
    _write_config(cfg_path, cfg)
    return state


def discover_fdmnes_executable(*, preferred_path: str = "",
                               cfg_path: str = "") -> str:
    """Locate an FDMNES executable.

    Resolution order:
        1. Caller-supplied preferred_path (if it exists)
        2. Stored exe_path in ~/.binah_config.json
        3. Managed install dir (~/.binah_tools/fdmnes/)
        4. Standard PATH lookup for fdmnes_win64.exe / fdmnes.exe / fdmnes
        5. Returns "" if nothing found.
    """
    if preferred_path and os.path.isfile(preferred_path):
        return preferred_path

    state = load_fdmnes_setup_state(cfg_path) if cfg_path else _default_state()
    stored = str(state.get("exe_path", "")).strip()
    if stored and os.path.isfile(stored):
        return stored

    install_dir = Path(str(state.get("install_dir", _default_install_dir())))
    if install_dir.is_dir():
        for name in PATH_CANDIDATES:
            cand = install_dir / name
            if cand.is_file():
                return str(cand)

    for name in PATH_CANDIDATES:
        resolved = shutil.which(name)
        if resolved:
            return resolved

    return ""


def verify_fdmnes_executable(exe_path: str,
                             timeout: int = 30) -> tuple[bool, str]:
    """Run a quick smoke test on an FDMNES binary.

    FDMNES doesn't have a clean ``--version`` flag — it prints a banner with
    the version on launch, then waits for fdmfile.txt input. We invoke it
    with no input file in a temp directory and capture the banner from
    stdout/stderr.

    Returns (ok, message). On success ``message`` is the version string (or a
    sensible default); on failure it's an error description suitable to show
    the user.
    """
    if not exe_path or not os.path.isfile(exe_path):
        return False, f"File not found: {exe_path}"

    # Run in a throwaway temp dir so FDMNES doesn't pick up stray inputs.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="fdmnes_verify_") as tmp:
        try:
            proc = subprocess.run(
                [exe_path],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                input="",
            )
        except subprocess.TimeoutExpired:
            # FDMNES will hang waiting for input if it can't find fdmfile.txt;
            # a timeout here actually means the binary launched fine.
            return True, "FDMNES launched (timed out waiting for input — that's fine)."
        except OSError as exc:
            return False, f"Could not execute: {exc}"

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    out = out.strip()
    # Look for an "FDMNES" banner line.
    for line in out.splitlines():
        if "fdmnes" in line.lower():
            return True, line.strip()
    if out:
        return True, "FDMNES launched (no version banner detected)."
    return False, "Binary produced no output — likely not FDMNES."


def pick_and_install_fdmnes_executable(cfg_path: str, picked_path: str,
                                       log: Callable[[str], None]) -> dict:
    """Record the user-picked FDMNES executable, smoke-test it, persist state.

    We do NOT copy the binary into a managed location — FDMNES ships its own
    runtime data files alongside the .exe, so moving just the exe would break
    it. We record the picked path as-is and verify it runs.

    Returns a dict shaped like other Binah managers' install results:
        {ok: bool, exe_path: str, version: str, message: str}
    """
    log(f"Recording FDMNES executable: {picked_path}")
    if not picked_path or not os.path.isfile(picked_path):
        msg = f"File not found: {picked_path}"
        log(f"  {msg}")
        update_fdmnes_setup_state(cfg_path, {
            "last_status": "needs-attention",
            "last_error":  msg,
        })
        return {"ok": False, "exe_path": "", "version": "", "message": msg}

    log("Running smoke test ...")
    ok, message = verify_fdmnes_executable(picked_path)
    log(f"  {message}")

    state_updates = {
        "exe_path":      picked_path,
        "install_dir":   str(Path(picked_path).parent),
        "version":       message if ok else "",
        "last_verified": _dt.datetime.now().isoformat(timespec="seconds"),
        "last_status":   "ready" if ok else "needs-attention",
        "last_error":    "" if ok else message,
    }
    update_fdmnes_setup_state(cfg_path, state_updates)

    return {
        "ok":       ok,
        "exe_path": picked_path,
        "version":  message if ok else "",
        "message":  ("FDMNES is ready to use from Binah."
                     if ok else f"FDMNES verify failed: {message}"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parallel FDMNES (WSL) support
#
# The FDMNES download page ships a Linux ELF binary plus a bash launcher
# (`mpirun_fdmnes`) that sources a bundled Intel MPI runtime. To run that on
# Windows we shell out via `wsl bash <launcher> -np N` — provided the user
# has WSL installed.
#
# This entire layer is gated on the presence of `wsl.exe` on PATH. When WSL
# isn't installed, callers fall back to the serial Windows binary and the UI
# disables the "Parallel" toggle.
# ─────────────────────────────────────────────────────────────────────────────
import re as _re


def windows_to_wsl_path(p: str) -> str:
    """Translate a Windows path to its WSL equivalent.

    ``C:\\foo\\bar`` -> ``/mnt/c/foo/bar``. Already-POSIX paths are returned
    unchanged. UNC paths and non-drive letters get a best-effort translation.
    """
    if not p:
        return p
    p = str(p).replace("\\", "/")
    m = _re.match(r"^([A-Za-z]):/?(.*)$", p)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).lstrip("/")
        return f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}"
    return p


def discover_wsl_executable() -> str:
    """Return the absolute path to ``wsl.exe`` if present on PATH, else ""."""
    return shutil.which("wsl") or shutil.which("wsl.exe") or ""


def discover_parallel_fdmnes_launcher(*, cfg_path: str = "") -> str:
    """Locate the FDMNES Linux/parallel launcher (bash script).

    Search order:
        1. Stored ``parallel_launcher`` in ~/.binah_config.json
        2. Standard locations under C:\\FDMNES\\parallel_fdmnes\\ etc.
        3. Returns "" if nothing found.
    """
    state = load_fdmnes_setup_state(cfg_path) if cfg_path else _default_state()
    stored = str(state.get("parallel_launcher", "")).strip()
    if stored and os.path.isfile(stored):
        return stored
    for cand in PARALLEL_LAUNCHER_CANDIDATES:
        if os.path.isfile(cand):
            return cand
    return ""


def verify_parallel_fdmnes(launcher_path: str,
                           timeout: int = 60) -> tuple[bool, str]:
    """Run a smoke test on the parallel FDMNES launcher via WSL.

    We invoke ``wsl bash <launcher> -h`` and accept any output that:
      - prints the FDMNES / mpirun banner, OR
      - returns exit code 0 with non-empty output.

    Returns ``(ok, message)`` ready to display in the setup dialog log.
    """
    wsl = discover_wsl_executable()
    if not wsl:
        return False, ("wsl.exe not found on PATH. Install WSL first: "
                       "open an admin PowerShell and run `wsl --install`, "
                       "then reboot.")
    if not launcher_path or not os.path.isfile(launcher_path):
        return False, f"Launcher script not found: {launcher_path}"

    # Make sure the launcher and ELF binary are executable (clones from
    # zip via Windows lose POSIX exec bits unless metadata mounting is on).
    launcher_dir = Path(launcher_path).parent
    elf_binary = launcher_dir / "fdmnes_mpi_linux64"
    wsl_launcher = windows_to_wsl_path(str(launcher_path))
    wsl_elf = windows_to_wsl_path(str(elf_binary))

    bash_cmd = (
        f"chmod +x '{wsl_launcher}' 2>/dev/null; "
        f"chmod +x '{wsl_elf}' 2>/dev/null; "
        f"bash '{wsl_launcher}' -h 2>&1 | head -20"
    )
    try:
        proc = subprocess.run(
            [wsl, "bash", "-c", bash_cmd],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "Verification timed out (WSL slow to start?)."
    except OSError as exc:
        return False, f"Could not invoke wsl: {exc}"

    out = (proc.stdout or "").strip() + ("\n" + proc.stderr if proc.stderr else "")
    out = out.strip()
    lower = out.lower()
    if "fdmnes" in lower or "mpirun" in lower or "intel" in lower:
        first = next((ln for ln in out.splitlines() if ln.strip()),
                     "(no banner)")
        return True, first[:200]
    if proc.returncode == 0 and out:
        return True, out.splitlines()[0][:200]
    return False, (out or f"exit code {proc.returncode}")[:300]


def update_parallel_fdmnes_state(cfg_path: str, launcher_path: str,
                                 log: Callable[[str], None]) -> dict:
    """Record + verify the parallel FDMNES launcher path.

    Mirrors ``pick_and_install_fdmnes_executable`` but for the WSL launcher.
    """
    log(f"Recording parallel FDMNES launcher: {launcher_path}")
    ok, message = verify_parallel_fdmnes(launcher_path)
    log(f"  {message}")
    update_fdmnes_setup_state(cfg_path, {
        "parallel_launcher":     launcher_path,
        "parallel_last_status":  "ready" if ok else "needs-attention",
        "last_verified":         _dt.datetime.now().isoformat(timespec="seconds"),
    })
    return {"ok": ok, "launcher": launcher_path, "message": message}


def build_parallel_fdmnes_command(launcher_path: str, workdir: str,
                                  n_procs: int) -> list[str]:
    """Construct the argv list for ``subprocess.Popen``/``subprocess.run`` to
    launch parallel FDMNES via WSL.

    The bash payload `cd`s into the (translated) workdir, defensively chmods
    the launcher + ELF binary, then invokes ``bash <launcher> -np N``. This
    is robust to filesystem clones losing exec bits.
    """
    wsl_launcher = windows_to_wsl_path(launcher_path)
    wsl_workdir = windows_to_wsl_path(workdir)
    elf_binary_wsl = wsl_launcher.rsplit("/", 1)[0] + "/fdmnes_mpi_linux64"
    n = max(1, int(n_procs))
    bash_payload = (
        f"cd '{wsl_workdir}' && "
        f"chmod +x '{wsl_launcher}' '{elf_binary_wsl}' 2>/dev/null; "
        f"bash '{wsl_launcher}' -np {n}"
    )
    return ["wsl", "bash", "-c", bash_payload]
