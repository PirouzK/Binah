"""
Helpers for optional FEFF source setup inside Binah.

FEFF10 is distributed as source code. This module can:
  - remember whether the user wants startup prompts
  - download FEFF10 from GitHub (git clone/pull when available, zip fallback otherwise)
  - attempt a local build
  - expose the managed FEFF launcher path for the EXAFS tab
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable


FEFF_REPO_URL = "https://github.com/times-software/feff10.git"
FEFF_ZIP_URL = "https://github.com/times-software/feff10/archive/refs/heads/master.zip"
WINDOWS_SEQUENCE = (
    "rdinp",
    "dmdw",
    "atomic",
    "pot",
    "ldos",
    "screen",
    "crpa",
    "opconsat",
    "xsph",
    "fms",
    "mkgtr",
    "path",
    "genfmt",
    "ff2x",
    "sfconv",
    "compton",
    "eels",
    "rhorrp",
)
PATH_CANDIDATES = ("feff8l.exe", "feff.exe", "feff85l.exe", "feff9.exe", "feff")


def _default_install_dir() -> str:
    return str(Path.home() / ".binah_tools" / "feff10")


def _default_state() -> dict:
    return {
        "auto_prompt": True,
        "install_dir": _default_install_dir(),
        "repo_url": FEFF_REPO_URL,
        "exe_path": "",
        "source_method": "",
        "last_status": "",
        "last_error": "",
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


def load_setup_state(cfg_path: str) -> dict:
    cfg = _read_config(cfg_path)
    state = dict(_default_state())
    state.update(cfg.get("feff_setup", {}))
    return state


def update_setup_state(cfg_path: str, updates: dict) -> dict:
    cfg = _read_config(cfg_path)
    state = dict(_default_state())
    state.update(cfg.get("feff_setup", {}))
    state.update(updates)
    cfg["feff_setup"] = state
    _write_config(cfg_path, cfg)
    return state


def _managed_executable_candidates(install_dir: str) -> list[str]:
    root = Path(install_dir)
    return [
        str(root / "bin" / "feff.cmd"),
        str(root / "bin" / "feff.bat"),
        str(root / "bin" / "feff"),
    ]


def discover_feff_executable(*, preferred_path: str = "", cfg_path: str = "") -> str:
    if preferred_path and os.path.isfile(preferred_path):
        return preferred_path

    state = load_setup_state(cfg_path) if cfg_path else _default_state()
    stored = str(state.get("exe_path", "")).strip()
    if stored and os.path.isfile(stored):
        return stored

    for candidate in _managed_executable_candidates(str(state.get("install_dir", _default_install_dir()))):
        if os.path.isfile(candidate):
            return candidate

    for name in PATH_CANDIDATES:
        resolved = shutil.which(name)
        if resolved:
            return resolved

    return preferred_path


def should_offer_setup(cfg_path: str) -> bool:
    state = load_setup_state(cfg_path)
    if not bool(state.get("auto_prompt", True)):
        return False
    return not bool(discover_feff_executable(cfg_path=cfg_path))


def _log_output(log: Callable[[str], None], header: str, text: str, limit: int = 120) -> None:
    body = (text or "").strip()
    if not body:
        return
    lines = body.splitlines()
    log(header)
    for line in lines[:limit]:
        log(f"  {line}")
    if len(lines) > limit:
        log(f"  ... {len(lines) - limit} more line(s)")


def _run_subprocess(args: list[str], cwd: str, log: Callable[[str], None],
                    timeout: int = 3600) -> subprocess.CompletedProcess:
    log(f"$ {' '.join(args)}")
    proc = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    _log_output(log, "stdout:", proc.stdout)
    _log_output(log, "stderr:", proc.stderr)
    log(f"Return code: {proc.returncode}")
    return proc


def _download_from_git_or_zip(repo_dir: Path, log: Callable[[str], None]) -> str:
    git = shutil.which("git")
    if repo_dir.exists() and (repo_dir / ".git").exists() and git:
        log("Updating FEFF10 source with git pull --ff-only ...")
        proc = _run_subprocess([git, "-C", str(repo_dir), "pull", "--ff-only"], str(repo_dir), log)
        if proc.returncode != 0:
            raise RuntimeError("Git update failed.")
        return "git"

    if repo_dir.exists():
        log("Using existing FEFF10 source tree.")
        return "existing"

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if git:
        log("Cloning FEFF10 from GitHub ...")
        proc = _run_subprocess(
            [git, "clone", "--depth", "1", FEFF_REPO_URL, str(repo_dir)],
            str(repo_dir.parent),
            log,
        )
        if proc.returncode == 0:
            return "git"
        log("Git clone failed, falling back to GitHub zip download.")

    log("Downloading FEFF10 source snapshot from GitHub ...")
    temp_dir = Path(tempfile.mkdtemp(prefix="binah-feff-"))
    try:
        zip_path = temp_dir / "feff10.zip"
        urllib.request.urlretrieve(FEFF_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_dir)
        extracted = next((p for p in temp_dir.iterdir() if p.is_dir() and p.name.startswith("feff10-")), None)
        if extracted is None:
            raise RuntimeError("GitHub archive did not contain an extracted FEFF10 folder.")
        shutil.move(str(extracted), str(repo_dir))
        return "zip"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _write_posix_compiler_file(src_dir: Path, log: Callable[[str], None]) -> None:
    if shutil.which("ifort"):
        default_path = src_dir / "Compiler.mk.default"
        shutil.copyfile(default_path, src_dir / "Compiler.mk")
        log("Configured FEFF10 with the default ifort compiler settings.")
        return

    if shutil.which("gfortran"):
        (src_dir / "Compiler.mk").write_text(
            "F90 = gfortran\n"
            "FLAGS = -O3 -ffree-line-length-none -finit-local-zero -g\n"
            "MPIF90 = mpif90\n"
            "MPIFLAGS = -g -O3\n"
            "LDFLAGS = \n"
            "FCINCLUDE = \n"
            "DEPTYPE = \n",
            encoding="utf-8",
        )
        log("Configured FEFF10 for a best-effort gfortran build.")
        return

    raise RuntimeError(
        "No supported Fortran compiler was found. Install Intel ifort or gfortran first."
    )


def _build_posix(repo_dir: Path, log: Callable[[str], None]) -> dict:
    src_dir = repo_dir / "src"
    if not src_dir.is_dir():
        raise RuntimeError("FEFF10 source tree is missing the src directory.")
    if not shutil.which("make"):
        raise RuntimeError("make was not found on PATH.")
    if not shutil.which("bash"):
        raise RuntimeError("bash was not found on PATH.")

    _write_posix_compiler_file(src_dir, log)
    log("Building FEFF10 ...")
    proc = _run_subprocess(["make"], str(src_dir), log)
    exe_path = repo_dir / "bin" / "feff"
    if proc.returncode != 0 or not exe_path.is_file():
        raise RuntimeError("FEFF10 build did not produce bin/feff.")
    exe_path.chmod(exe_path.stat().st_mode | 0o111)
    return {"ok": True, "built": True, "exe_path": str(exe_path)}


def _write_windows_wrapper(repo_dir: Path, log: Callable[[str], None]) -> str:
    bin_dir = repo_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    wrapper = bin_dir / "feff.cmd"
    lines = [
        "@echo off",
        "setlocal",
        'set "FeffPath=%~dp0..\\mod\\win64"',
    ]
    for name in WINDOWS_SEQUENCE:
        exe_path = repo_dir / "mod" / "win64" / f"{name}.exe"
        if exe_path.exists():
            lines.append(f'call "%FeffPath%\\{name}.exe"')
            lines.append("if errorlevel 1 exit /b %errorlevel%")
    lines.append("endlocal")
    wrapper.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    log(f"Created Windows FEFF wrapper: {wrapper}")
    return str(wrapper)


def _build_windows(repo_dir: Path, log: Callable[[str], None]) -> dict:
    compile_dir = repo_dir / "mod" / "Seq"
    script = compile_dir / "Compile_win64.BAT"
    if not compile_dir.is_dir() or not script.is_file():
        raise RuntimeError("FEFF10 Windows compile scripts were not found.")
    if not (shutil.which("ifort.exe") or shutil.which("ifort")):
        raise RuntimeError(
            "Intel Fortran (ifort) was not found on PATH, so FEFF10 could not be built."
        )

    log("Building FEFF10 with Compile_win64.BAT ...")
    proc = _run_subprocess(["cmd", "/c", "Compile_win64.BAT"], str(compile_dir), log)
    required = [
        repo_dir / "mod" / "win64" / "rdinp.exe",
        repo_dir / "mod" / "win64" / "pot.exe",
        repo_dir / "mod" / "win64" / "ff2x.exe",
    ]
    if proc.returncode != 0 or not all(path.exists() for path in required):
        raise RuntimeError("FEFF10 Windows build did not produce the expected executables.")
    wrapper = _write_windows_wrapper(repo_dir, log)
    return {"ok": True, "built": True, "exe_path": wrapper}


def install_or_update_managed_feff(cfg_path: str, log: Callable[[str], None]) -> dict:
    state = load_setup_state(cfg_path)
    repo_dir = Path(str(state.get("install_dir", _default_install_dir())))
    source_method = ""
    try:
        source_method = _download_from_git_or_zip(repo_dir, log)
        if os.name == "nt":
            build = _build_windows(repo_dir, log)
        else:
            build = _build_posix(repo_dir, log)
        result = {
            "ok": True,
            "built": bool(build.get("built", False)),
            "repo_dir": str(repo_dir),
            "exe_path": str(build.get("exe_path", "")),
            "source_method": source_method,
            "message": "FEFF10 is ready to use from Binah.",
        }
        update_setup_state(
            cfg_path,
            {
                "install_dir": str(repo_dir),
                "repo_url": FEFF_REPO_URL,
                "exe_path": result["exe_path"],
                "source_method": source_method,
                "last_status": "ready",
                "last_error": "",
            },
        )
        return result
    except Exception as exc:
        result = {
            "ok": False,
            "built": False,
            "repo_dir": str(repo_dir),
            "exe_path": "",
            "source_method": source_method,
            "message": str(exc),
        }
        update_setup_state(
            cfg_path,
            {
                "install_dir": str(repo_dir),
                "repo_url": FEFF_REPO_URL,
                "source_method": source_method,
                "last_status": "needs-attention",
                "last_error": str(exc),
            },
        )
        return result
