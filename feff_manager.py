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

_JFEFF_BIN_CANDIDATES = (
    r"C:\Program Files (x86)\JFEFF_FEFF10\feff10\bin",
    r"C:\Program Files\JFEFF_FEFF10\feff10\bin",
)


def _find_jfeff_bin_dir() -> str | None:
    for candidate in _JFEFF_BIN_CANDIDATES:
        p = Path(candidate)
        if p.is_dir() and (p / "rdinp.exe").exists():
            return str(p)
    return None


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

    # JFEFF takes priority over our compiled-from-source copies in mod/win64 because
    # it ships pre-built, verified executables. Regenerate wrapper if it doesn't yet
    # reference the JFEFF bin directory.
    repo_dir = Path(str(state.get("install_dir", _default_install_dir())))
    jfeff_bin = _find_jfeff_bin_dir()
    if jfeff_bin:
        wrapper = repo_dir / "bin" / "feff.cmd"
        needs_update = not wrapper.exists() or (
            jfeff_bin.lower() not in wrapper.read_text(encoding="utf-8",
                                                       errors="replace").lower()
        )
        if needs_update:
            try:
                return _write_windows_wrapper(repo_dir, lambda _m: None,
                                              feff_bin_dir=jfeff_bin)
            except Exception:
                pass
        if wrapper.exists():
            return str(wrapper)

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


_ONEAPI_SETVARS_CANDIDATES = (
    r"C:\Program Files (x86)\Intel\oneAPI\setvars.bat",
    r"C:\Program Files\Intel\oneAPI\setvars.bat",
)
_ONEAPI_ROOTS = (
    r"C:\Program Files (x86)\Intel\oneAPI",
    r"C:\Program Files\Intel\oneAPI",
)


def _find_oneapi_setvars() -> str:
    for path in _ONEAPI_SETVARS_CANDIDATES:
        if os.path.isfile(path):
            return path
    return ""


def _find_ifx_directly() -> str:
    """Scan common Intel oneAPI install trees for ifx.exe without setvars.bat."""
    import glob as _glob
    for root in _ONEAPI_ROOTS:
        for pattern in (
            os.path.join(root, "compiler", "*", "bin", "ifx.exe"),
            os.path.join(root, "*", "bin", "ifx.exe"),
            os.path.join(root, "compiler", "*", "windows", "bin", "ifx.exe"),
        ):
            matches = sorted(_glob.glob(pattern))
            if matches:
                return matches[-1]  # take highest version
    return ""


def _find_ifx_lib_dirs() -> list[str]:
    """Return Intel Fortran runtime lib directories (for LIB variable)."""
    import glob as _glob
    dirs: list[str] = []
    for root in _ONEAPI_ROOTS:
        for pattern in (
            os.path.join(root, "compiler", "*", "lib"),
            os.path.join(root, "*", "lib"),
        ):
            for d in sorted(_glob.glob(pattern)):
                if os.path.isfile(os.path.join(d, "ifconsol.lib")):
                    dirs.append(d)
    return dirs


def _find_msvc_link_dir() -> str:
    """Return the MSVC HostX64/x64 bin directory containing the real link.exe."""
    import glob as _glob
    for vs_root in (
        r"C:\Program Files\Microsoft Visual Studio",
        r"C:\Program Files (x86)\Microsoft Visual Studio",
    ):
        pattern = os.path.join(vs_root, "*", "*", "VC", "Tools", "MSVC", "*",
                               "bin", "Hostx64", "x64")
        matches = sorted(_glob.glob(pattern))
        for m in reversed(matches):  # highest version first
            if os.path.isfile(os.path.join(m, "link.exe")):
                return m
    return ""


def _find_lld_link() -> str:
    """Return the path to lld-link.exe bundled with Intel oneAPI (no VS needed)."""
    import glob as _glob
    for root in _ONEAPI_ROOTS:
        for pattern in (
            os.path.join(root, "compiler", "*", "bin", "compiler", "lld-link.exe"),
            os.path.join(root, "*", "bin", "compiler", "lld-link.exe"),
        ):
            matches = sorted(_glob.glob(pattern))
            if matches:
                return matches[-1]
    return ""


def _find_vcvars64() -> str:
    """Return path to vcvars64.bat from the highest installed VS edition."""
    import glob as _glob
    editions = ("Enterprise", "Professional", "Community", "BuildTools")
    for vs_root in (
        r"C:\Program Files\Microsoft Visual Studio",
        r"C:\Program Files (x86)\Microsoft Visual Studio",
    ):
        for pattern in (
            os.path.join(vs_root, "*", "*", "VC", "Auxiliary", "Build", "vcvars64.bat"),
        ):
            matches = sorted(_glob.glob(pattern))
            # prefer Enterprise > Professional > Community > BuildTools
            def _prio(p: str) -> int:
                for i, ed in enumerate(editions):
                    if ed.lower() in p.lower():
                        return i
                return len(editions)
            matches.sort(key=_prio)
            if matches:
                return matches[0]
    return ""


def _apply_vcvars_env(env: dict, log) -> None:
    """Run vcvars64.bat and merge its environment into env."""
    vcvars = _find_vcvars64()
    if not vcvars:
        log("Warning: vcvars64.bat not found; LIB/INCLUDE may be incomplete.")
        return
    log(f"Configuring MSVC environment from: {vcvars}")
    # Use ""path\vcvars64.bat" && set" quoting — required when path has spaces.
    cmd = f'cmd /c ""{vcvars}" && set"'
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=120
        )
        for line in result.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    except Exception as exc:
        log(f"Warning: vcvars64.bat failed ({exc})")


def _oneapi_env(log: Callable[[str], None]) -> dict | None:
    """Return os.environ updated with Intel oneAPI compiler bin on PATH.

    Tries setvars.bat first, then falls back to direct ifx.exe scanning.
    Always ensures MSVC link.exe precedes Git's POSIX link utility on PATH.
    """
    env = dict(os.environ)

    setvars = _find_oneapi_setvars()
    if setvars:
        log(f"Initialising Intel oneAPI environment from: {setvars}")
        setvars_dir = os.path.dirname(setvars)
        # setvars.bat looks for VS in standard locations. BuildTools edition is
        # non-standard; hint via VS2022INSTALLDIR so setvars.bat finds link.exe.
        vs_hint = _find_vcvars64()
        vs_install_dir = ""
        if vs_hint:
            # vcvars64.bat lives at <vs_root>\VC\Auxiliary\Build\vcvars64.bat
            vs_install_dir = str(Path(vs_hint).parents[3])
        hint_prefix = f'set "VS2022INSTALLDIR={vs_install_dir}" && ' if vs_install_dir else ""
        cmd = f'cmd /c "{hint_prefix}cd /d "{setvars_dir}" && call setvars.bat --force && set"'
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=120
            )
            for line in result.stdout.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
        except Exception as exc:
            log(f"Warning: setvars.bat failed ({exc}); trying direct ifx search ...")

    # If ifx is not reachable after setvars.bat, scan for it directly.
    if not shutil.which("ifx", path=env.get("PATH", "")):
        ifx_path = _find_ifx_directly()
        if ifx_path:
            ifx_dir = os.path.dirname(ifx_path)
            log(f"Adding ifx directory directly to PATH: {ifx_dir}")
            env["PATH"] = ifx_dir + os.pathsep + env.get("PATH", "")
        else:
            return None

    # If LIB is not set (setvars.bat component scripts failed), bootstrap the
    # full MSVC environment via vcvars64.bat.  This sets LIB, INCLUDE, PATH,
    # and all other vars needed by link.exe to find libcmt.lib and the SDK.
    if not env.get("LIB"):
        _apply_vcvars_env(env, log)
        # Re-add ifx dir after vcvars overwrites PATH
        if not shutil.which("ifx", path=env.get("PATH", "")):
            ifx_path = _find_ifx_directly()
            if ifx_path:
                env["PATH"] = os.path.dirname(ifx_path) + os.pathsep + env.get("PATH", "")

    # Always add Intel Fortran runtime lib dirs to LIB so the linker can find
    # ifconsol.lib and other ifx-specific libraries (not in MSVC LIB paths).
    ifx_lib_dirs = _find_ifx_lib_dirs()
    if ifx_lib_dirs:
        existing_lib = env.get("LIB", "")
        extra = os.pathsep.join(
            d for d in ifx_lib_dirs if d not in existing_lib
        )
        if extra:
            env["LIB"] = extra + (os.pathsep if existing_lib else "") + existing_lib
            log(f"Added Intel Fortran lib dirs to LIB")

    # Ensure a real PE linker precedes Git's POSIX link utility.  When Git for
    # Windows is installed, its /usr/bin/link.EXE (a symlink tool) appears on
    # PATH before the MSVC linker, causing ifx to fail at the link step.
    # Prefer MSVC link.exe; fall back to the lld-link.exe bundled with oneAPI
    # (present even without a Visual Studio installation).
    msvc_link_dir = _find_msvc_link_dir()
    found_link = shutil.which("link", path=env.get("PATH", ""))
    link_is_unix = not found_link or "git" in found_link.lower() or "mingw" in found_link.lower()
    if msvc_link_dir and link_is_unix:
        log(f"Prepending MSVC linker directory to PATH: {msvc_link_dir}")
        env["PATH"] = msvc_link_dir + os.pathsep + env.get("PATH", "")
    elif link_is_unix:
        # No VS — use the lld-link.exe that ships with Intel oneAPI 2024+.
        lld_link = _find_lld_link()
        if lld_link:
            # ifx looks for a binary named "link.exe" on PATH; copy lld-link.exe
            # into a private temp dir under that name so the linker step succeeds.
            lld_shim_dir = os.path.join(
                tempfile.gettempdir(), "binah_lld_shim"
            )
            os.makedirs(lld_shim_dir, exist_ok=True)
            shim_link = os.path.join(lld_shim_dir, "link.exe")
            if not os.path.isfile(shim_link):
                shutil.copy2(lld_link, shim_link)
            log(f"No MSVC installation found — using Intel lld-link.exe as link.exe shim")
            env["PATH"] = lld_shim_dir + os.pathsep + env.get("PATH", "")
        else:
            log("Warning: no suitable link.exe found; ifx link step may fail.")

    return env


def _run_subprocess(args: list[str], cwd: str, log: Callable[[str], None],
                    timeout: int = 3600,
                    env: dict | None = None) -> subprocess.CompletedProcess:
    log(f"$ {' '.join(args)}")
    proc = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )
    _log_output(log, "stdout:", proc.stdout)
    _log_output(log, "stderr:", proc.stderr)
    log(f"Return code: {proc.returncode}")
    return proc


def _looks_like_feff_source_tree(repo_dir: Path) -> bool:
    return (
        (repo_dir / "src" / "install.txt").is_file()
        or (repo_dir / "mod" / "Seq" / "Compile_win64.BAT").is_file()
    )


def _remove_existing_snapshot(repo_dir: Path, log: Callable[[str], None]) -> None:
    resolved = repo_dir.resolve()
    if not resolved.exists():
        return
    if not _looks_like_feff_source_tree(resolved):
        raise RuntimeError(
            f"Install directory exists but is not a recognizable FEFF10 source tree: {resolved}"
        )
    log(f"Refreshing existing FEFF10 snapshot at {resolved} ...")
    shutil.rmtree(resolved)


def _download_from_git_or_zip(repo_dir: Path, log: Callable[[str], None]) -> str:
    git = shutil.which("git")
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists() and (repo_dir / ".git").exists() and git:
        log("Updating FEFF10 source with git pull --ff-only ...")
        proc = _run_subprocess([git, "-C", str(repo_dir), "pull", "--ff-only"], str(repo_dir), log)
        if proc.returncode != 0:
            raise RuntimeError("Git update failed.")
        return "git"

    if repo_dir.exists():
        _remove_existing_snapshot(repo_dir, log)

    if git:
        log("Cloning FEFF10 from GitHub ...")
        proc = _run_subprocess(
            [git, "clone", "--depth", "1", FEFF_REPO_URL, str(repo_dir)],
            str(repo_dir.parent),
            log,
        )
        if proc.returncode == 0:
            return "git"
        if repo_dir.exists():
            log("Cleaning up partial git clone before zip fallback ...")
            shutil.rmtree(repo_dir, ignore_errors=True)
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
        if repo_dir.exists():
            raise RuntimeError(f"Install directory already exists and cannot be replaced: {repo_dir}")
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


def _totalize_feff10_windows_source(
    src_dir: Path, out_dir: Path, log: Callable[[str], None]
) -> None:
    """
    Python replacement for make_modular_sourcefiles.pl.

    Reads each DEP/<module>.mk file, concatenates the listed Fortran source
    files (MODULESRC first, then SRC) into out_dir/<module>_tot.f90, and
    inlines one level of INCLUDE statements.  Handles CRLF safely.
    """
    import re

    _MODULES = [
        "atomic", "compton", "dmdw", "dym2feffinp", "eels",
        "ff2x", "fms", "genfmt", "ldos", "mkgtr", "opconsat",
        "path", "pot", "rdinp", "rhorrp", "screen", "sfconv", "xsph",
    ]

    def _read(path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")

    def _extract_files(mk_text: str, var: str) -> list[Path]:
        # Capture make variable with backslash-continuation lines.
        # Pattern: VAR = <first line ending in \><LF> ... <last line without \>
        m = re.search(rf"^{re.escape(var)}\s*=\s*((?:.*\\\n)*.*)", mk_text, re.M)
        if not m:
            return []
        block = re.sub(r"\\\n", " ", m.group(1))
        result = []
        for tok in re.findall(r"\S+\.f90", block):
            rel = tok.lstrip("./").replace("/", os.sep)
            p = src_dir / rel
            if p.exists():
                result.append(p)
            else:
                log(f"  Warning: {tok} listed in DEP but not found — skipping")
        return result

    def _inline_includes(src_path: Path, src_text: str) -> str:
        out_lines = []
        for line in src_text.splitlines():
            # Allow optional Fortran inline comment (!...) after the filename.
            m = re.match(r"^\s*[Ii][Nn][Cc][Ll][Uu][Dd][Ee]\s+['\"]?([^'\" \t!]+)['\"]?\s*(?:!.*)?$", line)
            if m:
                inc_name = m.group(1)
                for candidate in (src_path.parent / inc_name, src_dir / inc_name):
                    if candidate.exists():
                        try:
                            out_lines.append(_read(candidate))
                        except Exception:
                            out_lines.append(line)
                        break
                else:
                    out_lines.append(line)
            else:
                out_lines.append(line)
        return "\n".join(out_lines) + "\n"

    # Sequential build requires PAR/parallel.f90, which is not committed to the
    # repo — only PAR/sequential.src and PAR/parallel.src exist.  Copy the
    # sequential variant so that every module that references ./PAR/parallel.f90
    # can be included (otherwise those source lists resolve to empty and the
    # resulting *_tot.f90 files contain no Fortran, causing empty-stream errors).
    par_f90 = src_dir / "PAR" / "parallel.f90"
    seq_src = src_dir / "PAR" / "sequential.src"
    if not par_f90.exists() and seq_src.exists():
        log("Copying PAR/sequential.src -> PAR/parallel.f90 for sequential build ...")
        shutil.copy2(seq_src, par_f90)
    elif not par_f90.exists():
        log("Warning: PAR/parallel.f90 not found and sequential.src missing — modules may be incomplete")

    dep_dir = src_dir / "DEP"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Python totalization: writing *_tot.f90 into {out_dir} ...")

    # Copy non-f90 include files (*.cmn, *.cmv, *.cmf, *.h, *.sys) into
    # out_dir so that ifx.exe can find them during compilation (e.g. ciftbx.cmn).
    _INC_EXTS = {".cmn", ".cmv", ".cmf", ".h", ".sys"}
    _copied = 0
    for inc_file in src_dir.rglob("*"):
        if inc_file.suffix.lower() in _INC_EXTS and inc_file.is_file():
            dest = out_dir / inc_file.name
            if not dest.exists():
                shutil.copy2(inc_file, dest)
                _copied += 1
    if _copied:
        log(f"  Copied {_copied} include file(s) to compile directory")

    for mod in _MODULES:
        dep_path = dep_dir / f"{mod}.mk"
        if not dep_path.exists():
            log(f"  No DEP/{mod}.mk — skipping {mod}")
            continue
        mk_text = _read(dep_path)
        mod_src   = _extract_files(mk_text, f"{mod}_MODULESRC")
        prog_src  = _extract_files(mk_text, f"{mod}SRC")
        tot_path  = out_dir / f"{mod}_tot.f90"
        with open(tot_path, "w", encoding="utf-8", newline="\n") as fout:
            for fpath in mod_src + prog_src:
                try:
                    fout.write(_inline_includes(fpath, _read(fpath)))
                except Exception as exc:
                    log(f"  Warning: could not read {fpath.name}: {exc}")
        log(f"  {tot_path.name}  ({len(mod_src)} modules, {len(prog_src)} sources)")


def _write_windows_wrapper(repo_dir: Path, log: Callable[[str], None],
                           feff_bin_dir: str | None = None) -> str:
    bin_dir = repo_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    wrapper = bin_dir / "feff.cmd"

    # Build the module call list with progress echo markers.
    present = [n for n in WINDOWS_SEQUENCE
               if (Path(feff_bin_dir) / f"{n}.exe" if feff_bin_dir
                   else repo_dir / "mod" / "win64" / f"{n}.exe").exists()]
    total = len(present)
    module_calls: list[str] = []
    for i, name in enumerate(present, start=1):
        module_calls.append(f'echo [FEFF] {name} ({i}/{total})')
        module_calls.append(f'call "%FeffPath%\\{name}.exe"')
        module_calls.append("if errorlevel 1 goto :done")

    # Run FEFF in %TEMP% to avoid OneDrive / cloud-sync file-locking.
    # Input files (*.inp, *.cif, *.xyz) are copied in; all outputs are
    # copied back to the original working directory afterwards.
    feff_path_value = feff_bin_dir if feff_bin_dir else r"%~dp0..\mod\win64"
    lines = [
        "@echo off",
        "setlocal",
        f'set "FeffPath={feff_path_value}"',
        'set "OrigDir=%CD%"',
        "",
        ":: Use a private temp subdirectory to avoid cloud-sync file locking",
        'set "TmpDir=%TEMP%\\feff_run_%RANDOM%"',
        'mkdir "%TmpDir%"',
        "",
        ":: Copy only user-supplied input files (not FEFF-generated *.inp outputs)",
        'if exist "%OrigDir%\\feff.inp" copy /Y "%OrigDir%\\feff.inp" "%TmpDir%\\" >nul',
        'for %%F in ("%OrigDir%\\*.cif" "%OrigDir%\\*.xyz") do (',
        '    if exist "%%~F" copy /Y "%%~F" "%TmpDir%\\" >nul',
        ")",
        "",
        'pushd "%TmpDir%"',
        "",
    ] + module_calls + [
        "",
        ":done",
        'set "_ec=%errorlevel%"',
        "popd",
        "",
        ":: Copy all outputs back to original directory",
        'xcopy /Y /Q "%TmpDir%\\*" "%OrigDir%\\" >nul 2>&1',
        'rmdir /S /Q "%TmpDir%"',
        "",
        "endlocal",
        "exit /b %_ec%",
    ]
    wrapper.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    log(f"Created Windows FEFF wrapper: {wrapper}")
    return str(wrapper)


def _build_windows(repo_dir: Path, log: Callable[[str], None]) -> dict:
    # Prefer an existing JFEFF installation — no compilation needed.
    jfeff_bin = _find_jfeff_bin_dir()
    if jfeff_bin:
        log(f"JFEFF installation found at: {jfeff_bin} — using pre-built executables.")
        wrapper = _write_windows_wrapper(repo_dir, log, feff_bin_dir=jfeff_bin)
        return {"ok": True, "built": False, "exe_path": wrapper}

    win64_dir = repo_dir / "mod" / "win64"
    _required_names = ["rdinp.exe", "pot.exe", "ff2x.exe"]
    prebuilt = win64_dir.is_dir() and all(
        (win64_dir / name).exists() for name in _required_names
    )
    if prebuilt:
        log("Pre-compiled FEFF10 executables found in mod/win64 — skipping compilation.")
        wrapper = _write_windows_wrapper(repo_dir, log)
        return {"ok": True, "built": False, "exe_path": wrapper}

    compile_dir = repo_dir / "mod" / "Seq"
    script = compile_dir / "Compile_win64.BAT"
    if not compile_dir.is_dir() or not script.is_file():
        raise RuntimeError("FEFF10 Windows compile scripts were not found.")

    # Generate *_tot.f90 files if missing or empty (not committed to the FEFF10 repo).
    src_dir = repo_dir / "src"
    tot_marker = compile_dir / "rdinp_tot.f90"
    if not tot_marker.exists() or tot_marker.stat().st_size == 0:
        _totalize_feff10_windows_source(src_dir, compile_dir, log)
    if not tot_marker.exists() or tot_marker.stat().st_size == 0:
        raise RuntimeError(
            "Totalization did not create a valid rdinp_tot.f90. "
            "Check the log above for errors."
        )
    # Always try to load the oneAPI environment — it may be needed even if ifx is
    # on PATH, because child cmd.exe processes reset environment variables.
    build_env = _oneapi_env(log)
    _fc = next(
        (c for c in ("ifort", "ifort.exe", "ifx", "ifx.exe")
         if shutil.which(c, path=(build_env or {}).get("PATH") or os.environ.get("PATH", ""))),
        None,
    )
    if not _fc:
        # oneAPI not installed — fall back to whatever is already on PATH
        build_env = None
        _fc = next(
            (c for c in ("ifort", "ifort.exe", "ifx", "ifx.exe") if shutil.which(c)),
            None,
        )
    if not _fc:
        raise RuntimeError(
            "Pre-compiled FEFF10 executables were not found in mod/win64, and no Intel "
            "Fortran compiler (ifort / ifx) is on PATH — source build is not possible.\n\n"
            "Options:\n"
            "  1. Install Intel oneAPI HPC Toolkit (free) from intel.com/oneapi,\n"
            "     then re-run FEFF Setup — Binah will find it automatically.\n"
            "  2. conda install -c conda-forge fortran-compiler\n"
            "     (adds gfortran; FEFF10 must then be compiled manually with make)."
        )

    # Compile_win64.BAT hardcodes "ifort.exe". If we only have ifx, patch it first.
    fc_exe = (_fc if _fc.lower().endswith(".exe") else _fc + ".exe").lower()
    bat_path = script  # default: use original
    patched_bat: Path | None = None
    if "ifx" in fc_exe and "ifort" not in fc_exe:
        try:
            original = script.read_text(encoding="utf-8", errors="replace")
            patched_text = (
                original
                .replace("ifort.exe", "ifx.exe")
                .replace("ifort.EXE", "ifx.exe")
                .replace('"ifort"', '"ifx"')
            )
            if patched_text != original:
                patched_bat = compile_dir / "_compile_ifx.BAT"
                patched_bat.write_text(patched_text, encoding="utf-8")
                bat_path = patched_bat
                log("Patched Compile_win64.BAT: replaced ifort.exe -> ifx.exe")
        except Exception as exc:
            log(f"Warning: could not patch BAT file ({exc}); using original.")

    win64_dir.mkdir(parents=True, exist_ok=True)
    log(f"Building FEFF10 with {bat_path.name} (compiler: {_fc}) ...")
    try:
        # Use full path to the BAT file so cmd /c finds it regardless of PATH.
        proc = _run_subprocess(
            ["cmd", "/c", str(bat_path)], str(compile_dir), log, env=build_env
        )
    finally:
        if patched_bat and patched_bat.exists():
            try:
                patched_bat.unlink()
            except Exception:
                pass

    # ifx on Windows may produce extension-less binaries when -o specifies no
    # extension.  Rename any extension-less files to .exe so the wrapper works.
    for f in win64_dir.iterdir():
        if f.is_file() and f.suffix == "" and not f.name.startswith("."):
            renamed = f.with_suffix(".exe")
            if not renamed.exists():
                f.rename(renamed)
                log(f"  Renamed {f.name} -> {renamed.name}")

    required_paths = [win64_dir / name for name in _required_names]
    if proc.returncode != 0 or not all(p.exists() for p in required_paths):
        raise RuntimeError("FEFF10 Windows build did not produce the expected executables.")
    # Raise PE stack reserve so pot/screen/fms don't blow the default 1 MB
    # on realistic clusters.
    _bump_stack_size(win64_dir, log)
    wrapper = _write_windows_wrapper(repo_dir, log)
    return {"ok": True, "built": True, "exe_path": wrapper}


def _find_mpiifx() -> str:
    """Locate Intel oneAPI's mpiifx.bat (MPI Fortran compiler driver)."""
    import glob as _glob
    for root in _ONEAPI_ROOTS:
        for pattern in (
            os.path.join(root, "mpi", "*", "bin", "mpiifx.bat"),
        ):
            matches = sorted(_glob.glob(pattern))
            if matches:
                return matches[-1]
    return ""


def _find_mpiexec() -> str:
    """Locate Intel oneAPI's mpiexec.exe."""
    import glob as _glob
    for root in _ONEAPI_ROOTS:
        for pattern in (
            os.path.join(root, "mpi", "*", "bin", "mpiexec.exe"),
        ):
            matches = sorted(_glob.glob(pattern))
            if matches:
                return matches[-1]
    return ""


def _switch_par_source(src_dir: Path, mode: str, log: Callable[[str], None]) -> None:
    """Overwrite src/PAR/parallel.f90 with the MPI or sequential variant.

    FEFF10 modules INCLUDE ./PAR/parallel.f90; toggling this file controls
    whether the resulting *_tot.f90 are MPI-parallel or sequential dummies.
    """
    par_f90 = src_dir / "PAR" / "parallel.f90"
    src_file = src_dir / "PAR" / ("parallel.src" if mode == "mpi" else "sequential.src")
    if not src_file.exists():
        raise RuntimeError(f"PAR/{src_file.name} not found in FEFF10 source.")
    shutil.copy2(src_file, par_f90)
    log(f"Configured PAR/parallel.f90 from {src_file.name} ({mode} mode)")


def _write_parallel_wrapper(repo_dir: Path, win64_par_dir: Path,
                            log: Callable[[str], None]) -> str:
    """Write bin/feff_par.cmd that runs each module via mpiexec -n %FEFF_NPROC%."""
    bin_dir = repo_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    wrapper = bin_dir / "feff_par.cmd"

    mpiexec = _find_mpiexec()
    if not mpiexec:
        raise RuntimeError("mpiexec.exe not found in Intel oneAPI install.")
    mpi_bin_dir = os.path.dirname(mpiexec)
    setvars = _find_oneapi_setvars()

    present = [n for n in WINDOWS_SEQUENCE
               if (win64_par_dir / f"{n}.exe").exists()]
    total = len(present)

    lines = [
        "@echo off",
        "setlocal enabledelayedexpansion",
        "if not defined FEFF_NPROC set FEFF_NPROC=4",
        f'set "FeffPath={win64_par_dir}"',
        f'set "MpiExec={mpiexec}"',
        "",
        ":: Initialise Intel oneAPI MPI environment if not already loaded.",
        "if not defined I_MPI_ROOT (",
    ]
    if setvars:
        lines.append(f'    call "{setvars}" --force >nul 2>&1')
    lines += [
        ")",
        f'set "PATH={mpi_bin_dir};%PATH%"',
        "",
        ":: Run FEFF in %TEMP% to avoid OneDrive/cloud-sync file locking.",
        'set "OrigDir=%CD%"',
        'set "TmpDir=%TEMP%\\feff_par_run_%RANDOM%"',
        'mkdir "%TmpDir%"',
        'if exist "%OrigDir%\\feff.inp" copy /Y "%OrigDir%\\feff.inp" "%TmpDir%\\" >nul',
        'for %%F in ("%OrigDir%\\*.cif" "%OrigDir%\\*.xyz") do (',
        '    if exist "%%~F" copy /Y "%%~F" "%TmpDir%\\" >nul',
        ")",
        'pushd "%TmpDir%"',
        "",
    ]
    for i, name in enumerate(present, start=1):
        if name in _PARALLEL_RUN_MODULES:
            lines.append(f'echo [FEFF/MPI {i}/{total}] {name} on %FEFF_NPROC% procs')
            lines.append(
                f'call "%MpiExec%" -n %FEFF_NPROC% "%FeffPath%\\{name}.exe"'
            )
        else:
            # Serial modules: run directly. The MPI-linked binary still
            # initialises MPI internally with COMM_WORLD size 1, behaving
            # like the sequential build but avoiding shared-file races
            # (e.g. kmesh.dat) that occur when these modules run under
            # mpiexec -n N.
            lines.append(f'echo [FEFF/MPI {i}/{total}] {name} (serial)')
            lines.append(f'call "%FeffPath%\\{name}.exe"')
        lines.append("if errorlevel 1 goto :done")

    lines += [
        "",
        ":done",
        'set "_ec=%errorlevel%"',
        "popd",
        ":: Copy outputs back to original directory.",
        'xcopy /Y /Q "%TmpDir%\\*" "%OrigDir%\\" >nul 2>&1',
        'rmdir /S /Q "%TmpDir%"',
        "endlocal",
        "exit /b %_ec%",
    ]
    wrapper.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    log(f"Created parallel FEFF wrapper: {wrapper}")
    return str(wrapper)


# Modules to compile against MPI (so they link the parallel.f90 routines).
_PAR_MODULES = (
    "rdinp", "atomic", "pot", "ldos", "screen", "opconsat", "xsph",
    "fms", "mkgtr", "path", "genfmt", "ff2x", "sfconv", "compton",
    "eels", "rhorrp", "dmdw", "dym2feffinp",
)

# Modules that are SAFE to launch under mpiexec -n N at runtime.  Other modules
# either have shared-file write races under MPI (pot/ldos/screen write the same
# kmesh.dat) or are inherently serial (rdinp, mkgtr, path, etc.).  Those still
# run from the parallel build, but as single MPI processes (no mpiexec wrapper).
# fms is the dominant cost in XANES, so this captures most of the speedup.
_PARALLEL_RUN_MODULES = frozenset({"fms", "xsph"})

# Stack reserve to bake into each FEFF executable's PE header.  The Windows
# default of 1 MB is far too small for FEFF's SCF / FMS arrays — the pot module
# in particular blows the stack on realistic clusters.  256 MB is generous but
# only consumes virtual address space; physical memory is committed lazily.
_FEFF_STACK_BYTES = 0x10000000  # 256 MiB


def _find_editbin() -> str:
    """Return path to MSVC editbin.exe (for setting PE stack reserve)."""
    import glob as _glob
    for vs_root in (
        r"C:\Program Files\Microsoft Visual Studio",
        r"C:\Program Files (x86)\Microsoft Visual Studio",
    ):
        pattern = os.path.join(vs_root, "*", "*", "VC", "Tools", "MSVC", "*",
                               "bin", "Hostx64", "x64", "editbin.exe")
        matches = sorted(_glob.glob(pattern))
        if matches:
            return matches[-1]
    return ""


def _bump_stack_size(exe_dir: Path, log: Callable[[str], None]) -> int:
    """Use editbin to raise the PE stack reserve on every .exe in exe_dir.

    Returns the number of executables successfully patched.  Silently no-ops
    if editbin is unavailable (the build still produced executables; they
    just keep the default 1 MB stack and may crash on large clusters).
    """
    editbin = _find_editbin()
    if not editbin:
        log("Warning: editbin.exe not found — leaving default 1 MB stack "
            "(pot/screen/fms may overflow on large clusters).")
        return 0
    patched = 0
    for exe in sorted(exe_dir.glob("*.exe")):
        try:
            proc = subprocess.run(
                [editbin, f"/STACK:{_FEFF_STACK_BYTES}", str(exe)],
                capture_output=True, text=True, timeout=60, check=False,
            )
            if proc.returncode == 0:
                patched += 1
            else:
                log(f"  editbin failed on {exe.name}: {proc.stderr.strip()}")
        except Exception as exc:
            log(f"  editbin error on {exe.name}: {exc}")
    if patched:
        log(f"Set stack reserve to {_FEFF_STACK_BYTES // (1024 * 1024)} MB "
            f"on {patched} executable(s).")
    return patched


def _build_windows_parallel(repo_dir: Path, log: Callable[[str], None]) -> dict:
    """Build the MPI-parallel variant of FEFF10 into mod/win64_par/."""
    src_dir = repo_dir / "src"
    compile_dir = repo_dir / "mod" / "Par"
    win64_par_dir = repo_dir / "mod" / "win64_par"

    if not src_dir.is_dir():
        raise RuntimeError("FEFF10 source directory not found. Run FEFF Setup first.")

    mpiexec = _find_mpiexec()
    if not mpiexec:
        raise RuntimeError(
            "Intel MPI was not found. Install Intel oneAPI HPC Toolkit "
            "(includes Intel MPI) from intel.com/oneapi, then retry."
        )

    # Fast path: if all required executables already exist, skip recompile and
    # just regenerate the wrapper (lets us refresh wrapper logic without a
    # full ~5 min rebuild).  We still re-run editbin so a stale build with
    # the default 1 MB stack gets bumped on the next "Build" click.
    required_names = ["rdinp.exe", "pot.exe", "ff2x.exe", "fms.exe", "xsph.exe"]
    if win64_par_dir.is_dir() and all(
        (win64_par_dir / n).exists() for n in required_names
    ):
        log("Parallel FEFF10 executables already present — regenerating wrapper only.")
        _bump_stack_size(win64_par_dir, log)
        # Also bump the serial build, since feff.cmd may still be in use.
        win64_dir = repo_dir / "mod" / "win64"
        if win64_dir.is_dir():
            _bump_stack_size(win64_dir, log)
        wrapper = _write_parallel_wrapper(repo_dir, win64_par_dir, log)
        return {"ok": True, "built": False, "exe_path": wrapper}

    mpiifx = _find_mpiifx()
    if not mpiifx:
        raise RuntimeError("mpiifx.bat not found in Intel oneAPI MPI installation.")
    log(f"MPI Fortran driver: {mpiifx}")
    log(f"mpiexec:            {mpiexec}")

    build_env = _oneapi_env(log)
    if build_env is None:
        raise RuntimeError("Could not configure Intel oneAPI build environment.")

    # Switch PAR/parallel.f90 to the MPI variant, regenerate *_tot.f90, compile,
    # then restore the sequential dummy so future serial rebuilds still work.
    _switch_par_source(src_dir, "mpi", log)
    proc = None
    try:
        compile_dir.mkdir(parents=True, exist_ok=True)
        # Force regeneration: delete any stale tot files from a prior build.
        for stale in compile_dir.glob("*_tot.f90"):
            stale.unlink()
        _totalize_feff10_windows_source(src_dir, compile_dir, log)

        win64_par_dir.mkdir(parents=True, exist_ok=True)

        bat_lines = ["@echo off"]
        modules_built = []
        for mod in _PAR_MODULES:
            tot = compile_dir / f"{mod}_tot.f90"
            if not tot.exists() or tot.stat().st_size == 0:
                continue
            modules_built.append(mod)
            bat_lines.append(f'echo Compiling {mod} (MPI) ...')
            bat_lines.append(
                f'call "{mpiifx}" "{mod}_tot.f90" -o "..\\win64_par\\{mod}.exe"'
            )
            bat_lines.append("if errorlevel 1 exit /b 1")
        if not modules_built:
            raise RuntimeError("No *_tot.f90 sources to compile in MPI mode.")

        bat_path = compile_dir / "_compile_par.BAT"
        bat_path.write_text("\r\n".join(bat_lines) + "\r\n", encoding="utf-8")
        log(f"Compiling {len(modules_built)} modules with MPI ...")
        try:
            proc = _run_subprocess(
                ["cmd", "/c", str(bat_path)], str(compile_dir),
                log, env=build_env, timeout=3600,
            )
        finally:
            try:
                bat_path.unlink()
            except Exception:
                pass

        # Some -o variants drop the .exe suffix; normalise so the wrapper finds them.
        for f in win64_par_dir.iterdir():
            if f.is_file() and f.suffix == "" and not f.name.startswith("."):
                renamed = f.with_suffix(".exe")
                if not renamed.exists():
                    f.rename(renamed)
    finally:
        try:
            _switch_par_source(src_dir, "seq", log)
        except Exception as exc:
            log(f"Warning: could not restore sequential PAR/parallel.f90 ({exc})")

    if proc is None or proc.returncode != 0:
        raise RuntimeError("MPI FEFF10 compile failed (see log).")

    # Raise the PE stack reserve so pot/screen/fms don't blow the default 1 MB.
    _bump_stack_size(win64_par_dir, log)

    required = [win64_par_dir / f"{n}.exe" for n in ("rdinp", "pot", "ff2x")]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            f"MPI FEFF10 build is missing required executables: {', '.join(missing)}"
        )

    wrapper = _write_parallel_wrapper(repo_dir, win64_par_dir, log)
    return {"ok": True, "built": True, "exe_path": wrapper}


def install_parallel_managed_feff(cfg_path: str, n_procs: int,
                                  log: Callable[[str], None]) -> dict:
    """Public entry point: build the MPI parallel FEFF10 binaries and wrapper.

    Caller-friendly: returns a dict with ok / exe_path / message and updates
    the setup-state file with the parallel exe path and default proc count.
    """
    state = load_setup_state(cfg_path)
    repo_dir = Path(str(state.get("install_dir", _default_install_dir())))
    if not repo_dir.exists() or not (repo_dir / "src").is_dir():
        return {
            "ok": False,
            "exe_path": "",
            "message": (
                "FEFF10 source is not installed yet. Run "
                "Help -> FEFF Setup / Update first."
            ),
        }
    try:
        if os.name != "nt":
            raise RuntimeError("Parallel build is currently Windows-only.")
        build = _build_windows_parallel(repo_dir, log)
        exe_path = str(build.get("exe_path", ""))
        update_setup_state(cfg_path, {
            "mpi_exe_path": exe_path,
            "mpi_n_procs": int(max(1, n_procs)),
        })
        return {
            "ok": True,
            "exe_path": exe_path,
            "n_procs": int(max(1, n_procs)),
            "message": (
                f"Parallel FEFF10 ready. Wrapper: {exe_path}\n"
                f"Default processes: {n_procs} (override via FEFF_NPROC env var)."
            ),
        }
    except Exception as exc:
        return {"ok": False, "exe_path": "", "message": str(exc)}


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
