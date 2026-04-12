"""
sgm_xas_loader.py  —  SGM Beamline XAS Stack Loader & Preprocessor
====================================================================
Loads custom SGM (Canadian Light Source, Soft X-ray) beamline stack data:
    <timestamp>_<sample>_stack/
        <date>_<time>_stack_data.h5      energy axis + ring current (I0)
        <sample>_<energy>eV/             one subdir per energy point
            sdd1_0.bin … sdd4_0.bin      SDD detector: uint32 (81×N_ch, N_ch auto-detected)
            mcc_flyer_N.csv              analog channels: I0, TEY, PD

Supported signal modes
----------------------
  TFY  – Total Fluorescence Yield   (all SDD channels summed)
  PFY  – Partial Fluorescence Yield (user-selected SDD channel ROI)
  TEY  – Total Electron Yield        (MCC ch2, only at available energies)
  PD   – Photodiode / transmission   (MCC ch3, only at available energies)

I0 sources
----------
  Ring Current  – from H5  map_data/ring_current  (always available)
  MCC I0        – from MCC ch1                     (at available energies)

Export
------
  Two-column CSV (energy_eV, mu) → directly loadable by ledge_normalizer.py
  Optional "Open in L-Edge Normalizer" button for direct handoff.

Requires: numpy  matplotlib  h5py   (pip install numpy matplotlib h5py)
"""

from __future__ import annotations
import os, sys, re, json, csv as _csv, threading, subprocess, tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.cm as cm

try:
    import h5py
    _H5PY = True
except ImportError:
    _H5PY = False

# ══════════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════════

# Default to the directory from which the program is launched.
DEFAULT_DIR = os.path.abspath(os.getcwd())
THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
NORMALIZER  = os.path.join(THIS_DIR, "ledge_normalizer.py")

SPATIAL_PIXELS = 81
# SDD_CHANNELS is NOT hardcoded — actual channel count is auto-detected from
# the binary file size (raw.size // SPATIAL_PIXELS).  Datasets differ:
#   • 256-channel SDD (PP1-P_2_23a, etc.): calibration ≈ 9.6 eV/channel
#       Ni Lα (~849 eV) → ch ~88;  Ni Lβ (~866 eV) → ch ~90
#       → PFY ROI ch 75–115 covers the full Ni Lα/Lβ emission window
#   • Higher-channel SDDs (other beamtime datasets): Teak Boyko (CLS) noted
#       ch 250–550 gave excellent Ni L-edge PFY on those configurations.
#   Use the Auto-detect ROI button (heatmap window) to find the right range
#   for any new dataset automatically.

# Default ROI for Ni L-edge PFY on a 256-channel SGM SDD.
# Covers Ni Lα (~849 eV) and Ni Lβ (~866 eV) emission lines.
# Calibration: ~9.6 eV/channel  →  ch 75 ≈ 720 eV,  ch 115 ≈ 1104 eV
# (For higher-channel SDDs use ch 250–550; see comment above.)
DEFAULT_ROI_LO = 75
DEFAULT_ROI_HI = 115

# Cycle colors for spectra
_COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
           "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]


# ══════════════════════════════════════════════════════════════════════════════
#  Data classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StackInfo:
    """Lightweight metadata for one stack directory (read from H5)."""
    path:         str
    name:         str          # directory basename
    sample:       str          # scan_name from map_md
    date:         str
    energy_min:   float
    energy_max:   float
    n_h5_pts:     int          # energy points recorded in H5
    n_subdirs:    int          # energy subdirs with .bin data
    has_data:     bool         # True if .bin files are present


@dataclass
class Spectrum:
    """Processed XAS spectrum (energy axis + signal array)."""
    label:       str
    energy:      np.ndarray
    signal:      np.ndarray
    signal_type: str           # 'TFY','PFY','TEY','PD'
    norm_by:     str           # 'ring_current','mcc_i0','none'
    stack_path:  str           = ""
    color:       str           = "#1f77b4"
    visible:     bool          = True


# ══════════════════════════════════════════════════════════════════════════════
#  Low-level I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def _parse_energy(dirname: str) -> Optional[float]:
    """Extract energy (eV) from a subdir name like PP3-P_5_43a_852_10eV."""
    m = re.search(r'_(\d+)_(\d{1,2})[Ee][Vv]$', dirname)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    return None


def _list_energy_subdirs(stack_path: str) -> List[Tuple[str, str, float]]:
    """Return sorted ``(name, path, energy_eV)`` tuples for energy subdirs."""
    subdirs: List[Tuple[str, str, float]] = []
    try:
        with os.scandir(stack_path) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                e_ev = _parse_energy(entry.name)
                if e_ev is not None:
                    subdirs.append((entry.name, entry.path, e_ev))
    except OSError:
        return []
    subdirs.sort(key=lambda item: item[2])
    return subdirs


def _list_dir_files(dir_path: str) -> Dict[str, str]:
    """Return a lowercase-name -> absolute-path map for files in ``dir_path``."""
    files: Dict[str, str] = {}
    try:
        with os.scandir(dir_path) as it:
            for entry in it:
                if entry.is_file():
                    files[entry.name.lower()] = entry.path
    except OSError:
        pass
    return files


def _first_file_with_suffix(files: Dict[str, str], suffix: str) -> Optional[str]:
    """Return the first file path whose lowercase name ends with ``suffix``."""
    suffix = suffix.lower()
    for name, path in files.items():
        if name.endswith(suffix):
            return path
    return None


def _load_sdd_bin(path: str) -> Optional[np.ndarray]:
    """
    Load one SDD .bin file.
    Returns uint32 array of shape (n_pixels, n_channels).
    Channel count is auto-detected from file size (raw.size // SPATIAL_PIXELS)
    so this works correctly whether the SDD has 256, 1024, or any other count.
    Returns None if the file is empty / corrupt.
    """
    raw = np.fromfile(path, dtype=np.uint32)
    if raw.size == 0:
        return None                          # empty file — skip silently
    n_ch = raw.size // SPATIAL_PIXELS
    if n_ch < 1:
        return None                          # less than one full row — unusable
    return raw[:SPATIAL_PIXELS * n_ch].reshape(SPATIAL_PIXELS, n_ch)


def _sum_sdd_channels(path: str) -> Optional[np.ndarray]:
    """
    Load one SDD .bin file and sum over spatial pixels.

    Returns a float64 array of shape ``(n_channels,)`` without creating an
    additional full-size float64 copy of the entire 2-D detector matrix.
    """
    raw = np.fromfile(path, dtype=np.uint32)
    if raw.size == 0:
        return None
    n_ch = raw.size // SPATIAL_PIXELS
    if n_ch < 1:
        return None
    usable = raw[:SPATIAL_PIXELS * n_ch].reshape(SPATIAL_PIXELS, n_ch)
    return usable.sum(axis=0, dtype=np.float64)


def _load_mcc_csv(path: str) -> Optional[np.ndarray]:
    """
    Load mcc_flyer CSV.
    Returns float array of shape (N_rows, 8) or None on failure.
    Columns: ch1=I0, ch2=TEY, ch3=PD, ch4=Misc1, ch5-8=unused.
    """
    rows = []
    try:
        with open(path, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                vals = [float(v) for v in line.split(',')]
                if len(vals) >= 4:
                    rows.append(vals)
        if rows:
            return np.array(rows, dtype=float)
    except Exception:
        pass
    return None


def _load_h5_metadata(h5_path: str) -> dict:
    """
    Parse H5 metadata.
    Returns dict with keys: energy, ring_current, sample, amplifier_settings.
    """
    result = {
        'energy': np.array([]),
        'ring_current': np.array([]),
        'sample': '',
        'amplifier': {},
    }
    if not _H5PY:
        return result
    try:
        with h5py.File(h5_path, 'r') as hf:
            md = hf.get('map_data')
            if md is not None:
                result['energy']        = md['energy'][()]
                result['ring_current']  = md['ring_current'][()]
                mds = [json.loads(x) for x in md['map_md'][()]]
                if mds:
                    result['sample'] = mds[0].get('scan_name', '')
                amps = md.get('amplifier_settings')
                if amps is not None:
                    result['amplifier'] = json.loads(amps[0])
    except Exception:
        pass
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Stack scanning
# ══════════════════════════════════════════════════════════════════════════════

def scan_base_dir(base: str) -> List[StackInfo]:
    """
    Walk base directory and return StackInfo for every stack subfolder.
    Works with or without an H5 metadata file — energy axis and sample name
    are inferred from energy-subdir names when H5 is absent.
    """
    infos = []
    if not os.path.isdir(base):
        return infos

    try:
        entries = sorted(os.scandir(base), key=lambda item: item.name.lower())
    except OSError:
        return infos

    for entry in entries:
        if not entry.is_dir():
            continue
        full = entry.path

        # ── Collect energy subdirs (primary requirement) ────────────────────
        subdirs = _list_energy_subdirs(full)

        # Must have at least some energy subdirs to be a valid stack
        if not subdirs:
            continue

        has_data = any(
            _first_file_with_suffix(_list_dir_files(sd_path), '.bin') is not None
            for _, sd_path, _ in subdirs
        )

        # ── Try H5 for rich metadata ────────────────────────────────────────
        root_files = _list_dir_files(full)
        h5_path = _first_file_with_suffix(root_files, '.h5')
        if h5_path:
            meta = _load_h5_metadata(h5_path)
            en   = meta['energy']
            sample = meta['sample']
            dm = re.match(r'(\d{4}-\d{2}-\d{2})', os.path.basename(h5_path))
            date = dm.group(1) if dm else ''
        else:
            en     = np.array([])
            sample = ''
            date   = ''

        # ── Fallback: derive energy range from subdir names ─────────────────
        if en.size == 0:
            en = np.array([e_ev for _, _, e_ev in subdirs], dtype=float)

        # ── Fallback: derive sample name from subdir names ──────────────────
        if not sample:
            # Subdir names look like  <sample>_<int>_<frac>eV
            # Strip the trailing _NNN_NNeV to recover sample name
            m_sample = re.match(r'^(.+?)_\d+_\d+[Ee][Vv]$', subdirs[0][0])
            if m_sample:
                sample = m_sample.group(1)
            else:
                # Last resort: strip timestamp prefix from stack dir name
                m_dir = re.match(r'^\d{6}_(.+?)(?:_stack)?$', entry.name)
                sample = m_dir.group(1) if m_dir else entry.name

        infos.append(StackInfo(
            path      = full,
            name      = entry.name,
            sample    = sample,
            date      = date,
            energy_min= float(en.min()) if en.size else 0,
            energy_max= float(en.max()) if en.size else 0,
            n_h5_pts  = int(en.size),
            n_subdirs = len(subdirs),
            has_data  = has_data,
        ))
    return infos


# ══════════════════════════════════════════════════════════════════════════════
#  Core spectrum builder
# ══════════════════════════════════════════════════════════════════════════════

def build_spectrum(
        info:        StackInfo,
        signal_type: str   = 'TFY',        # 'TFY','PFY','TEY','PD'
        detectors            = None,        # list of ints e.g. [1,2,3,4] or [1,3]
        roi_lo:      int   = DEFAULT_ROI_LO,
        roi_hi:      int   = DEFAULT_ROI_HI,
        norm_by:     str   = 'ring_current',  # 'ring_current','mcc_i0','none'
        progress_cb         = None,    # callable(pct: int)
        stop_event          = None,    # threading.Event
) -> Spectrum:
    """
    Build a normalized XAS spectrum from one stack directory.

    Returns a Spectrum with energy (eV) and signal (normalized).
    """
    # ── Collect energy subdirs ──────────────────────────────────────────────
    stack = info.path
    subdirs = _list_energy_subdirs(stack)
    if not subdirs:
        raise ValueError(f"No energy subdirectories found in:\n{stack}")

    # ── Load H5 metadata ────────────────────────────────────────────────────
    root_files = _list_dir_files(stack)
    h5_path = _first_file_with_suffix(root_files, '.h5')
    meta    = _load_h5_metadata(h5_path) if h5_path else {}
    h5_en   = np.asarray(meta.get('energy', np.array([])), dtype=float)
    h5_rc   = np.asarray(meta.get('ring_current', np.array([])), dtype=float)
    if h5_en.size and h5_rc.size:
        order = np.argsort(h5_en)
        h5_en = h5_en[order]
        h5_rc = h5_rc[order]

    def _ring_current_at(e_ev: float) -> float:
        """Lookup ring current from H5 for a given energy."""
        if h5_en.size == 0 or h5_rc.size == 0:
            return 1.0
        idx = int(np.searchsorted(h5_en, e_ev))
        if idx <= 0:
            best = 0
        elif idx >= h5_en.size:
            best = h5_en.size - 1
        else:
            left = idx - 1
            right = idx
            best = right if abs(h5_en[right] - e_ev) < abs(e_ev - h5_en[left]) else left
        return max(float(h5_rc[best]), 1e-6)

    # ── Which SDD files to sum ───────────────────────────────────────────────
    if detectors is None:
        detectors = [1, 2, 3, 4]
    sdd_names = [f'sdd{n}_0.bin' for n in detectors]

    energy_list  = []
    signal_list  = []
    total = len(subdirs)

    for i, (_dname, subdir_path, e_ev) in enumerate(subdirs):
        if stop_event and stop_event.is_set():
            break
        if progress_cb:
            progress_cb(int(100 * i / total))
        subdir_files = _list_dir_files(subdir_path)

        # ── SDD-based signals (TFY / PFY) ───────────────────────────────────
        if signal_type in ('TFY', 'PFY'):
            sdd_sum = None   # sized lazily from first loaded file
            found   = False
            for sdd_name in sdd_names:
                bin_path = subdir_files.get(sdd_name.lower())
                if not bin_path:
                    continue
                summed = _sum_sdd_channels(bin_path)
                if summed is None:
                        continue             # empty / corrupt file — skip
                ch = summed.size
                if sdd_sum is None:
                    sdd_sum = np.zeros(ch, dtype=np.float64)
                elif sdd_sum.size != ch:
                    sdd_sum = np.zeros(ch, dtype=np.float64)
                sdd_sum += summed
                found = True
            if sdd_sum is None:
                sdd_sum = np.zeros(1, dtype=np.float64)
            n_ch_actual = sdd_sum.size
            clamped_lo = max(0,            min(roi_lo, n_ch_actual - 1))
            clamped_hi = max(clamped_lo + 1, min(roi_hi, n_ch_actual))
            if not found:
                continue

            if signal_type == 'TFY':
                raw_sig = sdd_sum.sum()
            else:  # PFY
                raw_sig = sdd_sum[clamped_lo:clamped_hi].sum()

            # Normalization
            if norm_by == 'ring_current':
                i0 = _ring_current_at(e_ev)
                sig = raw_sig / i0
            elif norm_by == 'mcc_i0':
                csv_path = _first_file_with_suffix(subdir_files, '.csv')
                if csv_path:
                    mcc = _load_mcc_csv(csv_path)
                    i0  = float(np.mean(mcc[:, 0])) if mcc is not None else _ring_current_at(e_ev)
                else:
                    i0 = _ring_current_at(e_ev)
                sig = raw_sig / max(i0, 1e-12)
            else:
                sig = raw_sig

        # ── MCC-based signals (TEY / PD / drain) ─────────────────────────────
        # SGM MCC channel assignments (confirmed from data):
        #   ch1 (idx 0) = I0 ion chamber
        #   ch2 (idx 1) = drain current — no signal in ambient mode
        #   ch3 (idx 2) = PD photodiode
        #   ch4 (idx 3) = TEY (working electron yield signal at SGM)
        elif signal_type in ('TEY', 'PD', 'AEY'):
            csv_path = _first_file_with_suffix(subdir_files, '.csv')
            if not csv_path:
                continue   # no MCC data at this energy
            mcc = _load_mcc_csv(csv_path)
            if mcc is None:
                continue
            ch_map = {'TEY': 3, 'PD': 2, 'AEY': 1}   # TEY=ch4, PD=ch3, AEY=ch2 (drain)
            ch_idx = ch_map[signal_type]
            if mcc.shape[1] <= ch_idx:
                continue   # channel not present in this file
            i0_raw  = float(np.mean(mcc[:, 0]))
            sig_raw = float(np.mean(mcc[:, ch_idx]))

            if norm_by == 'ring_current':
                i0 = _ring_current_at(e_ev)
                sig = sig_raw / max(i0, 1e-12)
            elif norm_by == 'mcc_i0':
                sig = sig_raw / max(i0_raw, 1e-12)
            else:
                sig = sig_raw

        else:
            continue

        energy_list.append(e_ev)
        signal_list.append(sig)

    if progress_cb:
        progress_cb(100)

    if not energy_list:
        raise ValueError(
            f"No data could be extracted for signal '{signal_type}' "
            f"from:\n{stack}\n\n"
            "For TEY/PD, mcc_flyer CSV files must be present."
        )

    en  = np.array(energy_list,  dtype=float)
    sig = np.array(signal_list,  dtype=float)

    # Sort by energy (should already be sorted, but just in case)
    idx = np.argsort(en)
    en  = en[idx]
    sig = sig[idx]

    return Spectrum(
        label       = info.sample,
        energy      = en,
        signal      = sig,
        signal_type = signal_type,
        norm_by     = norm_by,
        stack_path  = stack,
    )


def average_spectra(spectra: List[Spectrum]) -> Spectrum:
    """
    Interpolate all spectra to a common energy grid and return their mean.
    Uses the union of all energy points with the finest resolution.
    """
    if not spectra:
        raise ValueError("No spectra to average.")
    if len(spectra) == 1:
        sp = spectra[0]
        return Spectrum(
            label       = sp.label + " (avg)",
            energy      = sp.energy.copy(),
            signal      = sp.signal.copy(),
            signal_type = sp.signal_type,
            norm_by     = sp.norm_by,
        )

    # Common grid: range of intersection, step = finest among all spectra
    en_min = max(sp.energy.min() for sp in spectra)
    en_max = min(sp.energy.max() for sp in spectra)
    steps  = [np.min(np.diff(sp.energy)) for sp in spectra if sp.energy.size > 1]
    step   = min(steps) if steps else 0.1

    if en_max <= en_min:
        # Fallback: use union range
        en_min = min(sp.energy.min() for sp in spectra)
        en_max = max(sp.energy.max() for sp in spectra)

    en_grid = np.arange(en_min, en_max + step * 0.5, step)

    interp_signals = []
    for sp in spectra:
        interp_signals.append(np.interp(en_grid, sp.energy, sp.signal))

    avg_sig = np.mean(interp_signals, axis=0)
    label   = spectra[0].label

    return Spectrum(
        label       = f"{label} (n={len(spectra)} avg)",
        energy      = en_grid,
        signal      = avg_sig,
        signal_type = spectra[0].signal_type,
        norm_by     = spectra[0].norm_by,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Emission / Excitation matrix builder
#  Inspired by SGMPython StackScan (github.com/Beamlines-CanadianLightSource)
# ══════════════════════════════════════════════════════════════════════════════

def build_exem_matrix(
        info:        StackInfo,
        detectors    = None,      # list[int] e.g. [1,2,3,4]
        progress_cb  = None,      # callable(pct: int)
        stop_event   = None,      # threading.Event
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the emission/excitation (Ex/Em) matrix from a stack directory.

    Mirrors the per-energy SDD loading in SGMPython's StackScan class:
    for each incident energy, all SDD binary files for the selected detectors
    are loaded and summed spatially (over SPATIAL_PIXELS rows), yielding a
    1-D emission spectrum.  Stacking these across energy produces a 2-D matrix.

    Returns
    -------
    energies : np.ndarray, shape (n_energies,)
        Incident photon energies in eV.
    matrix : np.ndarray, shape (n_energies, n_channels)
        Spatially-summed SDD counts per emission channel per energy step.
        Channel count is auto-detected from file size (see _load_sdd_bin).
    """
    if detectors is None:
        detectors = [1, 2, 3, 4]
    sdd_names = [f'sdd{n}_0.bin' for n in detectors]

    stack   = info.path
    subdirs = _list_energy_subdirs(stack)

    energies: List[float]      = []
    rows:     List[np.ndarray] = []
    total = len(subdirs)

    for i, (_dname, subdir_path, e_ev) in enumerate(subdirs):
        if stop_event and stop_event.is_set():
            break
        if progress_cb:
            progress_cb(int(100 * i / max(total, 1)))
        subdir_files = _list_dir_files(subdir_path)
        sdd_sum: Optional[np.ndarray] = None

        for sdd_name in sdd_names:
            bin_path = subdir_files.get(sdd_name.lower())
            if not bin_path:
                continue
            summed = _sum_sdd_channels(bin_path)
            if summed is None:
                continue
            ch = summed.size
            if sdd_sum is None:
                sdd_sum = np.zeros(ch, dtype=np.float64)
            elif sdd_sum.size != ch:
                sdd_sum = np.zeros(ch, dtype=np.float64)
            sdd_sum += summed

        if sdd_sum is not None:
            energies.append(e_ev)
            rows.append(sdd_sum)

    if progress_cb:
        progress_cb(100)

    if not rows:
        return np.array([]), np.zeros((0, 1)), 0, 0

    # ── Reconcile rows that have different channel counts ───────────────────
    # Truncated .bin files (scan interrupted mid-write) produce fewer channels
    # than intact files.  Find the mode (most common length) — that is the
    # true SDD channel count — and zero-pad any shorter rows to match.
    # Rows longer than the mode are trimmed (shouldn't happen, but defensive).
    lengths = [r.size for r in rows]
    n_ch_mode = int(np.bincount(lengths).argmax())

    n_truncated = sum(1 for l in lengths if l != n_ch_mode)
    if n_truncated:
        fixed = []
        for r in rows:
            if r.size == n_ch_mode:
                fixed.append(r)
            elif r.size < n_ch_mode:
                # Zero-pad: missing channels contribute 0 counts (safe)
                padded = np.zeros(n_ch_mode, dtype=np.float64)
                padded[:r.size] = r
                fixed.append(padded)
            else:
                # Longer than expected — trim to mode length
                fixed.append(r[:n_ch_mode])
        rows = fixed

    matrix = np.vstack(rows)
    return np.array(energies), matrix, n_truncated, n_ch_mode


# ══════════════════════════════════════════════════════════════════════════════
#  Emission / Excitation Heatmap Window
# ══════════════════════════════════════════════════════════════════════════════

class ExEmHeatmapWindow(tk.Toplevel):
    """
    Interactive emission/excitation matrix heatmap.

    Based on the spatial + spectral visualization in SGMPython's
    StackScan.plot_pca_kmeans() and MapScan.plot_overview():
      X-axis  →  SDD emission channel
      Y-axis  →  Incident photon energy  (excitation)
      Colour  →  Spatially-summed fluorescence intensity

    Features
    --------
    - Linear / Log₁ₚ / √ intensity scaling
    - Interchangeable colormaps
    - Per-detector toggle (Refresh rebuilds from disk)
    - PFY ROI overlay lines (dashed red) matching the loader settings
    - Horizontal energy cursor (click plot → highlights that row)
    - Save PNG / Export full matrix as CSV
    """

    _CMAPS = ['viridis', 'plasma', 'inferno', 'magma', 'hot',
              'Blues_r', 'YlOrRd', 'RdYlBu_r', 'coolwarm', 'gray']

    def __init__(self, master, info: StackInfo,
                 detectors=None, roi_lo: Optional[int] = None,
                 roi_hi: Optional[int] = None):
        super().__init__(master)
        self.title(f'Ex/Em Matrix — {info.name}')
        self.geometry('1020x720')
        self.minsize(700, 480)

        self._info      = info
        self._detectors = list(detectors) if detectors else [1, 2, 3, 4]
        self._roi_lo    = roi_lo
        self._roi_hi    = roi_hi
        self._energies: Optional[np.ndarray] = None
        self._matrix:   Optional[np.ndarray] = None
        self._load_gen  = 0          # generation counter — incremented each refresh

        # ── Controls vars ──────────────────────────────────────────────────
        self._cmap_var   = tk.StringVar(value='viridis')
        self._scale_var  = tk.StringVar(value='Linear')
        self._roi_show   = tk.BooleanVar(value=True)
        self._cursor_var = tk.StringVar(value='—')
        self._det_vars   = [tk.BooleanVar(value=(i + 1 in self._detectors))
                            for i in range(4)]
        self._status_var = tk.StringVar(value='Loading…')

        self._build()
        self.after(80, self._load_data)

    # ── UI construction ────────────────────────────────────────────────────

    def _build(self):
        # ── Toolbar row 1: colormap / scale / ROI ──────────────────────────
        tb1 = ttk.Frame(self)
        tb1.pack(fill='x', padx=6, pady=(6, 2))

        ttk.Label(tb1, text='Colormap:').pack(side='left')
        cm_cb = ttk.Combobox(tb1, textvariable=self._cmap_var,
                             values=self._CMAPS, state='readonly', width=12)
        cm_cb.pack(side='left', padx=(2, 10))
        cm_cb.bind('<<ComboboxSelected>>', lambda *_: self._redraw())

        ttk.Label(tb1, text='Scale:').pack(side='left')
        sc_cb = ttk.Combobox(tb1, textvariable=self._scale_var,
                             values=['Linear', 'Log₁ₚ', '√'], state='readonly', width=8)
        sc_cb.pack(side='left', padx=(2, 10))
        sc_cb.bind('<<ComboboxSelected>>', lambda *_: self._redraw())

        ttk.Checkbutton(tb1, text='Show ROI lines',
                        variable=self._roi_show,
                        command=self._toggle_roi).pack(side='left', padx=6)

        ttk.Label(tb1, text='Cursor E:').pack(side='left', padx=(10, 2))
        ttk.Label(tb1, textvariable=self._cursor_var,
                  width=10, relief='sunken', anchor='center').pack(side='left')

        # ── Toolbar row 1b: colour clipping ────────────────────────────────
        ttk.Label(tb1, text='Colour clip:').pack(side='left', padx=(12, 2))
        self._clip_var = tk.StringVar(value='99')
        ttk.Spinbox(tb1, textvariable=self._clip_var,
                    from_=50, to=100, increment=1, width=5,
                    command=self._redraw).pack(side='left')
        ttk.Label(tb1, text='%ile', font=('TkDefaultFont', 8)
                  ).pack(side='left', padx=(0, 4))
        self._clip_var.trace_add('write', lambda *_: self._redraw())

        # ── Toolbar row 2: detectors / actions ─────────────────────────────
        tb2 = ttk.Frame(self)
        tb2.pack(fill='x', padx=6, pady=(0, 4))

        ttk.Label(tb2, text='Detectors:').pack(side='left')
        for i, var in enumerate(self._det_vars):
            ttk.Checkbutton(tb2, text=f'SDD{i+1}', variable=var,
                            ).pack(side='left', padx=2)

        ttk.Separator(tb2, orient='vertical').pack(side='left', fill='y',
                                                    padx=8)

        ttk.Button(tb2, text='↺ Refresh',
                   command=self._load_data).pack(side='left', padx=3)
        ttk.Button(tb2, text='Auto-detect ROI',
                   command=self._auto_roi).pack(side='left', padx=3)
        ttk.Button(tb2, text='Save PNG…',
                   command=self._save_png).pack(side='left', padx=3)
        ttk.Button(tb2, text='Export CSV…',
                   command=self._export_csv).pack(side='left', padx=3)

        # ── Status bar ─────────────────────────────────────────────────────
        ttk.Label(self, textvariable=self._status_var,
                  anchor='w', foreground='#444',
                  font=('TkDefaultFont', 8)
                  ).pack(fill='x', padx=8, pady=(0, 2))

        # ── Progress bar (shown while loading) ─────────────────────────────
        self._prog = ttk.Progressbar(self, mode='determinate', maximum=100)
        self._prog.pack(fill='x', padx=8)

        # ── Matplotlib figure ───────────────────────────────────────────────
        self._fig  = Figure(figsize=(10, 6), dpi=96, tight_layout=True)
        self._ax   = self._fig.add_subplot(111)
        self._cvs  = FigureCanvasTkAgg(self._fig, master=self)
        self._cvs.get_tk_widget().pack(fill='both', expand=True)
        NavigationToolbar2Tk(self._cvs, self)

        self._cbar      = None
        self._pc        = None
        self._roi_lines = []
        self._h_cursor  = None

        # Click-to-query energy cursor
        self._fig.canvas.mpl_connect('button_press_event', self._on_click)

    # ── Data loading ───────────────────────────────────────────────────────

    def _get_active_detectors(self) -> List[int]:
        active = [i + 1 for i, v in enumerate(self._det_vars) if v.get()]
        return active if active else [1, 2, 3, 4]

    def _load_data(self):
        # Generation counter: any stale worker whose gen != current is ignored.
        # This avoids the threading.Event race where set() on the initial event
        # would prevent the very first worker from delivering its results.
        self._load_gen += 1
        my_gen = self._load_gen

        self._status_var.set('Building Ex/Em matrix from stack…')
        self._prog['value'] = 0
        self.update_idletasks()

        dets = self._get_active_detectors()

        def _cb(pct):
            if self._load_gen == my_gen:
                self.after(0, lambda: self._prog.configure(value=pct))

        def _worker():
            try:
                energies, matrix, n_trunc, n_ch = build_exem_matrix(
                    self._info, detectors=dets, progress_cb=_cb)
            except Exception as exc:
                err_msg = str(exc)
                if self._load_gen == my_gen:
                    self.after(0, lambda: self._status_var.set(
                        f'Error building matrix: {err_msg}'))
                return
            if self._load_gen == my_gen:
                self.after(0, lambda: self._on_loaded(
                    energies, matrix, my_gen, n_trunc, n_ch))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_loaded(self, energies: np.ndarray, matrix: np.ndarray,
                   gen: int, n_trunc: int = 0, n_ch_mode: int = 0):
        if self._load_gen != gen:
            return          # superseded by a newer Refresh — discard
        self._prog['value'] = 100
        if energies.size == 0:
            self._status_var.set(
                'No SDD data found. Check that the stack directory '
                'contains energy subdirs with sdd*.bin files.')
            return
        self._energies = energies
        self._matrix   = matrix
        n_en, n_ch = matrix.shape

        trunc_note = ''
        if n_trunc:
            trunc_note = (f'  |  ⚠ {n_trunc} truncated file(s) zero-padded '
                          f'to {n_ch_mode} ch')

        self._status_var.set(
            f'{n_en} energy points  ×  {n_ch} SDD channels  |  '
            f'Excitation E = {energies[0]:.2f} – {energies[-1]:.2f} eV'
            f'{trunc_note}')
        try:
            self._redraw()
        except Exception as exc:
            self._status_var.set(f'Render error: {exc}')

    # ── Rendering ──────────────────────────────────────────────────────────

    def _apply_scale(self, data: np.ndarray) -> np.ndarray:
        scale = self._scale_var.get()
        if scale == 'Log₁ₚ':
            return np.log1p(np.maximum(data, 0))
        if scale == '√':
            return np.sqrt(np.maximum(data, 0))
        return data.copy()

    def _get_clip_pct(self) -> float:
        try:
            return max(50.0, min(100.0, float(self._clip_var.get())))
        except (ValueError, AttributeError):
            return 99.0

    def _redraw(self):
        if self._matrix is None:
            return

        # ── Clear axes and old colorbar cleanly ────────────────────────────
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        self._ax.cla()

        data       = self._apply_scale(self._matrix)
        n_en, n_ch = data.shape
        en         = self._energies

        # ── Percentile colour clipping ──────────────────────────────────────
        # The low-channel detector artifact (elastic scatter at ch ~15) can be
        # orders of magnitude brighter than the fluorescence signal, crushing
        # the colourmap.  Clipping at the Nth percentile keeps the signal
        # channels visible even when outliers are present.
        pct   = self._get_clip_pct()
        vmax  = float(np.percentile(data, pct))
        vmin  = float(np.percentile(data,  1))   # 1st pct for dead-pixel floor

        # ── Pixel-accurate bin edges ────────────────────────────────────────
        ch_edges = np.arange(n_ch + 1, dtype=float) - 0.5
        if n_en > 1:
            de       = np.diff(en)
            half     = de / 2.0
            en_lo    = np.concatenate([[en[0] - de[0] / 2], en[:-1] + half])
            en_hi    = np.concatenate([en[:-1] + half,      [en[-1] + de[-1] / 2]])
            en_edges = np.concatenate([en_lo, [en_hi[-1]]])
        else:
            en_edges = np.array([en[0] - 0.5, en[0] + 0.5])

        self._pc = self._ax.pcolormesh(
            ch_edges, en_edges, data,
            cmap=self._cmap_var.get(), shading='flat',
            vmin=vmin, vmax=vmax, rasterized=True)

        self._cbar = self._fig.colorbar(
            self._pc, ax=self._ax,
            label=f'Fluorescence Counts ({self._scale_var.get()}, '
                  f'clipped ≤{pct:.0f}th %ile)',
            fraction=0.046, pad=0.04)

        # ── Axis labels — physically meaningful names ───────────────────────
        self._ax.set_xlabel('Emission Energy  (SDD channel number)', fontsize=10)
        self._ax.set_ylabel('Excitation Energy  (incident photon, eV)', fontsize=10)
        self._ax.set_title(
            f'Emission / Excitation Matrix — {self._info.name}\n'
            f'Colour = fluorescence counts at each (emission ch, excitation eV)',
            fontsize=10)
        self._ax.set_xlim(ch_edges[0], ch_edges[-1])
        self._ax.set_ylim(en_edges[0], en_edges[-1])

        # ── ROI overlay lines ───────────────────────────────────────────────
        # Clamp to actual channel range; warn in legend if values were outside.
        self._roi_lines = []
        if self._roi_lo is not None and self._roi_show.get():
            lo_raw = self._roi_lo
            hi_raw = self._roi_hi
            lo     = max(0, min(lo_raw, n_ch - 1))
            hi     = max(lo + 1, min(hi_raw, n_ch))

            # Build legend labels — append warning if clamped
            lo_lbl = (f'ROI lo = {lo_raw}'
                      if lo_raw == lo else
                      f'ROI lo = {lo_raw}  (clamped to {lo})')
            hi_lbl = (f'ROI hi = {hi_raw}'
                      if hi_raw == hi else
                      f'ROI hi = {hi_raw}  (clamped to {hi})')

            kw = dict(color='#ff3333', lw=1.8, ls='--', alpha=0.9)
            self._roi_lines.append(
                self._ax.axvline(lo, **kw, label=lo_lbl))
            self._roi_lines.append(
                self._ax.axvline(hi, **kw, label=hi_lbl))

            # Shade the ROI band so it's easy to see
            self._ax.axvspan(lo, hi, alpha=0.08, color='#ff3333',
                             label='_nolegend_')
            self._ax.legend(fontsize=8, loc='upper right')

        self._h_cursor = None
        self._fig.tight_layout()
        self._cvs.draw()

    # ── Interactivity ──────────────────────────────────────────────────────

    def _toggle_roi(self):
        for ln in self._roi_lines:
            ln.set_visible(self._roi_show.get())
        self._cvs.draw_idle()

    def _auto_roi(self):
        """
        Automatically detect the best PFY ROI from the emission/excitation matrix.

        Strategy (mirrors the approach used in SGMPython's StackScan):
          1. Compute the variance of each emission channel across all incident
             energies.  Channels that respond to the L-edge have HIGH variance;
             background channels and dead/noisy channels have LOW variance.
          2. Exclude the lowest 10% of channels (often dominated by low-energy
             noise / elastic scatter artifact).
          3. Find the contiguous window of high-variance channels that contains
             the global maximum and extend ±15 channels as margin.
          4. Update self._roi_lo / self._roi_hi and redraw.
          5. If a parent SettingsPanel exists, sync the spinboxes so the new ROI
             is used automatically on the next Process run.
        """
        if self._matrix is None:
            self._status_var.set('Load data first before auto-detecting ROI.')
            return

        n_en, n_ch = self._matrix.shape

        # Per-channel variance across all incident energies
        var = np.var(self._matrix, axis=0)

        # Exclude lowest 10% of channels (noise / elastic scatter region)
        skip = max(1, n_ch // 10)
        var[:skip] = 0.0

        # Channel of maximum variance = centre of the fluorescence peak
        peak_ch = int(np.argmax(var))

        # Find the full width at half-maximum of the variance profile
        half_max  = var[peak_ch] / 2.0
        left  = peak_ch
        right = peak_ch
        while left  > skip      and var[left  - 1] >= half_max:
            left  -= 1
        while right < n_ch - 1  and var[right + 1] >= half_max:
            right += 1

        # Add ±15 ch margin, clamp to valid range
        margin   = 15
        roi_lo   = max(skip,     left  - margin)
        roi_hi   = min(n_ch - 1, right + margin)

        self._roi_lo = roi_lo
        self._roi_hi = roi_hi

        self._status_var.set(
            f'Auto-detected ROI: channels {roi_lo} – {roi_hi}  '
            f'(peak variance at ch {peak_ch})')

        # ── Sync back to the SettingsPanel spinboxes if accessible ──────────
        try:
            app = self.master
            if hasattr(app, '_settings'):
                app._settings._roi_lo.set(roi_lo)
                app._settings._roi_hi.set(roi_hi)
        except Exception:
            pass   # not critical — heatmap ROI lines are already updated

        self._redraw()

    def _on_click(self, event):
        """Click on the heatmap → show energy of that row."""
        if event.inaxes is not self._ax or self._energies is None:
            return
        e_click = event.ydata
        if e_click is None:
            return
        idx = int(np.argmin(np.abs(self._energies - e_click)))
        e_actual = self._energies[idx]
        self._cursor_var.set(f'{e_actual:.2f} eV')

        # Draw / update horizontal cursor line
        if self._h_cursor is not None:
            try:
                self._h_cursor.remove()
            except Exception:
                pass
        self._h_cursor = self._ax.axhline(
            e_actual, color='white', lw=1.2, ls='-', alpha=0.75)
        self._cvs.draw_idle()

    # ── Export ─────────────────────────────────────────────────────────────

    def _save_png(self):
        path = filedialog.asksaveasfilename(
            parent=self, title='Save heatmap',
            defaultextension='.png',
            filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'),
                       ('SVG', '*.svg'), ('All', '*.*')])
        if path:
            self._fig.savefig(path, dpi=150, bbox_inches='tight')
            self._status_var.set(f'Saved → {os.path.basename(path)}')

    def _export_csv(self):
        if self._matrix is None:
            messagebox.showwarning('No data', 'Load data first.', parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self, title='Export Ex/Em matrix as CSV',
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv')])
        if not path:
            return
        try:
            n_en, n_ch = self._matrix.shape
            with open(path, 'w', newline='') as fh:
                w = _csv.writer(fh)
                w.writerow(['# Ex/Em matrix: rows=energy, cols=SDD channel'])
                w.writerow(['# Stack:', self._info.name])
                if self._roi_lo is not None:
                    w.writerow([f'# PFY ROI: {self._roi_lo} – {self._roi_hi}'])
                w.writerow(['energy_eV'] + [f'ch{c}' for c in range(n_ch)])
                for e, row in zip(self._energies, self._matrix):
                    w.writerow([f'{e:.4f}'] +
                                [str(int(v)) for v in row])
            self._status_var.set(
                f'Exported {n_en} × {n_ch} matrix → {os.path.basename(path)}')
        except Exception as exc:
            messagebox.showerror('Export error', str(exc), parent=self)


# ══════════════════════════════════════════════════════════════════════════════
#  GUI — Stack browser (left pane)
# ══════════════════════════════════════════════════════════════════════════════

class StackBrowser(ttk.Frame):
    """
    Left panel: shows all stacks grouped by sample name.
    Stacks with .bin data are shown in normal weight;
    stacks without data are dimmed.
    """

    def __init__(self, master, on_selection_change=None, **kw):
        super().__init__(master, **kw)
        self._cb    = on_selection_change
        self._infos : List[StackInfo] = []
        self._build()

    # ── Construction ───────────────────────────────────────────────────────

    def _build(self):
        # Directory row
        dir_fr = ttk.Frame(self)
        dir_fr.pack(fill='x', padx=4, pady=(4,0))
        ttk.Label(dir_fr, text="Data directory:").pack(side='left')
        self._dir_var = tk.StringVar(value=DEFAULT_DIR)
        e = ttk.Entry(dir_fr, textvariable=self._dir_var, width=28)
        e.pack(side='left', fill='x', expand=True, padx=2)
        ttk.Button(dir_fr, text="…", width=2,
                   command=self._browse_dir).pack(side='left')

        ttk.Button(self, text="Refresh  ↺",
                   command=self.refresh).pack(fill='x', padx=4, pady=2)

        # Search/filter
        sf = ttk.Frame(self)
        sf.pack(fill='x', padx=4)
        ttk.Label(sf, text="Filter:").pack(side='left')
        self._filter_var = tk.StringVar()
        self._filter_var.trace_add('write', lambda *_: self._apply_filter())
        ttk.Entry(sf, textvariable=self._filter_var).pack(
            side='left', fill='x', expand=True, padx=2)

        # Treeview
        cols = ('sample', 'date', 'energy', 'pts')
        self._tree = ttk.Treeview(self, columns=cols, show='tree headings',
                                  selectmode='extended')
        self._tree.heading('#0',       text='Stack / Sample')
        self._tree.heading('sample',   text='Sample')
        self._tree.heading('date',     text='Date')
        self._tree.heading('energy',   text='Energy range')
        self._tree.heading('pts',      text='Pts')
        self._tree.column('#0',      width=150, anchor='w')
        self._tree.column('sample',  width=110, anchor='w')
        self._tree.column('date',    width=78,  anchor='c')
        self._tree.column('energy',  width=110, anchor='c')
        self._tree.column('pts',     width=35,  anchor='c')

        self._tree.tag_configure('no_data', foreground='#888888')
        self._tree.tag_configure('has_data', foreground='#000000', font=('TkDefaultFont', 9, 'bold'))
        self._tree.tag_configure('group',    foreground='#003580', font=('TkDefaultFont', 9, 'bold'))

        vsb = ttk.Scrollbar(self, orient='vertical', command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side='left', fill='both', expand=True, padx=(4,0), pady=4)
        vsb.pack(side='left', fill='y', pady=4)

        self._tree.bind('<<TreeviewSelect>>', self._on_select)

        # Info label at bottom
        self._info_var = tk.StringVar(value="Select a stack to begin.")
        ttk.Label(self, textvariable=self._info_var,
                  wraplength=280, justify='left',
                  font=('TkDefaultFont', 8)).pack(
            fill='x', padx=4, pady=2)

        self.refresh()

    def _browse_dir(self):
        d = filedialog.askdirectory(
            initialdir=self._dir_var.get(),
            title="Select SGM Data Directory")
        if d:
            self._dir_var.set(d)
            self.refresh()

    # ── Data population ────────────────────────────────────────────────────

    def refresh(self):
        base = self._dir_var.get()
        self._infos = scan_base_dir(base)
        self._populate()

    def _populate(self):
        flt = self._filter_var.get().lower()
        self._tree.delete(*self._tree.get_children())
        self._iid_map: Dict[str, StackInfo] = {}

        # Group by sample name
        groups: Dict[str, List[StackInfo]] = {}
        for info in self._infos:
            if flt and flt not in info.sample.lower() and flt not in info.name.lower():
                continue
            groups.setdefault(info.sample, []).append(info)

        for sample, items in groups.items():
            has_any = any(i.has_data for i in items)
            g_iid = self._tree.insert(
                '', 'end',
                text=sample,
                values=(sample, '', '', ''),
                open=True,
                tags=('group',) if has_any else ('no_data',)
            )
            for info in items:
                en_str = f"{info.energy_min:.1f} – {info.energy_max:.1f}"
                pts    = f"{info.n_subdirs}/{info.n_h5_pts}"
                tag    = 'has_data' if info.has_data else 'no_data'
                iid = self._tree.insert(
                    g_iid, 'end',
                    text=info.name,
                    values=(info.sample, info.date, en_str, pts),
                    tags=(tag,)
                )
                self._iid_map[iid] = info

    def _apply_filter(self):
        self._populate()

    def _on_select(self, _evt=None):
        selected = self.selected_infos()
        if selected:
            s = selected[-1]
            status = (
                f"Sample: {s.sample}\n"
                f"Date:   {s.date}\n"
                f"Energy: {s.energy_min:.1f} – {s.energy_max:.1f} eV\n"
                f"Points: {s.n_subdirs} subdirs / {s.n_h5_pts} H5 pts\n"
                f"Data:   {'✓ SDD bins present' if s.has_data else '✗ no .bin files'}\n"
                f"Path:   {s.path}"
            )
            self._info_var.set(status)
        if self._cb:
            self._cb(selected)

    # ── Public API ─────────────────────────────────────────────────────────

    def selected_infos(self) -> List[StackInfo]:
        return [self._iid_map[iid]
                for iid in self._tree.selection()
                if iid in self._iid_map]

    @property
    def base_dir(self) -> str:
        return self._dir_var.get()


# ══════════════════════════════════════════════════════════════════════════════
#  GUI — Settings panel
# ══════════════════════════════════════════════════════════════════════════════

class SettingsPanel(ttk.LabelFrame):
    def __init__(self, master, **kw):
        super().__init__(master, text="Processing Settings", **kw)
        self._build()

    def _build(self):
        pad = dict(padx=6, pady=3)

        # Signal type
        ttk.Label(self, text="Signal:").grid(row=0, column=0, sticky='w', **pad)
        self._signal_var = tk.StringVar(value='TFY')
        sig_fr = ttk.Frame(self)
        sig_fr.grid(row=0, column=1, sticky='w', **pad)
        for s, tip in [('TFY', 'Total Fluorescence Yield (all SDD channels)'),
                       ('PFY', 'Partial Fluorescence Yield (SDD ROI)'),
                       ('TEY', 'Total Electron Yield — MCC ch4 (working signal at SGM)'),
                       ('PD',  'Photodiode — MCC ch3'),
                       ('AEY', 'Drain current — MCC ch2 (no signal in ambient mode)')]:
            rb = ttk.Radiobutton(sig_fr, text=s, variable=self._signal_var,
                                 value=s, command=self._on_signal)
            rb.pack(side='left', padx=2)
            rb._tooltip_text = tip

        # Detector selector — "All" master toggle + individual SDD checkboxes
        ttk.Label(self, text="Detectors:").grid(row=1, column=0, sticky='w', **pad)
        det_fr = ttk.Frame(self)
        det_fr.grid(row=1, column=1, sticky='w', **pad)

        self._sdd_all_var  = tk.BooleanVar(value=True)
        self._sdd_vars     = [tk.BooleanVar(value=True) for _ in range(4)]

        def _on_all_toggle():
            state = self._sdd_all_var.get()
            for v in self._sdd_vars:
                v.set(state)

        def _on_individual_toggle():
            # Sync "All" checkbox: checked only when every SDD is checked
            self._sdd_all_var.set(all(v.get() for v in self._sdd_vars))

        ttk.Checkbutton(det_fr, text="All", variable=self._sdd_all_var,
                        command=_on_all_toggle).pack(side='left', padx=(0, 6))
        ttk.Separator(det_fr, orient='vertical').pack(side='left', fill='y',
                                                      padx=4, pady=2)
        for i, var in enumerate(self._sdd_vars, start=1):
            ttk.Checkbutton(det_fr, text=f"SDD{i}", variable=var,
                            command=_on_individual_toggle).pack(side='left', padx=2)

        # ROI (PFY mode).  Channel count is auto-detected from file size,
        # so ROI values can be much larger than 256.  Use 4095 as a safe upper
        # bound for the spinbox (covers 1024 and 4096-channel SDDs).
        _EV_PER_CH = 9.6   # ≈ 9.6 eV/ch for 256-channel SDD (Ni Lα 849 eV → ch ~88)
        self._roi_frame = ttk.Frame(self)
        self._roi_frame.grid(row=2, column=0, columnspan=2, sticky='we', **pad)
        ttk.Label(self._roi_frame, text="SDD ROI — lo ch:").pack(side='left')
        self._roi_lo = tk.IntVar(value=DEFAULT_ROI_LO)
        self._roi_hi = tk.IntVar(value=DEFAULT_ROI_HI)

        def _clamp_roi(*_):
            lo = self._roi_lo.get()
            hi = self._roi_hi.get()
            if hi <= lo:
                self._roi_hi.set(lo + 1)
            lo_ev = int(self._roi_lo.get() * _EV_PER_CH)
            hi_ev = int(self._roi_hi.get() * _EV_PER_CH)
            self._roi_ev_lbl.config(text=f"≈ {lo_ev}–{hi_ev} eV")

        ttk.Spinbox(self._roi_frame, from_=0, to=4095,
                    textvariable=self._roi_lo, width=6,
                    command=_clamp_roi).pack(side='left', padx=2)
        self._roi_lo.trace_add('write', _clamp_roi)
        ttk.Label(self._roi_frame, text="hi:").pack(side='left')
        ttk.Spinbox(self._roi_frame, from_=1, to=4096,
                    textvariable=self._roi_hi, width=6,
                    command=_clamp_roi).pack(side='left', padx=2)
        self._roi_hi.trace_add('write', _clamp_roi)
        self._roi_ev_lbl = ttk.Label(
            self._roi_frame,
            text=f"≈ {int(DEFAULT_ROI_LO*_EV_PER_CH)}–{int(DEFAULT_ROI_HI*_EV_PER_CH)} eV  "
                 f"(Ni Lα/Lβ ~849/866 eV → ch 75–115 for 256-ch SDD)",
            font=('TkDefaultFont', 8), foreground='#555')
        self._roi_ev_lbl.pack(side='left', padx=4)

        # Normalization — default to MCC I0 because H5 files are often absent
        ttk.Label(self, text="Normalize by:").grid(row=3, column=0, sticky='w', **pad)
        self._norm_var = tk.StringVar(value='mcc_i0')
        norm_fr = ttk.Frame(self)
        norm_fr.grid(row=3, column=1, sticky='w', **pad)
        ttk.Radiobutton(norm_fr, text="Ring current (H5)",
                        variable=self._norm_var,
                        value='ring_current').pack(side='left', padx=2)
        ttk.Radiobutton(norm_fr, text="MCC I0 (ch1)",
                        variable=self._norm_var,
                        value='mcc_i0').pack(side='left', padx=2)
        ttk.Radiobutton(norm_fr, text="None",
                        variable=self._norm_var,
                        value='none').pack(side='left', padx=2)

        # Average selected scans?
        self._avg_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Average all selected scans",
                        variable=self._avg_var).grid(
            row=4, column=0, columnspan=2, sticky='w', **pad)

        # Show individual scans alongside average?
        self._show_ind_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Show individual scans (faint)",
                        variable=self._show_ind_var).grid(
            row=5, column=0, columnspan=2, sticky='w', **pad)

        self._on_signal()

    def _on_signal(self):
        # ROI controls only apply to PFY mode
        if self._signal_var.get() == 'PFY':
            # Pass explicit grid options — safer than relying on grid() with no
            # args, which can silently no-op on some tkinter/Tcl/Tk builds.
            self._roi_frame.grid(row=2, column=0, columnspan=2,
                                 sticky='we', padx=6, pady=3)
        else:
            self._roi_frame.grid_remove()

    @property
    def signal_type(self): return self._signal_var.get()
    @property
    def detectors(self):
        """Return list of selected SDD numbers, e.g. [1, 3] or [1,2,3,4]."""
        sel = [i + 1 for i, v in enumerate(self._sdd_vars) if v.get()]
        return sel if sel else [1, 2, 3, 4]   # fallback: all if none checked
    @property
    def roi_lo(self):      return int(self._roi_lo.get())
    @property
    def roi_hi(self):      return int(self._roi_hi.get())
    @property
    def norm_by(self):     return self._norm_var.get()
    @property
    def do_average(self):  return self._avg_var.get()
    @property
    def show_individual(self): return self._show_ind_var.get()


# ══════════════════════════════════════════════════════════════════════════════
#  GUI — Plot panel
# ══════════════════════════════════════════════════════════════════════════════

class PlotPanel(ttk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._spectra:  List[Spectrum] = []
        self._averaged: Optional[Spectrum] = None
        self._build()

    def _build(self):
        # ── Title editing bar ──────────────────────────────────────────────
        title_fr = ttk.Frame(self)
        title_fr.pack(fill='x', padx=6, pady=(4, 0))
        ttk.Label(title_fr, text="Plot title:").pack(side='left', padx=(0, 4))
        self._title_var = tk.StringVar(value="XAS Spectrum")
        title_ent = ttk.Entry(title_fr, textvariable=self._title_var, width=50)
        title_ent.pack(side='left', fill='x', expand=True)

        def _apply_title(*_):
            self._ax.set_title(self._title_var.get())
            self._fig.tight_layout()
            self._canvas.draw_idle()

        title_ent.bind('<Return>',   _apply_title)
        title_ent.bind('<FocusOut>', _apply_title)
        title_ent.bind('<Tab>',      _apply_title)
        ttk.Button(title_fr, text="Apply", width=6,
                   command=_apply_title).pack(side='left', padx=(4, 0))

        # ── Matplotlib figure ──────────────────────────────────────────────
        self._fig  = Figure(figsize=(7, 4.5), dpi=96)
        self._ax   = self._fig.add_subplot(111)
        self._ax.set_xlabel("Photon Energy (eV)")
        self._ax.set_ylabel("Normalized Intensity (arb. u.)")
        self._ax.set_title(self._title_var.get())
        self._fig.tight_layout()

        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)
        toolbar_fr = ttk.Frame(self)
        toolbar_fr.pack(fill='x')
        NavigationToolbar2Tk(self._canvas, toolbar_fr)

    def plot(self, spectra: List[Spectrum],
             averaged:    Optional[Spectrum] = None,
             show_individual: bool = True):
        self._spectra  = spectra
        self._averaged = averaged
        ax = self._ax
        ax.clear()

        if show_individual:
            for i, sp in enumerate(spectra):
                col = _COLORS[i % len(_COLORS)]
                ax.plot(sp.energy, sp.signal,
                        color=col, alpha=0.35, linewidth=0.9,
                        label=sp.label + " (raw)")

        if averaged is not None:
            ax.plot(averaged.energy, averaged.signal,
                    color='#d62728', linewidth=2.0,
                    label=averaged.label)
        elif spectra and not show_individual:
            # Single spectrum, show it normally
            ax.plot(spectra[0].energy, spectra[0].signal,
                    color='#1f77b4', linewidth=1.5,
                    label=spectra[0].label)
        elif spectra:
            # no average requested, show individual normally
            for i, sp in enumerate(spectra):
                col = _COLORS[i % len(_COLORS)]
                ax.plot(sp.energy, sp.signal,
                        color=col, linewidth=1.5,
                        label=sp.label)

        ax.set_xlabel("Photon Energy (eV)")
        norm_lbl = {'ring_current': '/ Ring Current', 'mcc_i0': '/ MCC I₀', 'none': '(raw counts)'}
        sp_ref = averaged or (spectra[0] if spectra else None)
        if sp_ref:
            sig_lbl  = sp_ref.signal_type
            norm_str = norm_lbl.get(sp_ref.norm_by, '')
            ax.set_ylabel(f"{sig_lbl} Intensity {norm_str}")
            # Auto-generate title only if the user hasn't typed a custom one
            auto_title = f"XAS Spectrum — {sp_ref.signal_type}"
            current    = self._title_var.get().strip()
            if not current or current == getattr(self, '_last_auto_title', ''):
                self._title_var.set(auto_title)
                ax.set_title(auto_title)
            else:
                ax.set_title(current)   # keep whatever the user typed
            self._last_auto_title = auto_title

        if spectra or averaged:
            ax.legend(fontsize=7, loc='upper right')

        ax.grid(True, alpha=0.25)
        self._fig.tight_layout()
        self._canvas.draw()

    def clear(self):
        self._ax.clear()
        self._canvas.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  GUI — Export panel
# ══════════════════════════════════════════════════════════════════════════════

class ExportPanel(ttk.Frame):
    def __init__(self, master, get_spectra_cb, **kw):
        super().__init__(master, **kw)
        self._get_spectra = get_spectra_cb
        self._build()

    def _build(self):
        pad = dict(padx=8, pady=4)

        ttk.Label(self,
                  text="Export processed spectrum(a) for use in the L-Edge Normalizer.",
                  wraplength=500, justify='left').pack(anchor='w', **pad)

        # Export options frame
        opt_fr = ttk.LabelFrame(self, text="Export Options")
        opt_fr.pack(fill='x', **pad)

        # What to export
        ttk.Label(opt_fr, text="Export:").grid(row=0, column=0, sticky='w', padx=6, pady=3)
        self._export_mode = tk.StringVar(value='averaged')
        ttk.Radiobutton(opt_fr, text="Averaged spectrum only",
                        variable=self._export_mode, value='averaged').grid(
            row=0, column=1, sticky='w')
        ttk.Radiobutton(opt_fr, text="Each individual scan (one CSV per scan)",
                        variable=self._export_mode, value='individual').grid(
            row=1, column=1, sticky='w')
        ttk.Radiobutton(opt_fr, text="Both averaged and individual",
                        variable=self._export_mode, value='both').grid(
            row=2, column=1, sticky='w')

        # Header row toggle
        self._include_header = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_fr, text="Include header row in CSV",
                        variable=self._include_header).grid(
            row=3, column=1, sticky='w', pady=(4, 0))

        # Output directory
        ttk.Label(opt_fr, text="Output folder:").grid(row=4, column=0, sticky='w',
                                                        padx=6, pady=3)
        out_fr = ttk.Frame(opt_fr)
        out_fr.grid(row=4, column=1, sticky='we', padx=2)
        self._out_dir = tk.StringVar(value=os.path.expanduser("~\\Desktop"))
        ttk.Entry(out_fr, textvariable=self._out_dir, width=36).pack(
            side='left', padx=2)
        ttk.Button(out_fr, text="…", width=2,
                   command=self._browse_out).pack(side='left')

        # Buttons
        btn_fr = ttk.Frame(self)
        btn_fr.pack(fill='x', **pad)
        ttk.Button(btn_fr, text="💾  Export CSV(s)",
                   command=self._export).pack(side='left', padx=4)
        ttk.Button(btn_fr, text="🔬  Export & Open in L-Edge Normalizer",
                   command=self._export_and_open).pack(side='left', padx=4)

        # Status
        self._status = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._status,
                  wraplength=500, foreground='#006600',
                  font=('TkDefaultFont', 8)).pack(anchor='w', padx=8)

        # Preview table
        tbl_fr = ttk.LabelFrame(self, text="Export Preview (first 15 rows)")
        tbl_fr.pack(fill='both', expand=True, **pad)
        cols = ('energy', 'signal')
        self._tbl = ttk.Treeview(tbl_fr, columns=cols, show='headings', height=12)
        self._tbl.heading('energy', text='Energy (eV)')
        self._tbl.heading('signal', text='Signal (norm.)')
        self._tbl.column('energy', width=120, anchor='center')
        self._tbl.column('signal', width=160, anchor='center')
        sb = ttk.Scrollbar(tbl_fr, command=self._tbl.yview)
        self._tbl.configure(yscrollcommand=sb.set)
        self._tbl.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

    def _browse_out(self):
        d = filedialog.askdirectory(
            initialdir=self._out_dir.get(),
            title="Choose output folder")
        if d:
            self._out_dir.set(d)

    def _write_csv(self, sp: Spectrum, path: str):
        with open(path, 'w', newline='') as fh:
            writer = _csv.writer(fh)
            if self._include_header.get():
                writer.writerow(['# energy_eV', sp.signal_type])
            for e, s in zip(sp.energy, sp.signal):
                writer.writerow([f"{e:.4f}", f"{s:.8g}"])

    def _export(self) -> List[str]:
        spectra, averaged = self._get_spectra()
        if not spectra and averaged is None:
            messagebox.showwarning("No data", "Process some stacks first.")
            return []

        out_dir = self._out_dir.get()
        os.makedirs(out_dir, exist_ok=True)
        mode  = self._export_mode.get()
        saved = []

        def _safe_name(s: str) -> str:
            return re.sub(r'[^\w\-.]', '_', s)

        if mode in ('averaged', 'both') and averaged is not None:
            fname = _safe_name(averaged.label) + f"_{averaged.signal_type}_avg.csv"
            path  = os.path.join(out_dir, fname)
            self._write_csv(averaged, path)
            saved.append(path)

        if mode in ('individual', 'both'):
            for i, sp in enumerate(spectra):
                fname = _safe_name(sp.label) + f"_{sp.signal_type}_{i+1:02d}.csv"
                path  = os.path.join(out_dir, fname)
                self._write_csv(sp, path)
                saved.append(path)

        if not saved and spectra:
            # If only individual scans exist (no average), save them
            for i, sp in enumerate(spectra):
                fname = _safe_name(sp.label) + f"_{sp.signal_type}_{i+1:02d}.csv"
                path  = os.path.join(out_dir, fname)
                self._write_csv(sp, path)
                saved.append(path)

        if saved:
            self._status.set("Saved:\n" + "\n".join(saved))

        # Populate preview table
        ref = averaged if averaged is not None else (spectra[0] if spectra else None)
        self._tbl.delete(*self._tbl.get_children())
        if ref is not None:
            for e, s in zip(ref.energy[:15], ref.signal[:15]):
                self._tbl.insert('', 'end', values=(f"{e:.4f}", f"{s:.6g}"))

        return saved

    def _export_and_open(self):
        saved = self._export()
        if not saved:
            return
        if not os.path.isfile(NORMALIZER):
            messagebox.showerror(
                "Not found",
                f"ledge_normalizer.py not found at:\n{NORMALIZER}")
            return
        # Launch normalizer in new process, passing CSV files as arguments
        subprocess.Popen(
            [sys.executable, NORMALIZER] + saved,
            cwd=THIS_DIR
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════════════════════════════

class SGMLoaderApp(tk.Toplevel):
    """Main window: SGM XAS Stack Loader.

    Can be used standalone (pass no master) or embedded inside another Tk app
    such as Binah (pass master=<parent_window>).

    Parameters
    ----------
    master : tk.Tk or tk.Widget, optional
        Parent window.  If None a hidden Tk root is created so the class can
        run its own mainloop() — this is the standalone mode used by
        ``if __name__ == "__main__"``.
    on_load_cb : callable, optional
        If provided, a "→ Send to Binah" button appears in the Export tab.
        Called with an ExperimentalScan-like object for each exported spectrum.
        Signature: on_load_cb(scan) where scan has .label, .energy_ev, .mu,
        .source_file, .e0, .is_normalized, .scan_type attributes.
    """

    def __init__(self, master=None, on_load_cb=None):
        # ── Standalone mode: create a hidden Tk root to own the event loop ──
        if master is None:
            self._hidden_root = tk.Tk()
            self._hidden_root.withdraw()
            master = self._hidden_root
        else:
            self._hidden_root = None

        super().__init__(master)
        self.title("SGM XAS Loader  —  Canadian Light Source")
        self.geometry("1280x760")
        self.resizable(True, True)
        self._on_load_cb = on_load_cb   # callback → Binah

        # Bring window to front
        self.lift()
        self.focus_force()

        if not _H5PY:
            messagebox.showwarning(
                "h5py not installed",
                "h5py is required to read .h5 files.\n\n"
                "Install with:  pip install h5py",
                parent=self)

        self._spectra:  List[Spectrum]      = []
        self._averaged: Optional[Spectrum]  = None
        self._load_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._build_menu()
        self._build_ui()

    def mainloop(self):
        """Run the event loop.  In standalone mode uses the hidden root."""
        if self._hidden_root is not None:
            self._hidden_root.mainloop()
        # Embedded mode: caller uses wait_window() — nothing extra needed

    # ── Menu ───────────────────────────────────────────────────────────────

    def _build_menu(self):
        mb = tk.Menu(self)
        self.config(menu=mb)

        fm = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="File", menu=fm)
        fm.add_command(label="Refresh stack list  Ctrl+R",
                       command=self._browser.refresh
                       if hasattr(self, '_browser') else lambda: None)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.destroy)

        hm = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="Help", menu=hm)
        hm.add_command(label="About", command=self._about)

    def _about(self):
        messagebox.showinfo(
            "About SGM XAS Loader",
            "SGM XAS Stack Loader\n\n"
            "Loads SGM beamline (CLS) stack data:\n"
            "  • HDF5 metadata (energy axis, ring current)\n"
            "  • SDD detector binary files (.bin)\n"
            "  • MCC analog channels (mcc_flyer.csv)\n\n"
            "Supported signals: TFY, PFY, TEY, PD\n"
            "Normalization: ring current or MCC I₀\n\n"
            "Export to CSV for ledge_normalizer.py."
        )

    # ── Layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        pw = ttk.PanedWindow(self, orient='horizontal')
        pw.pack(fill='both', expand=True)

        # ── Left pane: Stack browser
        left_fr = ttk.Frame(pw, width=310)
        left_fr.pack_propagate(False)
        pw.add(left_fr, weight=0)

        self._browser = StackBrowser(
            left_fr,
            on_selection_change=self._on_selection_change)
        self._browser.pack(fill='both', expand=True)

        # ── Right pane: Notebook
        right_fr = ttk.Frame(pw)
        pw.add(right_fr, weight=1)

        self._nb = ttk.Notebook(right_fr)
        self._nb.pack(fill='both', expand=True)

        # Tab 1: Process
        proc_fr = ttk.Frame(self._nb)
        self._nb.add(proc_fr, text="  Process  ")
        self._build_process_tab(proc_fr)

        # Tab 2: Spectra
        plot_fr = ttk.Frame(self._nb)
        self._nb.add(plot_fr, text="  Spectra  ")
        self._plot_panel = PlotPanel(plot_fr)
        self._plot_panel.pack(fill='both', expand=True)

        # Tab 3: Export
        exp_fr = ttk.Frame(self._nb)
        self._nb.add(exp_fr, text="  Export  ")
        self._export_panel = ExportPanel(
            exp_fr,
            get_spectra_cb=lambda: (self._spectra, self._averaged))
        self._export_panel.pack(fill='both', expand=True)

        # "Send to Binah" button — only shown when launched from Binah
        if self._on_load_cb is not None:
            _btn_bar = tk.Frame(exp_fr, bg="#003366", pady=4)
            _btn_bar.pack(fill='x', side='bottom')
            tk.Button(
                _btn_bar, text="\u2794  Send to Binah",
                font=("", 10, "bold"), bg="#FFB347", fg="black",
                activebackground="#FFA000",
                command=self._send_to_binah
            ).pack(padx=8, pady=2)

        # Rebuild menu now that _browser is available
        self._build_menu()

        # Status bar
        self._status_var = tk.StringVar(value="Ready.")
        sb = ttk.Label(self, textvariable=self._status_var,
                       relief='sunken', anchor='w')
        sb.pack(side='bottom', fill='x')
        self._prog = ttk.Progressbar(self, mode='determinate', maximum=100)
        self._prog.pack(side='bottom', fill='x')

    def _build_process_tab(self, parent):
        # Settings
        self._settings = SettingsPanel(parent)
        self._settings.pack(fill='x', padx=8, pady=8)

        # Selection summary
        self._sel_info_var = tk.StringVar(
            value="No stacks selected.  Select stacks in the left panel, "
                  "then click Process.")
        ttk.Label(parent, textvariable=self._sel_info_var,
                  wraplength=600, justify='left',
                  font=('TkDefaultFont', 8), foreground='#444').pack(
            anchor='w', padx=12)

        # Process / Stop buttons
        btn_fr = ttk.Frame(parent)
        btn_fr.pack(anchor='w', padx=12, pady=6)
        self._proc_btn = ttk.Button(btn_fr, text="▶  Process Selected Stacks",
                                    command=self._start_processing)
        self._proc_btn.pack(side='left', padx=4)
        self._stop_btn = ttk.Button(btn_fr, text="■  Stop",
                                    command=self._stop_processing, state='disabled')
        self._stop_btn.pack(side='left', padx=4)
        ttk.Button(btn_fr, text="Clear",
                   command=self._clear).pack(side='left', padx=4)

        ttk.Separator(btn_fr, orient='vertical').pack(side='left', fill='y',
                                                       padx=10)
        ttk.Button(btn_fr,
                   text="Ex/Em Heatmap",
                   command=self._open_exem_heatmap,
                   ).pack(side='left', padx=4)

        # Log
        log_fr = ttk.LabelFrame(parent, text="Processing Log")
        log_fr.pack(fill='both', expand=True, padx=8, pady=4)
        self._log = tk.Text(log_fr, height=8, state='disabled',
                            font=('Consolas', 8), bg='#f5f5f5')
        log_sb = ttk.Scrollbar(log_fr, command=self._log.yview)
        self._log.configure(yscrollcommand=log_sb.set)
        self._log.pack(side='left', fill='both', expand=True)
        log_sb.pack(side='right', fill='y')

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _on_selection_change(self, selected: List[StackInfo]):
        n_data = sum(1 for s in selected if s.has_data)
        n_tot  = len(selected)
        if n_tot == 0:
            self._sel_info_var.set(
                "No stacks selected. Select stacks in the left panel.")
        else:
            self._sel_info_var.set(
                f"{n_tot} stack(s) selected — "
                f"{n_data} have .bin data available, "
                f"{n_tot - n_data} without local data (will be skipped).")

    def _log_msg(self, msg: str):
        self._log.configure(state='normal')
        self._log.insert('end', msg + '\n')
        self._log.see('end')
        self._log.configure(state='disabled')

    def _set_status(self, msg: str):
        self._status_var.set(msg)
        self.update_idletasks()

    def _set_progress(self, pct: int):
        self._prog['value'] = pct
        self.update_idletasks()

    def _clear(self):
        self._spectra  = []
        self._averaged = None
        self._plot_panel.clear()
        self._log.configure(state='normal')
        self._log.delete('1.0', 'end')
        self._log.configure(state='disabled')
        self._set_status("Cleared.")
        self._set_progress(0)

    # ── Processing ─────────────────────────────────────────────────────────

    def _start_processing(self):
        selected = self._browser.selected_infos()
        to_process = [s for s in selected if s.has_data]

        if not to_process:
            messagebox.showwarning(
                "No data",
                "None of the selected stacks have local .bin data.\n\n"
                "Select stacks shown in bold — those have SDD detector files.")
            return

        self._stop_event.clear()
        self._proc_btn.configure(state='disabled')
        self._stop_btn.configure(state='normal')

        sig  = self._settings.signal_type
        det  = self._settings.detectors
        roi_lo = self._settings.roi_lo
        roi_hi = self._settings.roi_hi
        norm = self._settings.norm_by
        do_avg = self._settings.do_average
        show_ind = self._settings.show_individual

        def _worker():
            new_spectra = []
            for k, info in enumerate(to_process):
                if self._stop_event.is_set():
                    break
                self.after(0, self._log_msg, f"[{k+1}/{len(to_process)}] {info.sample}  ({info.name})")
                self.after(0, self._set_status, f"Processing {info.sample}…")

                def _prog_cb(pct, _k=k, _n=len(to_process)):
                    overall = int(100 * (_k + pct / 100) / _n)
                    self.after(0, self._set_progress, overall)

                try:
                    sp = build_spectrum(
                        info,
                        signal_type = sig,
                        detectors   = det,
                        roi_lo      = roi_lo,
                        roi_hi      = roi_hi,
                        norm_by     = norm,
                        progress_cb = _prog_cb,
                        stop_event  = self._stop_event,
                    )
                    sp.color = _COLORS[len(new_spectra) % len(_COLORS)]
                    new_spectra.append(sp)
                    self.after(0, self._log_msg,
                               f"  → {len(sp.energy)} pts, "
                               f"E={sp.energy.min():.1f}–{sp.energy.max():.1f} eV, "
                               f"signal range {sp.signal.min():.4g}–{sp.signal.max():.4g}")
                except Exception as exc:
                    self.after(0, self._log_msg, f"  ✗ ERROR: {exc}")

            # Compute average
            avg = None
            if do_avg and len(new_spectra) > 1:
                try:
                    avg = average_spectra(new_spectra)
                    self.after(0, self._log_msg,
                               f"Averaged {len(new_spectra)} spectra → "
                               f"{len(avg.energy)} pts on common grid.")
                except Exception as exc:
                    self.after(0, self._log_msg, f"  ✗ Average failed: {exc}")

            self.after(0, self._finish_processing, new_spectra, avg, show_ind)

        self._load_thread = threading.Thread(target=_worker, daemon=True)
        self._load_thread.start()

    def _finish_processing(self, spectra, averaged, show_individual):
        self._spectra   = spectra
        self._averaged  = averaged

        self._proc_btn.configure(state='normal')
        self._stop_btn.configure(state='disabled')
        self._set_progress(100)
        self._set_status(
            f"Done — {len(spectra)} scan(s) processed"
            + (f", averaged." if averaged else "."))

        self._log_msg("─" * 60)

        # Update plot
        show_ind = show_individual and len(spectra) > 1
        self._plot_panel.plot(spectra, averaged, show_individual=show_ind)

        # Switch to Spectra tab
        self._nb.select(1)

    def _open_exem_heatmap(self):
        """
        Open the Emission/Excitation matrix heatmap for the first selected
        stack that has binary SDD data.

        Uses the current detector selection and PFY ROI from SettingsPanel
        so the ROI overlay lines match the spectrum being processed.
        """
        selected  = self._browser.selected_infos()
        with_data = [s for s in selected if s.has_data]
        if not with_data:
            messagebox.showwarning(
                "No data",
                "Select at least one stack shown in bold (with .bin data) "
                "to open the Ex/Em heatmap.",
                parent=self)
            return

        info = with_data[0]
        if len(with_data) > 1:
            self._log_msg(
                f"Ex/Em Heatmap: multiple stacks selected — "
                f"showing '{info.name}' (first with data).")

        det   = self._settings.detectors
        lo    = self._settings.roi_lo
        hi    = self._settings.roi_hi

        ExEmHeatmapWindow(self, info,
                          detectors=det,
                          roi_lo=lo,
                          roi_hi=hi)

    def _stop_processing(self):
        self._stop_event.set()
        self._set_status("Stopping…")
        self._stop_btn.configure(state='disabled')

    def _send_to_binah(self):
        """Send processed spectra back to Binah via the on_load_cb callback."""
        if self._on_load_cb is None:
            return

        spectra, averaged = self._spectra, self._averaged
        to_send = []

        if averaged is not None:
            to_send.append(averaged)
        elif spectra:
            to_send.extend(spectra)

        if not to_send:
            messagebox.showwarning("No data",
                                   "Process some stacks first, then send to Binah.",
                                   parent=self)
            return

        # Build minimal ExperimentalScan-compatible objects and call back
        try:
            from experimental_parser import ExperimentalScan
        except ImportError:
            # Build a lightweight stand-in with the required attributes
            from dataclasses import dataclass
            import numpy as _np

            @dataclass
            class ExperimentalScan:
                label: str
                source_file: str
                energy_ev: _np.ndarray
                mu: _np.ndarray
                e0: float = 0.0
                is_normalized: bool = False
                scan_type: str = "SGM"

        n = 0
        for sp in to_send:
            mu = sp.signal.copy()
            mu = mu - mu.min()   # baseline at 0
            scan = ExperimentalScan(
                label=sp.label,
                source_file=sp.stack_path,
                energy_ev=sp.energy.copy(),
                mu=mu,
                e0=0.0,
                is_normalized=False,
                scan_type=f"SGM {sp.signal_type}",
            )
            try:
                self._on_load_cb(scan)
                n += 1
            except Exception as exc:
                messagebox.showerror("Send Error",
                                     f"Failed to send '{sp.label}':\n{exc}",
                                     parent=self)

        if n:
            self._set_status(f"Sent {n} spectrum/spectra to Binah.")
            messagebox.showinfo("Sent to Binah",
                                f"{n} spectrum/spectra added to Binah.\n"
                                "Check the Spectra tab.",
                                parent=self)

    def destroy(self):
        """Destroy this window and, in standalone mode, the hidden root."""
        super().destroy()
        if self._hidden_root is not None:
            try:
                self._hidden_root.destroy()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = SGMLoaderApp()
    app.mainloop()


if __name__ == "__main__":
    main()
