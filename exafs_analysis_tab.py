"""
Dedicated EXAFS studio for Binah.

This module adds a focused EXAFS workspace with q-space, R-space,
windowing, q/R overlap, and FEFF working-directory support.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

import matplotlib
import matplotlib.ticker as mticker
import numpy as np
import xas_analysis_tab as xas_core
import feff_manager
from experimental_parser import ExperimentalScan
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from structure_converter import export_xyz_as_feff_bundle, parse_xyz_file

matplotlib.use("TkAgg")

try:
    import seaborn as sns

    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


WINDOW_TYPES = ("Hanning", "Sine", "Welch", "Parzen")
FEFF_EXE_CANDIDATES = ("feff8l.exe", "feff.exe", "feff85l.exe", "feff9.exe", "feff")


@dataclass
class FeffPathData:
    index: int
    filename: str
    label: str
    reff: float
    degen: float
    nleg: int
    q: np.ndarray
    amp: np.ndarray


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_optional_radius(value) -> float | None:
    """Parse a radius entry: empty/blank -> None, valid number -> float."""
    text = str(value).strip()
    if not text:
        return None
    try:
        r = float(text)
    except Exception:
        return None
    return r if r > 0 else None


def _resolve_absorber_index(text, structure) -> tuple[int, str]:
    """Resolve a free-form absorber spec to a 1-based atom index.

    Accepts:
      * "" / None     -> default to atom 1
      * "5" / "  3 "  -> use as 1-based index (after bounds check)
      * "Ni" / "ni"   -> first atom whose element symbol matches (case-
                         insensitive). If multiple atoms match, the first one
                         is used and `note` describes how many were found.

    Returns (index_1based, note). Raises ValueError on invalid input or when
    an element symbol has no match in the structure.
    """
    if structure is None:
        raise ValueError("Load an XYZ structure first.")

    raw = "" if text is None else str(text).strip()
    n = int(structure.atom_count)
    if not raw:
        return 1, ""

    # Numeric path: explicit atom index.
    try:
        idx = int(raw)
    except ValueError:
        idx = None
    if idx is not None:
        if idx < 1 or idx > n:
            raise ValueError(
                f"Absorber index {idx} is out of range (1..{n})."
            )
        sym = structure.symbols[idx - 1]
        return idx, f"using atom #{idx} ({sym})"

    # Element-symbol path. Canonicalize so e.g. "ni" -> "Ni".
    token = raw
    if len(token) == 1:
        target = token.upper()
    else:
        target = token[0].upper() + token[1:].lower()

    matches = [i + 1 for i, s in enumerate(structure.symbols) if s == target]
    if not matches:
        raise ValueError(
            f"No '{target}' atom in structure (elements present: "
            f"{', '.join(sorted(set(structure.symbols)))})."
        )
    chosen = matches[0]
    if len(matches) == 1:
        note = f"using {target} at atom #{chosen}"
    else:
        note = (f"using first {target} at atom #{chosen} "
                f"(also at #{', #'.join(str(m) for m in matches[1:])})")
    return chosen, note


def _next_pow_two(value: int) -> int:
    out = 1
    while out < value:
        out <<= 1
    return out


def _window_ramp(frac: np.ndarray, kind: str) -> np.ndarray:
    frac = np.clip(np.asarray(frac, dtype=float), 0.0, 1.0)
    key = str(kind).strip().lower()
    if key == "sine":
        return np.sin(0.5 * np.pi * frac)
    if key == "welch":
        return 1.0 - (1.0 - frac) ** 2
    if key == "parzen":
        return frac * frac * (3.0 - 2.0 * frac)
    return 0.5 - 0.5 * np.cos(np.pi * frac)


def build_tapered_window(axis: np.ndarray, lo: float, hi: float,
                         taper: float, kind: str) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    out = np.zeros_like(axis)
    if len(axis) == 0:
        return out

    lo_eff = max(float(lo), float(axis[0]))
    hi_eff = min(float(hi), float(axis[-1]))
    if hi_eff <= lo_eff:
        return out

    taper = max(float(taper), 0.0)
    span = hi_eff - lo_eff
    if taper <= 1e-12:
        out[(axis >= lo_eff) & (axis <= hi_eff)] = 1.0
        return out

    if span <= 2.0 * taper:
        mid = 0.5 * (lo_eff + hi_eff)
        half = 0.5 * span
        if half <= 1e-12:
            return out
        frac = 1.0 - np.abs(axis - mid) / half
        mask = frac > 0.0
        out[mask] = _window_ramp(frac[mask], kind)
        return np.clip(out, 0.0, 1.0)

    left_flat = lo_eff + taper
    right_flat = hi_eff - taper
    core = (axis >= left_flat) & (axis <= right_flat)
    out[core] = 1.0

    left = (axis >= lo_eff) & (axis < left_flat)
    if np.any(left):
        out[left] = _window_ramp((axis[left] - lo_eff) / taper, kind)

    right = (axis > right_flat) & (axis <= hi_eff)
    if np.any(right):
        out[right] = _window_ramp((hi_eff - axis[right]) / taper, kind)

    return np.clip(out, 0.0, 1.0)


def compute_transform_bundle(q: np.ndarray, chi: np.ndarray,
                             qmin: float, qmax: float, dq: float,
                             qweight: int, qwin_kind: str,
                             rmin: float, rmax: float, dr: float,
                             rwin_kind: str) -> dict:
    q = np.asarray(q, dtype=float)
    chi = np.asarray(chi, dtype=float)
    finite = np.isfinite(q) & np.isfinite(chi)
    q = q[finite]
    chi = chi[finite]

    if len(q) < 4:
        return {
            "q_uniform": np.array([], dtype=float),
            "chi_uniform": np.array([], dtype=float),
            "chi_weighted": np.array([], dtype=float),
            "q_window": np.array([], dtype=float),
            "r": np.array([], dtype=float),
            "chir": np.array([], dtype=complex),
            "chi_r_mag": np.array([], dtype=float),
            "chi_r_selected_mag": np.array([], dtype=float),
            "r_window": np.array([], dtype=float),
            "chi_back": np.array([], dtype=float),
            "chi_weighted_back": np.array([], dtype=float),
        }

    order = np.argsort(q)
    q = q[order]
    chi = chi[order]
    q, unique_idx = np.unique(q, return_index=True)
    chi = chi[unique_idx]

    q_step = np.median(np.diff(q)) if len(q) > 1 else 0.05
    q_step = max(float(q_step), 0.01)

    q_uniform = np.arange(max(0.0, float(q[0])), float(q[-1]) + q_step * 0.1, q_step)
    chi_uniform = np.interp(q_uniform, q, chi)
    q_window = build_tapered_window(q_uniform, qmin, qmax, dq, qwin_kind)

    if int(qweight) == 0:
        chi_weighted = chi_uniform.copy()
    else:
        chi_weighted = chi_uniform * np.power(q_uniform, int(qweight))

    nfft = max(2048, _next_pow_two(max(16, len(q_uniform) * 4)))
    fft_in = np.zeros(nfft, dtype=float)
    npts = min(len(q_uniform), nfft)
    fft_in[:npts] = chi_weighted[:npts] * q_window[:npts]

    chir = np.fft.rfft(fft_in) * q_step / np.sqrt(np.pi)
    r_step = np.pi / (q_step * nfft)
    r = r_step * np.arange(len(chir))
    r_window = build_tapered_window(r, rmin, rmax, dr, rwin_kind)
    chir_selected = chir * r_window

    chi_weighted_back = np.fft.irfft(chir_selected, n=nfft) * np.sqrt(np.pi) / q_step
    chi_weighted_back = chi_weighted_back[:len(q_uniform)]

    chi_back = np.zeros_like(q_uniform)
    if int(qweight) == 0:
        chi_back = chi_weighted_back.copy()
    else:
        safe = q_uniform > 1e-9
        chi_back[safe] = chi_weighted_back[safe] / np.power(q_uniform[safe], int(qweight))

    return {
        "q_uniform": q_uniform,
        "chi_uniform": chi_uniform,
        "chi_weighted": chi_weighted,
        "q_window": q_window,
        "r": r,
        "chir": chir,
        "chi_r_mag": np.abs(chir),
        "chi_r_selected_mag": np.abs(chir_selected),
        "r_window": r_window,
        "chi_back": chi_back,
        "chi_weighted_back": chi_weighted_back,
    }


def parse_feff_path_file(path: str) -> FeffPathData:
    file_path = Path(path)
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    index_match = re.search(r"(\d+)", file_path.stem)
    index = int(index_match.group(1)) if index_match else 0

    label = file_path.stem
    reff = float("nan")
    degen = float("nan")
    nleg = 0
    amp_col = 2
    data_started = False
    q_vals: list[float] = []
    amp_vals: list[float] = []

    for raw in lines:
        line = raw.strip()
        clean = line.lstrip("#").strip()
        if not clean:
            continue

        if "path" in clean.lower() and ":" in clean:
            left, right = clean.split(":", 1)
            if "path" in left.lower() and right.strip():
                label = right.strip()

        for key, dest in (("reff", "reff"), ("degen", "degen"), ("nleg", "nleg")):
            match = re.search(rf"\b{key}\b\s*=\s*([^\s,]+)", clean, re.IGNORECASE)
            if not match:
                continue
            val = match.group(1)
            if dest == "nleg":
                try:
                    nleg = int(float(val))
                except Exception:
                    pass
            elif dest == "degen":
                degen = _coerce_float(val, degen)
            else:
                reff = _coerce_float(val, reff)

        header_candidate = re.split(r"\s+", clean)
        if header_candidate and header_candidate[0].lower() == "k" and any("mag" in tok.lower() for tok in header_candidate):
            for i, tok in enumerate(header_candidate):
                if "mag" in tok.lower():
                    amp_col = i
                    break
            data_started = True
            continue

        if not data_started:
            numeric_head = re.match(
                r"^\s*(\d+)\s+(\d+)\s+([+-]?\d+(?:\.\d*)?(?:[EeDd][+-]?\d+)?)\s+([+-]?\d+(?:\.\d*)?(?:[EeDd][+-]?\d+)?)",
                raw,
            )
            if numeric_head and (np.isnan(degen) or np.isnan(reff) or nleg <= 0):
                try:
                    nleg = int(numeric_head.group(2))
                    degen = float(numeric_head.group(3).replace("D", "E").replace("d", "e"))
                    reff = float(numeric_head.group(4).replace("D", "E").replace("d", "e"))
                except Exception:
                    pass
            continue

        if re.match(r"^[A-Za-z]", clean):
            continue
        parts = re.split(r"\s+", clean.replace("D", "E").replace("d", "e"))
        if len(parts) < 2:
            continue
        try:
            vals = [float(part) for part in parts]
        except Exception:
            continue
        q_vals.append(vals[0])
        amp_vals.append(vals[min(amp_col, len(vals) - 1)])

    q_arr = np.asarray(q_vals, dtype=float)
    amp_arr = np.asarray(amp_vals, dtype=float)
    if not np.isfinite(reff):
        reff = 0.0
    if not np.isfinite(degen):
        degen = 0.0

    return FeffPathData(
        index=index,
        filename=file_path.name,
        label=label,
        reff=float(reff),
        degen=float(degen),
        nleg=int(nleg),
        q=q_arr,
        amp=amp_arr,
    )


class EXAFSAnalysisTab(tk.Frame):
    _SCAN_COLOURS = xas_core._PALETTE

    def __init__(self, parent, get_scans_fn: Callable,
                 replot_fn: Optional[Callable] = None,
                 add_exp_scan_fn: Optional[Callable] = None,
                 *,
                 show_analysis_panel: bool = True,
                 show_feff_panel: bool = True,
                 feff_paths_provider: Optional[Callable] = None):
        super().__init__(parent)
        self._get_scans = get_scans_fn
        self._replot_fn = replot_fn
        # Callback for handing a freshly-computed FEFF spectrum back to the
        # main app so it appears as an experimental scan in the plot widget
        # (and is therefore captured by project save).
        self._add_exp_scan_fn = add_exp_scan_fn

        # Panel visibility flags. The same class is now used in two places:
        # EXAFS Studio tab (analysis only) and Simulation Studio tab (FEFF
        # engine only). Defaults preserve the original "everything visible"
        # behaviour for any external caller.
        self._show_analysis_panel = bool(show_analysis_panel)
        self._show_feff_panel = bool(show_feff_panel)
        # Optional callback that returns the active FEFF path list when the
        # FEFF panel lives in another tab (Simulation Studio). EXAFS Studio's
        # R-space marker overlay reads from this provider so it can keep
        # rendering FEFF reference markers even though the FEFF UI moved.
        self._feff_paths_provider = feff_paths_provider

        self._results: dict = {}
        self._selected_labels: list[str] = []
        self._scan_vis_vars: dict[str, tk.BooleanVar] = {}
        self._feff_paths: list[FeffPathData] = []
        self._xyz_structure = None
        self._pending_selected_labels: list[str] = []
        self._build_ui()

    # Public hook so binah.py can wire the marker provider after both tabs
    # exist (Simulation Studio is constructed AFTER EXAFS Studio).
    def set_feff_paths_provider(self, fn: Optional[Callable]) -> None:
        self._feff_paths_provider = fn

    # Public accessor used as the marker provider when this tab owns the FEFF
    # panel (Simulation Studio's inner FEFF instance is the live source).
    def get_feff_paths(self) -> list[FeffPathData]:
        return list(self._feff_paths)

    def _build_ui(self):
        top = tk.Frame(self, bd=1, relief=tk.GROOVE, padx=4, pady=3)
        top.pack(side=tk.TOP, fill=tk.X)

        # Always create the scan combobox — the FEFF panel uses it for some
        # contextual operations even when the analysis panel is hidden, and
        # refresh_scan_list() touches it unconditionally.
        tk.Label(top, text="Scan:", font=("", 9, "bold")).pack(side=tk.LEFT)
        self._scan_var = tk.StringVar()
        self._scan_cb = ttk.Combobox(top, textvariable=self._scan_var,
                                     state="readonly", width=38)
        self._scan_cb.pack(side=tk.LEFT, padx=(4, 8))
        self._scan_cb.bind("<<ComboboxSelected>>", lambda _e: self._auto_fill_e0())

        tk.Button(top, text="Refresh Scans", font=("", 8),
                  command=self.refresh_scan_list).pack(side=tk.LEFT, padx=2)
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # Analysis-only toolbar buttons. Hidden when this instance is being
        # hosted inside Simulation Studio (FEFF-only mode).
        if self._show_analysis_panel:
            tk.Button(top, text="Run EXAFS", bg="#003366", fg="black",
                      font=("", 9, "bold"), command=self._run).pack(side=tk.LEFT, padx=2)
            tk.Button(top, text="Update Views", font=("", 8),
                      command=self._redraw).pack(side=tk.LEFT, padx=2)
            tk.Button(top, text="+ Add to Overlay", font=("", 8),
                      command=self._add_overlay).pack(side=tk.LEFT, padx=2)
            tk.Button(top, text="Clear Overlay", font=("", 8),
                      command=self._clear_overlay).pack(side=tk.LEFT, padx=2)

        status_msg = (
            "Load experimental scans first, then run EXAFS on the scan of interest."
            if self._show_analysis_panel else
            "Pick a workdir + executable, load an XYZ, then write the bundle and run."
        )
        self._status_lbl = tk.Label(
            top, text=status_msg, fg="gray", font=("", 8),
        )
        self._status_lbl.pack(side=tk.LEFT, padx=10)

        body = tk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        if self._show_analysis_panel:
            self._build_params(body)
            self._build_scan_list(body)
        self._build_views(body)

    def _build_params(self, parent):
        pf = tk.Frame(parent, width=260, bd=1, relief=tk.SUNKEN, padx=5, pady=5)
        pf.pack(side=tk.LEFT, fill=tk.Y, padx=(2, 0), pady=2)
        pf.pack_propagate(False)

        def lbl(text):
            tk.Label(pf, text=text, font=("", 8, "bold"), fg="#333333",
                     anchor="w").pack(fill=tk.X, pady=(6, 0))

        def row(text, var, from_=None, to=None, inc=None, fmt=None, width=8):
            frame = tk.Frame(pf)
            frame.pack(fill=tk.X, pady=1)
            tk.Label(frame, text=text, width=14, anchor="w",
                     font=("", 8)).pack(side=tk.LEFT)
            if from_ is not None:
                ttk.Spinbox(
                    frame,
                    textvariable=var,
                    from_=from_,
                    to=to,
                    increment=inc,
                    format=fmt or "%.2f",
                    width=width,
                    font=("Courier", 8),
                ).pack(side=tk.LEFT)
            else:
                ttk.Entry(frame, textvariable=var, width=width,
                          font=("Courier", 8)).pack(side=tk.LEFT)

        self._e0_var = tk.DoubleVar(value=8333.0)
        self._pre1_var = tk.DoubleVar(value=float(xas_core._NORM_DEFAULTS["pre1"]))
        self._pre2_var = tk.DoubleVar(value=float(xas_core._NORM_DEFAULTS["pre2"]))
        self._nor1_var = tk.DoubleVar(value=float(xas_core._NORM_DEFAULTS["nor1"]))
        self._nor2_var = tk.DoubleVar(value=float(xas_core._NORM_DEFAULTS["nor2"]))
        self._nnorm_var = tk.IntVar(value=int(xas_core._NORM_DEFAULTS["nnorm"]))
        self._rbkg_var = tk.DoubleVar(value=float(xas_core._NORM_DEFAULTS["rbkg"]))
        self._kmin_bkg_var = tk.DoubleVar(value=float(xas_core._NORM_DEFAULTS["kmin_bkg"]))

        self._qmin_var = tk.DoubleVar(value=float(xas_core._NORM_DEFAULTS["kmin"]))
        self._qmax_var = tk.DoubleVar(value=float(xas_core._NORM_DEFAULTS["kmax"]))
        self._dq_var = tk.DoubleVar(value=float(xas_core._NORM_DEFAULTS["dk"]))
        self._qweight_var = tk.IntVar(value=int(xas_core._NORM_DEFAULTS["kw"]))
        self._qwin_var = tk.StringVar(value="Hanning")

        self._rmin_var = tk.DoubleVar(value=1.0)
        self._rmax_var = tk.DoubleVar(value=3.2)
        self._dr_var = tk.DoubleVar(value=0.5)
        self._rwin_var = tk.StringVar(value="Hanning")
        self._rdisplay_var = tk.DoubleVar(value=float(xas_core._NORM_DEFAULTS["rmax"]))

        self._style_var = tk.StringVar(value="ticks")
        self._context_var = tk.StringVar(value="paper")
        self._use_q_label_var = tk.BooleanVar(value=True)
        self._show_q_window_var = tk.BooleanVar(value=True)
        self._show_r_window_var = tk.BooleanVar(value=True)
        self._show_feff_markers_var = tk.BooleanVar(value=True)

        lbl("Edge / Background")
        row("E0 (eV):", self._e0_var, 100, 40000, 0.5, "%.1f")
        row("pre1 (eV):", self._pre1_var, -300, -1, 5.0, "%.0f")
        row("pre2 (eV):", self._pre2_var, -200, -1, 5.0, "%.0f")
        row("nor1 (eV):", self._nor1_var, 1, 500, 5.0, "%.0f")
        row("nor2 (eV):", self._nor2_var, 1, 1000, 5.0, "%.0f")
        row("rbkg (A):", self._rbkg_var, 0.3, 3.0, 0.1, "%.1f")
        row("kmin bkg:", self._kmin_bkg_var, 0.0, 5.0, 0.5, "%.1f")

        nnorm_frame = tk.Frame(pf)
        nnorm_frame.pack(fill=tk.X, pady=1)
        tk.Label(nnorm_frame, text="Norm order:", width=14, anchor="w",
                 font=("", 8)).pack(side=tk.LEFT)
        for value in (1, 2):
            tk.Radiobutton(nnorm_frame, text=str(value), value=value,
                           variable=self._nnorm_var,
                           font=("", 8)).pack(side=tk.LEFT)

        lbl("Q Space / Window")
        row("q min (A^-1):", self._qmin_var, 0.0, 20.0, 0.5, "%.1f")
        row("q max (A^-1):", self._qmax_var, 1.0, 24.0, 0.5, "%.1f")
        row("dq taper:", self._dq_var, 0.1, 4.0, 0.1, "%.1f")

        qweight_frame = tk.Frame(pf)
        qweight_frame.pack(fill=tk.X, pady=1)
        tk.Label(qweight_frame, text="q-weight:", width=14, anchor="w",
                 font=("", 8)).pack(side=tk.LEFT)
        for value in (1, 2, 3):
            tk.Radiobutton(qweight_frame, text=str(value), value=value,
                           variable=self._qweight_var,
                           font=("", 8)).pack(side=tk.LEFT)

        qwin_frame = tk.Frame(pf)
        qwin_frame.pack(fill=tk.X, pady=1)
        tk.Label(qwin_frame, text="q window:", width=14, anchor="w",
                 font=("", 8)).pack(side=tk.LEFT)
        ttk.Combobox(qwin_frame, textvariable=self._qwin_var, width=11,
                     state="readonly", values=WINDOW_TYPES).pack(side=tk.LEFT)

        qbtn = tk.Frame(pf)
        qbtn.pack(fill=tk.X, pady=(2, 1))
        tk.Button(qbtn, text="q from Plot", font=("", 8),
                  command=self._capture_q_window_from_plot).pack(side=tk.LEFT)
        tk.Button(qbtn, text="Default q", font=("", 8),
                  command=self._reset_q_window).pack(side=tk.LEFT, padx=4)

        lbl("R Space / Window")
        row("R min (A):", self._rmin_var, 0.0, 8.0, 0.1, "%.1f")
        row("R max (A):", self._rmax_var, 0.5, 12.0, 0.1, "%.1f")
        row("dR taper:", self._dr_var, 0.05, 2.0, 0.05, "%.2f", width=9)
        row("R display:", self._rdisplay_var, 2.0, 12.0, 0.5, "%.1f")

        rwin_frame = tk.Frame(pf)
        rwin_frame.pack(fill=tk.X, pady=1)
        tk.Label(rwin_frame, text="R window:", width=14, anchor="w",
                 font=("", 8)).pack(side=tk.LEFT)
        ttk.Combobox(rwin_frame, textvariable=self._rwin_var, width=11,
                     state="readonly", values=WINDOW_TYPES).pack(side=tk.LEFT)

        rbtn = tk.Frame(pf)
        rbtn.pack(fill=tk.X, pady=(2, 1))
        tk.Button(rbtn, text="R from Plot", font=("", 8),
                  command=self._capture_r_window_from_plot).pack(side=tk.LEFT)
        tk.Button(rbtn, text="Default R", font=("", 8),
                  command=self._reset_r_window).pack(side=tk.LEFT, padx=4)

        lbl("Display")
        style_frame = tk.Frame(pf)
        style_frame.pack(fill=tk.X, pady=1)
        tk.Label(style_frame, text="Style:", width=14, anchor="w",
                 font=("", 8)).pack(side=tk.LEFT)
        ttk.Combobox(style_frame, textvariable=self._style_var, width=11,
                     state="readonly",
                     values=["ticks", "whitegrid", "darkgrid", "white", "dark"]
                     ).pack(side=tk.LEFT)

        context_frame = tk.Frame(pf)
        context_frame.pack(fill=tk.X, pady=1)
        tk.Label(context_frame, text="Context:", width=14, anchor="w",
                 font=("", 8)).pack(side=tk.LEFT)
        ttk.Combobox(context_frame, textvariable=self._context_var, width=11,
                     state="readonly",
                     values=["paper", "notebook", "talk", "poster"]
                     ).pack(side=tk.LEFT)

        for text, var in [
            ("Label k-space as q", self._use_q_label_var),
            ("Show q window", self._show_q_window_var),
            ("Show R window", self._show_r_window_var),
            ("Show FEFF markers", self._show_feff_markers_var),
        ]:
            tk.Checkbutton(pf, text=text, variable=var,
                           command=self._redraw,
                           font=("", 8)).pack(anchor="w", pady=1)

        tk.Button(
            pf,
            text="Run / Refresh EXAFS",
            font=("", 9, "bold"),
            bg="#003366",
            fg="black",
            activebackground="#0055aa",
            command=self._run,
        ).pack(fill=tk.X, pady=(8, 2))
        tk.Button(
            pf,
            text="Redraw Windows Only",
            font=("", 8),
            command=self._redraw,
        ).pack(fill=tk.X, pady=(0, 2))

    def _build_scan_list(self, parent):
        outer = tk.Frame(parent, width=190, bd=1, relief=tk.SUNKEN)
        outer.pack(side=tk.LEFT, fill=tk.Y, padx=(2, 0), pady=2)
        outer.pack_propagate(False)

        hdr = tk.Frame(outer, bg="#003366", pady=3)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Loaded Scans", font=("", 8, "bold"),
                 bg="#003366", fg="black").pack(side=tk.LEFT, padx=6)
        tk.Button(hdr, text="All", font=("", 7), pady=0, padx=3,
                  command=self._show_all_scans).pack(side=tk.RIGHT, padx=2)
        tk.Button(hdr, text="None", font=("", 7), pady=0, padx=3,
                  command=self._hide_all_scans).pack(side=tk.RIGHT, padx=1)

        wrap = tk.Frame(outer)
        wrap.pack(fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(wrap, orient=tk.VERTICAL)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._scan_list_canvas = tk.Canvas(
            wrap, yscrollcommand=vsb.set, bg="white", highlightthickness=0
        )
        self._scan_list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.config(command=self._scan_list_canvas.yview)

        self._scan_list_inner = tk.Frame(self._scan_list_canvas, bg="white")
        self._scan_list_window = self._scan_list_canvas.create_window(
            (0, 0), window=self._scan_list_inner, anchor="nw"
        )

        self._scan_list_inner.bind(
            "<Configure>",
            lambda _e: self._scan_list_canvas.configure(
                scrollregion=self._scan_list_canvas.bbox("all"))
        )
        self._scan_list_canvas.bind(
            "<Configure>",
            lambda e: self._scan_list_canvas.itemconfig(
                self._scan_list_window, width=e.width)
        )
        self._scan_list_canvas.bind(
            "<MouseWheel>",
            lambda e: self._scan_list_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        )

    def _build_views(self, parent):
        outer = tk.Frame(parent)
        outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)

        self._views_nb = ttk.Notebook(outer)
        self._views_nb.pack(fill=tk.BOTH, expand=True)

        # EXAFS Workspace (analysis-only) tab.
        if self._show_analysis_panel:
            workspace = tk.Frame(self._views_nb)
            self._views_nb.add(workspace, text="EXAFS Workspace")

            ws_toolbar_frame = tk.Frame(workspace)
            ws_toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)

            self._fig_workspace = Figure(figsize=(8.4, 7.0), dpi=96, facecolor="white")
            gs = GridSpec(3, 1, figure=self._fig_workspace,
                          hspace=0.42, top=0.96, bottom=0.07, left=0.11, right=0.95)
            self._ax_q = self._fig_workspace.add_subplot(gs[0])
            self._ax_r = self._fig_workspace.add_subplot(gs[1])
            self._ax_overlap = self._fig_workspace.add_subplot(gs[2])

            self._canvas_workspace = FigureCanvasTkAgg(self._fig_workspace, master=workspace)
            self._canvas_workspace.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self._toolbar_workspace = NavigationToolbar2Tk(self._canvas_workspace, ws_toolbar_frame)
            self._toolbar_workspace.update()

        # FEFF tab (now hosted in Simulation Studio when this instance is
        # constructed with show_analysis_panel=False).
        if self._show_feff_panel:
            feff_tab = tk.Frame(self._views_nb)
            self._views_nb.add(feff_tab, text="FEFF")
            self._build_feff_tab(feff_tab)

        if self._show_analysis_panel:
            self._draw_empty_workspace()

    def _draw_empty_workspace(self):
        q_label = self._q_axis_symbol()
        for ax, title, xlabel, ylabel in [
            (self._ax_q, f"{q_label}-space  -  weighted EXAFS",
             f"{q_label}  (A^-1)", f"{q_label}^{self._qweight_var.get()} chi({q_label})"),
            (self._ax_r, "R-space  -  Fourier magnitude", "R  (A)", "|chi(R)|"),
            (self._ax_overlap, "Q / R overlap  -  R-window backtransform",
             f"{q_label}  (A^-1)", f"{q_label}^{self._qweight_var.get()} chi({q_label})"),
        ]:
            ax.clear()
            ax.set_title(title, fontsize=9, loc="left", pad=3)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.text(0.5, 0.5, "Run EXAFS on a loaded scan to populate this view.",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="lightgray")
            ax.tick_params(labelsize=7)
        self._canvas_workspace.draw_idle()

    def _build_feff_tab(self, parent):
        top = tk.Frame(parent, bd=1, relief=tk.GROOVE, padx=5, pady=4)
        top.pack(side=tk.TOP, fill=tk.X)

        self._feff_dir_var = tk.StringVar(value="")
        self._feff_exe_var = tk.StringVar(value="")
        self._feff_info_var = tk.StringVar(value="No FEFF directory selected.")
        self._xyz_path_var = tk.StringVar(value="")
        self._bundle_base_var = tk.StringVar(value="")
        self._xyz_info_var = tk.StringVar(
            value="No XYZ structure loaded. CIF export uses a boxed P1 cell."
        )
        self._xyz_padding_var = tk.DoubleVar(value=6.0)
        self._xyz_cubic_var = tk.BooleanVar(value=False)
        # Accepts either a 1-based atom index ("5") or an element symbol
        # ("Ni") that gets resolved against the loaded XYZ.
        self._xyz_absorber_var = tk.StringVar(value="Ni")
        self._xyz_edge_var = tk.StringVar(value="K")
        self._xyz_spectrum_var = tk.StringVar(value="EXAFS")
        self._xyz_kmesh_var = tk.IntVar(value=200)
        # Molecular mode: skip CIF/RECIPROCAL/KMESH and emit a real-space ATOMS
        # card. Drastically faster for isolated molecules and almost always the
        # right choice for XYZ-derived inputs.
        self._xyz_molecular_mode_var = tk.BooleanVar(value=True)
        # Cluster radius (Å): empty = include all atoms; >0 = crop to a sphere
        # around the absorber and tag the output filenames with the radius.
        self._xyz_cluster_radius_var = tk.StringVar(value="")
        self._xyz_equiv_var = tk.IntVar(value=2)
        self._xyz_xanes_emin_var = tk.DoubleVar(value=-30.0)
        self._xyz_xanes_emax_var = tk.DoubleVar(value=250.0)
        self._xyz_xanes_estep_var = tk.DoubleVar(value=0.25)
        # Advanced FEFF options (exposed in pop-out dialog)
        self._xyz_s02_var = tk.DoubleVar(value=1.0)
        self._xyz_corehole_var = tk.StringVar(value="RPA")
        self._xyz_exchange_var = tk.IntVar(value=0)
        self._xyz_exchange_vr_var = tk.DoubleVar(value=0.0)
        self._xyz_exchange_vi_var = tk.DoubleVar(value=0.0)
        self._xyz_scf_radius_var = tk.DoubleVar(value=4.0)
        self._xyz_scf_nscf_var = tk.IntVar(value=30)
        self._xyz_scf_ca_var = tk.DoubleVar(value=0.2)
        self._xyz_fms_radius_var = tk.DoubleVar(value=6.0)
        self._xyz_rpath_var = tk.DoubleVar(value=8.0)
        self._xyz_nleg_var = tk.IntVar(value=0)
        self._xyz_exafs_kmax_var = tk.DoubleVar(value=20.0)
        self._xyz_multipole_lmax_var = tk.IntVar(value=0)
        self._xyz_multipole_iorder_var = tk.IntVar(value=2)
        self._xyz_polarization_var = tk.BooleanVar(value=False)
        self._xyz_pol_x_var = tk.DoubleVar(value=1.0)
        self._xyz_pol_y_var = tk.DoubleVar(value=0.0)
        self._xyz_pol_z_var = tk.DoubleVar(value=0.0)
        self._xyz_ellip_var = tk.BooleanVar(value=False)
        self._xyz_ellip_val_var = tk.DoubleVar(value=0.0)
        self._xyz_ellip_x_var = tk.DoubleVar(value=0.0)
        self._xyz_ellip_y_var = tk.DoubleVar(value=0.0)
        self._xyz_ellip_z_var = tk.DoubleVar(value=1.0)
        self._xyz_debye_var = tk.BooleanVar(value=False)
        self._xyz_debye_temp_var = tk.DoubleVar(value=300.0)
        self._xyz_debye_dtemp_var = tk.DoubleVar(value=400.0)

        tk.Label(top, text="Workdir:", font=("", 8, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self._feff_dir_var, width=52).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        tk.Button(top, text="Browse", font=("", 8),
                  command=self._browse_feff_dir).grid(row=0, column=2, padx=2)
        tk.Button(top, text="Load Paths", font=("", 8),
                  command=self._load_feff_paths).grid(row=0, column=3, padx=2)

        tk.Label(top, text="Executable:", font=("", 8, "bold")).grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self._feff_exe_var, width=52).grid(
            row=1, column=1, sticky="ew", padx=4, pady=(3, 0)
        )
        tk.Button(top, text="Browse", font=("", 8),
                  command=self._browse_feff_exe).grid(row=1, column=2, padx=2, pady=(3, 0))
        self._run_feff_btn = tk.Button(top, text="Run FEFF", font=("", 8, "bold"),
                                       bg="#6B0000", fg="black",
                                       activebackground="#8B0000",
                                       command=self._run_feff)
        self._run_feff_btn.grid(row=1, column=3, padx=2, pady=(3, 0))

        tk.Label(top, textvariable=self._feff_info_var, fg="#003366",
                 font=("", 8)).grid(row=2, column=0, columnspan=4,
                                    sticky="w", pady=(5, 0))
        self._feff_status_var = tk.StringVar(value="")
        tk.Label(top, textvariable=self._feff_status_var, fg="#7B3F00",
                 font=("", 8, "italic")).grid(row=3, column=0, columnspan=4,
                                              sticky="w")
        top.columnconfigure(1, weight=1)

        xyz_box = tk.LabelFrame(parent, text="XYZ -> FEFF Bundle", padx=5, pady=4)
        xyz_box.pack(side=tk.TOP, fill=tk.X, padx=1, pady=(4, 0))

        tk.Label(xyz_box, text="XYZ file:", font=("", 8, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(xyz_box, textvariable=self._xyz_path_var, width=52).grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=4
        )
        tk.Button(xyz_box, text="Browse", font=("", 8),
                  command=self._browse_xyz_file).grid(row=0, column=4, padx=2)
        tk.Button(xyz_box, text="Load XYZ", font=("", 8),
                  command=self._load_xyz_structure).grid(row=0, column=5, padx=2)

        tk.Label(xyz_box, text="Base:", font=("", 8, "bold")).grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )
        ttk.Entry(xyz_box, textvariable=self._bundle_base_var, width=20).grid(
            row=1, column=1, sticky="w", padx=4, pady=(4, 0)
        )
        tk.Label(xyz_box, text="Padding (A):", font=("", 8, "bold")).grid(
            row=1, column=2, sticky="e", pady=(4, 0)
        )
        ttk.Entry(xyz_box, textvariable=self._xyz_padding_var, width=8).grid(
            row=1, column=3, sticky="w", padx=4, pady=(4, 0)
        )
        tk.Checkbutton(
            xyz_box,
            text="Force cubic cell",
            variable=self._xyz_cubic_var,
            font=("", 8),
        ).grid(row=1, column=4, columnspan=2, sticky="w", padx=(4, 0), pady=(4, 0))

        tk.Label(xyz_box, text="Absorber:", font=("", 8, "bold")).grid(
            row=2, column=0, sticky="w", pady=(4, 0)
        )
        # Accepts an element symbol ("Ni") or a 1-based atom index ("5").
        ttk.Entry(
            xyz_box, textvariable=self._xyz_absorber_var, width=8,
        ).grid(row=2, column=1, sticky="w", padx=4, pady=(4, 0))
        tk.Label(xyz_box, text="Edge:", font=("", 8, "bold")).grid(
            row=2, column=2, sticky="e", pady=(4, 0)
        )
        ttk.Combobox(
            xyz_box,
            textvariable=self._xyz_edge_var,
            values=("K", "L1", "L2", "L3"),
            state="readonly",
            width=7,
        ).grid(row=2, column=3, sticky="w", padx=4, pady=(4, 0))
        tk.Label(xyz_box, text="Spectrum:", font=("", 8, "bold")).grid(
            row=2, column=4, sticky="e", pady=(4, 0)
        )
        ttk.Combobox(
            xyz_box,
            textvariable=self._xyz_spectrum_var,
            values=("EXAFS", "XANES"),
            state="readonly",
            width=10,
        ).grid(row=2, column=5, sticky="w", padx=4, pady=(4, 0))

        ttk.Checkbutton(
            xyz_box, text="Molecular mode (real-space ATOMS, much faster)",
            variable=self._xyz_molecular_mode_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))

        tk.Label(xyz_box, text="Cluster (Å):", font=("", 8)).grid(
            row=3, column=3, sticky="e", pady=(4, 0)
        )
        ttk.Entry(xyz_box, textvariable=self._xyz_cluster_radius_var, width=6).grid(
            row=3, column=4, sticky="w", padx=4, pady=(4, 0)
        )
        tk.Label(xyz_box, text="KMESH:", font=("", 8)).grid(
            row=3, column=5, sticky="e", pady=(4, 0)
        )
        ttk.Entry(xyz_box, textvariable=self._xyz_kmesh_var, width=6).grid(
            row=3, column=6, sticky="w", padx=4, pady=(4, 0)
        )
        tk.Label(xyz_box, text="Equivalence:", font=("", 8)).grid(
            row=3, column=7, sticky="e", pady=(4, 0)
        )
        ttk.Entry(xyz_box, textvariable=self._xyz_equiv_var, width=6).grid(
            row=3, column=8, sticky="w", padx=4, pady=(4, 0)
        )
        btn_frame = tk.Frame(xyz_box)
        btn_frame.grid(row=3, column=9, columnspan=2, sticky="ew", padx=(8, 0), pady=(4, 0))
        tk.Button(
            btn_frame, text="FEFF Options...", font=("", 8),
            command=self._open_feff_options,
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(
            btn_frame, text="Write FEFF Bundle", font=("", 8, "bold"),
            bg="#003366", fg="black", activebackground="#004C99",
            command=self._write_xyz_feff_bundle,
        ).pack(side=tk.LEFT)
        self._batch_btn = tk.Button(
            btn_frame, text="Batch Run...", font=("", 8, "bold"),
            bg="#5C2A0E", fg="black", activebackground="#7A3D17",
            command=self._on_batch_run_clicked,
        )
        self._batch_btn.pack(side=tk.LEFT, padx=(4, 0))

        tk.Label(xyz_box, text="XANES E min (eV):", font=("", 8, "bold")).grid(
            row=4, column=0, sticky="w", pady=(4, 0)
        )
        ttk.Entry(xyz_box, textvariable=self._xyz_xanes_emin_var, width=8).grid(
            row=4, column=1, sticky="w", padx=4, pady=(4, 0)
        )
        tk.Label(xyz_box, text="E max (eV):", font=("", 8, "bold")).grid(
            row=4, column=2, sticky="e", pady=(4, 0)
        )
        ttk.Entry(xyz_box, textvariable=self._xyz_xanes_emax_var, width=8).grid(
            row=4, column=3, sticky="w", padx=4, pady=(4, 0)
        )
        tk.Label(xyz_box, text="Step (eV):", font=("", 8, "bold")).grid(
            row=4, column=4, sticky="e", pady=(4, 0)
        )
        ttk.Entry(xyz_box, textvariable=self._xyz_xanes_estep_var, width=8).grid(
            row=4, column=5, sticky="w", padx=4, pady=(4, 0)
        )

        tk.Label(
            xyz_box,
            textvariable=self._xyz_info_var,
            fg="#003366",
            font=("", 8),
            justify="left",
        ).grid(row=5, column=0, columnspan=6, sticky="w", pady=(6, 0))
        xyz_box.columnconfigure(1, weight=1)
        xyz_box.columnconfigure(3, weight=0)

        # Vertical paned window: body (paths + preview) on top, log on bottom.
        # The sash between them is user-draggable so the log area can be
        # extended when needed for long FEFF output.
        vbody = tk.PanedWindow(parent, orient=tk.VERTICAL,
                               sashwidth=5, sashrelief=tk.RAISED)
        vbody.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        body = tk.PanedWindow(vbody, orient=tk.HORIZONTAL, sashwidth=5, sashrelief=tk.RAISED)
        vbody.add(body, minsize=200, stretch="always")

        left = tk.Frame(body, bd=1, relief=tk.SUNKEN)
        body.add(left, minsize=260)

        tk.Label(left, text="Parsed FEFF Paths", font=("", 8, "bold"),
                 anchor="w").pack(fill=tk.X, padx=6, pady=(5, 2))

        tree_wrap = tk.Frame(left)
        tree_wrap.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        tree_scroll = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._feff_tree = ttk.Treeview(
            tree_wrap,
            columns=("index", "reff", "degen", "nleg"),
            show="headings",
            selectmode="extended",
            yscrollcommand=tree_scroll.set,
        )
        for col, width, text in [
            ("index", 60, "Path"),
            ("reff", 70, "Reff"),
            ("degen", 70, "Deg."),
            ("nleg", 60, "Legs"),
        ]:
            self._feff_tree.heading(col, text=text)
            self._feff_tree.column(col, width=width, anchor="center")
        self._feff_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.config(command=self._feff_tree.yview)
        self._feff_tree.bind("<<TreeviewSelect>>", lambda _e: self._on_feff_selection())

        right = tk.Frame(body)
        body.add(right, minsize=380)

        preview_toolbar = tk.Frame(right)
        preview_toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self._fig_feff = Figure(figsize=(7.0, 5.0), dpi=96, facecolor="white")
        self._ax_feff = self._fig_feff.add_subplot(111)
        self._fig_feff.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)

        self._canvas_feff = FigureCanvasTkAgg(self._fig_feff, master=right)
        self._canvas_feff.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._toolbar_feff = NavigationToolbar2Tk(self._canvas_feff, preview_toolbar)
        self._toolbar_feff.update()

        # Log panel — added to the vertical paned window so the user can drag
        # the sash above it to grow/shrink the visible log area.
        log_frame = tk.Frame(vbody)
        self._feff_log = tk.Text(log_frame, height=8, font=("Consolas", 10),
                                 wrap=tk.WORD, state=tk.DISABLED)
        log_scroll = tk.Scrollbar(log_frame, orient=tk.VERTICAL,
                                  command=self._feff_log.yview)
        self._feff_log.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._feff_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                            padx=(4, 0), pady=(3, 0))
        vbody.add(log_frame, minsize=60, stretch="never")

        self._load_feff_tab_settings()
        self._draw_empty_feff_preview()

    # ------------------------------------------------------------------ #
    #  FEFF-tab settings persistence                                       #
    # ------------------------------------------------------------------ #
    _CFG_PATH = os.path.join(os.path.expanduser("~"), ".binah_config.json")

    def _load_feff_tab_settings(self):
        import json
        try:
            if not os.path.exists(self._CFG_PATH):
                return
            with open(self._CFG_PATH, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            saved = cfg.get("feff_tab", {})
            if not saved:
                return

            def _str(var, key):
                v = saved.get(key, "")
                if v:
                    var.set(str(v))

            def _num(var, key):
                try:
                    var.set(type(var.get())(saved[key]))
                except Exception:
                    pass

            def _bool(var, key):
                try:
                    var.set(bool(saved[key]))
                except Exception:
                    pass

            _str(self._feff_dir_var,        "feff_dir")
            _str(self._feff_exe_var,         "feff_exe")
            _str(self._xyz_path_var,         "xyz_path")
            _str(self._bundle_base_var,      "bundle_base")
            _str(self._xyz_edge_var,         "xyz_edge")
            _str(self._xyz_spectrum_var,     "xyz_spectrum")
            _num(self._xyz_padding_var,      "xyz_padding")
            _bool(self._xyz_cubic_var,       "xyz_cubic")
            _num(self._xyz_absorber_var,     "xyz_absorber")
            _num(self._xyz_kmesh_var,        "xyz_kmesh")
            _num(self._xyz_equiv_var,        "xyz_equivalence")
            _num(self._xyz_xanes_emin_var,   "xyz_xanes_emin")
            _num(self._xyz_xanes_emax_var,   "xyz_xanes_emax")
            _num(self._xyz_xanes_estep_var,  "xyz_xanes_estep")
            if "xyz_molecular_mode" in saved:
                try:
                    self._xyz_molecular_mode_var.set(bool(saved["xyz_molecular_mode"]))
                except Exception:
                    pass
            if "xyz_cluster_radius" in saved:
                try:
                    self._xyz_cluster_radius_var.set(str(saved["xyz_cluster_radius"]))
                except Exception:
                    pass

            if self._feff_dir_var.get().strip():
                self._load_feff_paths(silent=True)
            if self._xyz_path_var.get().strip():
                self._load_xyz_structure(silent=True)
        except Exception:
            pass
        self._setup_feff_persistence_traces()

    def _save_feff_tab_settings(self):
        """Persist the EXAFS-tab state to ~/.binah_config.json.

        Each variable is read defensively — a single bad/blank entry must not
        abort the whole save (otherwise persistence silently breaks the
        moment the user clears a numeric field).
        """
        import json

        def _safe(var):
            try:
                return var.get()
            except Exception:
                return ""

        feff_tab_data = {
            "feff_dir":         _safe(self._feff_dir_var),
            "feff_exe":         _safe(self._feff_exe_var),
            "xyz_path":         _safe(self._xyz_path_var),
            "bundle_base":      _safe(self._bundle_base_var),
            "xyz_edge":         _safe(self._xyz_edge_var),
            "xyz_spectrum":     _safe(self._xyz_spectrum_var),
            "xyz_padding":      _safe(self._xyz_padding_var),
            "xyz_cubic":        _safe(self._xyz_cubic_var),
            "xyz_absorber":     _safe(self._xyz_absorber_var),
            "xyz_kmesh":        _safe(self._xyz_kmesh_var),
            "xyz_equivalence":  _safe(self._xyz_equiv_var),
            "xyz_xanes_emin":   _safe(self._xyz_xanes_emin_var),
            "xyz_xanes_emax":   _safe(self._xyz_xanes_emax_var),
            "xyz_xanes_estep":  _safe(self._xyz_xanes_estep_var),
            "xyz_molecular_mode": bool(_safe(self._xyz_molecular_mode_var)),
            "xyz_cluster_radius": str(_safe(self._xyz_cluster_radius_var)),
        }
        try:
            cfg = {}
            if os.path.exists(self._CFG_PATH):
                try:
                    with open(self._CFG_PATH, "r", encoding="utf-8") as fh:
                        cfg = json.load(fh)
                except Exception:
                    cfg = {}
            cfg["feff_tab"] = feff_tab_data
            with open(self._CFG_PATH, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2)
        except Exception as exc:
            # Surface failures rather than burying them — the EXAFS log is the
            # natural channel.  Don't crash if logging itself isn't ready yet.
            try:
                self._append_feff_log(f"Could not save EXAFS tab settings: {exc}")
            except Exception:
                pass

    def _schedule_feff_save(self, *_):
        if hasattr(self, "_feff_save_id"):
            try:
                self.after_cancel(self._feff_save_id)
            except Exception:
                pass
        self._feff_save_id = self.after(800, self._save_feff_tab_settings)

    def _setup_feff_persistence_traces(self):
        for var in (
            self._feff_dir_var, self._feff_exe_var,
            self._xyz_path_var, self._bundle_base_var,
            self._xyz_edge_var, self._xyz_spectrum_var,
            self._xyz_padding_var, self._xyz_cubic_var,
            self._xyz_absorber_var, self._xyz_kmesh_var, self._xyz_equiv_var,
            self._xyz_xanes_emin_var, self._xyz_xanes_emax_var, self._xyz_xanes_estep_var,
            self._xyz_molecular_mode_var, self._xyz_cluster_radius_var,
        ):
            var.trace_add("write", self._schedule_feff_save)

    def _draw_empty_feff_preview(self):
        self._ax_feff.clear()
        self._ax_feff.set_title("FEFF path preview", fontsize=9, loc="left", pad=3)
        self._ax_feff.set_xlabel(f"{self._q_axis_symbol()}  (A^-1)", fontsize=8)
        self._ax_feff.set_ylabel("Path amplitude", fontsize=8)
        self._ax_feff.text(
            0.5, 0.5,
            "Load FEFF path files or run FEFF in a working directory.",
            transform=self._ax_feff.transAxes,
            ha="center", va="center",
            fontsize=9, color="lightgray",
        )
        self._canvas_feff.draw_idle()

    def _q_axis_symbol(self) -> str:
        # Defaults to "k" when this instance has no analysis panel (the var
        # is created in _build_params, which is skipped in feff-only mode).
        var = getattr(self, "_use_q_label_var", None)
        if var is None:
            return "k"
        try:
            return "q" if var.get() else "k"
        except Exception:
            return "k"

    def _apply_theme(self):
        if _HAS_SNS:
            sns.set_theme(
                style=self._style_var.get(),
                context=self._context_var.get(),
                palette=xas_core._PALETTE,
            )

    def _ensure_scan_backup(self, scan) -> None:
        meta = getattr(scan, "metadata", None)
        if meta is None:
            scan.metadata = {}
            meta = scan.metadata
        if "_binah_original_energy" not in meta:
            meta["_binah_original_energy"] = np.asarray(scan.energy_ev, dtype=float).copy()
            meta["_binah_original_mu"] = np.asarray(scan.mu, dtype=float).copy()
            meta["_binah_original_e0"] = float(scan.e0)
            meta["_binah_original_norm"] = bool(scan.is_normalized)

    def _source_arrays(self, scan) -> tuple[np.ndarray, np.ndarray, float]:
        self._ensure_scan_backup(scan)
        meta = getattr(scan, "metadata", {}) or {}
        energy = np.asarray(meta.get("_binah_original_energy", scan.energy_ev), dtype=float).copy()
        mu = np.asarray(meta.get("_binah_original_mu", scan.mu), dtype=float).copy()
        e0 = float(meta.get("_binah_original_e0", scan.e0 or 0.0))
        return energy, mu, e0

    def _get_scan_by_label(self, label: str):
        for scan_label, scan, *_ in self._get_scans():
            if scan_label == label:
                return scan
        return None

    def _auto_fill_e0(self):
        # Only meaningful when the analysis panel (with E0 entry) exists.
        if not getattr(self, "_e0_var", None):
            return
        label = self._scan_var.get()
        scan = self._get_scan_by_label(label)
        if scan is None:
            return
        energy, mu, stored_e0 = self._source_arrays(scan)
        e0 = stored_e0 if stored_e0 > 100 else xas_core.find_e0(energy, mu)
        self._e0_var.set(float(e0))

    def _rebuild_scan_list_rows(self):
        # No-op when the analysis panel is hidden — the scan-list widget
        # never got built.
        if not getattr(self, "_scan_list_inner", None):
            return
        for widget in self._scan_list_inner.winfo_children():
            widget.destroy()

        scans = self._get_scans()
        for i, (label, _scan, *_rest) in enumerate(scans):
            col = xas_core._PALETTE[i % len(xas_core._PALETTE)]
            if label not in self._scan_vis_vars:
                self._scan_vis_vars[label] = tk.BooleanVar(value=False)
            var = self._scan_vis_vars[label]

            row = tk.Frame(self._scan_list_inner, bg="white")
            row.pack(fill=tk.X, pady=1, padx=2)

            tk.Label(row, bg=col, width=2, relief=tk.FLAT).pack(side=tk.LEFT, padx=(2, 3))
            tk.Checkbutton(
                row,
                variable=var,
                bg="white",
                pady=0,
                command=lambda lbl=label: self._toggle_scan_vis(lbl),
            ).pack(side=tk.LEFT)

            short = label if len(label) <= 22 else label[:20] + "..."
            lbl_w = tk.Label(row, text=short, anchor="w", bg="white",
                             font=("", 8), cursor="hand2", fg="#003366")
            lbl_w.pack(side=tk.LEFT, fill=tk.X, expand=True)
            lbl_w.bind("<Button-1>", lambda _e, lbl=label: self._select_scan(lbl))
            lbl_w.bind("<Enter>", lambda _e, w=lbl_w: w.config(fg="#0066CC", font=("", 8, "underline")))
            lbl_w.bind("<Leave>", lambda _e, w=lbl_w: w.config(fg="#003366", font=("", 8)))

        if not scans:
            tk.Label(self._scan_list_inner, text="No scans loaded",
                     fg="gray", font=("", 8, "italic"), bg="white").pack(pady=10)

    def _select_scan(self, label: str):
        self._scan_var.set(label)
        self._auto_fill_e0()
        self._run()

    def _toggle_scan_vis(self, label: str):
        var = self._scan_vis_vars.get(label)
        if var is None:
            return
        if var.get():
            if label not in self._results:
                scan = self._get_scan_by_label(label)
                if scan is not None:
                    self._scan_var.set(label)
                    self._auto_fill_e0()
                    self._run_single(label, scan)
            if label not in self._selected_labels:
                self._selected_labels.append(label)
        else:
            if label in self._selected_labels:
                self._selected_labels.remove(label)
        self._redraw()

    def _show_all_scans(self):
        for label, _scan, *_ in self._get_scans():
            if label not in self._scan_vis_vars:
                self._scan_vis_vars[label] = tk.BooleanVar(value=True)
            self._scan_vis_vars[label].set(True)
            if label not in self._results:
                scan = self._get_scan_by_label(label)
                if scan is not None:
                    self._run_single(label, scan)
            if label not in self._selected_labels:
                self._selected_labels.append(label)
        self._redraw()

    def _hide_all_scans(self):
        for var in self._scan_vis_vars.values():
            var.set(False)
        self._selected_labels.clear()
        self._redraw()

    def refresh_scan_list(self):
        scans = self._get_scans()
        labels = [label for label, *_ in scans]
        self._scan_cb["values"] = labels
        if labels:
            current = self._scan_var.get()
            if current not in labels:
                current = labels[0]
            self._scan_var.set(current)
            self._auto_fill_e0()
        else:
            self._scan_var.set("")

        self._results = {label: res for label, res in self._results.items() if label in labels}
        self._selected_labels = [label for label in self._selected_labels if label in labels]
        self._scan_vis_vars = {
            label: var for label, var in self._scan_vis_vars.items() if label in labels
        }

        for label in self._pending_selected_labels:
            if label in labels and label not in self._selected_labels:
                self._selected_labels.append(label)
            if label in labels and label not in self._scan_vis_vars:
                self._scan_vis_vars[label] = tk.BooleanVar(value=True)
            if label in labels:
                self._scan_vis_vars[label].set(True)
        self._pending_selected_labels = []

        self._rebuild_scan_list_rows()
        if not labels and self._show_analysis_panel:
            self._draw_empty_workspace()

    def auto_run_all(self):
        # No-op when running in FEFF-only mode (no analysis machinery).
        if not self._show_analysis_panel:
            return
        scans = self._get_scans()
        if not scans:
            return
        self.refresh_scan_list()
        for label, scan, *_ in scans:
            if label not in self._results:
                self._run_single(label, scan)
            if label not in self._selected_labels:
                self._selected_labels.append(label)
            if label not in self._scan_vis_vars:
                self._scan_vis_vars[label] = tk.BooleanVar(value=True)
            self._scan_vis_vars[label].set(True)
        self._rebuild_scan_list_rows()
        self._redraw()

    def _add_overlay(self):
        label = self._scan_var.get()
        if not label:
            self._status_lbl.config(text="Select a scan first.", fg="#993300")
            return
        scan = self._get_scan_by_label(label)
        if scan is None:
            self._status_lbl.config(text="Scan not found. Refresh the scan list.", fg="#993300")
            return
        if label not in self._results:
            self._run_single(label, scan)
        if label not in self._selected_labels:
            self._selected_labels.append(label)
        if label not in self._scan_vis_vars:
            self._scan_vis_vars[label] = tk.BooleanVar(value=True)
        self._scan_vis_vars[label].set(True)
        self._rebuild_scan_list_rows()
        self._redraw()

    def _clear_overlay(self):
        self._selected_labels.clear()
        for var in self._scan_vis_vars.values():
            var.set(False)
        self._redraw()

    def _capture_q_window_from_plot(self):
        lo, hi = self._ax_q.get_xlim()
        if hi > lo:
            self._qmin_var.set(max(0.0, float(lo)))
            self._qmax_var.set(max(0.0, float(hi)))
            self._status_lbl.config(text="Captured q window from the current q-space plot.",
                                    fg="#003366")
            self._redraw()

    def _capture_r_window_from_plot(self):
        lo, hi = self._ax_r.get_xlim()
        if hi > lo:
            self._rmin_var.set(max(0.0, float(lo)))
            self._rmax_var.set(max(0.0, float(hi)))
            self._status_lbl.config(text="Captured R window from the current R-space plot.",
                                    fg="#003366")
            self._redraw()

    def _reset_q_window(self):
        self._qmin_var.set(float(xas_core._NORM_DEFAULTS["kmin"]))
        self._qmax_var.set(float(xas_core._NORM_DEFAULTS["kmax"]))
        self._dq_var.set(float(xas_core._NORM_DEFAULTS["dk"]))
        self._redraw()

    def _reset_r_window(self):
        self._rmin_var.set(1.0)
        self._rmax_var.set(3.2)
        self._dr_var.set(0.5)
        self._rdisplay_var.set(float(xas_core._NORM_DEFAULTS["rmax"]))
        self._redraw()

    def _run(self):
        label = self._scan_var.get()
        if not label:
            self._status_lbl.config(text="Select a scan first.", fg="#993300")
            return
        scan = self._get_scan_by_label(label)
        if scan is None:
            self._status_lbl.config(text="Scan not found. Refresh the scan list.", fg="#993300")
            return
        if self._run_single(label, scan):
            if label not in self._selected_labels:
                self._selected_labels.append(label)
            if label not in self._scan_vis_vars:
                self._scan_vis_vars[label] = tk.BooleanVar(value=True)
            self._scan_vis_vars[label].set(True)
            self._rebuild_scan_list_rows()
            self._redraw()

    def _run_single(self, label: str, scan) -> bool:
        energy, mu_raw, stored_e0 = self._source_arrays(scan)
        if len(energy) < 6:
            self._status_lbl.config(text=f"{label}: not enough points for EXAFS analysis.",
                                    fg="#993300")
            return False

        e0 = stored_e0 if stored_e0 > 100 else float(self._e0_var.get())
        if e0 <= 100:
            e0 = float(xas_core.find_e0(energy, mu_raw))

        pre1 = float(self._pre1_var.get())
        pre2 = float(self._pre2_var.get())
        nor1 = float(self._nor1_var.get())
        nor2 = float(self._nor2_var.get())
        nnorm = int(self._nnorm_var.get())
        rbkg = float(self._rbkg_var.get())
        kmin_bkg = float(self._kmin_bkg_var.get())

        try:
            use_larch = bool(
                getattr(xas_core, "_HAS_LARCH", False)
                and getattr(xas_core, "LarchGroup", None) is not None
                and hasattr(xas_core, "_larch_pre_edge")
                and hasattr(xas_core, "_larch_autobk")
            )

            if use_larch:
                session = xas_core._get_larch_session()
                grp = xas_core.LarchGroup(energy=energy.copy(), mu=mu_raw.copy())
                xas_core._larch_pre_edge(
                    grp,
                    _larch=session,
                    e0=float(e0),
                    pre1=pre1,
                    pre2=pre2,
                    norm1=nor1,
                    norm2=nor2,
                    nnorm=nnorm,
                )
                xas_core._larch_autobk(
                    grp,
                    _larch=session,
                    rbkg=rbkg,
                    kmin=kmin_bkg,
                )
                mu_norm = np.asarray(getattr(grp, "flat", grp.norm), dtype=float)
                q = np.asarray(getattr(grp, "k", np.array([], dtype=float)), dtype=float)
                chi = np.asarray(getattr(grp, "chi", np.array([], dtype=float)), dtype=float)
                bkg_e = np.asarray(getattr(grp, "bkg", np.zeros_like(energy)), dtype=float)
                pre_line = np.asarray(getattr(grp, "pre_edge", np.zeros_like(energy)), dtype=float)
                e0 = float(grp.e0)
                edge_step = float(getattr(grp, "edge_step", 1.0))
                engine = "larch"
            else:
                mu_norm, edge_step, pre_line = xas_core.normalize_xanes(
                    energy, mu_raw, e0, pre1, pre2, nor1, nor2, nnorm
                )
                q, chi, bkg_e = xas_core.autobk(
                    energy, mu_norm, e0, rbkg=rbkg, kmin_bkg=kmin_bkg
                )
                engine = "binah"
        except Exception as exc:
            self._status_lbl.config(text=f"{label}: EXAFS analysis failed ({exc}).",
                                    fg="#993300")
            return False

        self._results[label] = {
            "energy": np.asarray(energy, dtype=float),
            "mu_raw": np.asarray(mu_raw, dtype=float),
            "mu_norm": np.asarray(mu_norm, dtype=float),
            "pre_line": np.asarray(pre_line, dtype=float),
            "bkg_e": np.asarray(bkg_e, dtype=float),
            "q": np.asarray(q, dtype=float),
            "chi": np.asarray(chi, dtype=float),
            "e0": float(e0),
            "edge_step": float(edge_step),
            "engine": engine,
        }

        msg = (
            f"[{engine}] {label} | E0={e0:.1f} eV | edge step={edge_step:.4f} | "
            f"{self._q_axis_symbol()} range={self._qmin_var.get():.1f}-{self._qmax_var.get():.1f} A^-1"
        )
        if xas_core._is_l_edge_e0(e0):
            msg += " | warning: soft X-ray edge, EXAFS range may be too short"
        self._status_lbl.config(text=msg, fg="#003366" if engine == "larch" else "#664400")
        return True

    def _transform_for_label(self, label: str) -> dict:
        res = self._results.get(label)
        if res is None:
            return compute_transform_bundle(
                np.array([], dtype=float),
                np.array([], dtype=float),
                0.0, 0.0, 0.1, 2, "Hanning", 0.0, 0.0, 0.1, "Hanning",
            )
        return compute_transform_bundle(
            res["q"],
            res["chi"],
            float(self._qmin_var.get()),
            float(self._qmax_var.get()),
            float(self._dq_var.get()),
            int(self._qweight_var.get()),
            self._qwin_var.get(),
            float(self._rmin_var.get()),
            float(self._rmax_var.get()),
            float(self._dr_var.get()),
            self._rwin_var.get(),
        )

    def _active_label(self) -> str:
        label = self._scan_var.get()
        if label in self._selected_labels and label in self._results:
            return label
        for candidate in self._selected_labels:
            if candidate in self._results:
                return candidate
        return ""

    def _selected_feff_paths(self) -> list[FeffPathData]:
        if not hasattr(self, "_feff_tree"):
            return []
        selected_ids = self._feff_tree.selection()
        selected = []
        for item_id in selected_ids:
            try:
                idx = int(item_id)
            except Exception:
                continue
            if 0 <= idx < len(self._feff_paths):
                selected.append(self._feff_paths[idx])
        return selected

    def _resolve_feff_paths_for_overlay(self) -> list[FeffPathData]:
        """Return the FEFF paths to draw on R-space.

        When this instance owns the FEFF panel, return our own list. When it
        doesn't (analysis-only mode in EXAFS Studio after the FEFF panel
        moved to Simulation Studio), pull from the provider callback.
        """
        if self._feff_paths:
            return self._feff_paths
        provider = self._feff_paths_provider
        if provider is not None:
            try:
                paths = provider()
                return list(paths) if paths else []
            except Exception:
                return []
        return []

    def _draw_feff_markers(self, ax):
        if not self._show_feff_markers_var.get():
            return
        all_paths = self._resolve_feff_paths_for_overlay()
        if not all_paths:
            return
        selected = self._selected_feff_paths()
        paths = selected if selected else all_paths
        alpha = 0.6 if selected else 0.18
        ymax = ax.get_ylim()[1]
        for i, path in enumerate(paths[:20]):
            if path.reff <= 0:
                continue
            colour = "#880000" if selected else "#555555"
            ax.axvline(path.reff, color=colour, lw=1.0, ls=":", alpha=alpha, zorder=1)
            y_text = ymax * (0.88 - 0.06 * (i % 4))
            ax.text(path.reff, y_text, f"P{path.index:04d}",
                    rotation=90, va="top", ha="right", fontsize=6,
                    color=colour, alpha=min(1.0, alpha + 0.2))

    def _redraw(self):
        self._apply_theme()

        labels = [label for label in self._selected_labels if label in self._results]
        active = self._active_label()
        if not labels:
            self._draw_empty_workspace()
            return

        transforms = {label: self._transform_for_label(label) for label in labels}
        active_bundle = transforms.get(active) if active else None

        q_name = self._q_axis_symbol()
        q_weight = int(self._qweight_var.get())
        ax_q = self._ax_q
        ax_r = self._ax_r
        ax_overlap = self._ax_overlap
        ax_q.clear()
        ax_r.clear()
        ax_overlap.clear()

        if active and xas_core._is_l_edge_e0(self._results[active]["e0"]):
            message = (
                "Soft X-ray / L-edge data detected.\n\n"
                "The available post-edge range is usually too short for a robust EXAFS fit.\n"
                "Use this view cautiously and treat any q/R structure as qualitative."
            )
            for ax in (ax_q, ax_r, ax_overlap):
                ax.set_facecolor("#fffff8")
                ax.text(
                    0.5, 0.5, message, transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="#885500",
                    bbox=dict(boxstyle="round,pad=0.6",
                              facecolor="#fff8e1", edgecolor="#ccaa00", alpha=0.9),
                )
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            self._canvas_workspace.draw_idle()
            return

        for i, label in enumerate(labels):
            bundle = transforms[label]
            if len(bundle["q_uniform"]) < 2:
                continue
            colour = xas_core._PALETTE[i % len(xas_core._PALETTE)]
            short = label if len(label) <= 30 else label[:28] + "..."

            ax_q.plot(bundle["q_uniform"], bundle["chi_weighted"],
                      color=colour, lw=1.5, label=short, zorder=3)
            ax_r.plot(bundle["r"], bundle["chi_r_mag"],
                      color=colour, lw=1.6, label=short, zorder=3)

            if label == active:
                if self._show_q_window_var.get():
                    amp = max(np.max(np.abs(bundle["chi_weighted"])), 1e-6)
                    ax_q.fill_between(
                        bundle["q_uniform"],
                        -bundle["q_window"] * amp,
                        bundle["q_window"] * amp,
                        color="orange",
                        alpha=0.10,
                        label="q window",
                    )
                ax_r.plot(bundle["r"], bundle["chi_r_selected_mag"],
                          color=colour, lw=1.3, ls="--", alpha=0.95,
                          label=f"{short} (R window)")
                if self._show_r_window_var.get():
                    amp_r = max(np.max(bundle["chi_r_mag"]), 1e-6)
                    ax_r.fill_between(
                        bundle["r"], 0.0, bundle["r_window"] * amp_r,
                        color="#C85A17", alpha=0.10, label="R window"
                    )

        if active_bundle is not None and len(active_bundle["q_uniform"]) > 1:
            colour = "#1f4e79"
            ax_overlap.plot(
                active_bundle["q_uniform"],
                active_bundle["chi_weighted"],
                color=colour,
                lw=1.8,
                label=f"Original {q_name}^{q_weight} chi({q_name})",
            )
            ax_overlap.plot(
                active_bundle["q_uniform"],
                active_bundle["chi_weighted_back"],
                color="#D1495B",
                lw=1.5,
                ls="--",
                label="Backtransform from selected R window",
            )
            if self._show_q_window_var.get():
                amp = max(
                    np.max(np.abs(active_bundle["chi_weighted"])),
                    np.max(np.abs(active_bundle["chi_weighted_back"])),
                    1e-6,
                )
                ax_overlap.fill_between(
                    active_bundle["q_uniform"],
                    0.0,
                    active_bundle["q_window"] * amp,
                    color="orange",
                    alpha=0.08,
                    label="Active q window",
                )
            ax_overlap.text(
                0.02, 0.96,
                f"R window: {self._rmin_var.get():.2f}-{self._rmax_var.get():.2f} A"
                f"  |  q window: {self._qmin_var.get():.2f}-{self._qmax_var.get():.2f} A^-1",
                transform=ax_overlap.transAxes,
                ha="left", va="top", fontsize=7, color="#333333",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor="#cccccc", alpha=0.9),
            )

        ax_q.set_title(f"{q_name}-space  -  weighted EXAFS", fontsize=9, loc="left", pad=3)
        ax_q.set_xlabel(f"{q_name}  (A^-1)", fontsize=8)
        ax_q.set_ylabel(f"{q_name}^{q_weight} chi({q_name})", fontsize=8)
        ax_q.axhline(0.0, color="gray", lw=0.5, ls="--", alpha=0.40)
        ax_q.set_xlim(left=0.0)

        ax_r.set_title("R-space  -  Fourier magnitude", fontsize=9, loc="left", pad=3)
        ax_r.set_xlabel("R  (A)", fontsize=8)
        ax_r.set_ylabel("|chi(R)|", fontsize=8)
        ax_r.set_xlim(0.0, float(self._rdisplay_var.get()))
        ax_r.axhline(0.0, color="gray", lw=0.5, ls="--", alpha=0.35)
        self._draw_feff_markers(ax_r)

        ax_overlap.set_title("Q / R overlap  -  R-window backtransform", fontsize=9, loc="left", pad=3)
        ax_overlap.set_xlabel(f"{q_name}  (A^-1)", fontsize=8)
        ax_overlap.set_ylabel(f"{q_name}^{q_weight} chi({q_name})", fontsize=8)
        ax_overlap.axhline(0.0, color="gray", lw=0.5, ls="--", alpha=0.40)
        ax_overlap.set_xlim(left=0.0)

        for ax in (ax_q, ax_r, ax_overlap):
            ax.tick_params(labelsize=7)
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
            if _HAS_SNS:
                sns.despine(ax=ax, offset=4)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.85)

        self._canvas_workspace.draw_idle()

    def _browse_feff_dir(self):
        path = filedialog.askdirectory(title="Select FEFF Working Directory")
        if path:
            self._feff_dir_var.set(path)
            self._load_feff_paths(silent=True)

    def _browse_feff_exe(self):
        path = filedialog.askopenfilename(
            title="Select FEFF Executable",
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
        )
        if path:
            self._feff_exe_var.set(path)

    def _open_feff_options(self):
        win = tk.Toplevel(self)
        win.title("FEFF10 Options")
        win.resizable(True, True)
        win.grab_set()

        nb = ttk.Notebook(win)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        def _row(parent, label, widget_factory, row, col=0, tooltip=None):
            tk.Label(parent, text=label, font=("", 8, "bold"), anchor="w").grid(
                row=row, column=col, sticky="w", padx=(4, 2), pady=3)
            w = widget_factory(parent)
            w.grid(row=row, column=col + 1, sticky="w", padx=(0, 8), pady=3)
            if tooltip:
                tk.Label(parent, text=tooltip, fg="gray", font=("", 7),
                         anchor="w").grid(row=row, column=col + 2, sticky="w")
            return w

        def _entry(var, width=10):
            return lambda p: ttk.Entry(p, textvariable=var, width=width)

        def _combo(var, values, width=12):
            return lambda p: ttk.Combobox(p, textvariable=var, values=values,
                                          state="readonly", width=width)

        def _check(var, text=""):
            return lambda p: tk.Checkbutton(p, variable=var, text=text, font=("", 8))

        # ── Tab 1: Core ──────────────────────────────────────────────────────
        t1 = tk.Frame(nb, padx=6, pady=6)
        nb.add(t1, text="Core")

        _row(t1, "S02:", _entry(self._xyz_s02_var, 8), 0,
             tooltip="Passive electron reduction factor (0–1)")
        _row(t1, "Core hole:", _combo(self._xyz_corehole_var,
             ("RPA", "FMS", "HALF", "NONE"), 8), 1,
             tooltip="RPA = full screening (recommended for XANES)")
        tk.Label(t1, text="Exchange-correlation:", font=("", 8, "bold")).grid(
            row=2, column=0, sticky="w", padx=4, pady=(10, 2), columnspan=3)
        _row(t1, "Type:", _combo(self._xyz_exchange_var,
             (0, 1, 2, 5), 6), 3,
             tooltip="0=Hedin-Lundqvist  1=Dirac-Hara  2=ground state  5=LDA+C")
        _row(t1, "Vr shift (eV):", _entry(self._xyz_exchange_vr_var, 8), 4,
             tooltip="Real part of optical potential shift")
        _row(t1, "Vi shift (eV):", _entry(self._xyz_exchange_vi_var, 8), 5,
             tooltip="Imaginary broadening shift")

        tk.Label(t1, text="Self-consistent field (SCF):", font=("", 8, "bold")).grid(
            row=6, column=0, sticky="w", padx=4, pady=(10, 2), columnspan=3)
        _row(t1, "SCF radius (Å):", _entry(self._xyz_scf_radius_var, 8), 7,
             tooltip="Cluster radius for self-consistency (~1st shell)")
        _row(t1, "Iterations:", _entry(self._xyz_scf_nscf_var, 8), 8,
             tooltip="Max SCF iterations (default 30)")
        _row(t1, "Mixing (ca):", _entry(self._xyz_scf_ca_var, 8), 9,
             tooltip="Charge mixing fraction (0.1–0.5)")
        _row(t1, "FMS radius (Å):", _entry(self._xyz_fms_radius_var, 8), 10,
             tooltip="Full multiple scattering cluster radius")

        # ── Tab 2: Spectrum ───────────────────────────────────────────────────
        t2 = tk.Frame(nb, padx=6, pady=6)
        nb.add(t2, text="Spectrum")

        _row(t2, "RPATH (Å):", _entry(self._xyz_rpath_var, 8), 0,
             tooltip="Max path length for EXAFS paths")
        _row(t2, "EXAFS kmax (Å⁻¹):", _entry(self._xyz_exafs_kmax_var, 8), 1,
             tooltip="Max k for EXAFS energy grid (EXAFS card)")
        _row(t2, "NLEG:", _entry(self._xyz_nleg_var, 8), 2,
             tooltip="Max path legs (0 = FEFF default; 4–8 typical)")

        tk.Label(t2, text="XANES energy grid:", font=("", 8, "bold")).grid(
            row=3, column=0, sticky="w", padx=4, pady=(10, 2), columnspan=3)
        _row(t2, "E min (eV):", _entry(self._xyz_xanes_emin_var, 8), 4)
        _row(t2, "E max (eV):", _entry(self._xyz_xanes_emax_var, 8), 5)
        _row(t2, "Step (eV):", _entry(self._xyz_xanes_estep_var, 8), 6,
             tooltip="0.1–0.5 eV typical (smaller = slower)")

        # ── Tab 3: Multipole ─────────────────────────────────────────────────
        t3 = tk.Frame(nb, padx=6, pady=6)
        nb.add(t3, text="Multipole")

        tk.Label(t3, text="Quadrupole / higher multipoles (MULTIPOLE card)",
                 font=("", 8, "bold")).grid(row=0, column=0, columnspan=3,
                                            sticky="w", padx=4, pady=(4, 8))
        _row(t3, "lmax:", _combo(self._xyz_multipole_lmax_var, (0, 1, 2, 3), 6), 1,
             tooltip="0/1=dipole only  2=+quadrupole  3=+octupole")
        _row(t3, "iorder:", _combo(self._xyz_multipole_iorder_var, (0, 1, 2), 6), 2,
             tooltip="Perturbation order for higher multipoles (2=default)")
        tk.Label(t3, text="lmax=0 omits the MULTIPOLE card (pure dipole, fastest).",
                 fg="gray", font=("", 7)).grid(row=3, column=0, columnspan=3,
                                               sticky="w", padx=4, pady=(0, 10))

        tk.Label(t3, text="Polarization (POLARIZATION card)",
                 font=("", 8, "bold")).grid(row=4, column=0, columnspan=3,
                                            sticky="w", padx=4, pady=(4, 4))
        _row(t3, "Enable:", _check(self._xyz_polarization_var), 5)
        _row(t3, "x:", _entry(self._xyz_pol_x_var, 8), 6)
        _row(t3, "y:", _entry(self._xyz_pol_y_var, 8), 7)
        _row(t3, "z:", _entry(self._xyz_pol_z_var, 8), 8,
             tooltip="E-field direction vector (need not be normalised)")

        tk.Label(t3, text="Ellipticity (ELLIPTICITY card)",
                 font=("", 8, "bold")).grid(row=9, column=0, columnspan=3,
                                            sticky="w", padx=4, pady=(10, 4))
        _row(t3, "Enable:", _check(self._xyz_ellip_var), 10)
        _row(t3, "ellip:", _entry(self._xyz_ellip_val_var, 8), 11,
             tooltip="0=linear  1=circular  fractional = elliptical")
        _row(t3, "x:", _entry(self._xyz_ellip_x_var, 8), 12)
        _row(t3, "y:", _entry(self._xyz_ellip_y_var, 8), 13)
        _row(t3, "z:", _entry(self._xyz_ellip_z_var, 8), 14,
             tooltip="Propagation direction vector")

        # ── Tab 4: Advanced ──────────────────────────────────────────────────
        t4 = tk.Frame(nb, padx=6, pady=6)
        nb.add(t4, text="Advanced")

        tk.Label(t4, text="Debye-Waller via correlated Debye model (DEBYE card)",
                 font=("", 8, "bold")).grid(row=0, column=0, columnspan=3,
                                            sticky="w", padx=4, pady=(4, 4))
        _row(t4, "Enable:", _check(self._xyz_debye_var), 1)
        _row(t4, "Temperature (K):", _entry(self._xyz_debye_temp_var, 8), 2,
             tooltip="Sample measurement temperature")
        _row(t4, "Debye temp (K):", _entry(self._xyz_debye_dtemp_var, 8), 3,
             tooltip="Characteristic Debye temperature of the material")

        tk.Button(win, text="Close", command=win.destroy,
                  font=("", 8)).pack(pady=(0, 8))

    def _browse_xyz_file(self):
        path = filedialog.askopenfilename(
            title="Select XYZ Structure",
            filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")],
        )
        if path:
            self._xyz_path_var.set(path)
            self._load_xyz_structure()

    def _load_xyz_structure(self, silent: bool = False):
        xyz_path = self._xyz_path_var.get().strip()
        if not xyz_path or not os.path.isfile(xyz_path):
            self._xyz_structure = None
            self._xyz_info_var.set("No XYZ structure loaded. CIF export uses a boxed P1 cell.")
            if not silent:
                messagebox.showwarning(
                    "XYZ Import",
                    "Select a valid .xyz structure file first.",
                    parent=self,
                )
            return

        try:
            structure = parse_xyz_file(xyz_path)
        except Exception as exc:
            self._xyz_structure = None
            self._xyz_info_var.set("Could not parse the selected XYZ file.")
            if not silent:
                messagebox.showerror("XYZ Import", str(exc), parent=self)
            return

        self._xyz_structure = structure
        if not self._bundle_base_var.get().strip():
            self._bundle_base_var.set(structure.basename)

        # Resolve the absorber spec ("Ni", "5", etc.) against the loaded
        # structure. On invalid input, fall back to atom #1 silently in this
        # display path; the bundle-write path raises a clear error instead.
        try:
            resolved_idx, note = _resolve_absorber_index(
                self._xyz_absorber_var.get(), structure
            )
        except Exception:
            resolved_idx, note = 1, "default to atom #1"

        padding = _coerce_float(self._xyz_padding_var.get(), 6.0)
        self._xyz_info_var.set(
            f"Loaded {structure.atom_count} atoms ({structure.formula}). "
            f"Absorber: {note}. Export uses a boxed P1 CIF with "
            f"{padding:.1f} A padding."
        )
        if not silent:
            self._append_feff_log(
                f"Loaded XYZ structure: {os.path.basename(xyz_path)} "
                f"({structure.atom_count} atoms, formula {structure.formula})"
            )

    def _write_xyz_feff_bundle(self):
        xyz_path = self._xyz_path_var.get().strip()
        workdir = self._feff_dir_var.get().strip()
        if not workdir:
            messagebox.showwarning(
                "FEFF Bundle",
                "Choose a FEFF working directory first.",
                parent=self,
            )
            return

        self._load_xyz_structure(silent=True)
        if self._xyz_structure is None:
            messagebox.showwarning(
                "FEFF Bundle",
                "Load a valid XYZ structure first.",
                parent=self,
            )
            return

        base = self._bundle_base_var.get().strip() or self._xyz_structure.basename
        safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("._") or "structure"
        collisions = []
        for candidate in (
            Path(workdir) / "feff.inp",
            Path(workdir) / f"{safe_base}.cif",
            Path(workdir) / f"{safe_base}.xyz",
        ):
            if candidate.exists():
                collisions.append(candidate.name)
        if collisions:
            should_overwrite = messagebox.askyesno(
                "Overwrite FEFF Bundle?",
                "The selected workdir already contains:\n"
                + "\n".join(collisions)
                + "\n\nOverwrite these files with the new XYZ-derived bundle?",
                parent=self,
            )
            if not should_overwrite:
                return

        pol = (
            (self._xyz_pol_x_var.get(), self._xyz_pol_y_var.get(), self._xyz_pol_z_var.get())
            if self._xyz_polarization_var.get() else None
        )
        ellip = (
            (self._xyz_ellip_val_var.get(), self._xyz_ellip_x_var.get(),
             self._xyz_ellip_y_var.get(), self._xyz_ellip_z_var.get())
            if self._xyz_ellip_var.get() else None
        )
        deb = (
            (self._xyz_debye_temp_var.get(), self._xyz_debye_dtemp_var.get())
            if self._xyz_debye_var.get() else None
        )
        # Resolve "Ni" / "5" / etc. against the loaded structure. Fail loud
        # if it can't be resolved — silently defaulting would silently send
        # the wrong absorber to FEFF and waste another long run.
        try:
            absorber_idx, absorber_note = _resolve_absorber_index(
                self._xyz_absorber_var.get(), self._xyz_structure
            )
        except Exception as exc:
            messagebox.showerror("Absorber", str(exc), parent=self)
            self._append_feff_log(f"Absorber error: {exc}")
            return
        self._append_feff_log(f"Absorber: {absorber_note}")

        try:
            bundle = export_xyz_as_feff_bundle(
                xyz_path,
                workdir,
                basename=base,
                padding=_coerce_float(self._xyz_padding_var.get(), 6.0),
                cubic=bool(self._xyz_cubic_var.get()),
                absorber_index=absorber_idx,
                edge=self._xyz_edge_var.get(),
                spectrum=self._xyz_spectrum_var.get(),
                kmesh=max(1, int(self._xyz_kmesh_var.get())),
                equivalence=max(1, min(4, int(self._xyz_equiv_var.get()))),
                xanes_emin=_coerce_float(self._xyz_xanes_emin_var.get(), -30.0),
                xanes_emax=_coerce_float(self._xyz_xanes_emax_var.get(), 250.0),
                xanes_estep=max(0.01, _coerce_float(self._xyz_xanes_estep_var.get(), 0.25)),
                s02=_coerce_float(self._xyz_s02_var.get(), 1.0),
                corehole=self._xyz_corehole_var.get(),
                exchange=int(self._xyz_exchange_var.get()),
                exchange_vr=_coerce_float(self._xyz_exchange_vr_var.get(), 0.0),
                exchange_vi=_coerce_float(self._xyz_exchange_vi_var.get(), 0.0),
                scf_radius=_coerce_float(self._xyz_scf_radius_var.get(), 4.0),
                scf_nscf=max(1, int(self._xyz_scf_nscf_var.get())),
                scf_ca=_coerce_float(self._xyz_scf_ca_var.get(), 0.2),
                fms_radius=_coerce_float(self._xyz_fms_radius_var.get(), 6.0),
                rpath=_coerce_float(self._xyz_rpath_var.get(), 8.0),
                nleg=int(self._xyz_nleg_var.get()),
                exafs_kmax=_coerce_float(self._xyz_exafs_kmax_var.get(), 20.0),
                multipole_lmax=int(self._xyz_multipole_lmax_var.get()),
                multipole_iorder=int(self._xyz_multipole_iorder_var.get()),
                polarization=pol,
                ellipticity=ellip,
                debye=deb,
                molecular_mode=bool(self._xyz_molecular_mode_var.get()),
                cluster_radius=_parse_optional_radius(
                    self._xyz_cluster_radius_var.get()
                ),
            )
        except Exception as exc:
            messagebox.showerror("FEFF Bundle", str(exc), parent=self)
            self._append_feff_log(f"XYZ -> FEFF bundle failed: {exc}")
            return

        self._xyz_structure = bundle["structure"]
        self._bundle_base_var.set(Path(bundle["cif_path"]).stem)
        cell = np.asarray(bundle["cell_lengths"], dtype=float)
        self._xyz_info_var.set(
            "Wrote FEFF bundle: "
            f"{os.path.basename(bundle['cif_path'])}, feff.inp, and XYZ copy. "
            f"Cell = {cell[0]:.2f} x {cell[1]:.2f} x {cell[2]:.2f} A (P1)."
        )
        self._append_feff_log(
            f"Wrote FEFF bundle from {os.path.basename(xyz_path)} into {workdir}"
        )
        self._append_feff_log(f"  CIF: {bundle['cif_path']}")
        self._append_feff_log(f"  FEFF input: {bundle['feff_inp_path']}")
        if bundle.get("molecular_mode"):
            self._append_feff_log(
                "  Mode: molecular (real-space ATOMS card; no RECIPROCAL/KMESH)."
            )
        else:
            self._append_feff_log(
                "  Mode: periodic (boxed P1 CIF + RECIPROCAL + KMESH). "
                "For isolated molecules, enabling Molecular mode is much faster."
            )
        if bundle.get("cluster_radius") is not None:
            self._append_feff_log(
                f"  Cluster: kept {bundle.get('atoms_used', '?')} of "
                f"{bundle.get('atoms_total_input', '?')} atoms within "
                f"{bundle['cluster_radius']:.1f} Å of absorber "
                f"(dropped {bundle.get('atoms_dropped', 0)})."
            )
        exe = self._feff_exe_var.get().strip()
        if exe and self._exe_is_feff8l(exe):
            self._append_feff_log(
                "WARNING: feff8l is EXAFS-only and cannot run this bundle "
                "(CIF / RECIPROCAL workflow requires FEFF10)."
            )
            if self._xyz_spectrum_var.get().upper() == "XANES":
                self._append_feff_log(
                    "         XANES also requires FEFF10. Use Help → FEFF Setup / Update."
                )
        self._load_feff_paths(silent=True)

    def _append_feff_log(self, text: str):
        self._feff_log.config(state=tk.NORMAL)
        self._feff_log.insert(tk.END, text.rstrip() + "\n")
        self._feff_log.see(tk.END)
        self._feff_log.config(state=tk.DISABLED)

    def _refresh_feff_tree(self):
        for item in self._feff_tree.get_children():
            self._feff_tree.delete(item)
        for i, path in enumerate(self._feff_paths):
            self._feff_tree.insert(
                "",
                "end",
                iid=str(i),
                values=(f"{path.index:04d}", f"{path.reff:.3f}",
                        f"{path.degen:.2f}", f"{path.nleg:d}"),
            )
        if self._feff_paths:
            self._feff_info_var.set(
                f"Loaded {len(self._feff_paths)} FEFF path file(s) from {self._feff_dir_var.get()}."
            )
        else:
            self._feff_info_var.set("No FEFF path files loaded.")

    def _preview_selected_feff_paths(self):
        self._apply_theme()
        paths = self._selected_feff_paths()
        if not paths and self._feff_paths:
            paths = [self._feff_paths[0]]
        if not paths:
            self._draw_empty_feff_preview()
            return

        self._ax_feff.clear()
        q_name = self._q_axis_symbol()
        for i, path in enumerate(paths[:6]):
            colour = xas_core._PALETTE[i % len(xas_core._PALETTE)]
            if len(path.q) > 1 and len(path.amp) == len(path.q):
                label = f"P{path.index:04d} | Reff={path.reff:.3f} A"
                self._ax_feff.plot(path.q, np.abs(path.amp), color=colour, lw=1.5, label=label)
        self._ax_feff.set_title("FEFF path amplitude preview", fontsize=9, loc="left", pad=3)
        self._ax_feff.set_xlabel(f"{q_name}  (A^-1)", fontsize=8)
        self._ax_feff.set_ylabel("Path amplitude", fontsize=8)
        self._ax_feff.tick_params(labelsize=7)
        self._ax_feff.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        if self._ax_feff.get_legend_handles_labels()[0]:
            self._ax_feff.legend(fontsize=7, loc="upper right", framealpha=0.85)
        if _HAS_SNS:
            sns.despine(ax=self._ax_feff, offset=4)
        self._canvas_feff.draw_idle()

    def _on_feff_selection(self):
        self._preview_selected_feff_paths()
        self._redraw()

    def _load_feff_paths(self, silent: bool = False):
        workdir = self._feff_dir_var.get().strip()
        if not workdir or not os.path.isdir(workdir):
            self._feff_paths = []
            self._refresh_feff_tree()
            self._preview_selected_feff_paths()
            self._redraw()
            if not silent:
                messagebox.showwarning("FEFF", "Select a valid FEFF working directory first.",
                                       parent=self)
            return

        pattern_paths = sorted(Path(workdir).glob("feff*.dat"))
        parsed: list[FeffPathData] = []
        failures: list[str] = []
        for path in pattern_paths:
            try:
                parsed.append(parse_feff_path_file(str(path)))
            except Exception as exc:
                failures.append(f"{path.name}: {exc}")

        parsed.sort(key=lambda item: item.index)
        self._feff_paths = parsed
        self._refresh_feff_tree()
        self._preview_selected_feff_paths()
        self._redraw()

        if parsed:
            self._append_feff_log(f"Loaded {len(parsed)} FEFF path file(s) from {workdir}")
        elif not silent:
            self._append_feff_log(f"No feff*.dat files found in {workdir}")

        if failures and not silent:
            self._append_feff_log("Some FEFF path files could not be parsed:")
            for failure in failures[:10]:
                self._append_feff_log(f"  - {failure}")

    @staticmethod
    def _exe_is_feff8l(exe: str) -> bool:
        return "feff8l" in os.path.basename(str(exe)).lower()

    @staticmethod
    def _feff_inp_flags(workdir: str) -> tuple[bool, bool]:
        """Return (is_xanes, needs_feff10) by scanning feff.inp."""
        inp = os.path.join(workdir, "feff.inp")
        try:
            text = Path(inp).read_text(encoding="utf-8", errors="replace").upper()
        except Exception:
            return False, False
        is_xanes = "XANES" in text
        needs_feff10 = any(
            kw in text for kw in ("CIF ", "\nCIF\n", "RECIPROCAL", "KMESH", "TARGET ", "EQUIVALENCE")
        )
        return is_xanes, needs_feff10

    def _resolve_feff_executable(self) -> str:
        user_path = self._feff_exe_var.get().strip()
        cfg_path = os.path.join(os.path.expanduser("~"), ".binah_config.json")
        managed = feff_manager.discover_feff_executable(
            preferred_path=user_path,
            cfg_path=cfg_path,
        )
        if managed and os.path.isfile(managed):
            return managed
        for candidate in FEFF_EXE_CANDIDATES:
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        return user_path

    def _run_feff(self):
        workdir = self._feff_dir_var.get().strip()
        if not workdir or not os.path.isdir(workdir):
            messagebox.showwarning("FEFF", "Select a valid FEFF working directory first.",
                                   parent=self)
            return
        if not os.path.exists(os.path.join(workdir, "feff.inp")):
            messagebox.showwarning(
                "FEFF",
                "This folder does not contain feff.inp.\nChoose a FEFF input directory first.",
                parent=self,
            )
            return

        exe = self._resolve_feff_executable()
        if not exe or not os.path.exists(exe):
            messagebox.showwarning(
                "FEFF",
                "No FEFF executable was found.\nBrowse to feff.exe / feff8l.exe first.",
                parent=self,
            )
            return

        if self._exe_is_feff8l(exe):
            is_xanes, needs_feff10 = self._feff_inp_flags(workdir)
            issues = []
            if is_xanes:
                issues.append("• XANES keyword detected — feff8l is an EXAFS-only build")
            if needs_feff10:
                issues.append("• CIF / RECIPROCAL / KMESH workflow requires FEFF10")
            if issues:
                messagebox.showerror(
                    "feff8l — Incompatible Input",
                    "The selected executable (feff8l) cannot run this feff.inp:\n\n"
                    + "\n".join(issues)
                    + "\n\nSolutions:\n"
                    "  • Use Help → FEFF Setup / Update to install FEFF10\n"
                    "  • Browse to a FEFF10 'feff.exe' or 'feff.cmd' instead",
                    parent=self,
                )
                return

        self._feff_exe_var.set(exe)
        self._run_feff_btn.config(state=tk.DISABLED)
        self._append_feff_log(f"Running FEFF: {exe}")
        self._append_feff_log(f"  workdir: {workdir}")

        thread = threading.Thread(
            target=self._run_feff_worker,
            args=(exe, workdir),
            daemon=True,
        )
        thread.start()

    def _run_feff_worker(self, exe: str, workdir: str):
        try:
            lower = exe.lower()
            command = ["cmd", "/c", exe] if lower.endswith((".cmd", ".bat")) else [exe]
            proc = subprocess.Popen(
                command,
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for raw in proc.stdout:
                line = raw.rstrip()
                if line.startswith("[FEFF]"):
                    status = line[6:].strip()
                    self.after(0, lambda s=status: self._feff_status_var.set(s))
                else:
                    self.after(0, lambda s=line: self._append_feff_log(f"  {s}"))
            proc.wait()
            self.after(0, lambda rc=proc.returncode: self._finish_feff_run(rc))
        except Exception as exc:
            self.after(0, lambda e=str(exc): self._finish_feff_run(-1, e))

    def _finish_feff_run(self, returncode: int, error: str = ""):
        self._run_feff_btn.config(state=tk.NORMAL)
        self._feff_status_var.set("")
        if error:
            self._append_feff_log(f"FEFF error: {error}")
        self._append_feff_log(f"FEFF finished (return code {returncode})")
        self._load_feff_paths(silent=True)
        # Rename FEFF outputs to a descriptive name and load the spectrum
        # into the main plot.  Best-effort: any failure here is logged but
        # doesn't break the run.
        if returncode == 0:
            try:
                self._archive_and_load_feff_outputs()
            except Exception as exc:
                self._append_feff_log(f"  Could not auto-load FEFF output: {exc}")

    # ------------------------------------------------------------------ #
    #  FEFF output handling                                                #
    # ------------------------------------------------------------------ #
    def _feff_output_label(self) -> str:
        """Return a descriptive label like 'PK-26a_6.0A' for naming outputs."""
        base = (self._bundle_base_var.get().strip()
                or (self._xyz_structure.basename if self._xyz_structure else "")
                or "feff_run")
        radius = _parse_optional_radius(self._xyz_cluster_radius_var.get())
        if radius is not None:
            base = f"{base}_{radius:.1f}A"
        return base

    def _archive_and_load_feff_outputs(self,
                                       workdir: str | None = None,
                                       label: str | None = None,
                                       emin: float | None = None,
                                       emax: float | None = None,
                                       is_xanes: bool | None = None):
        """After a successful FEFF run:
          * copy xmu.dat / chi.dat to <base>[_<R>A]_xmu.dat etc. in the workdir
          * load the spectrum into Binah's plot via the main app

        All arguments default to the instance state so the single-run flow
        keeps working unchanged; the batch worker passes explicit values per
        XYZ since the instance state is shared across the whole batch.
        """
        if workdir is None:
            workdir = self._feff_dir_var.get().strip()
        if not workdir or not os.path.isdir(workdir):
            return

        if label is None:
            label = self._feff_output_label()
        if emin is None:
            emin = _coerce_float(self._xyz_xanes_emin_var.get(), -30.0)
        if emax is None:
            emax = _coerce_float(self._xyz_xanes_emax_var.get(), 250.0)
        if is_xanes is None:
            is_xanes = self._xyz_spectrum_var.get().strip().upper() == "XANES"

        # Map of FEFF output -> (descriptive suffix, parser).
        # xmu.dat is XANES/total absorption, chi.dat is EXAFS chi(k).
        candidates = [
            ("xmu.dat", "_xmu.dat", self._scan_from_xmu_dat),
            ("chi.dat", "_chi.dat", self._scan_from_chi_dat),
        ]

        archived: list[Path] = []
        loaded_scan: ExperimentalScan | None = None
        for src_name, suffix, parser in candidates:
            src = Path(workdir) / src_name
            if not src.exists():
                continue
            dst = Path(workdir) / f"{label}{suffix}"
            try:
                shutil.copy2(src, dst)
                archived.append(dst)
            except Exception as exc:
                self._append_feff_log(f"  Could not copy {src.name}: {exc}")
                continue
            # Prefer xmu.dat for the plot (it has total absorption mu(E)).
            if loaded_scan is None:
                try:
                    if src_name == "xmu.dat":
                        loaded_scan = self._scan_from_xmu_dat(
                            dst, label,
                            emin=emin if is_xanes else None,
                            emax=emax if is_xanes else None,
                        )
                    else:
                        loaded_scan = parser(dst, label)
                except Exception as exc:
                    self._append_feff_log(
                        f"  Could not parse {dst.name} as a scan: {exc}"
                    )

        if archived:
            self._append_feff_log("  Archived outputs:")
            for p in archived:
                self._append_feff_log(f"    {p.name}")

        if loaded_scan is not None and self._add_exp_scan_fn is not None:
            try:
                self._add_exp_scan_fn(loaded_scan)
                self._append_feff_log(
                    f"  Loaded into plot as: {loaded_scan.label}"
                )
            except Exception as exc:
                self._append_feff_log(f"  Could not push to plot: {exc}")

    @staticmethod
    def _scan_from_xmu_dat(path: Path, label: str,
                           emin: float | None = None,
                           emax: float | None = None) -> ExperimentalScan:
        """Parse FEFF10's xmu.dat into an ExperimentalScan.

        FEFF10 columns (after the # header block):
            col 0: omega   absolute photon energy in eV
            col 1: e       energy relative to threshold (E - E0) in eV
            col 2: k       Å⁻¹
            col 3: mu      total absorption (what we plot)
            col 4: mu0     smooth atomic background
            col 5: chi     k^kweight * (mu - mu0)/mu0

        We plot vs absolute energy (col 0) and, when `emin`/`emax` are given
        (the user's XANES window in relative eV), clip rows where col 1 is
        outside that window — otherwise the plot covers the full k=0..k_max
        range FEFF always writes (~1500 eV) and the XANES detail vanishes.
        """
        data = np.loadtxt(str(path), comments="#")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 4:
            raise ValueError(
                f"xmu.dat has only {data.shape[1]} columns, expected at least 4."
            )

        omega    = np.asarray(data[:, 0], dtype=float)  # absolute energy (eV)
        e_rel    = np.asarray(data[:, 1], dtype=float)  # relative energy (eV)
        mu_total = np.asarray(data[:, 3], dtype=float)

        # E0 = the omega where the relative energy crosses zero (threshold).
        try:
            zero_idx = int(np.argmin(np.abs(e_rel)))
            e0 = float(omega[zero_idx])
        except Exception:
            e0 = float(omega[0])

        # Clip to the user's XANES window so the rising edge isn't squished
        # into the leftmost 1.5% of a 1500 eV plot.
        if emin is not None and emax is not None:
            keep = (e_rel >= float(emin)) & (e_rel <= float(emax))
            if keep.any():
                omega    = omega[keep]
                e_rel    = e_rel[keep]
                mu_total = mu_total[keep]

        return ExperimentalScan(
            label=f"FEFF: {label}",
            source_file=str(path),
            energy_ev=omega,
            mu=mu_total,
            e0=e0,
            is_normalized=True,
            scan_type="FEFF calculation",
            metadata={
                "source": "feff",
                "filename": path.name,
                "e0_eV": e0,
                "n_points": int(omega.size),
                "energy_range_eV": [float(omega[0]), float(omega[-1])],
            },
        )

    @staticmethod
    def _scan_from_chi_dat(path: Path, label: str) -> ExperimentalScan:
        """Parse FEFF10's chi.dat (k, chi(k))."""
        data = np.loadtxt(str(path), comments="#")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # chi.dat columns: k[Å⁻¹], chi[k], mag, phase ...
        k = np.asarray(data[:, 0], dtype=float)
        chi = np.asarray(data[:, 1], dtype=float) if data.shape[1] > 1 else np.zeros_like(k)
        return ExperimentalScan(
            label=f"FEFF χ(k): {label}",
            source_file=str(path),
            energy_ev=k,
            mu=chi,
            e0=0.0,
            is_normalized=True,
            scan_type="FEFF chi(k)",
            metadata={"source": "feff", "filename": path.name, "k_axis": True},
        )

    # ------------------------------------------------------------------ #
    #  Batch FEFF runs                                                     #
    # ------------------------------------------------------------------ #
    def _on_batch_run_clicked(self):
        """Pick N .xyz files and queue a sequential FEFF run for each."""
        paths = filedialog.askopenfilenames(
            title="Pick .xyz files for batch FEFF",
            filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")],
            parent=self,
        )
        if not paths:
            return

        base_workdir = self._feff_dir_var.get().strip()
        if not base_workdir:
            messagebox.showwarning(
                "Batch FEFF",
                "Pick a FEFF working directory first (Workdir at the top).",
                parent=self,
            )
            return
        exe = self._feff_exe_var.get().strip()
        if not exe or not os.path.isfile(exe):
            messagebox.showwarning(
                "Batch FEFF",
                "Pick a valid FEFF executable first (Executable at the top).",
                parent=self,
            )
            return

        radius = _parse_optional_radius(self._xyz_cluster_radius_var.get())
        radius_text = f"{radius:.1f} Å" if radius is not None else "(no crop)"
        absorber_spec = self._xyz_absorber_var.get().strip() or "1"

        if not messagebox.askyesno(
            "Batch FEFF",
            f"Run FEFF sequentially on {len(paths)} XYZ file(s)?\n\n"
            f"  Workdir:   {base_workdir}\n"
            f"  Executable: {os.path.basename(exe)}\n"
            f"  Absorber:   {absorber_spec}\n"
            f"  Cluster:    {radius_text}\n"
            f"  Spectrum:   {self._xyz_spectrum_var.get()}\n"
            f"  Mol. mode:  {bool(self._xyz_molecular_mode_var.get())}\n\n"
            "Each XYZ goes into its own subdirectory of the workdir.\n"
            "Failed runs are logged and skipped — the batch keeps going.",
            parent=self,
        ):
            return

        # Snapshot every relevant setting on the main thread so the worker
        # never reaches into Tk state (and so the user can keep editing the
        # form without affecting in-flight runs).
        try:
            settings = {
                "padding":       _coerce_float(self._xyz_padding_var.get(), 6.0),
                "cubic":         bool(self._xyz_cubic_var.get()),
                "absorber_spec": absorber_spec,
                "edge":          self._xyz_edge_var.get(),
                "spectrum":      self._xyz_spectrum_var.get(),
                "kmesh":         max(1, int(self._xyz_kmesh_var.get())),
                "equivalence":   max(1, min(4, int(self._xyz_equiv_var.get()))),
                "xanes_emin":    _coerce_float(self._xyz_xanes_emin_var.get(), -30.0),
                "xanes_emax":    _coerce_float(self._xyz_xanes_emax_var.get(), 250.0),
                "xanes_estep":   max(0.01, _coerce_float(self._xyz_xanes_estep_var.get(), 0.25)),
                "s02":           _coerce_float(self._xyz_s02_var.get(), 1.0),
                "corehole":      self._xyz_corehole_var.get(),
                "exchange":      int(self._xyz_exchange_var.get()),
                "exchange_vr":   _coerce_float(self._xyz_exchange_vr_var.get(), 0.0),
                "exchange_vi":   _coerce_float(self._xyz_exchange_vi_var.get(), 0.0),
                "scf_radius":    _coerce_float(self._xyz_scf_radius_var.get(), 4.0),
                "scf_nscf":      max(1, int(self._xyz_scf_nscf_var.get())),
                "scf_ca":        _coerce_float(self._xyz_scf_ca_var.get(), 0.2),
                "fms_radius":    _coerce_float(self._xyz_fms_radius_var.get(), 6.0),
                "rpath":         _coerce_float(self._xyz_rpath_var.get(), 8.0),
                "nleg":          int(self._xyz_nleg_var.get()),
                "exafs_kmax":    _coerce_float(self._xyz_exafs_kmax_var.get(), 20.0),
                "multipole_lmax":   int(self._xyz_multipole_lmax_var.get()),
                "multipole_iorder": int(self._xyz_multipole_iorder_var.get()),
                "molecular_mode":   bool(self._xyz_molecular_mode_var.get()),
                "cluster_radius":   radius,
            }
        except Exception as exc:
            messagebox.showerror(
                "Batch FEFF",
                f"Could not read XYZ-tab settings:\n{exc}\n\n"
                "Fix any invalid numeric fields and try again.",
                parent=self,
            )
            return

        self._run_feff_btn.config(state=tk.DISABLED)
        self._batch_btn.config(state=tk.DISABLED)
        self._append_feff_log(
            f"=== Batch FEFF: {len(paths)} XYZ file(s) queued ==="
        )
        threading.Thread(
            target=self._batch_feff_worker,
            args=(list(paths), base_workdir, exe, settings),
            daemon=True,
        ).start()

    def _batch_feff_worker(self, xyz_paths: list[str], base_workdir: str,
                           exe: str, settings: dict):
        """Background thread: run FEFF on each XYZ in turn and load each
        result into the plot.  Tk vars are NOT touched from here — UI
        updates go through self.after()."""

        def log(msg: str):
            self.after(0, lambda m=msg: self._append_feff_log(m))

        def push_scan(scan):
            if scan is not None and self._add_exp_scan_fn is not None:
                self.after(0, lambda s=scan: self._add_exp_scan_fn(s))

        n = len(xyz_paths)
        succeeded = 0
        failed: list[str] = []

        for idx, xyz_path in enumerate(xyz_paths, start=1):
            xyz_name = os.path.basename(xyz_path)
            log("")
            log(f"--- [{idx}/{n}] {xyz_name} ---")

            try:
                # Per-XYZ subdirectory keeps outputs separated.  When a
                # cluster radius is set, tag the folder with it so multiple
                # radii of the same molecule sit side-by-side without
                # clobbering each other (PK-26a_5.0A/, PK-26a_7.0A/, …).
                xyz_stem = re.sub(
                    r"[^A-Za-z0-9_.-]+", "_",
                    Path(xyz_path).stem,
                ).strip("._") or f"xyz_{idx}"
                radius = settings["cluster_radius"]
                folder_name = (f"{xyz_stem}_{radius:.1f}A"
                               if radius is not None else xyz_stem)
                sub = Path(base_workdir) / folder_name
                sub.mkdir(parents=True, exist_ok=True)

                # Resolve absorber against THIS XYZ (not the loaded structure
                # in the UI) so element-symbol lookups work per-file.
                structure = parse_xyz_file(xyz_path)
                absorber_idx, absorber_note = _resolve_absorber_index(
                    settings["absorber_spec"], structure
                )
                log(f"  Absorber: {absorber_note}")

                bundle = export_xyz_as_feff_bundle(
                    xyz_path,
                    str(sub),
                    basename=xyz_stem,
                    padding=settings["padding"],
                    cubic=settings["cubic"],
                    absorber_index=absorber_idx,
                    edge=settings["edge"],
                    spectrum=settings["spectrum"],
                    kmesh=settings["kmesh"],
                    equivalence=settings["equivalence"],
                    xanes_emin=settings["xanes_emin"],
                    xanes_emax=settings["xanes_emax"],
                    xanes_estep=settings["xanes_estep"],
                    s02=settings["s02"],
                    corehole=settings["corehole"],
                    exchange=settings["exchange"],
                    exchange_vr=settings["exchange_vr"],
                    exchange_vi=settings["exchange_vi"],
                    scf_radius=settings["scf_radius"],
                    scf_nscf=settings["scf_nscf"],
                    scf_ca=settings["scf_ca"],
                    fms_radius=settings["fms_radius"],
                    rpath=settings["rpath"],
                    nleg=settings["nleg"],
                    exafs_kmax=settings["exafs_kmax"],
                    multipole_lmax=settings["multipole_lmax"],
                    multipole_iorder=settings["multipole_iorder"],
                    molecular_mode=settings["molecular_mode"],
                    cluster_radius=settings["cluster_radius"],
                )
                kept = bundle.get("atoms_used", "?")
                total = bundle.get("atoms_total_input", "?")
                log(f"  Bundle written ({kept}/{total} atoms used)")

                # Run FEFF synchronously — never spawn parallel calculations,
                # since each FEFF run is already CPU-saturated.
                lower = exe.lower()
                cmd = ["cmd", "/c", exe] if lower.endswith((".cmd", ".bat")) else [exe]
                log(f"  Running FEFF in {sub} ...")
                proc = subprocess.run(
                    cmd, cwd=str(sub), capture_output=True, text=True,
                    check=False,
                )
                if proc.returncode != 0:
                    log(f"  FEFF returned {proc.returncode}; skipping load.")
                    if proc.stderr:
                        for ln in proc.stderr.splitlines()[-5:]:
                            log(f"    {ln}")
                    failed.append(xyz_name)
                    continue

                log(f"  FEFF finished (rc=0)")

                # Build a descriptive label like "PK-26a_6.0A".
                radius = settings["cluster_radius"]
                label = (f"{xyz_stem}_{radius:.1f}A"
                         if radius is not None else xyz_stem)

                # Archive and parse on the worker (pure file IO + numpy).
                # Only the plot-add must happen on the main thread.
                self._archive_and_load_feff_outputs_threadsafe(
                    workdir=str(sub),
                    label=label,
                    settings=settings,
                    log=log,
                    push_scan=push_scan,
                )
                succeeded += 1
            except Exception as exc:
                log(f"  FAILED: {exc}")
                failed.append(xyz_name)

        log("")
        log(f"=== Batch complete: {succeeded}/{n} succeeded"
            + (f", failed: {', '.join(failed)}" if failed else "")
            + " ===")
        self.after(0, self._on_batch_finished)

    def _archive_and_load_feff_outputs_threadsafe(self, workdir: str,
                                                   label: str, settings: dict,
                                                   log, push_scan):
        """Worker-thread variant: archive xmu.dat/chi.dat under <label>_xmu.dat
        etc., parse the spectrum, and hand it to the main thread for plotting.
        Mirrors `_archive_and_load_feff_outputs` but is safe to call off the
        Tk thread."""
        is_xanes = settings["spectrum"].strip().upper() == "XANES"
        emin = settings["xanes_emin"]
        emax = settings["xanes_emax"]

        archived = []
        loaded_scan = None
        for src_name, suffix in (("xmu.dat", "_xmu.dat"),
                                 ("chi.dat", "_chi.dat")):
            src = Path(workdir) / src_name
            if not src.exists():
                continue
            dst = Path(workdir) / f"{label}{suffix}"
            try:
                shutil.copy2(src, dst)
                archived.append(dst.name)
            except Exception as exc:
                log(f"    Could not copy {src.name}: {exc}")
                continue
            if loaded_scan is None:
                try:
                    if src_name == "xmu.dat":
                        loaded_scan = self._scan_from_xmu_dat(
                            dst, label,
                            emin=emin if is_xanes else None,
                            emax=emax if is_xanes else None,
                        )
                    else:
                        loaded_scan = self._scan_from_chi_dat(dst, label)
                except Exception as exc:
                    log(f"    Could not parse {dst.name}: {exc}")

        if archived:
            log(f"  Archived: {', '.join(archived)}")
        if loaded_scan is not None:
            push_scan(loaded_scan)
            log(f"  Pushed to plot: {loaded_scan.label}")

    def _on_batch_finished(self):
        """Re-enable the action buttons after a batch completes."""
        try:
            self._run_feff_btn.config(state=tk.NORMAL)
            self._batch_btn.config(state=tk.NORMAL)
        except Exception:
            pass

    def get_params(self) -> dict:
        # Defensive read: when a panel is hidden (analysis-only or FEFF-only
        # mode), the corresponding Tk vars don't exist. We skip them silently
        # so the project file only contains what actually exists.
        def _g(attr_name, default=None):
            var = getattr(self, attr_name, None)
            if var is None:
                return default
            try:
                return var.get()
            except Exception:
                return default

        out: dict = {}
        # Analysis params (only present when show_analysis_panel=True)
        analysis_keys = [
            ("e0", "_e0_var"), ("pre1", "_pre1_var"), ("pre2", "_pre2_var"),
            ("nor1", "_nor1_var"), ("nor2", "_nor2_var"),
            ("nnorm", "_nnorm_var"), ("rbkg", "_rbkg_var"),
            ("kmin_bkg", "_kmin_bkg_var"),
            ("qmin", "_qmin_var"), ("qmax", "_qmax_var"), ("dq", "_dq_var"),
            ("qweight", "_qweight_var"), ("q_window", "_qwin_var"),
            ("rmin", "_rmin_var"), ("rmax", "_rmax_var"), ("dr", "_dr_var"),
            ("r_window", "_rwin_var"), ("r_display", "_rdisplay_var"),
            ("style", "_style_var"), ("context", "_context_var"),
            ("use_q_label", "_use_q_label_var"),
            ("show_q_window", "_show_q_window_var"),
            ("show_r_window", "_show_r_window_var"),
            ("show_feff_markers", "_show_feff_markers_var"),
        ]
        for key, attr in analysis_keys:
            val = _g(attr)
            if val is not None:
                out[key] = val
        if hasattr(self, "_selected_labels"):
            out["selected_labels"] = list(self._selected_labels)

        # FEFF params (only present when show_feff_panel=True)
        feff_keys = [
            ("feff_dir", "_feff_dir_var"), ("feff_exe", "_feff_exe_var"),
            ("xyz_path", "_xyz_path_var"),
            ("bundle_base", "_bundle_base_var"),
            ("xyz_padding", "_xyz_padding_var"),
            ("xyz_cubic", "_xyz_cubic_var"),
            ("xyz_absorber", "_xyz_absorber_var"),
            ("xyz_edge", "_xyz_edge_var"),
            ("xyz_spectrum", "_xyz_spectrum_var"),
            ("xyz_kmesh", "_xyz_kmesh_var"),
            ("xyz_equivalence", "_xyz_equiv_var"),
            ("xyz_xanes_emin", "_xyz_xanes_emin_var"),
            ("xyz_xanes_emax", "_xyz_xanes_emax_var"),
            ("xyz_xanes_estep", "_xyz_xanes_estep_var"),
        ]
        for key, attr in feff_keys:
            val = _g(attr)
            if val is not None:
                out[key] = val
        if hasattr(self, "_xyz_molecular_mode_var"):
            try:
                out["xyz_molecular_mode"] = bool(self._xyz_molecular_mode_var.get())
            except Exception:
                pass
        if hasattr(self, "_xyz_cluster_radius_var"):
            try:
                out["xyz_cluster_radius"] = str(self._xyz_cluster_radius_var.get())
            except Exception:
                pass
        return out

    def set_params(self, data: dict) -> None:
        # Defensive write: any var that doesn't exist on this instance (the
        # other panel was hidden) is silently skipped, so the same payload
        # can be safely applied to either flavour of EXAFSAnalysisTab.
        def _set(var, key, cast=float):
            if var is None or key not in data:
                return
            try:
                var.set(cast(data[key]))
            except Exception:
                pass

        def _v(attr_name):
            return getattr(self, attr_name, None)

        def _set_str(attr_name, key):
            var = _v(attr_name)
            if var is not None and key in data:
                try:
                    var.set(str(data[key]))
                except Exception:
                    pass

        def _set_bool(attr_name, key):
            var = _v(attr_name)
            if var is not None and key in data:
                try:
                    var.set(bool(data[key]))
                except Exception:
                    pass

        # Analysis params (silently skipped when this instance hides analysis).
        _set(_v("_e0_var"), "e0")
        _set(_v("_pre1_var"), "pre1")
        _set(_v("_pre2_var"), "pre2")
        _set(_v("_nor1_var"), "nor1")
        _set(_v("_nor2_var"), "nor2")
        _set(_v("_nnorm_var"), "nnorm", int)
        _set(_v("_rbkg_var"), "rbkg")
        _set(_v("_kmin_bkg_var"), "kmin_bkg")
        _set(_v("_qmin_var"), "qmin")
        _set(_v("_qmax_var"), "qmax")
        _set(_v("_dq_var"), "dq")
        _set(_v("_qweight_var"), "qweight", int)
        _set_str("_qwin_var", "q_window")
        _set(_v("_rmin_var"), "rmin")
        _set(_v("_rmax_var"), "rmax")
        _set(_v("_dr_var"), "dr")
        _set_str("_rwin_var", "r_window")
        _set(_v("_rdisplay_var"), "r_display")
        _set_str("_style_var", "style")
        _set_str("_context_var", "context")
        _set_bool("_use_q_label_var", "use_q_label")
        _set_bool("_show_q_window_var", "show_q_window")
        _set_bool("_show_r_window_var", "show_r_window")
        _set_bool("_show_feff_markers_var", "show_feff_markers")

        if hasattr(self, "_selected_labels"):
            self._pending_selected_labels = list(data.get("selected_labels", []))

        # FEFF params (silently skipped when this instance hides FEFF).
        _set_str("_feff_dir_var", "feff_dir")
        _set_str("_feff_exe_var", "feff_exe")
        _set_str("_xyz_path_var", "xyz_path")
        _set_str("_bundle_base_var", "bundle_base")
        _set(_v("_xyz_padding_var"), "xyz_padding")
        _set_bool("_xyz_cubic_var", "xyz_cubic")
        # absorber may be stored as int (legacy) or string ("Ni"); always
        # round-trip as a string so element symbols are preserved.
        _set_str("_xyz_absorber_var", "xyz_absorber")
        _set_str("_xyz_edge_var", "xyz_edge")
        _set_str("_xyz_spectrum_var", "xyz_spectrum")
        _set(_v("_xyz_kmesh_var"), "xyz_kmesh", int)
        _set(_v("_xyz_equiv_var"), "xyz_equivalence", int)
        _set(_v("_xyz_xanes_emin_var"), "xyz_xanes_emin")
        _set(_v("_xyz_xanes_emax_var"), "xyz_xanes_emax")
        _set(_v("_xyz_xanes_estep_var"), "xyz_xanes_estep")
        _set_bool("_xyz_molecular_mode_var", "xyz_molecular_mode")
        _set_str("_xyz_cluster_radius_var", "xyz_cluster_radius")

        # Auto-load follow-ups, but only if the corresponding panel exists.
        xyz_var = _v("_xyz_path_var")
        if xyz_var is not None and str(xyz_var.get()).strip():
            try:
                self._load_xyz_structure(silent=True)
            except Exception:
                pass
        feff_dir_var = _v("_feff_dir_var")
        if feff_dir_var is not None and str(feff_dir_var.get()).strip():
            try:
                self._load_feff_paths(silent=True)
            except Exception:
                pass
        if self._show_analysis_panel:
            try:
                self._redraw()
            except Exception:
                pass
