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
                 replot_fn: Optional[Callable] = None):
        super().__init__(parent)
        self._get_scans = get_scans_fn
        self._replot_fn = replot_fn

        self._results: dict = {}
        self._selected_labels: list[str] = []
        self._scan_vis_vars: dict[str, tk.BooleanVar] = {}
        self._feff_paths: list[FeffPathData] = []
        self._xyz_structure = None
        self._pending_selected_labels: list[str] = []
        self._build_ui()

    def _build_ui(self):
        top = tk.Frame(self, bd=1, relief=tk.GROOVE, padx=4, pady=3)
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Label(top, text="Scan:", font=("", 9, "bold")).pack(side=tk.LEFT)
        self._scan_var = tk.StringVar()
        self._scan_cb = ttk.Combobox(top, textvariable=self._scan_var,
                                     state="readonly", width=38)
        self._scan_cb.pack(side=tk.LEFT, padx=(4, 8))
        self._scan_cb.bind("<<ComboboxSelected>>", lambda _e: self._auto_fill_e0())

        tk.Button(top, text="Refresh Scans", font=("", 8),
                  command=self.refresh_scan_list).pack(side=tk.LEFT, padx=2)
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        tk.Button(top, text="Run EXAFS", bg="#003366", fg="white",
                  font=("", 9, "bold"), command=self._run).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="Update Views", font=("", 8),
                  command=self._redraw).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="+ Add to Overlay", font=("", 8),
                  command=self._add_overlay).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="Clear Overlay", font=("", 8),
                  command=self._clear_overlay).pack(side=tk.LEFT, padx=2)

        self._status_lbl = tk.Label(
            top,
            text="Load experimental scans first, then run EXAFS on the scan of interest.",
            fg="gray",
            font=("", 8),
        )
        self._status_lbl.pack(side=tk.LEFT, padx=10)

        body = tk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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
            fg="white",
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
                 bg="#003366", fg="white").pack(side=tk.LEFT, padx=6)
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

        feff_tab = tk.Frame(self._views_nb)
        self._views_nb.add(feff_tab, text="FEFF")
        self._build_feff_tab(feff_tab)

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
        self._xyz_absorber_var = tk.IntVar(value=1)
        self._xyz_edge_var = tk.StringVar(value="K")
        self._xyz_spectrum_var = tk.StringVar(value="EXAFS")
        self._xyz_kmesh_var = tk.IntVar(value=200)
        self._xyz_equiv_var = tk.IntVar(value=2)

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
                                       bg="#6B0000", fg="white",
                                       activebackground="#8B0000",
                                       command=self._run_feff)
        self._run_feff_btn.grid(row=1, column=3, padx=2, pady=(3, 0))

        tk.Label(top, textvariable=self._feff_info_var, fg="#003366",
                 font=("", 8)).grid(row=2, column=0, columnspan=4,
                                    sticky="w", pady=(5, 0))
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

        tk.Label(xyz_box, text="Absorber #:", font=("", 8, "bold")).grid(
            row=2, column=0, sticky="w", pady=(4, 0)
        )
        ttk.Entry(xyz_box, textvariable=self._xyz_absorber_var, width=8).grid(
            row=2, column=1, sticky="w", padx=4, pady=(4, 0)
        )
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

        tk.Label(xyz_box, text="KMESH:", font=("", 8, "bold")).grid(
            row=3, column=0, sticky="w", pady=(4, 0)
        )
        ttk.Entry(xyz_box, textvariable=self._xyz_kmesh_var, width=8).grid(
            row=3, column=1, sticky="w", padx=4, pady=(4, 0)
        )
        tk.Label(xyz_box, text="Equivalence:", font=("", 8, "bold")).grid(
            row=3, column=2, sticky="e", pady=(4, 0)
        )
        ttk.Entry(xyz_box, textvariable=self._xyz_equiv_var, width=8).grid(
            row=3, column=3, sticky="w", padx=4, pady=(4, 0)
        )
        tk.Button(
            xyz_box,
            text="Write FEFF Bundle",
            font=("", 8, "bold"),
            bg="#003366",
            fg="white",
            activebackground="#004C99",
            command=self._write_xyz_feff_bundle,
        ).grid(row=3, column=4, columnspan=2, sticky="ew", padx=(8, 0), pady=(4, 0))

        tk.Label(
            xyz_box,
            textvariable=self._xyz_info_var,
            fg="#003366",
            font=("", 8),
            justify="left",
        ).grid(row=4, column=0, columnspan=6, sticky="w", pady=(6, 0))
        xyz_box.columnconfigure(1, weight=1)
        xyz_box.columnconfigure(3, weight=0)

        body = tk.PanedWindow(parent, orient=tk.HORIZONTAL, sashwidth=5, sashrelief=tk.RAISED)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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

        self._feff_log = tk.Text(parent, height=6, font=("Courier", 8), state=tk.DISABLED)
        self._feff_log.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=(3, 0))

        self._draw_empty_feff_preview()

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
        return "q" if self._use_q_label_var.get() else "k"

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
        label = self._scan_var.get()
        scan = self._get_scan_by_label(label)
        if scan is None:
            return
        energy, mu, stored_e0 = self._source_arrays(scan)
        e0 = stored_e0 if stored_e0 > 100 else xas_core.find_e0(energy, mu)
        self._e0_var.set(float(e0))

    def _rebuild_scan_list_rows(self):
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
        if not labels:
            self._draw_empty_workspace()

    def auto_run_all(self):
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

    def _draw_feff_markers(self, ax):
        if not self._show_feff_markers_var.get() or not self._feff_paths:
            return
        selected = self._selected_feff_paths()
        paths = selected if selected else self._feff_paths
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
        absorber = self._xyz_absorber_var.get()
        if absorber < 1 or absorber > structure.atom_count:
            self._xyz_absorber_var.set(1)
        padding = _coerce_float(self._xyz_padding_var.get(), 6.0)
        self._xyz_info_var.set(
            f"Loaded {structure.atom_count} atoms ({structure.formula}). "
            f"Export will write a boxed P1 CIF with {padding:.1f} A padding. "
            "TARGET uses the 1-based XYZ atom index."
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

        try:
            bundle = export_xyz_as_feff_bundle(
                xyz_path,
                workdir,
                basename=base,
                padding=_coerce_float(self._xyz_padding_var.get(), 6.0),
                cubic=bool(self._xyz_cubic_var.get()),
                absorber_index=max(1, int(self._xyz_absorber_var.get())),
                edge=self._xyz_edge_var.get(),
                spectrum=self._xyz_spectrum_var.get(),
                kmesh=max(1, int(self._xyz_kmesh_var.get())),
                equivalence=max(1, min(4, int(self._xyz_equiv_var.get()))),
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
        self._append_feff_log(
            "  Note: XYZ -> CIF uses a boxed P1 cell because XYZ files do not contain lattice metadata."
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
            command = [exe]
            lower = exe.lower()
            if lower.endswith(".cmd") or lower.endswith(".bat"):
                command = ["cmd", "/c", exe]
            proc = subprocess.run(
                command,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            self.after(0, lambda: self._finish_feff_run(proc.returncode, proc.stdout, proc.stderr))
        except Exception as exc:
            self.after(0, lambda: self._finish_feff_run(-1, "", str(exc)))

    def _finish_feff_run(self, returncode: int, stdout: str, stderr: str):
        self._run_feff_btn.config(state=tk.NORMAL)
        self._append_feff_log(f"FEFF finished with return code {returncode}")
        if stdout.strip():
            self._append_feff_log("stdout:")
            for line in stdout.strip().splitlines()[:40]:
                self._append_feff_log(f"  {line}")
        if stderr.strip():
            self._append_feff_log("stderr:")
            for line in stderr.strip().splitlines()[:40]:
                self._append_feff_log(f"  {line}")
        self._load_feff_paths(silent=True)

    def get_params(self) -> dict:
        return {
            "e0": self._e0_var.get(),
            "pre1": self._pre1_var.get(),
            "pre2": self._pre2_var.get(),
            "nor1": self._nor1_var.get(),
            "nor2": self._nor2_var.get(),
            "nnorm": self._nnorm_var.get(),
            "rbkg": self._rbkg_var.get(),
            "kmin_bkg": self._kmin_bkg_var.get(),
            "qmin": self._qmin_var.get(),
            "qmax": self._qmax_var.get(),
            "dq": self._dq_var.get(),
            "qweight": self._qweight_var.get(),
            "q_window": self._qwin_var.get(),
            "rmin": self._rmin_var.get(),
            "rmax": self._rmax_var.get(),
            "dr": self._dr_var.get(),
            "r_window": self._rwin_var.get(),
            "r_display": self._rdisplay_var.get(),
            "style": self._style_var.get(),
            "context": self._context_var.get(),
            "use_q_label": self._use_q_label_var.get(),
            "show_q_window": self._show_q_window_var.get(),
            "show_r_window": self._show_r_window_var.get(),
            "show_feff_markers": self._show_feff_markers_var.get(),
            "selected_labels": list(self._selected_labels),
            "feff_dir": self._feff_dir_var.get(),
            "feff_exe": self._feff_exe_var.get(),
            "xyz_path": self._xyz_path_var.get(),
            "bundle_base": self._bundle_base_var.get(),
            "xyz_padding": self._xyz_padding_var.get(),
            "xyz_cubic": self._xyz_cubic_var.get(),
            "xyz_absorber": self._xyz_absorber_var.get(),
            "xyz_edge": self._xyz_edge_var.get(),
            "xyz_spectrum": self._xyz_spectrum_var.get(),
            "xyz_kmesh": self._xyz_kmesh_var.get(),
            "xyz_equivalence": self._xyz_equiv_var.get(),
        }

    def set_params(self, data: dict) -> None:
        def _set(var, key, cast=float):
            if key not in data:
                return
            try:
                var.set(cast(data[key]))
            except Exception:
                pass

        _set(self._e0_var, "e0")
        _set(self._pre1_var, "pre1")
        _set(self._pre2_var, "pre2")
        _set(self._nor1_var, "nor1")
        _set(self._nor2_var, "nor2")
        _set(self._nnorm_var, "nnorm", int)
        _set(self._rbkg_var, "rbkg")
        _set(self._kmin_bkg_var, "kmin_bkg")
        _set(self._qmin_var, "qmin")
        _set(self._qmax_var, "qmax")
        _set(self._dq_var, "dq")
        _set(self._qweight_var, "qweight", int)
        if "q_window" in data:
            self._qwin_var.set(str(data["q_window"]))
        _set(self._rmin_var, "rmin")
        _set(self._rmax_var, "rmax")
        _set(self._dr_var, "dr")
        if "r_window" in data:
            self._rwin_var.set(str(data["r_window"]))
        _set(self._rdisplay_var, "r_display")

        if "style" in data:
            self._style_var.set(str(data["style"]))
        if "context" in data:
            self._context_var.set(str(data["context"]))
        if "use_q_label" in data:
            self._use_q_label_var.set(bool(data["use_q_label"]))
        if "show_q_window" in data:
            self._show_q_window_var.set(bool(data["show_q_window"]))
        if "show_r_window" in data:
            self._show_r_window_var.set(bool(data["show_r_window"]))
        if "show_feff_markers" in data:
            self._show_feff_markers_var.set(bool(data["show_feff_markers"]))

        self._pending_selected_labels = list(data.get("selected_labels", []))
        if "feff_dir" in data:
            self._feff_dir_var.set(str(data["feff_dir"]))
        if "feff_exe" in data:
            self._feff_exe_var.set(str(data["feff_exe"]))
        if "xyz_path" in data:
            self._xyz_path_var.set(str(data["xyz_path"]))
        if "bundle_base" in data:
            self._bundle_base_var.set(str(data["bundle_base"]))
        _set(self._xyz_padding_var, "xyz_padding")
        if "xyz_cubic" in data:
            self._xyz_cubic_var.set(bool(data["xyz_cubic"]))
        _set(self._xyz_absorber_var, "xyz_absorber", int)
        if "xyz_edge" in data:
            self._xyz_edge_var.set(str(data["xyz_edge"]))
        if "xyz_spectrum" in data:
            self._xyz_spectrum_var.set(str(data["xyz_spectrum"]))
        _set(self._xyz_kmesh_var, "xyz_kmesh", int)
        _set(self._xyz_equiv_var, "xyz_equivalence", int)
        if self._xyz_path_var.get().strip():
            self._load_xyz_structure(silent=True)
        if self._feff_dir_var.get().strip():
            self._load_feff_paths(silent=True)
        self._redraw()
