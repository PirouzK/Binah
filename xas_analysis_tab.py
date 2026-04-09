"""
xas_analysis_tab.py  –  Full XAS Analysis panel
Uses native xraylarch functions (pre_edge, autobk, xftf) when available,
falls back to scipy reimplementations transparently.

Larch functions used (xraylarch >= 2026.1):
  larch.xafs.pre_edge   – XANES normalization + E0 detection
  larch.xafs.autobk     – AUTOBK spline background removal
  larch.xafs.xftf       – Hanning-windowed Fourier transform

Seaborn used for figure styling (themes, contexts, palette).
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, List, Optional, Tuple
import json
import os
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

# ── Persistent config (norm defaults survive restarts) ────────────────────────
_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".orca_tddft_viewer_config.json")

# Hard-coded factory defaults
_NORM_FACTORY: dict = {
    "pre1": -150.0, "pre2": -30.0,
    "nor1":  150.0, "nor2": 400.0, "nnorm": 1,
    "rbkg": 1.0,  "kmin_bkg": 0.5,
    "kmin": 2.0,  "kmax": 12.0, "dk": 1.0, "kw": 2, "rmax": 6.0,
}

def _load_norm_defaults() -> dict:
    """Load saved norm defaults from config file; fall back to factory values."""
    try:
        if os.path.exists(_CONFIG_PATH):
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            saved = cfg.get("xas_norm_defaults", {})
            merged = dict(_NORM_FACTORY)
            merged.update({k: v for k, v in saved.items() if k in _NORM_FACTORY})
            return merged
    except Exception:
        pass
    return dict(_NORM_FACTORY)

def _save_norm_defaults(vals: dict) -> None:
    """Persist norm defaults to config file."""
    try:
        cfg = {}
        if os.path.exists(_CONFIG_PATH):
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        cfg["xas_norm_defaults"] = {k: vals[k] for k in _NORM_FACTORY if k in vals}
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass

# Load once at import time
_NORM_DEFAULTS: dict = _load_norm_defaults()

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

# ── xraylarch (optional but preferred) ────────────────────────────────────────
try:
    from larch import Group as LarchGroup, Interpreter as LarchInterpreter
    from larch.xafs import (pre_edge  as _larch_pre_edge,
                             autobk   as _larch_autobk,
                             xftf     as _larch_xftf)
    _HAS_LARCH = True
    _LARCH_SESSION: Optional[LarchInterpreter] = None   # created on first use
except ImportError:
    _HAS_LARCH = False
    _LARCH_SESSION = None


def _get_larch_session():
    """Return (or lazily create) the shared Larch Interpreter session."""
    global _LARCH_SESSION
    if _LARCH_SESSION is None and _HAS_LARCH:
        _LARCH_SESSION = LarchInterpreter()
    return _LARCH_SESSION


from experimental_parser import ExperimentalScan

# ── Physical constant ──────────────────────────────────────────────────────────
# k [Å⁻¹] = sqrt(ETOK * (E-E0) [eV])    where  ETOK = 2m/ℏ²  in eV⁻¹·Å⁻²
ETOK = 0.26246840

def etok(delta_e: np.ndarray) -> np.ndarray:
    """Energy above edge (eV)  →  k (Å⁻¹).  Negative values clipped to 0."""
    return np.sqrt(np.maximum(delta_e, 0.0) * ETOK)

def ktoe(k: np.ndarray) -> np.ndarray:
    """k (Å⁻¹)  →  energy above edge (eV)."""
    return k ** 2 / ETOK


# ── Seaborn theme helper ───────────────────────────────────────────────────────
_SNS_STYLE = "ticks"
_SNS_CONTEXT = "paper"
_PALETTE = ["#2C7BB6", "#1A9641", "#D7191C", "#FDAE61", "#762A83", "#4DAC26"]

def _apply_seaborn_style(fig):
    """Apply seaborn theme to a matplotlib figure already created."""
    if not _HAS_SNS:
        return
    with sns.axes_style(_SNS_STYLE):
        for ax in fig.axes:
            sns.despine(ax=ax, offset=5, trim=False)


# ═════════════════════════════════════════════════════════════════════════════
#  Core XAS algorithms
# ═════════════════════════════════════════════════════════════════════════════

def find_e0(energy: np.ndarray, mu: np.ndarray) -> float:
    """Estimate E0 as the energy of the maximum first derivative."""
    if len(energy) < 4:
        return float(energy[len(energy) // 2])
    # Smooth gradient to reduce noise influence
    grad = np.gradient(mu, energy)
    # Look only in the main rising-edge region (middle 80% of scan)
    lo = len(energy) // 10
    hi = len(energy) * 9 // 10
    idx = int(np.argmax(grad[lo:hi])) + lo
    return float(energy[idx])


def normalize_xanes(
    energy: np.ndarray,
    mu: np.ndarray,
    e0: float,
    pre1: float = -150.0,
    pre2: float = -30.0,
    nor1: float = 150.0,
    nor2: float = 400.0,
    nnor: int = 1,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Athena-style XANES normalization.

    Returns
    -------
    mu_norm   : normalized mu (0 before edge, ~1 in post-edge)
    edge_step : the normalization factor
    pre_line  : the pre-edge background line evaluated at all energies
    """
    # Pre-edge linear fit
    pre_mask = (energy >= e0 + pre1) & (energy <= e0 + pre2)
    if pre_mask.sum() >= 2:
        p_pre = np.polyfit(energy[pre_mask], mu[pre_mask], 1)
    else:
        p_pre = np.polyfit(energy[:max(3, len(energy)//5)], mu[:max(3, len(energy)//5)], 1)
    pre_line = np.polyval(p_pre, energy)
    mu_sub = mu - pre_line

    # Post-edge polynomial fit.
    # Use flat normalization: divide mu_sub by the polynomial evaluated at
    # *each energy point* (not just the constant edge_step at e0).
    # This removes the smooth E^-3 background curvature so the post-edge
    # stays flat at 1.0 across the whole energy range — identical to what
    # Athena/Demeter calls "flat normalized mu(E)".
    nor_mask = (energy >= e0 + nor1) & (energy <= e0 + nor2)
    if nor_mask.sum() >= nnor + 1:
        p_nor     = np.polyfit(energy[nor_mask], mu_sub[nor_mask], nnor)
        edge_step = float(np.polyval(p_nor, e0))          # constant at e0 (for reporting)
        post_poly = np.polyval(p_nor, energy)              # polynomial at every E (for flat norm)
    elif nor_mask.sum() >= 2:
        edge_step = float(mu_sub[nor_mask].mean())
        post_poly = np.full_like(energy, edge_step)
    else:
        n_tail    = max(5, len(mu_sub) // 10)
        edge_step = float(mu_sub[-n_tail:].mean())
        post_poly = np.full_like(energy, edge_step)

    if abs(edge_step) < 1e-10:
        edge_step = 1.0
        post_poly = np.full_like(energy, 1.0)

    # Guard: never let the denominator collapse near zero far from the edge
    sign      = 1.0 if edge_step > 0 else -1.0
    floor_val = sign * max(abs(edge_step) * 0.05, 1e-10)
    post_poly = np.where(post_poly * sign < abs(floor_val), floor_val, post_poly)

    return mu_sub / post_poly, edge_step, pre_line


def autobk(
    energy: np.ndarray,
    mu_norm: np.ndarray,
    e0: float,
    rbkg: float = 1.0,
    kmin_bkg: float = 0.0,
    kmax_bkg: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simplified AUTOBK background removal.

    Fits a cubic spline to the post-edge mu_norm in k-space with knot
    spacing pi/rbkg — coarse enough to not follow EXAFS oscillations.

    Returns
    -------
    k_arr    : k array (Å⁻¹) for the post-edge region
    chi      : chi(k) = (mu_norm - background) / 1   [already normalised]
    bkg_e    : background evaluated on original energy grid (for plotting)
    """
    post_mask = energy >= e0
    E_post = energy[post_mask]
    mu_post = mu_norm[post_mask]

    if len(E_post) < 5:
        return np.array([]), np.array([]), np.zeros_like(energy)

    k_post = etok(E_post - e0)

    if kmax_bkg is None:
        kmax_bkg = k_post.max()

    # Remove duplicate k values (can happen near E0)
    _, uidx = np.unique(k_post, return_index=True)
    k_u = k_post[uidx]
    mu_u = mu_post[uidx]

    if len(k_u) < 4:
        bkg_e = np.zeros_like(energy)
        bkg_e[post_mask] = mu_post
        return k_u, np.zeros_like(k_u), bkg_e

    # Knot spacing in k-space: pi / rbkg
    dk_knot = np.pi / max(rbkg, 0.3)
    knots_k = np.arange(dk_knot, k_u[-1] - dk_knot / 2, dk_knot)
    # Only keep knots inside data range (with buffer)
    knots_k = knots_k[(knots_k > k_u[1]) & (knots_k < k_u[-2])]

    try:
        if len(knots_k) >= 1:
            spl = UnivariateSpline(k_u, mu_u, k=3, t=knots_k, ext=3)
        else:
            # Too few knots → heavy smoothing spline
            s = float(len(k_u)) * 0.1
            spl = UnivariateSpline(k_u, mu_u, k=3, s=s, ext=3)
    except Exception:
        # Fallback: simple polynomial background
        deg = min(5, max(2, len(knots_k) + 1))
        p = np.polyfit(k_u, mu_u, deg)
        bkg_k = np.polyval(p, k_u)
        chi = mu_u - bkg_k
        bkg_e = np.zeros_like(energy)
        bkg_e[post_mask] = mu_post - chi
        return k_u, chi, bkg_e

    bkg_k = spl(k_u)
    chi = mu_u - bkg_k

    # Map background back onto original energy grid
    bkg_e = np.zeros_like(energy)
    bkg_e[post_mask] = bkg_k

    return k_u, chi, bkg_e


def xftf(
    k: np.ndarray,
    chi: np.ndarray,
    kmin: float = 2.0,
    kmax: float = 12.0,
    dk: float = 1.0,
    kweight: int = 2,
    nfft: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward Fourier Transform: chi(k) → chi_tilde(R).

    Returns
    -------
    r          : R array (Å)
    chi_r_mag  : |chi_tilde(R)|
    chi_r_re   : Re[chi_tilde(R)]
    chi_r_im   : Im[chi_tilde(R)]
    """
    if len(k) < 4:
        return np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])

    # Interpolate onto a uniform k-grid
    k_step = float(np.diff(k).mean()) if len(k) > 1 else 0.05
    k_step = max(k_step, 0.01)
    k_uni = np.arange(k[0], k[-1] + k_step * 0.1, k_step)
    chi_uni = np.interp(k_uni, k, chi)

    # k-weighting
    chi_kw = k_uni ** kweight * chi_uni

    # Hanning window with dk taper at each edge
    win = np.zeros_like(k_uni)
    kmin_eff = max(kmin, k_uni[0])
    kmax_eff = min(kmax, k_uni[-1])
    dk = max(dk, k_step)

    flat = (k_uni >= kmin_eff + dk) & (k_uni <= kmax_eff - dk)
    win[flat] = 1.0
    t_in = (k_uni >= kmin_eff) & (k_uni < kmin_eff + dk)
    if t_in.any():
        win[t_in] = 0.5 * (1 - np.cos(np.pi * (k_uni[t_in] - kmin_eff) / dk))
    t_out = (k_uni > kmax_eff - dk) & (k_uni <= kmax_eff)
    if t_out.any():
        win[t_out] = 0.5 * (1 + np.cos(np.pi * (k_uni[t_out] - (kmax_eff - dk)) / dk))

    # Zero-pad and FFT
    npad = max(nfft, 4 * len(chi_kw))
    arr = np.zeros(npad)
    n = min(len(chi_kw), npad)
    arr[:n] = chi_kw[:n] * win[:n]

    cft = np.fft.rfft(arr) * k_step / np.sqrt(np.pi)

    # R grid
    dr = np.pi / (k_step * npad)
    r = dr * np.arange(len(cft))

    return r, np.abs(cft), cft.real, cft.imag


# ═════════════════════════════════════════════════════════════════════════════
#  UI Widget
# ═════════════════════════════════════════════════════════════════════════════

class XASAnalysisTab(tk.Frame):
    """
    Full Larch-style XAS analysis panel.

    Parameters
    ----------
    parent : tk parent widget
    get_scans_fn : callable returning List[(label, ExperimentalScan, enabled_var, style_dict)]
                   (the format stored in PlotWidget._exp_scans)
    """

    _SCAN_COLOURS = _PALETTE

    def __init__(self, parent, get_scans_fn: Callable,
                 replot_fn: Optional[Callable] = None):
        super().__init__(parent)
        self._get_scans = get_scans_fn
        self._replot_fn = replot_fn   # called after apply-all to refresh Spectra tab

        # Analysis results cache per scan label
        self._results: dict = {}

        # Which scans are selected for overlay
        self._selected_labels: List[str] = []

        self._build_ui()

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top: scan selector + run button
        top = tk.Frame(self, bd=1, relief=tk.GROOVE, padx=4, pady=3)
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Label(top, text="Scan:", font=("", 9, "bold")).pack(side=tk.LEFT)
        self._scan_var = tk.StringVar()
        self._scan_cb = ttk.Combobox(top, textvariable=self._scan_var,
                                      state="readonly", width=40)
        self._scan_cb.pack(side=tk.LEFT, padx=(4, 8))
        self._scan_cb.bind("<<ComboboxSelected>>", lambda _e: self._auto_fill_e0())

        tk.Button(top, text="\u21bb Refresh Scans", font=("", 8),
                  command=self.refresh_scan_list).pack(side=tk.LEFT, padx=2)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        tk.Button(top, text="\u25b6  Run Analysis", bg="#003366", fg="white",
                  font=("", 9, "bold"), command=self._run).pack(side=tk.LEFT, padx=2)

        tk.Button(top, text="+ Add to Overlay", font=("", 8),
                  command=self._add_overlay).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="Clear Overlay", font=("", 8),
                  command=self._clear_overlay).pack(side=tk.LEFT, padx=2)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        tk.Button(top, text="\u2713 Apply norm to ALL scans",
                  bg="#1a5c1a", fg="white", font=("", 9, "bold"),
                  command=self._apply_norm_all).pack(side=tk.LEFT, padx=2)

        self._status_lbl = tk.Label(top, text="Load experimental scans first (File \u2192 Load Exp. Data)",
                                     fg="gray", font=("", 8))
        self._status_lbl.pack(side=tk.LEFT, padx=10)

        # Main body: params left, plot right
        body = tk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._build_params(body)
        self._build_plot(body)

    def _build_params(self, parent):
        pf = tk.Frame(parent, width=210, bd=1, relief=tk.SUNKEN, padx=4, pady=4)
        pf.pack(side=tk.LEFT, fill=tk.Y, padx=(2, 0), pady=2)
        pf.pack_propagate(False)

        def lbl(text):
            tk.Label(pf, text=text, font=("", 8, "bold"), fg="#333333",
                     anchor="w").pack(fill=tk.X, pady=(6, 0))

        def row(text, var, from_=None, to=None, inc=None, fmt=None, width=7):
            f = tk.Frame(pf); f.pack(fill=tk.X, pady=1)
            tk.Label(f, text=text, width=14, anchor="w", font=("", 8)).pack(side=tk.LEFT)
            kw = dict(textvariable=var, width=width, font=("Courier", 8))
            if from_ is not None:
                kw.update(from_=from_, to=to, increment=inc, format=fmt or "%.2f")
                ttk.Spinbox(f, **kw).pack(side=tk.LEFT)
            else:
                ttk.Entry(f, **kw).pack(side=tk.LEFT)

        # ── Edge / Normalization ──────────────────────────────────────────
        lbl("\u2500\u2500 Edge / Normalization \u2500\u2500\u2500\u2500\u2500")
        nd = _NORM_DEFAULTS   # shorthand
        self._e0_var    = tk.DoubleVar(value=8333.0)
        self._pre1_var  = tk.DoubleVar(value=nd["pre1"])
        self._pre2_var  = tk.DoubleVar(value=nd["pre2"])
        self._nor1_var  = tk.DoubleVar(value=nd["nor1"])
        self._nor2_var  = tk.DoubleVar(value=nd["nor2"])
        self._nnor_var  = tk.IntVar(value=nd["nnorm"])

        row("E0 (eV):",    self._e0_var,   7000, 10000, 0.5,  "%.1f")
        row("pre1 (eV):",  self._pre1_var, -300,  -5,   5.0,  "%.0f")
        row("pre2 (eV):",  self._pre2_var, -200,  -5,   5.0,  "%.0f")
        row("nor1 (eV):",  self._nor1_var,   10,  500,  10.0, "%.0f")
        row("nor2 (eV):",  self._nor2_var,   10, 1000,  10.0, "%.0f")

        f_nnor = tk.Frame(pf); f_nnor.pack(fill=tk.X, pady=1)
        tk.Label(f_nnor, text="Nor. order:", width=14, anchor="w",
                 font=("", 8)).pack(side=tk.LEFT)
        for v, t in [(1, "1"), (2, "2")]:
            tk.Radiobutton(f_nnor, text=t, variable=self._nnor_var,
                           value=v, font=("", 8)).pack(side=tk.LEFT)

        # "Set as Default" — saves current norm ranges to config file
        tk.Button(pf, text="\u2605 Set Norm as Default", font=("", 8, "bold"),
                  bg="#003366", fg="white", activebackground="#0055aa",
                  command=self._set_norm_default).pack(fill=tk.X, pady=(4, 2))

        # ── AUTOBK ───────────────────────────────────────────────────────
        lbl("\u2500\u2500 AUTOBK (background) \u2500\u2500\u2500\u2500\u2500\u2500")
        self._rbkg_var     = tk.DoubleVar(value=nd["rbkg"])
        self._kmin_bkg_var = tk.DoubleVar(value=nd["kmin_bkg"])
        row("rbkg (A):",   self._rbkg_var,   0.3, 3.0, 0.1, "%.1f")
        row("kmin_bkg:",   self._kmin_bkg_var, 0, 5.0, 0.5, "%.1f")

        # ── XFTF ─────────────────────────────────────────────────────────
        lbl("\u2500\u2500 Fourier Transform \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        self._kmin_var   = tk.DoubleVar(value=nd["kmin"])
        self._kmax_var   = tk.DoubleVar(value=nd["kmax"])
        self._dk_var     = tk.DoubleVar(value=nd["dk"])
        self._kw_var     = tk.IntVar(value=nd["kw"])
        self._rmax_var   = tk.DoubleVar(value=nd["rmax"])

        row("kmin (A^-1):", self._kmin_var, 0,  6,   0.5, "%.1f")
        row("kmax (A^-1):", self._kmax_var, 4,  20,  0.5, "%.1f")
        row("dk (A^-1):",   self._dk_var,   0.1, 3,  0.1, "%.1f")
        row("R max (A):",   self._rmax_var, 2,  12,  0.5, "%.1f")

        f_kw = tk.Frame(pf); f_kw.pack(fill=tk.X, pady=1)
        tk.Label(f_kw, text="k-weight:", width=14, anchor="w",
                 font=("", 8)).pack(side=tk.LEFT)
        for v in [1, 2, 3]:
            tk.Radiobutton(f_kw, text=str(v), variable=self._kw_var,
                           value=v, font=("", 8)).pack(side=tk.LEFT)

        # ── Plot style ────────────────────────────────────────────────────
        lbl("\u2500\u2500 Plot Style \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        self._style_var = tk.StringVar(value="ticks")
        self._context_var = tk.StringVar(value="paper")
        f_sty = tk.Frame(pf); f_sty.pack(fill=tk.X, pady=1)
        tk.Label(f_sty, text="Style:", width=14, anchor="w",
                 font=("", 8)).pack(side=tk.LEFT)
        ttk.Combobox(f_sty, textvariable=self._style_var, width=10,
                     state="readonly",
                     values=["ticks", "whitegrid", "darkgrid", "white", "dark"]
                     ).pack(side=tk.LEFT)
        f_ctx = tk.Frame(pf); f_ctx.pack(fill=tk.X, pady=1)
        tk.Label(f_ctx, text="Context:", width=14, anchor="w",
                 font=("", 8)).pack(side=tk.LEFT)
        ttk.Combobox(f_ctx, textvariable=self._context_var, width=10,
                     state="readonly",
                     values=["paper", "notebook", "talk", "poster"]
                     ).pack(side=tk.LEFT)

        self._show_bkg_var = tk.BooleanVar(value=True)
        tk.Checkbutton(pf, text="Show background on mu(E)",
                       variable=self._show_bkg_var,
                       font=("", 8)).pack(anchor="w", pady=2)
        self._show_win_var = tk.BooleanVar(value=True)
        tk.Checkbutton(pf, text="Show FT window on chi(k)",
                       variable=self._show_win_var,
                       font=("", 8)).pack(anchor="w", pady=2)

    def _build_plot(self, parent):
        plot_area = tk.Frame(parent)
        plot_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Toolbar frame at top
        tb_frame = tk.Frame(plot_area)
        tb_frame.pack(side=tk.TOP, fill=tk.X)

        # Figure
        self._fig = Figure(figsize=(9, 7), dpi=96, facecolor="white")
        self._fig.subplots_adjust(hspace=0.38, left=0.09, right=0.96,
                                   top=0.94, bottom=0.07)
        self._axes: List = []

        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_area)
        self._canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self._canvas, tb_frame)
        self.toolbar.update()

        self._draw_empty_figure()

    def _draw_empty_figure(self):
        self._fig.clear()
        self._axes = []
        gs = GridSpec(3, 1, figure=self._fig,
                      hspace=0.42, top=0.93, bottom=0.07,
                      left=0.10, right=0.95)
        titles = ["mu(E)  — normalized XANES",
                  "chi(k)  — EXAFS oscillations",
                  "|chi_tilde(R)|  — Radial distribution"]
        ylabels = ["mu(E) normalized", "chi(k) * k^n", "|chi_tilde(R)|  (A^-3)"]
        xlabels = ["Energy (eV)", "k  (A^-1)", "R  (A)"]
        for i in range(3):
            ax = self._fig.add_subplot(gs[i])
            ax.set_title(titles[i], fontsize=9, loc="left", pad=3)
            ax.set_xlabel(xlabels[i], fontsize=8)
            ax.set_ylabel(ylabels[i], fontsize=8)
            ax.tick_params(labelsize=7)
            ax.text(0.5, 0.45, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="lightgray")
            self._axes.append(ax)
        _apply_seaborn_style(self._fig)
        self._canvas.draw_idle()

    # ── Scan list management ─────────────────────────────────────────────────

    def refresh_scan_list(self):
        """Re-populate the scan combobox from currently loaded experimental scans."""
        scans = self._get_scans()
        labels = [lbl for lbl, *_ in scans]
        self._scan_cb["values"] = labels
        if labels and self._scan_var.get() not in labels:
            self._scan_var.set(labels[0])
            self._auto_fill_e0()
        n = len(labels)
        self._status_lbl.config(
            text=f"{n} scan{'s' if n != 1 else ''} available.",
            fg="gray")

    def _get_scan_by_label(self, label: str) -> Optional[ExperimentalScan]:
        for lbl, scan, *_ in self._get_scans():
            if lbl == label:
                return scan
        return None

    def _auto_fill_e0(self):
        """Auto-detect E0 from the selected scan and fill the spinbox."""
        label = self._scan_var.get()
        scan = self._get_scan_by_label(label)
        if scan is None:
            return

        src = "stored"
        if scan.e0 and scan.e0 > 100:
            e0 = scan.e0
        elif _HAS_LARCH:
            # Use larch's derivative-based E0 finder (more robust)
            try:
                session = _get_larch_session()
                grp = LarchGroup(energy=scan.energy_ev.copy(), mu=scan.mu.copy())
                _larch_pre_edge(grp, _larch=session)
                e0 = float(grp.e0)
                src = "larch"
            except Exception:
                e0 = find_e0(scan.energy_ev, scan.mu)
                src = "scipy"
        else:
            e0 = find_e0(scan.energy_ev, scan.mu)
            src = "scipy"

        self._e0_var.set(round(e0, 1))
        self._status_lbl.config(
            text=f"E\u2080 = {e0:.1f} eV  (auto-detected via {src})",
            fg="#005500")

    # ── Overlay management ───────────────────────────────────────────────────

    def _add_overlay(self):
        label = self._scan_var.get()
        if label and label not in self._selected_labels:
            self._selected_labels.append(label)
            self._run()

    def _clear_overlay(self):
        self._selected_labels.clear()
        self._results.clear()

    # ── Norm defaults ─────────────────────────────────────────────────────────

    def _set_norm_default(self):
        """Save current norm/FT parameters as the new persistent defaults."""
        vals = {
            "pre1":    self._pre1_var.get(),
            "pre2":    self._pre2_var.get(),
            "nor1":    self._nor1_var.get(),
            "nor2":    self._nor2_var.get(),
            "nnorm":   self._nnor_var.get(),
            "rbkg":    self._rbkg_var.get(),
            "kmin_bkg": self._kmin_bkg_var.get(),
            "kmin":    self._kmin_var.get(),
            "kmax":    self._kmax_var.get(),
            "dk":      self._dk_var.get(),
            "kw":      self._kw_var.get(),
            "rmax":    self._rmax_var.get(),
        }
        _NORM_DEFAULTS.update(vals)
        _save_norm_defaults(vals)
        messagebox.showinfo(
            "Defaults Saved",
            "Normalization & FT parameters saved as defaults.\n"
            "These will be loaded automatically whenever you open the program.",
            parent=self,
        )

    # ── Project save / load helpers ───────────────────────────────────────────

    def get_params(self) -> dict:
        """Return all analysis parameters as a plain dict (for project save)."""
        return {
            "e0":       self._e0_var.get(),
            "pre1":     self._pre1_var.get(),
            "pre2":     self._pre2_var.get(),
            "nor1":     self._nor1_var.get(),
            "nor2":     self._nor2_var.get(),
            "nnorm":    self._nnor_var.get(),
            "rbkg":     self._rbkg_var.get(),
            "kmin_bkg": self._kmin_bkg_var.get(),
            "kmin":     self._kmin_var.get(),
            "kmax":     self._kmax_var.get(),
            "dk":       self._dk_var.get(),
            "kw":       self._kw_var.get(),
            "rmax":     self._rmax_var.get(),
            "style":    self._style_var.get(),
            "context":  self._context_var.get(),
            "show_bkg": self._show_bkg_var.get(),
            "show_win": self._show_win_var.get(),
        }

    def set_params(self, d: dict) -> None:
        """Restore analysis parameters from a dict (for project load)."""
        def _s(var, key, cast=float):
            if key in d:
                try:
                    var.set(cast(d[key]))
                except Exception:
                    pass
        _s(self._e0_var,       "e0")
        _s(self._pre1_var,     "pre1")
        _s(self._pre2_var,     "pre2")
        _s(self._nor1_var,     "nor1")
        _s(self._nor2_var,     "nor2")
        _s(self._nnor_var,     "nnorm", int)
        _s(self._rbkg_var,     "rbkg")
        _s(self._kmin_bkg_var, "kmin_bkg")
        _s(self._kmin_var,     "kmin")
        _s(self._kmax_var,     "kmax")
        _s(self._dk_var,       "dk")
        _s(self._kw_var,       "kw", int)
        _s(self._rmax_var,     "rmax")
        if "style" in d:
            self._style_var.set(d["style"])
        if "context" in d:
            self._context_var.set(d["context"])
        if "show_bkg" in d:
            self._show_bkg_var.set(bool(d["show_bkg"]))
        if "show_win" in d:
            self._show_win_var.set(bool(d["show_win"]))

    # ── Apply normalisation to every loaded scan ──────────────────────────────

    def _apply_norm_all(self):
        """Re-normalise ALL loaded experimental scans using the current panel
        parameters, then push the results back to the Spectra tab."""
        scans_raw = self._get_scans()
        if not scans_raw:
            self._status_lbl.config(
                text="No experimental scans loaded.", fg="#993300")
            return

        e0_ui   = self._e0_var.get()
        pre1    = self._pre1_var.get()
        pre2    = self._pre2_var.get()
        nor1    = self._nor1_var.get()
        nor2    = self._nor2_var.get()
        nnor    = self._nnor_var.get()

        ok = 0
        fail = 0

        for lbl, scan, *_ in scans_raw:
            energy = scan.energy_ev
            # Use stored e0 as starting point; override only if UI value looks
            # reasonable for this scan (within ±50 eV of stored edge energy).
            use_e0 = scan.e0 if scan.e0 > 100 else float(e0_ui)
            if abs(e0_ui - use_e0) < 50:
                use_e0 = float(e0_ui)

            try:
                if _HAS_LARCH:
                    session = _get_larch_session()
                    grp = LarchGroup(energy=energy.copy(), mu=scan.mu.copy())
                    _larch_pre_edge(grp, _larch=session,
                                    e0=use_e0,
                                    pre1=float(pre1), pre2=float(pre2),
                                    norm1=float(nor1), norm2=float(nor2),
                                    nnorm=int(nnor))
                    # Sanity-check — auto-retry with clamped ranges if bad
                    from experimental_parser import ExperimentalParser as _EP
                    flat0 = getattr(grp, "flat", grp.norm)
                    if not _EP._norm_is_valid(energy, flat0, float(grp.e0),
                                              (pre1, pre2), (nor1, nor2)):
                        safe_pre, safe_post = _EP._safe_ranges(
                            energy, float(grp.e0), (pre1, pre2), (nor1, nor2))
                        grp2 = LarchGroup(energy=energy.copy(), mu=scan.mu.copy())
                        _larch_pre_edge(grp2, _larch=session,
                                        e0=use_e0,
                                        pre1=safe_pre[0], pre2=safe_pre[1],
                                        norm1=safe_post[0], norm2=safe_post[1],
                                        nnorm=int(nnor))
                        grp = grp2
                    # Use flat normalization (polynomial at each E, not constant
                    # edge-step at e0) so the post-edge stays flat — same as Athena.
                    scan.mu  = getattr(grp, "flat", grp.norm)
                    scan.e0  = float(grp.e0)
                else:
                    from experimental_parser import ExperimentalParser as _EP
                    norm, new_e0 = _EP._normalize_poly(
                        energy, scan.mu.copy(), use_e0,
                        (pre1, pre2), (nor1, nor2), nnor)
                    scan.mu = norm
                    scan.e0 = new_e0
                scan.is_normalized = True
                ok += 1
            except Exception as exc:
                fail += 1

        msg = f"Re-normalised {ok} scan(s)"
        if fail:
            msg += f"  ({fail} failed — kept original)"
        self._status_lbl.config(text=msg, fg="#006600" if not fail else "#993300")

        # Invalidate cached analysis results (norm changed so chi/FT are stale)
        self._results.clear()

        # Refresh Spectra tab
        if self._replot_fn is not None:
            self._replot_fn()
        self._draw_empty_figure()
        self._status_lbl.config(text="Overlay cleared.", fg="gray")

    # ── Analysis ─────────────────────────────────────────────────────────────

    def _run(self):
        """Run the full analysis pipeline on the selected scan and redraw."""
        label = self._scan_var.get()
        if not label:
            self._status_lbl.config(text="Select a scan first.", fg="#993300")
            return
        scan = self._get_scan_by_label(label)
        if scan is None:
            self._status_lbl.config(
                text="Scan not found. Click \u21bb Refresh.", fg="red")
            return

        e0   = self._e0_var.get()
        pre1 = self._pre1_var.get()
        pre2 = self._pre2_var.get()
        nor1 = self._nor1_var.get()
        nor2 = self._nor2_var.get()
        nnor = self._nnor_var.get()
        rbkg = self._rbkg_var.get()
        kmin_bkg = self._kmin_bkg_var.get()
        kmin = self._kmin_var.get()
        kmax = self._kmax_var.get()
        dk   = self._dk_var.get()
        kw   = self._kw_var.get()

        energy = scan.energy_ev.copy()
        mu_raw = scan.mu.copy()

        engine = "scipy"

        if _HAS_LARCH:
            # ── Use native larch functions ─────────────────────────────────
            try:
                session = _get_larch_session()
                grp = LarchGroup(energy=energy, mu=mu_raw)

                # 1. pre_edge: normalization + E0
                _larch_pre_edge(grp, _larch=session,
                                e0=float(e0) if e0 > 100 else None,
                                pre1=float(pre1), pre2=float(pre2),
                                norm1=float(nor1), norm2=float(nor2),
                                nnorm=int(nnor))

                # Update E0 spinbox with larch's refined value
                self._e0_var.set(round(float(grp.e0), 1))
                e0 = float(grp.e0)

                # 2. autobk: EXAFS background removal
                _larch_autobk(grp, _larch=session,
                               rbkg=float(rbkg), kmin=float(kmin_bkg))

                # 3. xftf: Fourier transform
                _larch_xftf(grp, _larch=session,
                             kmin=float(kmin), kmax=float(kmax),
                             dk=float(dk), kweight=int(kw))

                self._results[label] = {
                    "energy":    grp.energy,
                    # grp.flat = (mu-pre_line)/(post_poly(E)-pre_line(E)) at each E
                    # — the "flat normalized" spectrum Athena displays, with a
                    #   perfectly flat post-edge.  Fall back to grp.norm only if
                    #   flat wasn't computed (older larch builds).
                    "mu_norm":   getattr(grp, "flat", grp.norm),
                    "bkg_e":     grp.bkg,        # background on energy grid
                    "k":         grp.k,
                    "chi":       grp.chi,
                    "r":         grp.r,
                    "chi_r":     grp.chir_mag,
                    "chi_r_re":  grp.chir_re,
                    "chi_r_im":  grp.chir_im,
                    "e0":        e0,
                    "edge_step": float(grp.edge_step),
                    "kw":        kw,
                }
                engine = "larch"

            except Exception as exc:
                # Fall through to scipy if larch fails for any reason
                engine = f"scipy (larch err: {type(exc).__name__})"
                _HAS_LARCH_local = False
            else:
                _HAS_LARCH_local = True
        else:
            _HAS_LARCH_local = False

        if not _HAS_LARCH or not _HAS_LARCH_local or engine.startswith("scipy"):
            # ── scipy fallback ─────────────────────────────────────────────
            mu_norm, edge_step, _ = normalize_xanes(
                energy, mu_raw, e0, pre1, pre2, nor1, nor2, nnor)
            k_arr, chi, bkg_e = autobk(energy, mu_norm, e0, rbkg, kmin_bkg)
            if len(k_arr) >= 4:
                r_arr, chi_r, chi_r_re, chi_r_im = xftf(
                    k_arr, chi, kmin, kmax, dk, kw)
            else:
                r_arr = np.array([0.0])
                chi_r = chi_r_re = chi_r_im = np.array([0.0])

            self._results[label] = {
                "energy": energy, "mu_norm": mu_norm, "bkg_e": bkg_e,
                "k": k_arr, "chi": chi,
                "r": r_arr, "chi_r": chi_r,
                "chi_r_re": chi_r_re, "chi_r_im": chi_r_im,
                "e0": e0, "edge_step": edge_step, "kw": kw,
            }

        # If not already in overlay list, reset to just this scan
        if label not in self._selected_labels:
            self._selected_labels = [label]

        res = self._results[label]
        self._redraw()
        self._status_lbl.config(
            text=(f"[{engine}]  {label}  |  E\u2080={res['e0']:.1f} eV  |  "
                  f"edge step={res['edge_step']:.4f}  |  "
                  f"k: {kmin:.1f}\u2013{kmax:.1f} \u00c5\u207b\u00b9"),
            fg="#003366" if engine == "larch" else "#664400")

    def _redraw(self):
        """Redraw all three panels with the current overlay list."""
        style = self._style_var.get() if _HAS_SNS else "default"
        context = self._context_var.get() if _HAS_SNS else "paper"

        if _HAS_SNS:
            sns.set_theme(style=style, context=context, palette=_PALETTE)

        self._fig.clear()
        self._axes = []
        gs = GridSpec(3, 1, figure=self._fig,
                      hspace=0.44, top=0.93, bottom=0.07,
                      left=0.11, right=0.95)

        ax_mu  = self._fig.add_subplot(gs[0])
        ax_chi = self._fig.add_subplot(gs[1])
        ax_r   = self._fig.add_subplot(gs[2])
        self._axes = [ax_mu, ax_chi, ax_r]

        kw_label = {1: "k^1", 2: "k^2", 3: "k^3"}.get(self._kw_var.get(), "k^n")

        for i, label in enumerate(self._selected_labels):
            res = self._results.get(label)
            if res is None:
                continue
            col = _PALETTE[i % len(_PALETTE)]
            lbl_short = label[:30] + ("\u2026" if len(label) > 30 else "")

            # ── mu(E) panel ────────────────────────────────────────────────
            ax_mu.plot(res["energy"], res["mu_norm"],
                       color=col, lw=1.6, label=lbl_short, zorder=3)
            if self._show_bkg_var.get() and res["bkg_e"] is not None:
                mask = res["energy"] >= res["e0"]
                ax_mu.plot(res["energy"][mask], res["bkg_e"][mask],
                           color=col, lw=1.0, ls="--", alpha=0.55,
                           label=f"bkg {lbl_short}" if i == 0 else "_nolegend_",
                           zorder=2)
            if i == 0:
                e0 = res["e0"]
                ax_mu.axvline(e0, color="gray", lw=0.8, ls=":", alpha=0.7,
                              label=f"E0 = {e0:.1f} eV")
                # Shade pre/post-edge norm regions
                p1, p2 = e0 + self._pre1_var.get(), e0 + self._pre2_var.get()
                n1, n2 = e0 + self._nor1_var.get(), e0 + self._nor2_var.get()
                ax_mu.axvspan(p1, p2, alpha=0.07, color="steelblue",
                               label="pre-edge")
                ax_mu.axvspan(n1, n2, alpha=0.07, color="seagreen",
                               label="post-edge norm")

            # ── chi(k) panel ────────────────────────────────────────────────
            if len(res["k"]) > 1:
                kw_val = res["kw"]
                chi_w = res["k"] ** kw_val * res["chi"]
                ax_chi.plot(res["k"], chi_w, color=col, lw=1.4,
                            label=lbl_short, zorder=3)
                if i == 0 and self._show_win_var.get():
                    # Show FT window
                    kmin = self._kmin_var.get()
                    kmax = self._kmax_var.get()
                    dk   = self._dk_var.get()
                    k_u  = np.linspace(res["k"][0], res["k"][-1], 400)
                    win  = np.zeros_like(k_u)
                    flat = (k_u >= kmin + dk) & (k_u <= kmax - dk)
                    win[flat] = 1.0
                    t_in = (k_u >= kmin) & (k_u < kmin + dk)
                    if t_in.any() and dk > 0:
                        win[t_in] = 0.5*(1-np.cos(np.pi*(k_u[t_in]-kmin)/dk))
                    t_out = (k_u > kmax - dk) & (k_u <= kmax)
                    if t_out.any() and dk > 0:
                        win[t_out] = 0.5*(1+np.cos(np.pi*(k_u[t_out]-(kmax-dk))/dk))
                    # Scale window to data amplitude for visibility
                    chi_amp = np.abs(chi_w).max() if len(chi_w) > 0 else 1.0
                    ax_chi.fill_between(k_u, -win * chi_amp, win * chi_amp,
                                        alpha=0.08, color="orange",
                                        label="FT window")

            # ── |chi(R)| panel ─────────────────────────────────────────────
            if len(res["r"]) > 1:
                rmax = self._rmax_var.get()
                r_mask = res["r"] <= rmax
                ax_r.plot(res["r"][r_mask], res["chi_r"][r_mask],
                          color=col, lw=1.6, label=lbl_short, zorder=3)
                ax_r.fill_between(res["r"][r_mask], 0, res["chi_r"][r_mask],
                                  alpha=0.12, color=col)

        # ── Labels & formatting ──────────────────────────────────────────────
        ax_mu.set_xlabel("Energy (eV)", fontsize=8)
        ax_mu.set_ylabel("mu(E)  normalized", fontsize=8)
        ax_mu.set_title("mu(E)  — XANES", fontsize=9, loc="left", pad=3)
        ax_mu.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.4)
        ax_mu.axhline(1, color="gray", lw=0.5, ls=":", alpha=0.4)
        if self._selected_labels:
            ax_mu.legend(fontsize=7, loc="lower right", framealpha=0.8)

        ax_chi.set_xlabel("k  (A^-1)", fontsize=8)
        ax_chi.set_ylabel(f"chi(k)*{kw_label}  (A^-n)", fontsize=8)
        ax_chi.set_title("chi(k)  — EXAFS oscillations", fontsize=9, loc="left", pad=3)
        ax_chi.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.4)
        if self._selected_labels:
            ax_chi.legend(fontsize=7, loc="upper right", framealpha=0.8)

        ax_r.set_xlabel("R  (A)", fontsize=8)
        ax_r.set_ylabel("|chi_tilde(R)|  (A^-n-1)", fontsize=8)
        ax_r.set_title("|chi_tilde(R)|  — Fourier transform", fontsize=9, loc="left", pad=3)
        ax_r.set_xlim(0, self._rmax_var.get())
        if self._selected_labels:
            ax_r.legend(fontsize=7, loc="upper right", framealpha=0.8)

        for ax in self._axes:
            ax.tick_params(labelsize=7)
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

        if _HAS_SNS:
            for ax in self._axes:
                sns.despine(ax=ax, offset=4, trim=False)

        self._canvas.draw_idle()
