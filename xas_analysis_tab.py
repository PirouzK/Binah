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
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

# ── Persistent config (norm defaults survive restarts) ────────────────────────
_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".binah_config.json")

# Hard-coded factory defaults — K-edge (hard X-ray, E0 > 2 keV)
_NORM_FACTORY: dict = {
    "pre1": -150.0, "pre2": -30.0,
    "nor1":  150.0, "nor2": 400.0, "nnorm": 1,
    "rbkg": 1.0,  "kmin_bkg": 0.5,
    "kmin": 2.0,  "kmax": 12.0, "dk": 1.0, "kw": 2, "rmax": 6.0,
}

# L-edge factory defaults (soft X-ray, E0 < 2 keV, e.g. Ni L₃ ≈ 853 eV)
# Pre-edge: fit line in region -30 to -5 eV before edge
# Post-edge: normalize in +10 to +30 eV above L₃ (below the L₂ edge, ~+17 eV for Ni)
# AUTOBK / XFTF not applied — k_max ≈ √(0.26 × 17) ≈ 2.1 Å⁻¹ is insufficient
_NORM_FACTORY_L: dict = {
    "pre1": -30.0, "pre2": -5.0,
    "nor1":  10.0, "nor2": 30.0, "nnorm": 1,
    "rbkg": 1.0, "kmin_bkg": 0.5,
    "kmin": 0.5, "kmax": 2.0, "dk": 0.3, "kw": 1, "rmax": 3.0,
}

def _is_l_edge_e0(e0: float) -> bool:
    """Return True if E0 indicates a soft X-ray / L-edge scan (< 2000 eV)."""
    return 100 < e0 < 2000

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


from experimental_parser import ExperimentalScan, align_and_average_scans

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
                 replot_fn: Optional[Callable] = None,
                 add_scan_fn: Optional[Callable] = None):
        super().__init__(parent)
        self._get_scans  = get_scans_fn
        self._replot_fn  = replot_fn    # called after apply-all to refresh Spectra tab
        self._add_scan_fn = add_scan_fn  # called to push an averaged scan back to Binah

        # Analysis results cache per scan label
        self._results: dict = {}

        # Which scans are selected for overlay
        self._selected_labels: List[str] = []
        self._click_mode = tk.StringVar(value="")

        # Scan list panel — BooleanVar per label (visible in overlay)
        self._scan_vis_vars: dict = {}   # label → tk.BooleanVar

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

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        tk.Button(top, text="\u2211 Average Scans\u2026",
                  bg="#5c1a5c", fg="white", font=("", 9, "bold"),
                  command=self._open_average_dialog).pack(side=tk.LEFT, padx=2)

        self._status_lbl = tk.Label(top, text="Load experimental scans first (File \u2192 Load Exp. Data)",
                                     fg="gray", font=("", 8))
        self._status_lbl.pack(side=tk.LEFT, padx=10)

        # Main body: params left, scan list centre-left, plot right
        body = tk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._build_params(body)
        self._build_scan_list(body)
        self._build_plot(body)

    def _build_scan_list(self, parent):
        """Scrollable scan list panel — one row per loaded scan with colour + checkbox."""
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

        # Scrollable inner area
        _wrap = tk.Frame(outer)
        _wrap.pack(fill=tk.BOTH, expand=True)
        _vsb = ttk.Scrollbar(_wrap, orient=tk.VERTICAL)
        _vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._scan_list_canvas = tk.Canvas(_wrap, yscrollcommand=_vsb.set,
                                            bg="white", highlightthickness=0)
        self._scan_list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _vsb.config(command=self._scan_list_canvas.yview)

        self._scan_list_inner = tk.Frame(self._scan_list_canvas, bg="white")
        self._scan_list_window = self._scan_list_canvas.create_window(
            (0, 0), window=self._scan_list_inner, anchor="nw")

        self._scan_list_inner.bind(
            "<Configure>",
            lambda _e: self._scan_list_canvas.configure(
                scrollregion=self._scan_list_canvas.bbox("all")))
        self._scan_list_canvas.bind(
            "<Configure>",
            lambda e: self._scan_list_canvas.itemconfig(
                self._scan_list_window, width=e.width))

        # Mouse-wheel scrolling
        def _on_wheel(e):
            self._scan_list_canvas.yview_scroll(
                int(-1 * (e.delta / 120)), "units")
        self._scan_list_canvas.bind("<MouseWheel>", _on_wheel)

    def _rebuild_scan_list_rows(self):
        """Destroy and recreate all rows in the scan list panel."""
        for w in self._scan_list_inner.winfo_children():
            w.destroy()

        scans = self._get_scans()
        for i, (label, scan, *_) in enumerate(scans):
            col = _PALETTE[i % len(_PALETTE)]

            # Ensure BooleanVar exists (default visible)
            if label not in self._scan_vis_vars:
                self._scan_vis_vars[label] = tk.BooleanVar(value=True)
            var = self._scan_vis_vars[label]

            row = tk.Frame(self._scan_list_inner, bg="white")
            row.pack(fill=tk.X, pady=1, padx=2)

            # Colour swatch
            tk.Label(row, bg=col, width=2, relief=tk.FLAT).pack(side=tk.LEFT, padx=(2, 3))

            # Visibility checkbox
            cb = tk.Checkbutton(row, variable=var, bg="white", pady=0,
                                command=lambda lbl=label: self._toggle_scan_vis(lbl))
            cb.pack(side=tk.LEFT)

            # Clickable label — selects scan + runs analysis
            short = label if len(label) <= 22 else label[:20] + "…"
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
        """Click on a scan name — select it in combobox, auto-fill E0, run analysis."""
        self._scan_var.set(label)
        self._auto_fill_e0()
        self._run()

    def _toggle_scan_vis(self, label: str):
        """Checkbox toggled — add or remove from overlay and redraw."""
        var = self._scan_vis_vars.get(label)
        if var is None:
            return
        if var.get():
            # Turned on — add to overlay if analysed, else run analysis
            if label in self._results:
                if label not in self._selected_labels:
                    self._selected_labels.append(label)
                self._redraw()
            else:
                self._scan_var.set(label)
                self._auto_fill_e0()
                self._run()
        else:
            # Turned off — remove from overlay
            if label in self._selected_labels:
                self._selected_labels.remove(label)
            self._redraw()

    def _show_all_scans(self):
        for label, var in self._scan_vis_vars.items():
            var.set(True)
            if label not in self._selected_labels:
                self._selected_labels.append(label)
        self._redraw()

    def _hide_all_scans(self):
        for var in self._scan_vis_vars.values():
            var.set(False)
        self._selected_labels.clear()
        self._redraw()

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

        # E0 range covers both L-edges (~100 eV) and heavy-atom K-edges (>30 keV)
        row("E0 (eV):",    self._e0_var,   100, 40000, 0.5,  "%.1f")
        row("pre1 (eV):",  self._pre1_var, -300,  -1,   5.0,  "%.0f")
        row("pre2 (eV):",  self._pre2_var, -200,  -1,   5.0,  "%.0f")
        row("nor1 (eV):",  self._nor1_var,    1,  500,   5.0, "%.0f")
        row("nor2 (eV):",  self._nor2_var,    1, 1000,   5.0, "%.0f")

        # Edge-type indicator — updated dynamically in _auto_fill_e0()
        self._edge_type_lbl = tk.Label(
            pf, text="", font=("", 8, "bold"), anchor="w",
            fg="#005500", wraplength=195, justify="left")
        self._edge_type_lbl.pack(fill=tk.X, pady=(1, 0))

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

        # ── AUTOBK ── K-edge only ─────────────────────────────────────────
        lbl("\u2500\u2500 AUTOBK \u2014 K-edge only \u2500\u2500\u2500\u2500")
        self._rbkg_var     = tk.DoubleVar(value=nd["rbkg"])
        self._kmin_bkg_var = tk.DoubleVar(value=nd["kmin_bkg"])
        row("rbkg (A):",   self._rbkg_var,   0.3, 3.0, 0.1, "%.1f")
        row("kmin_bkg:",   self._kmin_bkg_var, 0, 5.0, 0.5, "%.1f")

        # ── XFTF ── K-edge only ───────────────────────────────────────────
        lbl("\u2500\u2500 FT \u2014 K-edge only \u2500\u2500\u2500\u2500\u2500\u2500\u2500")
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

        # ── Show on XANES plot (Athena-style) — hidden when >1 scan shown ────
        self._show_section_frame = tk.Frame(pf)
        self._show_section_frame.pack(fill=tk.X)

        def _slbl(text):
            tk.Label(self._show_section_frame, text=text,
                     font=("", 8, "bold"), fg="#333333",
                     anchor="w").pack(fill=tk.X, pady=(6, 0))

        _slbl("── Show on XANES plot ───────")
        self._show_raw_var      = tk.BooleanVar(value=True)
        self._show_preline_var  = tk.BooleanVar(value=True)
        self._show_postline_var = tk.BooleanVar(value=True)
        self._show_bkg_var      = tk.BooleanVar(value=False)
        self._show_norm_var     = tk.BooleanVar(value=True)
        self._show_deriv_var    = tk.BooleanVar(value=False)
        self._show_win_var      = tk.BooleanVar(value=True)

        for _txt, _var in [
            ("\u03bc(E)  raw",         self._show_raw_var),
            ("Pre-edge fit",           self._show_preline_var),
            ("Post-edge fit",          self._show_postline_var),
            ("Background (bkg)",       self._show_bkg_var),
            ("Normalized \u03bc(E)",   self._show_norm_var),
            ("Derivative d\u03bc/dE",  self._show_deriv_var),
        ]:
            tk.Checkbutton(self._show_section_frame, text=_txt, variable=_var,
                           command=self._redraw_xanes,
                           font=("", 8)).pack(anchor="w", pady=1)

        lbl("\u2500\u2500 View Box \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        self._xanes_xmin_var = tk.StringVar(value="")
        self._xanes_xmax_var = tk.StringVar(value="")
        self._xanes_ymin_var = tk.StringVar(value="")
        self._xanes_ymax_var = tk.StringVar(value="")

        f_vx1 = tk.Frame(pf); f_vx1.pack(fill=tk.X, pady=1)
        tk.Label(f_vx1, text="X min:", width=14, anchor="w",
                 font=("", 8)).pack(side=tk.LEFT)
        ttk.Entry(f_vx1, textvariable=self._xanes_xmin_var,
                  width=9, font=("Courier", 8)).pack(side=tk.LEFT)

        f_vx2 = tk.Frame(pf); f_vx2.pack(fill=tk.X, pady=1)
        tk.Label(f_vx2, text="X max:", width=14, anchor="w",
                 font=("", 8)).pack(side=tk.LEFT)
        ttk.Entry(f_vx2, textvariable=self._xanes_xmax_var,
                  width=9, font=("Courier", 8)).pack(side=tk.LEFT)

        f_vy1 = tk.Frame(pf); f_vy1.pack(fill=tk.X, pady=1)
        tk.Label(f_vy1, text="Y min:", width=14, anchor="w",
                 font=("", 8)).pack(side=tk.LEFT)
        ttk.Entry(f_vy1, textvariable=self._xanes_ymin_var,
                  width=9, font=("Courier", 8)).pack(side=tk.LEFT)

        f_vy2 = tk.Frame(pf); f_vy2.pack(fill=tk.X, pady=1)
        tk.Label(f_vy2, text="Y max:", width=14, anchor="w",
                 font=("", 8)).pack(side=tk.LEFT)
        ttk.Entry(f_vy2, textvariable=self._xanes_ymax_var,
                  width=9, font=("Courier", 8)).pack(side=tk.LEFT)

        f_vbtn = tk.Frame(pf); f_vbtn.pack(fill=tk.X, pady=(2, 1))
        tk.Button(f_vbtn, text="Apply", font=("", 8),
                  command=self._apply_xanes_view_box).pack(side=tk.LEFT)
        tk.Button(f_vbtn, text="From Plot", font=("", 8),
                  command=self._capture_xanes_view_box).pack(side=tk.LEFT, padx=4)
        tk.Button(f_vbtn, text="Auto", font=("", 8),
                  command=self._reset_xanes_view_box).pack(side=tk.LEFT)

        lbl("\u2500\u2500 Processing \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        self._deglitch_sigma_var = tk.DoubleVar(value=6.0)
        self._deglitch_window_var = tk.IntVar(value=7)
        row("Glitch \u03c3:", self._deglitch_sigma_var, 2.0, 15.0, 0.5, "%.1f")
        row("Median pts:", self._deglitch_window_var, 3, 31, 2, "%.0f")
        f_deg = tk.Frame(pf); f_deg.pack(fill=tk.X, pady=(2, 1))
        tk.Button(f_deg, text="Auto Deglitch", font=("", 8),
                  command=self._auto_deglitch_current).pack(side=tk.LEFT)
        tk.Button(f_deg, text="Reset Scan", font=("", 8),
                  command=self._reset_current_scan_processing).pack(side=tk.LEFT, padx=4)

        self._smooth_window_var = tk.IntVar(value=7)
        self._smooth_poly_var = tk.IntVar(value=3)
        row("Smooth pts:", self._smooth_window_var, 5, 51, 2, "%.0f")
        row("Poly order:", self._smooth_poly_var, 2, 5, 1, "%.0f")
        f_smooth = tk.Frame(pf); f_smooth.pack(fill=tk.X, pady=(2, 1))
        tk.Button(f_smooth, text="Smooth Scan", font=("", 8),
                  command=self._smooth_current_scan).pack(side=tk.LEFT)

        self._energy_shift_var = tk.DoubleVar(value=0.0)
        row("Shift E (eV):", self._energy_shift_var, -20.0, 20.0, 0.1, "%.1f")
        f_shift = tk.Frame(pf); f_shift.pack(fill=tk.X, pady=(2, 1))
        tk.Button(f_shift, text="Shift Energy", font=("", 8),
                  command=self._shift_current_scan_energy).pack(side=tk.LEFT)

        tk.Checkbutton(pf, text="Show FT window on \u03c7(k)",
                       variable=self._show_win_var,
                       font=("", 8)).pack(anchor="w", pady=1)

    def _update_show_section_visibility(self):
        """Keep the XANES display toggles visible.

        The XAS Analysis tab auto-selects all loaded scans on entry, so hiding
        this section when multiple scans are visible makes the controls appear
        to vanish during normal use.
        """
        if not self._show_section_frame.winfo_manager():
            self._show_section_frame.pack(fill=tk.X)

    def _parse_view_limit(self, value: str) -> Optional[float]:
        value = str(value).strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _format_view_limit(self, value: float) -> str:
        return f"{value:.3f}".rstrip("0").rstrip(".")

    def _apply_xanes_view_limits(self):
        xmin = self._parse_view_limit(self._xanes_xmin_var.get())
        xmax = self._parse_view_limit(self._xanes_xmax_var.get())
        ymin = self._parse_view_limit(self._xanes_ymin_var.get())
        ymax = self._parse_view_limit(self._xanes_ymax_var.get())

        if xmin is not None and xmax is not None and xmin >= xmax:
            self._status_lbl.config(text="XANES view box: X min must be < X max.",
                                    fg="#993300")
            return
        if ymin is not None and ymax is not None and ymin >= ymax:
            self._status_lbl.config(text="XANES view box: Y min must be < Y max.",
                                    fg="#993300")
            return

        if xmin is not None or xmax is not None:
            self._ax_mu.set_xlim(left=xmin, right=xmax)
        if ymin is not None or ymax is not None:
            self._ax_mu.set_ylim(bottom=ymin, top=ymax)

    def _apply_xanes_view_box(self):
        self._redraw_xanes()
        self._status_lbl.config(text="Applied XANES view box.", fg="#003366")

    def _capture_xanes_view_box(self):
        xmin, xmax = self._ax_mu.get_xlim()
        ymin, ymax = self._ax_mu.get_ylim()
        self._xanes_xmin_var.set(self._format_view_limit(xmin))
        self._xanes_xmax_var.set(self._format_view_limit(xmax))
        self._xanes_ymin_var.set(self._format_view_limit(ymin))
        self._xanes_ymax_var.set(self._format_view_limit(ymax))
        self._status_lbl.config(text="Captured current XANES plot limits.",
                                fg="#003366")

    def _reset_xanes_view_box(self):
        self._xanes_xmin_var.set("")
        self._xanes_xmax_var.set("")
        self._xanes_ymin_var.set("")
        self._xanes_ymax_var.set("")
        self._redraw_xanes()
        self._status_lbl.config(text="Reset XANES view box to auto.",
                                fg="gray")

    def _ensure_scan_backup(self, scan) -> None:
        meta = getattr(scan, "metadata", None)
        if meta is None:
            scan.metadata = {}
            meta = scan.metadata
        if "_binah_original_energy" not in meta:
            meta["_binah_original_energy"] = scan.energy_ev.copy()
            meta["_binah_original_mu"] = scan.mu.copy()
            meta["_binah_original_e0"] = float(scan.e0)
            meta["_binah_original_norm"] = bool(scan.is_normalized)

    def _restore_scan_backup(self, scan) -> bool:
        self._ensure_scan_backup(scan)
        meta = scan.metadata
        if "_binah_original_energy" not in meta or "_binah_original_mu" not in meta:
            return False
        scan.energy_ev = np.array(meta["_binah_original_energy"], dtype=float).copy()
        scan.mu = np.array(meta["_binah_original_mu"], dtype=float).copy()
        scan.e0 = float(meta.get("_binah_original_e0", 0.0))
        scan.is_normalized = bool(meta.get("_binah_original_norm", scan.is_normalized))
        return True

    def _odd_int(self, value, minimum: int = 3) -> int:
        out = max(minimum, int(round(float(value))))
        if out % 2 == 0:
            out += 1
        return out

    def _get_current_scan(self):
        label = self._scan_var.get()
        if not label:
            self._status_lbl.config(text="Select a scan first.", fg="#993300")
            return None, None
        scan = self._get_scan_by_label(label)
        if scan is None:
            self._status_lbl.config(text="Scan not found. Click \u21bb Refresh.",
                                    fg="#993300")
            return label, None
        self._ensure_scan_backup(scan)
        return label, scan

    def _refresh_after_processing_change(self, focus_label: Optional[str] = None) -> None:
        current = focus_label or self._scan_var.get()
        self._results.clear()

        # Keep only labels that still exist.
        self._selected_labels = [
            lbl for lbl in self._selected_labels
            if self._get_scan_by_label(lbl) is not None
        ]

        rerun_labels = list(self._selected_labels)
        if current and not rerun_labels and self._get_scan_by_label(current) is not None:
            rerun_labels.append(current)
            if current not in self._selected_labels:
                self._selected_labels.append(current)
            if current not in self._scan_vis_vars:
                self._scan_vis_vars[current] = tk.BooleanVar(value=True)
            self._scan_vis_vars[current].set(True)

        self._rebuild_scan_list_rows()

        if current:
            self._scan_var.set(current)
            self._auto_fill_e0()

        for label in rerun_labels:
            scan = self._get_scan_by_label(label)
            if scan is not None:
                self._run_single(label, scan)

        if rerun_labels:
            self._redraw()
        else:
            self._draw_empty_xanes()
            self._draw_empty_exafs()
            if self._replot_fn is not None:
                self._replot_fn()

    def _deglitch_scan_arrays(self, energy: np.ndarray, mu: np.ndarray,
                              sigma: float, window_pts: int):
        if len(mu) < 5:
            return mu.copy(), np.zeros(len(mu), dtype=bool)

        kernel = min(self._odd_int(window_pts, minimum=3), len(mu) if len(mu) % 2 == 1 else len(mu) - 1)
        if kernel < 3:
            return mu.copy(), np.zeros(len(mu), dtype=bool)

        baseline = median_filter(mu, size=kernel, mode="nearest")
        resid = mu - baseline
        resid_med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - resid_med)))
        scale = 1.4826 * mad
        if scale <= 1e-12:
            scale = float(np.std(resid))
        if scale <= 1e-12:
            return mu.copy(), np.zeros(len(mu), dtype=bool)

        glitch_mask = np.abs(resid - resid_med) > max(float(sigma), 0.5) * scale
        if glitch_mask.sum() == 0:
            return mu.copy(), glitch_mask

        good = ~glitch_mask
        if good.sum() < 2:
            return mu.copy(), np.zeros(len(mu), dtype=bool)

        mu_fixed = mu.copy()
        mu_fixed[glitch_mask] = np.interp(energy[glitch_mask], energy[good], mu[good])
        return mu_fixed, glitch_mask

    def _auto_deglitch_current(self):
        label, scan = self._get_current_scan()
        if scan is None:
            return

        sigma = float(self._deglitch_sigma_var.get())
        window_pts = int(self._deglitch_window_var.get())
        mu_new, glitch_mask = self._deglitch_scan_arrays(
            scan.energy_ev, scan.mu, sigma=sigma, window_pts=window_pts)

        n_glitch = int(np.count_nonzero(glitch_mask))
        if n_glitch == 0:
            self._status_lbl.config(
                text=f"No glitches detected for {label}. Try a lower \u03c3 threshold.",
                fg="gray")
            return

        scan.mu = mu_new
        self._refresh_after_processing_change(label)
        self._status_lbl.config(
            text=f"Deglitched {label}: replaced {n_glitch} point(s).",
            fg="#003366")

    def _smooth_current_scan(self):
        label, scan = self._get_current_scan()
        if scan is None:
            return

        npts = len(scan.mu)
        if npts < 5:
            self._status_lbl.config(text="Not enough data points to smooth.",
                                    fg="#993300")
            return

        window = min(self._odd_int(self._smooth_window_var.get(), minimum=5),
                     npts if npts % 2 == 1 else npts - 1)
        poly = max(1, min(int(self._smooth_poly_var.get()), window - 1))
        if window < 5 or window <= poly:
            self._status_lbl.config(text="Smooth settings are not valid for this scan.",
                                    fg="#993300")
            return

        scan.mu = savgol_filter(scan.mu, window_length=window,
                                polyorder=poly, mode="interp")
        self._refresh_after_processing_change(label)
        self._status_lbl.config(
            text=f"Smoothed {label} with Savitzky-Golay ({window} pts, poly {poly}).",
            fg="#003366")

    def _shift_current_scan_energy(self):
        label, scan = self._get_current_scan()
        if scan is None:
            return

        shift = float(self._energy_shift_var.get())
        if abs(shift) < 1e-12:
            self._status_lbl.config(text="Energy shift is 0.0 eV; nothing changed.",
                                    fg="gray")
            return

        meta = getattr(scan, "metadata", {}) or {}
        link_group = meta.get("_binah_link_group")
        linked_scans = []
        if link_group:
            for other_label, other_scan, *_ in self._get_scans():
                other_meta = getattr(other_scan, "metadata", {}) or {}
                if other_meta.get("_binah_link_group") == link_group:
                    linked_scans.append((other_label, other_scan))
        else:
            linked_scans.append((label, scan))

        moved = 0
        for _lbl, _scan in linked_scans:
            _scan.energy_ev = _scan.energy_ev + shift
            if _scan.e0:
                _scan.e0 = float(_scan.e0 + shift)
            moved += 1

        self._refresh_after_processing_change(label)
        if moved > 1:
            self._status_lbl.config(
                text=f"Shifted {label} and {moved - 1} linked scan(s) by {shift:.2f} eV.",
                fg="#003366")
        else:
            self._status_lbl.config(
                text=f"Shifted {label} by {shift:.2f} eV.",
                fg="#003366")

    def _reset_current_scan_processing(self):
        label, scan = self._get_current_scan()
        if scan is None:
            return

        if not self._restore_scan_backup(scan):
            self._status_lbl.config(text="No saved original scan state found.",
                                    fg="#993300")
            return

        self._refresh_after_processing_change(label)
        self._status_lbl.config(
            text=f"Reset {label} to the originally loaded scan.",
            fg="gray")

    # ── Averaging dialog ─────────────────────────────────────────────────────

    def _open_average_dialog(self):
        """Open a dialog to select scans for averaging with optional I₂ alignment."""
        scans = self._get_scans()
        if not scans:
            messagebox.showinfo("No Scans", "Load experimental scans first.", parent=self)
            return

        dlg = tk.Toplevel(self)
        dlg.title("Average Scans")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.lift()

        # ── Header ──────────────────────────────────────────────────────────
        hdr = tk.Frame(dlg, bg="#5c1a5c", padx=10, pady=6)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="\u2211  Average XAS Scans", font=("", 11, "bold"),
                 bg="#5c1a5c", fg="white").pack(side=tk.LEFT)

        body = tk.Frame(dlg, padx=12, pady=8)
        body.pack(fill=tk.BOTH, expand=True)

        # ── Scan checklist ───────────────────────────────────────────────────
        tk.Label(body, text="Select scans to average:",
                 font=("", 9, "bold")).grid(row=0, column=0, columnspan=3,
                                             sticky="w", pady=(0, 4))
        tk.Label(body, text="Scan", font=("", 8, "bold"), fg="#333",
                 width=42, anchor="w").grid(row=1, column=0, sticky="w")
        tk.Label(body, text="Reference", font=("", 8, "bold"),
                 fg="#333").grid(row=1, column=1, sticky="w", padx=(8, 0))

        check_vars: List[tk.BooleanVar] = []
        scan_entries = []   # (label, scan) tuples

        scroll_fr = tk.Frame(body)
        scroll_fr.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=4)

        canvas = tk.Canvas(scroll_fr, width=480, height=min(220, len(scans)*26 + 10),
                           highlightthickness=0)
        sb = ttk.Scrollbar(scroll_fr, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        inner = tk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        for label, scan, *_ in scans:
            var = tk.BooleanVar(value=False)
            check_vars.append(var)
            scan_entries.append((label, scan))

            row_fr = tk.Frame(inner)
            row_fr.pack(fill=tk.X, pady=1)

            tk.Checkbutton(row_fr, variable=var, width=0).pack(side=tk.LEFT)
            # Colour swatch
            col = _PALETTE[len(scan_entries) % len(_PALETTE)]
            tk.Label(row_fr, bg=col, width=2, relief=tk.FLAT).pack(
                side=tk.LEFT, padx=(0, 4))
            # Scan label (truncated)
            short = label[:45] + ("…" if len(label) > 45 else "")
            tk.Label(row_fr, text=short, font=("", 8), anchor="w",
                     width=42).pack(side=tk.LEFT)
            # Reference indicator
            if scan.has_reference():
                ref_txt = f"✓ {scan.ref_label}" if scan.ref_label else "✓ ref"
                ref_fg  = "#005500"
            else:
                ref_txt = "—"
                ref_fg  = "#999999"
            tk.Label(row_fr, text=ref_txt, font=("", 8), fg=ref_fg,
                     width=14, anchor="w").pack(side=tk.LEFT, padx=(8, 0))

        # ── Options ─────────────────────────────────────────────────────────
        opt_fr = tk.Frame(body)
        opt_fr.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 2))

        use_ref_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_fr,
                       text="Use reference channel for energy alignment (I\u2082 / diode)",
                       variable=use_ref_var,
                       font=("", 8)).pack(side=tk.LEFT)

        lbl_fr = tk.Frame(body)
        lbl_fr.grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 2))
        tk.Label(lbl_fr, text="Output label:", font=("", 8)).pack(side=tk.LEFT)
        out_label_var = tk.StringVar(value="average")
        tk.Entry(lbl_fr, textvariable=out_label_var, width=30,
                 font=("", 8)).pack(side=tk.LEFT, padx=(6, 0))

        info_lbl = tk.Label(body, text="", font=("", 8), fg="#444",
                            wraplength=460, justify="left")
        info_lbl.grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # ── Buttons ─────────────────────────────────────────────────────────
        btn_fr = tk.Frame(body)
        btn_fr.grid(row=6, column=0, columnspan=3, pady=(10, 2))

        def _do_average():
            selected = [(label, scan)
                        for (label, scan), var in zip(scan_entries, check_vars)
                        if var.get()]
            if len(selected) < 2:
                info_lbl.config(text="⚠ Select at least 2 scans.", fg="#993300")
                return

            sel_scans = [scan for _, scan in selected]
            n_ref     = sum(1 for sc in sel_scans if sc.has_reference())
            use_ref   = use_ref_var.get() and n_ref == len(sel_scans)

            if use_ref_var.get() and n_ref < len(sel_scans):
                info_lbl.config(
                    text=(f"⚠ Only {n_ref}/{len(sel_scans)} scans have a reference "
                          f"channel — averaging without alignment."),
                    fg="#885500")
                use_ref = False

            try:
                averaged = align_and_average_scans(
                    sel_scans,
                    use_reference=use_ref,
                    label=out_label_var.get().strip() or "average",
                )
            except Exception as exc:
                info_lbl.config(text=f"Error: {exc}", fg="red")
                return

            meta = averaged.metadata
            shift_str = ""
            if use_ref and meta.get("shifts_ev"):
                shifts = meta["shifts_ev"]
                shift_str = (
                    f"  |  energy shifts: "
                    + ", ".join(f"{s:+.2f}" for s in shifts) + " eV"
                )

            info_lbl.config(
                text=(f"✓ Averaged {meta['n_averaged']} scans"
                      + (" with reference alignment" if use_ref else " (no alignment)")
                      + shift_str),
                fg="#005500")

            if self._add_scan_fn is not None:
                self._add_scan_fn(averaged)
                self.refresh_scan_list()
                dlg.after(800, dlg.destroy)
            else:
                info_lbl.config(
                    text="Averaged scan created but no add_scan_fn provided.",
                    fg="#885500")

        tk.Button(btn_fr, text="\u2211  Average & Add to Binah",
                  bg="#5c1a5c", fg="white", font=("", 9, "bold"),
                  command=_do_average).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_fr, text="Cancel",
                  command=dlg.destroy).pack(side=tk.LEFT, padx=4)

    def auto_run_all(self):
        """Run analysis on every loaded scan and show them all. Called when tab is opened."""
        scans = self._get_scans()
        if not scans:
            return
        self._rebuild_scan_list_rows()
        for label, scan, *_ in scans:
            if label not in self._scan_vis_vars:
                self._scan_vis_vars[label] = tk.BooleanVar(value=True)
            self._scan_vis_vars[label].set(True)
            # Run analysis only if not already cached
            if label not in self._results:
                self._scan_var.set(label)
                self._auto_fill_e0()
                self._run_single(label, scan)
            if label not in self._selected_labels:
                self._selected_labels.append(label)
        self._rebuild_scan_list_rows()
        self._redraw()

    def _run_single(self, label: str, scan):
        """Run analysis pipeline on one scan without touching the overlay/redraw."""
        self._ensure_scan_backup(scan)
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

        # Use stored e0 if valid
        if scan.e0 and scan.e0 > 100:
            e0 = scan.e0

        energy = scan.energy_ev.copy()
        mu_raw = scan.mu.copy()

        try:
            if _HAS_LARCH:
                session = _get_larch_session()
                grp = LarchGroup(energy=energy, mu=mu_raw)
                _larch_pre_edge(grp, _larch=session,
                                e0=float(e0) if e0 > 100 else None,
                                pre1=float(pre1), pre2=float(pre2),
                                norm1=float(nor1), norm2=float(nor2),
                                nnorm=int(nnor))
                _larch_autobk(grp, _larch=session,
                               rbkg=float(rbkg), kmin=float(kmin_bkg))
                _larch_xftf(grp, _larch=session,
                             kmin=float(kmin), kmax=float(kmax),
                             dk=float(dk), kweight=int(kw))
                self._results[label] = {
                    "energy":    grp.energy,
                    "mu_norm":   getattr(grp, "flat", grp.norm),
                    "bkg_e":     grp.bkg,
                    "k":         grp.k,
                    "chi":       grp.chi,
                    "r":         grp.r,
                    "chi_r":     grp.chir_mag,
                    "chi_r_re":  grp.chir_re,
                    "chi_r_im":  grp.chir_im,
                    "e0":        float(grp.e0),
                    "edge_step": float(grp.edge_step),
                    "kw":        kw,
                    "mu_raw":    mu_raw,
                    "pre_line":  getattr(grp, "pre_edge", np.zeros_like(energy)),
                }
                # Push back to scan
                scan.mu = getattr(grp, "flat", grp.norm).copy()
                scan.e0 = float(grp.e0)
                scan.is_normalized = True
            else:
                mu_norm, edge_step, pre_line = normalize_xanes(
                    energy, mu_raw, e0, pre1, pre2, nor1, nor2, nnor)
                k_arr, chi, bkg_e = autobk(energy, mu_norm, e0, rbkg, kmin_bkg)
                r_arr, chi_r, chi_r_re, chi_r_im = (
                    xftf(k_arr, chi, kmin, kmax, dk, kw)
                    if len(k_arr) >= 4
                    else (np.array([0.0]), np.array([0.0]),
                          np.array([0.0]), np.array([0.0])))
                self._results[label] = {
                    "energy": energy, "mu_norm": mu_norm, "bkg_e": bkg_e,
                    "k": k_arr, "chi": chi,
                    "r": r_arr, "chi_r": chi_r,
                    "chi_r_re": chi_r_re, "chi_r_im": chi_r_im,
                    "e0": e0, "edge_step": edge_step, "kw": kw,
                    "mu_raw": mu_raw, "pre_line": pre_line,
                }
                scan.mu = mu_norm.copy()
                scan.e0 = e0
                scan.is_normalized = True
        except Exception:
            pass   # silently skip failed scans in batch mode

    def _build_plot(self, parent):
        plot_area = tk.Frame(parent)
        plot_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)

        self._plot_nb = ttk.Notebook(plot_area)
        self._plot_nb.pack(fill=tk.BOTH, expand=True)

        # ── XANES tab ─────────────────────────────────────────────────────────
        xanes_outer = tk.Frame(self._plot_nb)
        self._plot_nb.add(xanes_outer, text="  XANES / \u03bc(E)  ")

        self._build_click_toolbar(xanes_outer)

        tb_xanes = tk.Frame(xanes_outer)
        tb_xanes.pack(side=tk.BOTTOM, fill=tk.X)

        self._fig_xanes = Figure(figsize=(8, 5), dpi=96, facecolor="white")
        self._ax_mu = self._fig_xanes.add_subplot(111)
        self._fig_xanes.subplots_adjust(left=0.10, right=0.96, top=0.93, bottom=0.10)

        self._canvas_xanes = FigureCanvasTkAgg(self._fig_xanes, master=xanes_outer)
        self._canvas_xanes.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._toolbar_xanes = NavigationToolbar2Tk(self._canvas_xanes, tb_xanes)
        self._toolbar_xanes.update()

        self._canvas_xanes.mpl_connect('button_press_event', self._on_plot_click)

        # ── EXAFS tab ─────────────────────────────────────────────────────────
        exafs_outer = tk.Frame(self._plot_nb)
        self._plot_nb.add(exafs_outer, text="  EXAFS  ")

        tb_exafs = tk.Frame(exafs_outer)
        tb_exafs.pack(side=tk.BOTTOM, fill=tk.X)

        self._fig_exafs = Figure(figsize=(8, 6), dpi=96, facecolor="white")
        _gs2 = GridSpec(2, 1, figure=self._fig_exafs,
                        hspace=0.42, top=0.94, bottom=0.08, left=0.11, right=0.95)
        self._ax_chi = self._fig_exafs.add_subplot(_gs2[0])
        self._ax_r   = self._fig_exafs.add_subplot(_gs2[1])

        self._canvas_exafs = FigureCanvasTkAgg(self._fig_exafs, master=exafs_outer)
        self._canvas_exafs.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._toolbar_exafs = NavigationToolbar2Tk(self._canvas_exafs, tb_exafs)
        self._toolbar_exafs.update()

        self._draw_empty_xanes()
        self._draw_empty_exafs()

    # ── Click-to-set toolbar ──────────────────────────────────────────────────

    def _build_click_toolbar(self, parent):
        bar = tk.Frame(parent, bd=1, relief=tk.GROOVE, bg="#ECECEC", pady=3)
        bar.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(2, 0))

        tk.Label(bar, text="Click to set:", font=("", 8, "bold"),
                 bg="#ECECEC").pack(side=tk.LEFT, padx=(4, 6))

        self._click_btns: dict = {}
        _btn_defs = [
            ("E0",   "#4A4A8A", "white"),
            ("pre1", "#2C7BB6", "white"),
            ("pre2", "#2C7BB6", "white"),
            ("nor1", "#1A9641", "white"),
            ("nor2", "#1A9641", "white"),
        ]
        for _key, _bg, _fg in _btn_defs:
            _b = tk.Button(bar, text=_key, width=5, font=("", 8),
                           relief=tk.RAISED, bg=_bg, fg=_fg,
                           activebackground=_bg, activeforeground=_fg,
                           command=lambda k=_key: self._set_click_mode(k))
            _b.pack(side=tk.LEFT, padx=2)
            self._click_btns[_key] = _b

        tk.Button(bar, text="\u2715 Cancel", font=("", 8), bg="#ECECEC",
                  command=lambda: self._set_click_mode("")).pack(side=tk.LEFT, padx=6)

        self._click_hint = tk.Label(
            bar, text="  Select a parameter above, then click on the plot",
            fg="gray", font=("", 8, "italic"), bg="#ECECEC")
        self._click_hint.pack(side=tk.LEFT, padx=4)

    def _set_click_mode(self, mode: str):
        self._click_mode.set(mode)
        _inactive = {"E0": "#4A4A8A", "pre1": "#2C7BB6", "pre2": "#2C7BB6",
                     "nor1": "#1A9641", "nor2": "#1A9641"}
        for _key, _btn in self._click_btns.items():
            if _key == mode:
                _btn.config(relief=tk.SUNKEN, font=("", 8, "bold"),
                            bg="#FFB347", fg="black")
            else:
                _btn.config(relief=tk.RAISED, font=("", 8),
                            bg=_inactive[_key], fg="white")
        if mode:
            self._click_hint.config(
                text=f"  \u2192 Click on plot to set  {mode}  (\u2715 to cancel)",
                fg="#003399")
        else:
            self._click_hint.config(
                text="  Select a parameter above, then click on the plot",
                fg="gray")

    def _on_plot_click(self, event):
        mode = self._click_mode.get()
        if not mode or event.button != 1:
            return
        if getattr(self._toolbar_xanes, 'mode', ''):
            return
        if event.inaxes is None or event.xdata is None:
            return

        x  = event.xdata
        e0 = self._e0_var.get()

        if mode == "E0":
            self._e0_var.set(round(x, 1))
        elif mode == "pre1":
            self._pre1_var.set(round(min(x - e0, self._pre2_var.get() - 5.0), 1))
        elif mode == "pre2":
            self._pre2_var.set(round(max(x - e0, self._pre1_var.get() + 5.0), 1))
        elif mode == "nor1":
            self._nor1_var.set(round(min(x - e0, self._nor2_var.get() - 10.0), 1))
        elif mode == "nor2":
            self._nor2_var.set(round(max(x - e0, self._nor1_var.get() + 10.0), 1))

        self._set_click_mode("")
        if self._scan_var.get():
            self._run()

    # ── Empty-state figures ───────────────────────────────────────────────────

    def _draw_empty_xanes(self):
        self._ax_mu.clear()
        self._ax_mu.set_title("\u03bc(E)  \u2014  XANES", fontsize=9, loc="left", pad=3)
        self._ax_mu.set_xlabel("Energy (eV)", fontsize=8)
        self._ax_mu.set_ylabel("\u03bc(E)", fontsize=8)
        self._ax_mu.tick_params(labelsize=7)
        self._ax_mu.text(0.5, 0.45, "Load a scan and click  \u25b6  Run Analysis",
                          transform=self._ax_mu.transAxes,
                          ha="center", va="center", fontsize=9, color="lightgray")
        self._apply_xanes_view_limits()
        if _HAS_SNS:
            sns.despine(ax=self._ax_mu, offset=4)
        self._canvas_xanes.draw_idle()

    def _draw_empty_exafs(self):
        for _ax, _title, _xl, _yl in [
            (self._ax_chi, "\u03c7(k)  \u2014  EXAFS oscillations",
             "k  (\u00c5\u207b\u00b9)", "\u03c7(k)\u00b7k\u207f"),
            (self._ax_r,   "|\u03c7\u0303(R)|  \u2014  Fourier transform",
             "R  (\u00c5)",    "|\u03c7\u0303(R)|"),
        ]:
            _ax.clear()
            _ax.set_title(_title, fontsize=9, loc="left", pad=3)
            _ax.set_xlabel(_xl, fontsize=8)
            _ax.set_ylabel(_yl, fontsize=8)
            _ax.tick_params(labelsize=7)
            _ax.text(0.5, 0.45, "No data", transform=_ax.transAxes,
                     ha="center", va="center", fontsize=9, color="lightgray")
            if _HAS_SNS:
                sns.despine(ax=_ax, offset=4)
        self._canvas_exafs.draw_idle()

    # ── Scan list management ─────────────────────────────────────────────────

    def refresh_scan_list(self):
        """Re-populate the scan combobox and scan list panel."""
        scans = self._get_scans()
        for _label, _scan, *_ in scans:
            self._ensure_scan_backup(_scan)
        labels = [lbl for lbl, *_ in scans]
        self._scan_cb["values"] = labels
        if labels and self._scan_var.get() not in labels:
            self._scan_var.set(labels[0])
            self._auto_fill_e0()
        # Remove vis vars for scans no longer loaded
        for gone in [l for l in list(self._scan_vis_vars) if l not in labels]:
            del self._scan_vis_vars[gone]
            if gone in self._selected_labels:
                self._selected_labels.remove(gone)
        # Rebuild the visual list
        self._rebuild_scan_list_rows()
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

        # ── Edge-type detection ───────────────────────────────────────────────
        is_l = _is_l_edge_e0(e0)
        if is_l:
            self._edge_type_lbl.config(
                text="\u26a0 L-edge (soft X-ray)  —  using tight norm windows",
                fg="#994400")
            # Auto-apply L-edge defaults only if the current nor2 still looks
            # like a K-edge value (> 80 eV) — don't clobber user's manual edits.
            if self._nor2_var.get() > 80:
                ld = _NORM_FACTORY_L
                self._pre1_var.set(ld["pre1"])
                self._pre2_var.set(ld["pre2"])
                self._nor1_var.set(ld["nor1"])
                self._nor2_var.set(ld["nor2"])
                self._kmin_var.set(ld["kmin"])
                self._kmax_var.set(ld["kmax"])
                self._dk_var.set(ld["dk"])
                self._kw_var.set(ld["kw"])
        else:
            self._edge_type_lbl.config(
                text="K-edge (hard X-ray)", fg="#005500")

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
            "xanes_xmin": self._xanes_xmin_var.get(),
            "xanes_xmax": self._xanes_xmax_var.get(),
            "xanes_ymin": self._xanes_ymin_var.get(),
            "xanes_ymax": self._xanes_ymax_var.get(),
            "deglitch_sigma": self._deglitch_sigma_var.get(),
            "deglitch_window": self._deglitch_window_var.get(),
            "smooth_window": self._smooth_window_var.get(),
            "smooth_poly": self._smooth_poly_var.get(),
            "energy_shift": self._energy_shift_var.get(),
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
        if "xanes_xmin" in d:
            self._xanes_xmin_var.set(str(d["xanes_xmin"]))
        if "xanes_xmax" in d:
            self._xanes_xmax_var.set(str(d["xanes_xmax"]))
        if "xanes_ymin" in d:
            self._xanes_ymin_var.set(str(d["xanes_ymin"]))
        if "xanes_ymax" in d:
            self._xanes_ymax_var.set(str(d["xanes_ymax"]))
        _s(self._deglitch_sigma_var, "deglitch_sigma")
        _s(self._deglitch_window_var, "deglitch_window", int)
        _s(self._smooth_window_var, "smooth_window", int)
        _s(self._smooth_poly_var, "smooth_poly", int)
        _s(self._energy_shift_var, "energy_shift")

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
            self._ensure_scan_backup(scan)
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
        self._draw_empty_xanes()
        self._draw_empty_exafs()
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
        self._ensure_scan_backup(scan)

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
                    "mu_raw":    mu_raw,
                    "pre_line":  getattr(grp, "pre_edge", np.zeros_like(energy)),
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
            mu_norm, edge_step, pre_line = normalize_xanes(
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
                "mu_raw":    mu_raw,
                "pre_line":  pre_line,
            }

        # Ensure scan is marked visible and in the overlay
        if label not in self._scan_vis_vars:
            self._scan_vis_vars[label] = tk.BooleanVar(value=True)
        self._scan_vis_vars[label].set(True)
        if label not in self._selected_labels:
            self._selected_labels.append(label)

        res = self._results[label]

        # ── Push normalized result back to scan object ────────────────────────
        scan.mu            = res["mu_norm"].copy()
        scan.e0            = res["e0"]
        scan.is_normalized = True

        self._redraw()   # also calls replot_fn to refresh Spectra tab

        self._status_lbl.config(
            text=(f"[{engine}]  {label}  |  E\u2080={res['e0']:.1f} eV  |  "
                  f"edge step={res['edge_step']:.4f}  |  "
                  f"k: {kmin:.1f}\u2013{kmax:.1f} \u00c5\u207b\u00b9"),
            fg="#003366" if engine == "larch" else "#664400")

    def _redraw(self):
        self._update_show_section_visibility()
        self._redraw_xanes()
        self._redraw_exafs()
        if self._replot_fn is not None:
            self._replot_fn()

    def _redraw_xanes(self):
        if _HAS_SNS:
            sns.set_theme(style=self._style_var.get(),
                          context=self._context_var.get(), palette=_PALETTE)

        ax = self._ax_mu
        ax.clear()

        for i, label in enumerate(self._selected_labels):
            res = self._results.get(label)
            if res is None:
                continue
            col   = _PALETTE[i % len(_PALETTE)]
            lbl_s = label[:30] + ("\u2026" if len(label) > 30 else "")

            energy    = res["energy"]
            e0        = res["e0"]
            mu_norm   = res["mu_norm"]
            mu_raw    = res.get("mu_raw")
            pre_line  = res.get("pre_line")
            edge_step = res.get("edge_step", 1.0)
            bkg_e     = res.get("bkg_e")

            # ── Raw μ(E) ───────────────────────────────────────────────────
            if self._show_raw_var.get() and mu_raw is not None:
                ax.plot(energy, mu_raw, color=col, lw=1.2, alpha=0.50,
                        ls="--",
                        label=f"\u03bc(E) raw  {lbl_s}" if i == 0 else "_nolegend_")

            # ── Pre-edge fit ───────────────────────────────────────────────
            if self._show_preline_var.get() and pre_line is not None:
                ax.plot(energy, pre_line, color="#2C7BB6", lw=1.0,
                        ls=":", alpha=0.80, zorder=2,
                        label="Pre-edge fit" if i == 0 else "_nolegend_")

            # ── Post-edge reference (constant at edge_step above pre-edge) ─
            if self._show_postline_var.get() and pre_line is not None:
                ax.plot(energy, pre_line + edge_step, color="#1A9641", lw=1.0,
                        ls=":", alpha=0.80, zorder=2,
                        label="Post-edge ref" if i == 0 else "_nolegend_")

            # ── Normalized μ(E) ────────────────────────────────────────────
            if self._show_norm_var.get():
                ax.plot(energy, mu_norm, color=col, lw=1.8,
                        label=lbl_s, zorder=4)

            # ── Background ────────────────────────────────────────────────
            if self._show_bkg_var.get() and bkg_e is not None:
                _mask = energy >= e0
                ax.plot(energy[_mask], bkg_e[_mask], color=col, lw=1.0,
                        ls="--", alpha=0.60, zorder=3,
                        label="bkg" if i == 0 else "_nolegend_")

            # ── Derivative ────────────────────────────────────────────────
            if self._show_deriv_var.get():
                _deriv = np.gradient(mu_norm, energy)
                _amp   = (np.max(mu_norm) - np.min(mu_norm)) / (np.max(np.abs(_deriv)) + 1e-10)
                ax.plot(energy, _deriv * _amp, color=col, lw=1.0,
                        ls="-.", alpha=0.65,
                        label="d\u03bc/dE (scaled)" if i == 0 else "_nolegend_")

            # ── Normalization region markers (first scan only) ─────────────
            if i == 0:
                _p1 = e0 + self._pre1_var.get()
                _p2 = e0 + self._pre2_var.get()
                _n1 = e0 + self._nor1_var.get()
                _n2 = e0 + self._nor2_var.get()

                ax.axvline(e0, color="#4A4A8A", lw=1.2, ls="--", alpha=0.85,
                           label=f"E\u2080 = {e0:.1f} eV", zorder=5)

                ax.axvspan(_p1, _p2, alpha=0.08, color="#2C7BB6",
                           label="pre-edge region")
                ax.axvline(_p1, color="#2C7BB6", lw=0.7, ls=":", alpha=0.55)
                ax.axvline(_p2, color="#2C7BB6", lw=0.7, ls=":", alpha=0.55)

                ax.axvspan(_n1, _n2, alpha=0.08, color="#1A9641",
                           label="post-edge region")
                ax.axvline(_n1, color="#1A9641", lw=0.7, ls=":", alpha=0.55)
                ax.axvline(_n2, color="#1A9641", lw=0.7, ls=":", alpha=0.55)

                if self._show_norm_var.get():
                    ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.40)
                    ax.axhline(1, color="gray", lw=0.5, ls=":",  alpha=0.40)

        # ── Axis decoration ────────────────────────────────────────────────────
        ax.set_xlabel("Energy (eV)", fontsize=8)
        if self._show_norm_var.get() and not self._show_raw_var.get():
            ax.set_ylabel("\u03bc(E)  normalized", fontsize=8)
        elif self._show_raw_var.get() and not self._show_norm_var.get():
            ax.set_ylabel("\u03bc(E)  raw", fontsize=8)
        else:
            ax.set_ylabel("\u03bc(E)", fontsize=8)

        ax.set_title("\u03bc(E)  \u2014  XANES", fontsize=9, loc="left", pad=3)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        self._apply_xanes_view_limits()

        if self._selected_labels:
            ax.legend(fontsize=7, loc="lower right", framealpha=0.8)
        else:
            ax.text(0.5, 0.45, "Load a scan and click  \u25b6  Run Analysis",
                    transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="lightgray")

        if _HAS_SNS:
            sns.despine(ax=ax, offset=4)

        self._canvas_xanes.draw_idle()

    def _redraw_exafs(self):
        if _HAS_SNS:
            sns.set_theme(style=self._style_var.get(),
                          context=self._context_var.get(), palette=_PALETTE)

        ax_chi = self._ax_chi
        ax_r   = self._ax_r
        ax_chi.clear()
        ax_r.clear()

        # ── L-edge guard: EXAFS is physically meaningless for soft X-ray data ──
        # For a transition-metal L-edge (E0 < 2000 eV) the L₂ edge sits only
        # ~15–20 eV above L₃, giving k_max ≈ √(0.26 × 17) ≈ 2.1 Å⁻¹ — far
        # too short for a real EXAFS analysis.  Warn the user and skip plotting.
        if _is_l_edge_e0(self._e0_var.get()):
            e0_v  = self._e0_var.get()
            k_max_est = (0.26246840 * 17.0) ** 0.5   # ~17 eV usable above L₃
            msg = (f"EXAFS analysis is not applicable\n"
                   f"for L-edge / soft X-ray data.\n\n"
                   f"E\u2080 \u2248 {e0_v:.0f} eV — the L\u2082 edge is\n"
                   f"only ~17 eV above L\u2083, giving\n"
                   f"k\u2098\u2090\u2093 \u2248 {k_max_est:.1f} \u00c5\u207b\u00b9 "
                   f"(insufficient for EXAFS).\n\n"
                   f"Use the \u03bc(E) tab for XANES analysis.")
            for _ax in [ax_chi, ax_r]:
                _ax.set_facecolor("#fffff8")
                _ax.text(0.5, 0.5, msg,
                         transform=_ax.transAxes,
                         ha="center", va="center", fontsize=9,
                         color="#885500", alpha=0.85,
                         bbox=dict(boxstyle="round,pad=0.6",
                                   facecolor="#fff8e1", edgecolor="#ccaa00",
                                   alpha=0.9))
                _ax.tick_params(left=False, bottom=False,
                                labelleft=False, labelbottom=False)
            self._canvas_exafs.draw_idle()
            return

        _kw_label = {1: "k\u00b9", 2: "k\u00b2", 3: "k\u00b3"}.get(
            self._kw_var.get(), "k\u207f")

        for i, label in enumerate(self._selected_labels):
            res = self._results.get(label)
            if res is None:
                continue
            col   = _PALETTE[i % len(_PALETTE)]
            lbl_s = label[:30] + ("\u2026" if len(label) > 30 else "")

            # ── chi(k) ────────────────────────────────────────────────────
            if len(res["k"]) > 1:
                _kw_val = res["kw"]
                _chi_w  = res["k"] ** _kw_val * res["chi"]
                ax_chi.plot(res["k"], _chi_w, color=col, lw=1.4,
                            label=lbl_s, zorder=3)
                if i == 0 and self._show_win_var.get():
                    _kmin = self._kmin_var.get()
                    _kmax = self._kmax_var.get()
                    _dk   = self._dk_var.get()
                    _k_u  = np.linspace(res["k"][0], res["k"][-1], 400)
                    _win  = np.zeros_like(_k_u)
                    _flat = (_k_u >= _kmin + _dk) & (_k_u <= _kmax - _dk)
                    _win[_flat] = 1.0
                    _t_in  = (_k_u >= _kmin) & (_k_u < _kmin + _dk)
                    if _t_in.any() and _dk > 0:
                        _win[_t_in] = 0.5 * (1 - np.cos(
                            np.pi * (_k_u[_t_in] - _kmin) / _dk))
                    _t_out = (_k_u > _kmax - _dk) & (_k_u <= _kmax)
                    if _t_out.any() and _dk > 0:
                        _win[_t_out] = 0.5 * (1 + np.cos(
                            np.pi * (_k_u[_t_out] - (_kmax - _dk)) / _dk))
                    _amp = np.abs(_chi_w).max() if len(_chi_w) > 0 else 1.0
                    ax_chi.fill_between(_k_u, -_win * _amp, _win * _amp,
                                        alpha=0.08, color="orange",
                                        label="FT window")

            # ── |chi(R)| ──────────────────────────────────────────────────
            if len(res["r"]) > 1:
                _rmax   = self._rmax_var.get()
                _r_mask = res["r"] <= _rmax
                ax_r.plot(res["r"][_r_mask], res["chi_r"][_r_mask],
                          color=col, lw=1.6, label=lbl_s, zorder=3)
                ax_r.fill_between(res["r"][_r_mask], 0,
                                  res["chi_r"][_r_mask],
                                  alpha=0.12, color=col)

        ax_chi.set_xlabel("k  (\u00c5\u207b\u00b9)", fontsize=8)
        ax_chi.set_ylabel(f"\u03c7(k)\u00b7{_kw_label}  (\u00c5\u207b\u207f)", fontsize=8)
        ax_chi.set_title("\u03c7(k)  \u2014  EXAFS oscillations", fontsize=9,
                         loc="left", pad=3)
        ax_chi.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.40)
        if self._selected_labels:
            ax_chi.legend(fontsize=7, loc="upper right", framealpha=0.8)
        else:
            ax_chi.text(0.5, 0.45, "No data", transform=ax_chi.transAxes,
                        ha="center", va="center", fontsize=9, color="lightgray")

        ax_r.set_xlabel("R  (\u00c5)", fontsize=8)
        ax_r.set_ylabel("|\u03c7\u0303(R)|", fontsize=8)
        ax_r.set_title("|\u03c7\u0303(R)|  \u2014  Fourier transform", fontsize=9,
                       loc="left", pad=3)
        ax_r.set_xlim(0, self._rmax_var.get())
        if self._selected_labels:
            ax_r.legend(fontsize=7, loc="upper right", framealpha=0.8)
        else:
            ax_r.text(0.5, 0.45, "No data", transform=ax_r.transAxes,
                      ha="center", va="center", fontsize=9, color="lightgray")

        for _ax in [ax_chi, ax_r]:
            _ax.tick_params(labelsize=7)
            _ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
            if _HAS_SNS:
                sns.despine(ax=_ax, offset=4)

        self._canvas_exafs.draw_idle()
