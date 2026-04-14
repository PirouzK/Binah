"""
Interactive TDDFT / XAS plot widget using matplotlib embedded in tkinter.
Supports single-spectrum view, multi-spectrum TDDFT overlay, and experimental
XAS scan overlay on a twin axis with ΔE energy-shift alignment.
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Optional, Tuple

from orca_parser import TDDFTSpectrum
from experimental_parser import ExperimentalScan


# ── Broadening functions ──────────────────────────────────────────────────────

def gaussian(x, center, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def lorentzian(x, center, fwhm):
    gamma = fwhm / 2.0
    return gamma**2 / ((x - center)**2 + gamma**2)


# ── Colour cycles ─────────────────────────────────────────────────────────────
OVERLAY_COLOURS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]

# Warm red/brown palette for experimental scans — distinct from TDDFT blues
EXP_COLOURS = [
    "#8B0000", "#B22222", "#CC4400", "#884400", "#6A1B9A", "#00695C",
]

# Matplotlib linestyle names → display names and mpl values
LS_OPTIONS = [
    ("Solid",    "solid"),
    ("Dashed",   "dashed"),
    ("Dotted",   "dotted"),
    ("Dash-dot", "dashdot"),
]


# ── Lightweight tooltip for panel widgets ────────────────────────────────────

class _ToolTip:
    """Shows a small tooltip label after a short hover delay."""
    def __init__(self, widget, text: str, delay: int = 600):
        self._widget = widget
        self._text   = text
        self._delay  = delay
        self._id     = None
        self._win    = None
        widget.bind("<Enter>",  self._schedule, add="+")
        widget.bind("<Leave>",  self._cancel,   add="+")
        widget.bind("<Button>", self._cancel,   add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._id = self._widget.after(self._delay, self._show)

    def _cancel(self, _event=None):
        if self._id:
            self._widget.after_cancel(self._id)
            self._id = None
        if self._win:
            self._win.destroy()
            self._win = None

    def _show(self):
        x = self._widget.winfo_rootx() + self._widget.winfo_width() + 4
        y = self._widget.winfo_rooty()
        self._win = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self._text, background="#ffffe0",
                 relief=tk.SOLID, borderwidth=1,
                 font=("", 8)).pack()


# ── Mutable default style stores ─────────────────────────────────────────────
# "Set as Default" writes to disk so preferences survive program restarts.
import json as _json
import os as _os

_STYLE_CONFIG_PATH = _os.path.join(
    _os.path.expanduser("~"), ".binah_config.json")

_EXP_STYLE_FACTORY: dict = {
    "linestyle":  "solid",
    "linewidth":  1.8,
    "fill":       True,
    "fill_alpha": 0.06,
    "color":      "",
}
_TDDFT_STYLE_FACTORY: dict = {
    "env_linewidth":   2.0,
    "env_fill":        True,
    "env_fill_alpha":  0.10,
    "stick_linewidth": 1.2,
    "stick_alpha":     0.75,
    "stick_markersize": 4,
    "stick_markers":   True,
}


def _load_style_config() -> tuple:
    """Load exp + tddft style defaults from the shared config file."""
    exp   = dict(_EXP_STYLE_FACTORY)
    tddft = dict(_TDDFT_STYLE_FACTORY)
    try:
        if _os.path.exists(_STYLE_CONFIG_PATH):
            with open(_STYLE_CONFIG_PATH, "r", encoding="utf-8") as _f:
                _cfg = _json.load(_f)
            for k, v in _cfg.get("exp_style_defaults", {}).items():
                if k in exp:
                    exp[k] = v
            for k, v in _cfg.get("tddft_style_defaults", {}).items():
                if k in tddft:
                    tddft[k] = v
    except Exception:
        pass
    return exp, tddft


def _save_style_config() -> None:
    """Persist current style defaults to the shared config file."""
    try:
        cfg = {}
        if _os.path.exists(_STYLE_CONFIG_PATH):
            with open(_STYLE_CONFIG_PATH, "r", encoding="utf-8") as _f:
                cfg = _json.load(_f)
        cfg["exp_style_defaults"]   = dict(_EXP_STYLE_DEFAULTS)
        cfg["tddft_style_defaults"] = dict(_TDDFT_STYLE_DEFAULTS)
        with open(_STYLE_CONFIG_PATH, "w", encoding="utf-8") as _f:
            _json.dump(cfg, _f, indent=2)
    except Exception:
        pass


# Initialise from disk (falls back to factory values if config absent)
_EXP_STYLE_DEFAULTS, _TDDFT_STYLE_DEFAULTS = _load_style_config()


def _default_exp_style() -> dict:
    """Return a fresh copy of the current experimental-scan defaults."""
    return dict(_EXP_STYLE_DEFAULTS)


def _default_tddft_style() -> dict:
    """Return a fresh copy of the current TDDFT defaults."""
    return dict(_TDDFT_STYLE_DEFAULTS)


# ═════════════════════════════════════════════════════════════════════════════
class PlotWidget(tk.Frame):
    """
    Embeds a matplotlib figure with a controls strip.
    Supports:
      - Single spectrum view + multi-spectrum TDDFT overlay (right y-axis)
      - Experimental XAS scans on left y-axis
      - ΔE shift: nudge all TDDFT positions to align with experimental
      - Per-overlay enable/disable, style control and label editing
      - Global TDDFT style (envelope/stick) and per-scan experimental style
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # Current single TDDFT spectrum
        self.spectrum: Optional[TDDFTSpectrum] = None

        # TDDFT overlay list: [(label, TDDFTSpectrum, enabled_BooleanVar, color_or_None), ...]
        # color_or_None is "" / None  → use default OVERLAY_COLOURS palette
        self._overlay_spectra: List[Tuple[str, TDDFTSpectrum, tk.BooleanVar, str]] = []
        self._overlay_mode = tk.BooleanVar(value=False)

        # Experimental scan list: [(label, ExperimentalScan, enabled_BooleanVar, style_dict), ...]
        self._exp_scans: List[Tuple[str, ExperimentalScan, tk.BooleanVar, dict]] = []
        self._exp_link_counter = 1

        # Global style dictionaries
        self._tddft_style = _default_tddft_style()

        # Controls state
        self._x_unit      = tk.StringVar(value="eV")
        self._broadening  = tk.StringVar(value="Gaussian")
        self._fwhm        = tk.DoubleVar(value=1.0)
        self._fwhm_str    = tk.StringVar(value="1.00")
        self._show_sticks = tk.BooleanVar(value=True)
        self._show_env    = tk.BooleanVar(value=True)
        self._show_trans  = tk.BooleanVar(value=False)
        self._normalise   = tk.BooleanVar(value=False)
        self._show_legend           = tk.BooleanVar(value=True)
        self._show_tddft_in_legend  = tk.BooleanVar(value=True)   # TDDFT handles in legend
        self._show_primary_in_legend = tk.BooleanVar(value=True)  # primary TDDFT in legend
        self._custom_title = tk.StringVar(value="")

        # ΔE energy shift (eV) applied to all TDDFT stick positions.
        # _delta_e holds the true (unbounded) value.
        # _de_slider_var is the slider display var, clamped to ±20 for the widget.
        self._delta_e       = tk.DoubleVar(value=0.0)
        self._delta_e_str   = tk.StringVar(value="+0.00")
        self._de_slider_var = tk.DoubleVar(value=0.0)

        # Intensity scale factor multiplied onto all TDDFT y-values.
        # Useful for visually matching TDDFT envelope height to experimental μ(E).
        # Entry accepts any positive float; slider covers 0.01–5× for fine-tuning.
        self._tddft_scale       = tk.DoubleVar(value=1.0)
        self._tddft_scale_str   = tk.StringVar(value="1.000")
        self._scale_slider_var  = tk.DoubleVar(value=1.0)

        # Plot appearance
        self._bg_colour  = "#ffffff"   # axes + figure background colour
        self._show_grid  = tk.BooleanVar(value=False)  # grid off by default

        # Combined-spectrum component toggles (shown/hidden by _update_comb_ui)
        self._comb_total = tk.BooleanVar(value=True)
        self._comb_d2    = tk.BooleanVar(value=False)
        self._comb_m2    = tk.BooleanVar(value=False)
        self._comb_q2    = tk.BooleanVar(value=False)
        self._comb_frame = None   # tk.Frame reference — built in _build_controls

        # Pop-out window refresh callbacks — each entry is a callable()
        # registered by _pop_out_graph and removed when the window closes.
        self._popout_callbacks: list = []

        # Legend drag persistence: (x, y) in figure-fraction coords, or None = "best"
        self._legend_bbox      = None   # (x, y) figure-fraction of upper-left corner
        self._legend_ref       = None
        self._legend_drag_cid  = None   # press
        self._legend_drag_cid2 = None   # motion
        self._legend_drag_cid3 = None   # release
        self._legend_dragging  = False  # True only while mouse is held & moved

        # Master visibility toggles
        self._show_tddft = tk.BooleanVar(value=True)

        # Manual axis-range overrides (empty string = auto-scale)
        self._xlim_lo = tk.StringVar(value="")
        self._xlim_hi = tk.StringVar(value="")
        self._ylim_lo = tk.StringVar(value="")   # right / TDDFT axis
        self._ylim_hi = tk.StringVar(value="")
        self._yleft_lo = tk.StringVar(value="")  # left / experimental axis
        self._yleft_hi = tk.StringVar(value="")

        # Font controls — sizes and bold toggles for each text element
        self._font_title_size   = tk.IntVar(value=11)
        self._font_title_bold   = tk.BooleanVar(value=True)
        self._font_xlabel_size  = tk.IntVar(value=11)
        self._font_xlabel_bold  = tk.BooleanVar(value=False)
        self._font_ylabel_size  = tk.IntVar(value=11)
        self._font_ylabel_bold  = tk.BooleanVar(value=False)
        self._font_tick_size    = tk.IntVar(value=9)
        self._font_legend_size  = tk.IntVar(value=9)

        # Y-axis label visibility + custom text overrides
        # Empty string = use the auto-computed label; non-empty = use the override.
        self._show_left_ylabel   = tk.BooleanVar(value=True)
        self._show_right_ylabel  = tk.BooleanVar(value=True)
        self._custom_left_ylabel  = tk.StringVar(value="")
        self._custom_right_ylabel = tk.StringVar(value="")

        # Inset (zoomed sub-plot) state
        self._inset_active      = False
        self._inset_xlim        = [None, None]   # data-unit (current x unit)
        self._inset_ylim        = [None, None]   # left-axis units
        self._inset_pos         = [0.54, 0.52, 0.40, 0.34]  # axes-fraction [l,b,w,h]
        self._inset_ax          = None
        self._inset_ax2         = None
        self._inset_drag        = None           # {x0, y0, pos0} while dragging
        self._inset_drag_cids: list = []
        self._inset_plot_data: list = []         # collected each _replot for re-draw
        self._inset_show_labels = tk.BooleanVar(value=True)  # tick labels on/off
        self._inset_indicator   = None   # (rect, connectors) from indicate_inset_zoom

        # Hover state
        self._annot         = None
        self._hover_cid     = None
        self._hover_x       = np.array([])
        self._hover_y       = np.array([])
        self._hover_ev      = np.array([])
        self._hover_cm      = np.array([])
        self._hover_states: list = []    # actual root numbers (sp.states)
        self._hover_labels: list = []    # transition labels (sp.transition_labels)

        self._build_controls()
        self._build_alignment_controls()
        self._build_view_controls()
        self._build_title_bar()
        self._build_figure()        # creates self.ax and self.ax2 once
        self._build_overlay_panel()

    # ══════════════════════════════════════════════════════════════════════════
    #  Scrollable control-bar factory
    # ══════════════════════════════════════════════════════════════════════════
    def _scrollable_bar(self, **kw) -> tk.Frame:
        """Return a Frame packed inside a horizontally-scrollable Canvas.

        Pass the same keyword arguments you would give to tk.Frame (bd, relief,
        padx, pady, bg …).  A thin horizontal scrollbar appears automatically
        whenever the bar content is wider than the window — so controls are
        never cut off on small screens — and hides itself on large screens.
        """
        outer = tk.Frame(self)
        outer.pack(side=tk.TOP, fill=tk.X)

        bg = kw.get("bg", None)
        cnv = tk.Canvas(outer, highlightthickness=0,
                        **({"bg": bg} if bg else {}))
        hbar = ttk.Scrollbar(outer, orient="horizontal", command=cnv.xview)
        cnv.configure(xscrollcommand=hbar.set)
        cnv.pack(side=tk.TOP, fill=tk.X, expand=True)

        inner = tk.Frame(cnv, **kw)
        cnv.create_window((0, 0), window=inner, anchor="nw")

        def _update(_event=None):
            cnv.configure(scrollregion=cnv.bbox("all"))
            req_h = inner.winfo_reqheight()
            if req_h > 1:
                cnv.configure(height=req_h)
            try:
                needed = inner.winfo_reqwidth() > outer.winfo_width() > 1
            except Exception:
                needed = False
            if needed:
                hbar.pack(side=tk.BOTTOM, fill=tk.X, before=cnv)
            else:
                hbar.pack_forget()

        inner.bind("<Configure>", lambda e: outer.after_idle(_update))
        outer.bind("<Configure>", lambda e: outer.after_idle(_update))
        return inner

    # ══════════════════════════════════════════════════════════════════════════
    #  Controls strip (row 1)
    # ══════════════════════════════════════════════════════════════════════════
    def _build_controls(self):
        ctrl = self._scrollable_bar(bd=1, relief=tk.SUNKEN, padx=4, pady=3)
        self._ctrl_frame = ctrl   # saved so _update_comb_ui can pack into it

        # X-axis unit
        tk.Label(ctrl, text="X axis:").pack(side=tk.LEFT)
        for unit in ("nm", "eV", "cm\u207b\u00b9"):
            tk.Radiobutton(ctrl, text=unit, variable=self._x_unit,
                           value=unit, command=self._on_unit_change).pack(side=tk.LEFT, padx=1)

        _sep(ctrl)

        # Broadening
        tk.Label(ctrl, text="Broadening:").pack(side=tk.LEFT)
        for b in ("Gaussian", "Lorentzian"):
            tk.Radiobutton(ctrl, text=b, variable=self._broadening,
                           value=b, command=self._replot).pack(side=tk.LEFT, padx=1)

        _sep(ctrl)

        # FWHM — entry + slider
        tk.Label(ctrl, text="FWHM:").pack(side=tk.LEFT)
        self._fwhm_entry = tk.Entry(ctrl, textvariable=self._fwhm_str, width=7,
                                    font=("Courier", 9))
        self._fwhm_entry.pack(side=tk.LEFT, padx=(2, 0))
        self._fwhm_entry.bind("<Return>",   self._on_fwhm_entry_commit)
        self._fwhm_entry.bind("<FocusOut>", self._on_fwhm_entry_commit)

        self._fwhm_unit_label = tk.Label(ctrl, text="eV", width=5, anchor="w")
        self._fwhm_unit_label.pack(side=tk.LEFT)

        self._fwhm_slider = tk.Scale(
            ctrl, from_=0.05, to=20.0, resolution=0.05,
            orient=tk.HORIZONTAL, length=140,
            variable=self._fwhm, showvalue=False,
            command=self._on_fwhm_slider_change
        )
        self._fwhm_slider.pack(side=tk.LEFT, padx=2)

        _sep(ctrl)

        # Toggles
        tk.Checkbutton(ctrl, text="Sticks",      variable=self._show_sticks, command=self._replot).pack(side=tk.LEFT)
        tk.Checkbutton(ctrl, text="Envelope",    variable=self._show_env,    command=self._replot).pack(side=tk.LEFT)
        tk.Checkbutton(ctrl, text="Transitions", variable=self._show_trans,  command=self._replot).pack(side=tk.LEFT)
        tk.Checkbutton(ctrl, text="Normalise",   variable=self._normalise,   command=self._replot).pack(side=tk.LEFT)

        _sep(ctrl)

        # ── Export / pop-out — placed early so they're always visible ────────
        tk.Button(ctrl, text="Save Fig",   command=self._save_figure,
                  font=("", 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="Export CSV", command=self._export_csv,
                  font=("", 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="\u29c9 Pop Out", command=self._pop_out_graph,
                  font=("", 8, "bold"), fg="#003399",
                  relief=tk.RAISED).pack(side=tk.LEFT, padx=(2, 4))

        _sep(ctrl)

        # Legend + label editor
        tk.Checkbutton(ctrl, text="Legend", variable=self._show_legend,
                       command=self._toggle_legend).pack(side=tk.LEFT)
        # Sub-toggle: include/exclude TDDFT entries from the legend
        tk.Checkbutton(ctrl, text="TDDFT", variable=self._show_tddft_in_legend,
                       command=self._replot,
                       font=("", 8), fg="navy").pack(side=tk.LEFT, padx=(0, 2))
        tk.Button(ctrl, text="Edit Labels\u2026", command=self._open_legend_editor,
                  font=("", 8)).pack(side=tk.LEFT, padx=(2, 0))

        _sep(ctrl)

        # Grid toggle + background colour picker
        tk.Checkbutton(ctrl, text="Grid", variable=self._show_grid,
                       command=self._replot).pack(side=tk.LEFT)
        self._bg_btn = tk.Button(
            ctrl, text="Plot BG", font=("", 8),
            bg=self._bg_colour, relief=tk.RAISED,
            command=self._choose_bg_colour
        )
        self._bg_btn.pack(side=tk.LEFT, padx=(2, 0))

        _sep(ctrl)

        # Style dialogs
        tk.Button(ctrl, text="TDDFT Style\u2026", command=self._open_tddft_style_dialog,
                  font=("", 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="Fonts\u2026", command=self._open_font_dialog,
                  font=("", 8)).pack(side=tk.LEFT, padx=2)

        _sep(ctrl)

        # Overlay toggle
        tk.Checkbutton(ctrl, text="Overlay Mode", variable=self._overlay_mode,
                       command=self._on_overlay_toggle, fg="darkblue",
                       font=("", 9, "bold")).pack(side=tk.LEFT, padx=2)

        # ── Combined-spectrum component selector (hidden until a combined spectrum loads)
        _sep(ctrl)
        self._comb_frame = tk.Frame(ctrl)
        # Packed/unpacked dynamically by _update_comb_ui — do not pack here
        tk.Label(self._comb_frame, text="Components:",
                 font=("", 9, "bold"), fg="#005580").pack(side=tk.LEFT, padx=(0, 2))
        tk.Checkbutton(self._comb_frame, text="Total",
                       variable=self._comb_total, command=self._replot,
                       fg="#333333").pack(side=tk.LEFT)
        tk.Checkbutton(self._comb_frame, text="Elec. Dipole (D\u00b2)",
                       variable=self._comb_d2, command=self._replot,
                       fg="#1f77b4").pack(side=tk.LEFT)
        tk.Checkbutton(self._comb_frame, text="Mag. Dipole (m\u00b2)",
                       variable=self._comb_m2, command=self._replot,
                       fg="#2ca02c").pack(side=tk.LEFT)
        tk.Checkbutton(self._comb_frame, text="Elec. Quad. (Q\u00b2)",
                       variable=self._comb_q2, command=self._replot,
                       fg="#d62728").pack(side=tk.LEFT)

    # ══════════════════════════════════════════════════════════════════════════
    #  Alignment controls (row 2) — ΔE shift, entry unbounded, slider ±20
    # ══════════════════════════════════════════════════════════════════════════
    def _build_alignment_controls(self):
        bar = self._scrollable_bar(bd=1, relief=tk.FLAT, padx=4, pady=2, bg="#f0f0e8")

        tk.Label(bar, text="TDDFT \u0394E shift:", font=("", 9),
                 bg="#f0f0e8").pack(side=tk.LEFT)

        self._de_entry = tk.Entry(bar, textvariable=self._delta_e_str, width=9,
                                  font=("Courier", 9))
        self._de_entry.pack(side=tk.LEFT, padx=(2, 0))
        self._de_entry.bind("<Return>",   self._on_de_entry_commit)
        self._de_entry.bind("<FocusOut>", self._on_de_entry_commit)

        tk.Label(bar, text="eV", font=("", 9), bg="#f0f0e8").pack(side=tk.LEFT, padx=(2, 4))

        self._de_slider = tk.Scale(
            bar, from_=-20.0, to=20.0, resolution=0.05,
            orient=tk.HORIZONTAL, length=200,
            variable=self._de_slider_var, showvalue=False,
            bg="#f0f0e8",
            command=self._on_de_slider_change
        )
        self._de_slider.pack(side=tk.LEFT, padx=2)

        tk.Button(bar, text="Reset \u0394E", font=("", 8), bg="#f0f0e8",
                  command=self._reset_delta_e).pack(side=tk.LEFT, padx=(4, 12))

        # ── TDDFT intensity scale ─────────────────────────────────────────────
        tk.Label(bar, text="Intensity scale:", font=("", 9),
                 bg="#f0f0e8").pack(side=tk.LEFT)

        self._scale_entry = tk.Entry(bar, textvariable=self._tddft_scale_str, width=8,
                                     font=("Courier", 9))
        self._scale_entry.pack(side=tk.LEFT, padx=(2, 0))
        self._scale_entry.bind("<Return>",   self._on_scale_entry_commit)
        self._scale_entry.bind("<FocusOut>", self._on_scale_entry_commit)

        tk.Label(bar, text="\u00d7", font=("", 10), bg="#f0f0e8").pack(
            side=tk.LEFT, padx=(2, 4))

        self._scale_slider = tk.Scale(
            bar, from_=0.01, to=5.0, resolution=0.01,
            orient=tk.HORIZONTAL, length=160,
            variable=self._scale_slider_var, showvalue=False,
            bg="#f0f0e8",
            command=self._on_scale_slider_change
        )
        self._scale_slider.pack(side=tk.LEFT, padx=2)

        tk.Button(bar, text="Reset \u00d7", font=("", 8), bg="#f0f0e8",
                  command=self._reset_scale).pack(side=tk.LEFT, padx=4)

        tk.Label(bar, text="(slider 0.01\u20135\u00d7  |  type any value for larger scale)",
                 font=("", 8), fg="gray", bg="#f0f0e8").pack(side=tk.LEFT, padx=4)

    # ══════════════════════════════════════════════════════════════════════════
    #  View-controls bar (row 3) — visibility, axis ranges, inset
    # ══════════════════════════════════════════════════════════════════════════
    def _build_view_controls(self):
        bar = self._scrollable_bar(bd=1, relief=tk.FLAT, padx=4, pady=2, bg="#e8f0e8")

        # ── Master visibility ──────────────────────────────────────────────
        tk.Checkbutton(bar, text="Show TDDFT", variable=self._show_tddft,
                       command=self._replot, bg="#e8f0e8",
                       font=("", 9, "bold"), fg="navy").pack(side=tk.LEFT, padx=(0, 4))

        def _sep2(): tk.Frame(bar, width=1, bg="#aabbaa").pack(
            side=tk.LEFT, fill=tk.Y, padx=5, pady=2)

        _sep2()

        # ── X axis range ──────────────────────────────────────────────────
        tk.Label(bar, text="X:", font=("", 9), bg="#e8f0e8").pack(side=tk.LEFT)
        ex_lo = tk.Entry(bar, textvariable=self._xlim_lo, width=8, font=("Courier", 9))
        ex_lo.pack(side=tk.LEFT, padx=(2, 0))
        ex_lo.bind("<Return>",   lambda e: self._replot())
        ex_lo.bind("<FocusOut>", lambda e: self._replot())
        tk.Label(bar, text="→", bg="#e8f0e8").pack(side=tk.LEFT, padx=1)
        ex_hi = tk.Entry(bar, textvariable=self._xlim_hi, width=8, font=("Courier", 9))
        ex_hi.pack(side=tk.LEFT)
        ex_hi.bind("<Return>",   lambda e: self._replot())
        ex_hi.bind("<FocusOut>", lambda e: self._replot())
        tk.Button(bar, text="Auto X", font=("", 8), bg="#e8f0e8",
                  command=self._auto_x).pack(side=tk.LEFT, padx=(2, 0))

        _sep2()

        # ── Y axis range (right / TDDFT) ──────────────────────────────────
        tk.Label(bar, text="Y right (TDDFT):", font=("", 9), bg="#e8f0e8").pack(side=tk.LEFT)
        ey_lo = tk.Entry(bar, textvariable=self._ylim_lo, width=7, font=("Courier", 9))
        ey_lo.pack(side=tk.LEFT, padx=(2, 0))
        ey_lo.bind("<Return>",   lambda e: self._replot())
        ey_lo.bind("<FocusOut>", lambda e: self._replot())
        tk.Label(bar, text="→", bg="#e8f0e8").pack(side=tk.LEFT, padx=1)
        ey_hi = tk.Entry(bar, textvariable=self._ylim_hi, width=7, font=("Courier", 9))
        ey_hi.pack(side=tk.LEFT)
        ey_hi.bind("<Return>",   lambda e: self._replot())
        ey_hi.bind("<FocusOut>", lambda e: self._replot())
        tk.Button(bar, text="Auto Y right", font=("", 8), bg="#e8f0e8",
                  command=self._auto_y).pack(side=tk.LEFT, padx=(2, 0))

        _sep2()

        # ── Inset ─────────────────────────────────────────────────────────
        tk.Label(bar, text="Y left (Exp):", font=("", 9), bg="#e8f0e8").pack(side=tk.LEFT)
        ey2_lo = tk.Entry(bar, textvariable=self._yleft_lo, width=7, font=("Courier", 9))
        ey2_lo.pack(side=tk.LEFT, padx=(2, 0))
        ey2_lo.bind("<Return>",   lambda e: self._replot())
        ey2_lo.bind("<FocusOut>", lambda e: self._replot())
        tk.Label(bar, text="→", bg="#e8f0e8").pack(side=tk.LEFT, padx=1)
        ey2_hi = tk.Entry(bar, textvariable=self._yleft_hi, width=7, font=("Courier", 9))
        ey2_hi.pack(side=tk.LEFT)
        ey2_hi.bind("<Return>",   lambda e: self._replot())
        ey2_hi.bind("<FocusOut>", lambda e: self._replot())
        tk.Button(bar, text="Auto Y left", font=("", 8), bg="#e8f0e8",
                  command=self._auto_y_left).pack(side=tk.LEFT, padx=(2, 0))

        _sep2()

        self._inset_btn = tk.Button(
            bar, text="+ Add Inset\u2026", font=("", 8, "bold"), bg="#e8f0e8",
            fg="darkgreen", command=self._open_inset_dialog)
        self._inset_btn.pack(side=tk.LEFT, padx=2)
        tk.Checkbutton(bar, text="Axis labels", variable=self._inset_show_labels,
                       command=self._on_inset_labels_toggle,
                       bg="#e8f0e8", font=("", 8)).pack(side=tk.LEFT, padx=(2, 0))

        _sep2()

        # ── Y-axis label visibility + custom text ─────────────────────────
        tk.Label(bar, text="Y labels:", font=("", 8, "bold"),
                 bg="#e8f0e8").pack(side=tk.LEFT, padx=(0, 2))

        # Left Y = experimental
        tk.Checkbutton(bar, text="Left (Exp):", variable=self._show_left_ylabel,
                       command=self._replot, bg="#e8f0e8",
                       font=("", 8)).pack(side=tk.LEFT)
        self._left_ylabel_entry = tk.Entry(
            bar, textvariable=self._custom_left_ylabel,
            width=18, font=("", 8), relief=tk.SUNKEN)
        self._left_ylabel_entry.pack(side=tk.LEFT, padx=(1, 6))
        self._left_ylabel_entry.bind("<Return>",   lambda _: self._replot())
        self._left_ylabel_entry.bind("<FocusOut>", lambda _: self._replot())
        _ToolTip(self._left_ylabel_entry,
                 "Custom left Y-axis label for experimental data.\nLeave blank to use the auto label.")

        # Right Y = TDDFT
        tk.Checkbutton(bar, text="Right (TDDFT):", variable=self._show_right_ylabel,
                       command=self._replot, bg="#e8f0e8",
                       font=("", 8)).pack(side=tk.LEFT)
        self._right_ylabel_entry = tk.Entry(
            bar, textvariable=self._custom_right_ylabel,
            width=18, font=("", 8), relief=tk.SUNKEN)
        self._right_ylabel_entry.pack(side=tk.LEFT, padx=(1, 4))
        self._right_ylabel_entry.bind("<Return>",   lambda _: self._replot())
        self._right_ylabel_entry.bind("<FocusOut>", lambda _: self._replot())
        _ToolTip(self._right_ylabel_entry,
                 "Custom right Y-axis label for TDDFT data.\nLeave blank to use the auto label.")

    def _auto_x(self):
        self._xlim_lo.set("")
        self._xlim_hi.set("")
        self._replot()

    def _auto_y(self):
        self._ylim_lo.set("")
        self._ylim_hi.set("")
        self._replot()

    def _auto_y_left(self):
        self._yleft_lo.set("")
        self._yleft_hi.set("")
        self._replot()

    def _on_inset_labels_toggle(self):
        if self._inset_active and self._inset_ax is not None:
            self._quick_move_inset()
        # No-op if inset not yet shown

    # ── ΔE handlers ──────────────────────────────────────────────────────────

    def _on_de_slider_change(self, val):
        """Slider moved → update actual delta_e and entry display."""
        v = float(val)
        self._delta_e.set(v)
        self._delta_e_str.set(f"{v:+.2f}")
        self._replot()

    def _on_de_entry_commit(self, event=None):
        """Entry committed → update delta_e (unbounded) and nudge slider."""
        raw = self._delta_e_str.get().strip()
        try:
            val = float(raw)
        except ValueError:
            self._delta_e_str.set(f"{self._delta_e.get():+.2f}")
            return
        # Store actual (unbounded) value
        self._delta_e.set(val)
        self._delta_e_str.set(f"{val:+.2f}")
        # Move slider to clamped representation for visual feedback
        self._de_slider_var.set(max(-20.0, min(20.0, val)))
        self._replot()

    def _reset_delta_e(self):
        self._delta_e.set(0.0)
        self._delta_e_str.set("+0.00")
        self._de_slider_var.set(0.0)
        self._replot()

    # ── Intensity scale handlers ──────────────────────────────────────────────

    def _on_scale_slider_change(self, val):
        v = float(val)
        self._tddft_scale.set(v)
        self._tddft_scale_str.set(f"{v:.3f}")
        self._replot()

    def _on_scale_entry_commit(self, event=None):
        raw = self._tddft_scale_str.get().strip()
        try:
            val = float(raw)
            if val <= 0:
                raise ValueError
        except ValueError:
            self._tddft_scale_str.set(f"{self._tddft_scale.get():.3f}")
            return
        self._tddft_scale.set(val)
        self._tddft_scale_str.set(f"{val:.3f}")
        # Move slider to clamped position for visual feedback
        self._scale_slider_var.set(max(0.01, min(5.0, val)))
        self._replot()

    def _reset_scale(self):
        self._tddft_scale.set(1.0)
        self._tddft_scale_str.set("1.000")
        self._scale_slider_var.set(1.0)
        self._replot()

    # ── Background colour ─────────────────────────────────────────────────────

    def _choose_bg_colour(self):
        from tkinter import colorchooser
        result = colorchooser.askcolor(color=self._bg_colour,
                                       title="Choose plot background colour")
        if result and result[1]:
            self._bg_colour = result[1]
            # Keep button face in sync so it acts as a swatch
            try:
                self._bg_btn.config(bg=self._bg_colour)
            except Exception:
                pass
            self._replot()

    # ══════════════════════════════════════════════════════════════════════════
    #  Title bar
    # ══════════════════════════════════════════════════════════════════════════
    def _build_title_bar(self):
        bar = self._scrollable_bar(padx=4, pady=2)
        tk.Label(bar, text="Plot title:", font=("", 9)).pack(side=tk.LEFT)
        entry = tk.Entry(bar, textvariable=self._custom_title, font=("", 10),
                         relief=tk.FLAT, bg="#f0f0f0", fg="#111111")
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))
        entry.bind("<Return>",   lambda _: self._replot())
        entry.bind("<FocusOut>", lambda _: self._replot())
        tk.Button(bar, text="Reset", font=("", 8),
                  command=self._reset_title).pack(side=tk.LEFT)

    def _reset_title(self):
        self._custom_title.set("")
        self._replot()

    def _auto_title(self) -> str:
        if self._overlay_mode.get():
            n_tddft  = (1 if self.spectrum else 0)
            n_tddft += sum(1 for _, _, v, _c in self._overlay_spectra if v.get())
            n_exp    = sum(1 for _, _, v, _ in self._exp_scans if v.get())
            parts = []
            if n_tddft > 1:
                parts.append(f"{n_tddft} TDDFT")
            elif n_tddft == 1 and self.spectrum:
                parts.append(getattr(self.spectrum, "_custom_label", None) or
                              self.spectrum.display_name())
            if n_exp:
                parts.append(f"{n_exp} Exp.")
            return ("Overlay: " + ", ".join(parts)) if parts else ""
        if self.spectrum:
            n_exp = sum(1 for _, _, v, _ in self._exp_scans if v.get())
            base  = getattr(self.spectrum, "_custom_label", None) or self.spectrum.display_name()
            return (f"{base} + {n_exp} Exp. scan{'s' if n_exp > 1 else ''}"
                    if n_exp else base)
        if self._exp_scans:
            n = sum(1 for _, _, v, _ in self._exp_scans if v.get())
            return f"Experimental scan{'s' if n > 1 else ''}"
        return ""

    # ── FWHM two-way binding ──────────────────────────────────────────────────

    def _on_fwhm_slider_change(self, val):
        fwhm = float(val)
        if self._x_unit.get() == "cm\u207b\u00b9":
            self._fwhm_str.set(f"{fwhm:.0f}")
        else:
            self._fwhm_str.set(f"{fwhm:.2f}")
        self._replot()

    def _on_fwhm_entry_commit(self, event=None):
        raw = self._fwhm_str.get().strip()
        try:
            val = float(raw)
        except ValueError:
            self._sync_fwhm_entry()
            return
        lo = self._fwhm_slider.cget("from")
        hi = self._fwhm_slider.cget("to")
        val = max(lo, min(hi, val))
        self._fwhm.set(val)
        self._sync_fwhm_entry()
        self._replot()

    def _sync_fwhm_entry(self):
        fwhm = self._fwhm.get()
        if self._x_unit.get() == "cm\u207b\u00b9":
            self._fwhm_str.set(f"{fwhm:.0f}")
        else:
            self._fwhm_str.set(f"{fwhm:.2f}")

    def _on_unit_change(self):
        unit = self._x_unit.get()
        if unit == "cm\u207b\u00b9":
            self._fwhm_unit_label.config(text="cm\u207b\u00b9")
            self._fwhm_slider.config(from_=50, to=8000, resolution=50)
            if self._fwhm.get() < 50:
                self._fwhm.set(3000)
        else:
            self._fwhm_unit_label.config(text="eV")
            self._fwhm_slider.config(from_=0.05, to=20.0, resolution=0.05)
            if self._fwhm.get() > 20:
                self._fwhm.set(1.0)
        self._sync_fwhm_entry()
        self._replot()

    # ══════════════════════════════════════════════════════════════════════════
    #  Figure — ax and ax2 created ONCE; never destroyed (fixes toolbar zoom)
    # ══════════════════════════════════════════════════════════════════════════
    def _build_figure(self):
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self.ax2 = None   # created on demand in _replot; never persistent

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        tb_frame = tk.Frame(self)
        tb_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, tb_frame)
        self.toolbar.update()

    # ax2 is created fresh in _replot when experimental scans are present,
    # and set to None otherwise. No hide/show helpers are needed.

    # ══════════════════════════════════════════════════════════════════════════
    #  Overlay panel
    # ══════════════════════════════════════════════════════════════════════════
    def _build_overlay_panel(self):
        self._overlay_panel = tk.Frame(self, bd=1, relief=tk.RIDGE, padx=4, pady=3)
        # Not packed yet

        hdr = tk.Frame(self._overlay_panel)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Overlays / Experimental:", font=("", 8, "bold")).pack(side=tk.LEFT)
        tk.Button(hdr, text="Clear TDDFT", command=self._clear_overlays,
                  width=10, font=("", 8)).pack(side=tk.RIGHT, padx=2)
        tk.Button(hdr, text="Clear Exp.", command=self._clear_exp_scans,
                  width=9, font=("", 8)).pack(side=tk.RIGHT, padx=2)
        tk.Button(hdr, text="Detach", command=self._unlink_selected_exp_scans,
                  width=7, font=("", 8)).pack(side=tk.RIGHT, padx=2)
        tk.Button(hdr, text="Attach Ref", command=self._link_selected_exp_scans,
                  width=9, font=("", 8)).pack(side=tk.RIGHT, padx=2)
        tk.Button(hdr, text="Merge Orig.", command=lambda: self._merge_exp_scans(use_original=True),
                  width=10, font=("", 8)).pack(side=tk.RIGHT, padx=2)
        tk.Button(hdr, text="Merge Curr.", command=lambda: self._merge_exp_scans(use_original=False),
                  width=10, font=("", 8)).pack(side=tk.RIGHT, padx=2)
        tk.Button(hdr, text="Edit Labels\u2026", command=self._open_legend_editor,
                  width=9, font=("", 8)).pack(side=tk.RIGHT, padx=2)

        container = tk.Frame(self._overlay_panel)
        container.pack(fill=tk.BOTH, expand=True)

        self._ov_canvas = tk.Canvas(container, height=90, bd=0, highlightthickness=0)
        ov_scroll = ttk.Scrollbar(container, orient=tk.VERTICAL,
                                  command=self._ov_canvas.yview)
        self._ov_inner = tk.Frame(self._ov_canvas)
        self._ov_inner.bind(
            "<Configure>",
            lambda e: self._ov_canvas.configure(
                scrollregion=self._ov_canvas.bbox("all"))
        )
        self._ov_canvas.create_window((0, 0), window=self._ov_inner, anchor="nw")
        self._ov_canvas.configure(yscrollcommand=ov_scroll.set)
        self._ov_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ov_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _make_panel_row(self, parent, label, var, colour, remove_cmd,
                        style_cmd=None, color_cmd=None):
        """Create a single checkbox row: [swatch] [checkbox label] [Style] [✕]
        If color_cmd is provided the swatch becomes a clickable colour-picker button.
        """
        row = tk.Frame(parent)
        row.pack(fill=tk.X, anchor="w")
        if color_cmd:
            swatch = tk.Button(row, bg=colour, width=2, relief=tk.RAISED,
                               cursor="hand2", command=color_cmd,
                               activebackground=colour)
            swatch.pack(side=tk.LEFT, padx=(2, 0))
            # Show a tooltip hint on hover
            _ToolTip(swatch, "Click to change colour")
        else:
            tk.Label(row, bg=colour, width=2, relief=tk.RAISED).pack(
                side=tk.LEFT, padx=(2, 0))
        tk.Checkbutton(
            row, text=label, variable=var, command=self._replot,
            anchor="w", font=("", 8), wraplength=340, justify=tk.LEFT
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        if style_cmd:
            tk.Button(row, text="Style\u2026", width=5, font=("", 7),
                      relief=tk.FLAT, command=style_cmd).pack(side=tk.RIGHT, padx=1)
        tk.Button(row, text="\u2715", width=2, font=("", 7), relief=tk.FLAT,
                  command=remove_cmd).pack(side=tk.RIGHT, padx=2)

    def _refresh_panel_content(self):
        """Rebuild all rows: TDDFT overlays (if in overlay mode) + experimental scans."""
        for w in self._ov_inner.winfo_children():
            w.destroy()

        # TDDFT overlay section
        if self._overlay_mode.get() and self._overlay_spectra:
            tk.Label(self._ov_inner, text="TDDFT Overlays:",
                     font=("", 8, "bold"), fg="navy").pack(anchor="w", padx=4, pady=(2, 0))
            for i, (label, sp, var, col) in enumerate(self._overlay_spectra):
                colour = col or OVERLAY_COLOURS[(i + 1) % len(OVERLAY_COLOURS)]
                self._make_panel_row(
                    self._ov_inner, label, var, colour,
                    remove_cmd=lambda idx=i: self._remove_overlay_idx(idx),
                    color_cmd=lambda idx=i: self._pick_overlay_colour(idx),
                )

        # Experimental scans section
        if self._exp_scans:
            if self._overlay_mode.get() and self._overlay_spectra:
                ttk.Separator(self._ov_inner, orient=tk.HORIZONTAL).pack(
                    fill=tk.X, pady=(3, 1))
            tk.Label(self._ov_inner, text="Experimental Scans (left axis \u2190):",
                     font=("", 8, "bold"), fg="darkred").pack(anchor="w", padx=4, pady=(2, 0))
            for i, (label, scan, var, style) in enumerate(self._exp_scans):
                colour = style.get("color") or EXP_COLOURS[i % len(EXP_COLOURS)]
                self._make_panel_row(
                    self._ov_inner, self._link_display_label(label, scan), var, colour,
                    remove_cmd=lambda idx=i: self._remove_exp_scan_idx(idx),
                    style_cmd=lambda idx=i: self._open_exp_style_dialog(idx),
                    color_cmd=lambda idx=i: self._pick_exp_colour(idx),
                )

        self._ov_inner.update_idletasks()
        self._ov_canvas.configure(scrollregion=self._ov_canvas.bbox("all"))
        self._update_overlay_panel_visibility()

    # Alias kept for external callers
    def _refresh_overlay_panel(self):
        self._refresh_panel_content()

    def _ensure_exp_scan_original_backup(self, scan: ExperimentalScan):
        """Persist the originally loaded curve so later processing stays reversible."""
        meta = getattr(scan, "metadata", None)
        if meta is None:
            scan.metadata = {}
            meta = scan.metadata
        if "_binah_original_energy" not in meta:
            meta["_binah_original_energy"] = np.array(scan.energy_ev, dtype=float).copy()
            meta["_binah_original_mu"] = np.array(scan.mu, dtype=float).copy()
            meta["_binah_original_e0"] = float(scan.e0)
            meta["_binah_original_norm"] = bool(scan.is_normalized)
            meta["_binah_original_scan_type"] = str(getattr(scan, "scan_type", ""))

    def _exp_link_meta(self, scan: ExperimentalScan) -> tuple:
        meta = getattr(scan, "metadata", {}) or {}
        return meta.get("_binah_link_group", ""), meta.get("_binah_link_role", "")

    def _selected_exp_entries(self):
        return [
            (i, label, scan, var, style)
            for i, (label, scan, var, style) in enumerate(self._exp_scans)
            if var.get()
        ]

    def _new_exp_link_group_id(self) -> str:
        existing = set()
        for _label, scan, _var, _style in self._exp_scans:
            meta = getattr(scan, "metadata", {}) or {}
            gid = meta.get("_binah_link_group")
            if gid:
                existing.add(str(gid))
        while True:
            gid = f"exp-link-{self._exp_link_counter}"
            self._exp_link_counter += 1
            if gid not in existing:
                return gid

    def _cleanup_exp_link_groups(self):
        groups = {}
        for _label, scan, _var, _style in self._exp_scans:
            meta = getattr(scan, "metadata", None) or {}
            gid = meta.get("_binah_link_group")
            if gid:
                groups.setdefault(gid, []).append(scan)

        for gid, scans in groups.items():
            if len(scans) < 2:
                for scan in scans:
                    meta = getattr(scan, "metadata", None) or {}
                    meta.pop("_binah_link_group", None)
                    meta.pop("_binah_link_role", None)
                continue

            ref_found = False
            for scan in scans:
                meta = getattr(scan, "metadata", None) or {}
                role = meta.get("_binah_link_role")
                if role == "reference" and not ref_found:
                    ref_found = True
                else:
                    meta["_binah_link_role"] = "linked"

            if not ref_found and scans:
                scans[0].metadata["_binah_link_role"] = "reference"

    def _link_display_label(self, label: str, scan: ExperimentalScan) -> str:
        _gid, role = self._exp_link_meta(scan)
        if role == "reference":
            return f"[Ref] {label}"
        if role == "linked":
            return f"[Linked] {label}"
        return label

    def _link_selected_exp_scans(self):
        selected = self._selected_exp_entries()
        if len(selected) < 2:
            messagebox.showinfo(
                "Attach Reference / Raw",
                "Enable at least two experimental scans first.\n\n"
                "The first enabled scan becomes the reference, and the others follow its energy calibration."
            )
            return

        gid = self._new_exp_link_group_id()

        for idx, (_row_idx, _label, scan, _var, _style) in enumerate(selected):
            self._ensure_exp_scan_original_backup(scan)
            meta = getattr(scan, "metadata", None)
            if meta is None:
                scan.metadata = {}
                meta = scan.metadata
            meta["_binah_link_group"] = gid
            meta["_binah_link_role"] = "reference" if idx == 0 else "linked"

        self._cleanup_exp_link_groups()
        ref_label = selected[0][1]
        n_linked = max(0, len(selected) - 1)
        self._refresh_panel_content()
        self._replot()
        messagebox.showinfo(
            "Attach Reference / Raw",
            f"Attached {len(selected)} scans into one calibration group.\n\n"
            f"Reference: {ref_label}\n"
            f"Followers: {n_linked}\n\n"
            f"Now when you shift the reference energy in XAS Analysis, the linked raw scans will move with it."
        )

    def _unlink_selected_exp_scans(self):
        selected = self._selected_exp_entries()
        if not selected:
            messagebox.showinfo(
                "Detach Linked Scans",
                "Enable one or more linked experimental scans to detach."
            )
            return

        changed = 0
        for _row_idx, _label, scan, _var, _style in selected:
            meta = getattr(scan, "metadata", None) or {}
            if "_binah_link_group" in meta or "_binah_link_role" in meta:
                meta.pop("_binah_link_group", None)
                meta.pop("_binah_link_role", None)
                changed += 1

        self._cleanup_exp_link_groups()
        self._refresh_panel_content()
        self._replot()
        if changed:
            messagebox.showinfo(
                "Detach Linked Scans",
                f"Detached {changed} scan(s) from calibration groups."
            )
        else:
            messagebox.showinfo(
                "Detach Linked Scans",
                "None of the enabled scans were linked."
            )

    def _update_overlay_panel_visibility(self):
        should_show = self._overlay_mode.get() or bool(self._exp_scans)
        if should_show:
            self._overlay_panel.pack(side=tk.TOP, fill=tk.X,
                                     before=self.canvas.get_tk_widget())
        else:
            self._overlay_panel.pack_forget()

    def _on_overlay_toggle(self):
        self._refresh_panel_content()
        self._replot()

    # ══════════════════════════════════════════════════════════════════════════
    #  Font settings dialog
    # ══════════════════════════════════════════════════════════════════════════
    def _open_font_dialog(self):
        """Non-modal dialog for controlling font sizes and bold on all text elements."""
        win = tk.Toplevel(self)
        win.title("Font Settings")
        win.resizable(False, False)

        frm = tk.Frame(win, padx=18, pady=14)
        frm.pack(fill=tk.BOTH)

        # Header row
        tk.Label(frm, text="Element", font=("", 9, "bold"),
                 width=16, anchor="w").grid(row=0, column=0, sticky="w")
        tk.Label(frm, text="Size", font=("", 9, "bold")).grid(
            row=0, column=1, columnspan=2, sticky="w", padx=(12, 0))
        tk.Label(frm, text="Bold", font=("", 9, "bold")).grid(
            row=0, column=3, sticky="w", padx=(16, 0))

        ttk.Separator(frm, orient=tk.HORIZONTAL).grid(
            row=1, column=0, columnspan=4, sticky="ew", pady=(2, 8))

        def _row(r, label, size_var, bold_var=None):
            tk.Label(frm, text=label, font=("", 9),
                     width=16, anchor="w").grid(row=r, column=0, sticky="w", pady=3)
            tk.Label(frm, text="Size:", font=("", 8),
                     fg="gray").grid(row=r, column=1, sticky="e", padx=(12, 2))
            sb = tk.Spinbox(
                frm, from_=5, to=36, increment=1, width=5,
                textvariable=size_var, font=("Courier", 9),
                command=self._replot,
            )
            sb.grid(row=r, column=2, sticky="w")
            sb.bind("<Return>",   lambda _e: self._replot())
            sb.bind("<FocusOut>", lambda _e: self._replot())
            if bold_var is not None:
                tk.Checkbutton(frm, text="Bold", variable=bold_var,
                               command=self._replot,
                               font=("", 9)).grid(row=r, column=3, sticky="w",
                                                   padx=(16, 0))
            else:
                tk.Label(frm, text="—", fg="gray",
                         font=("", 9)).grid(row=r, column=3, sticky="w", padx=(16, 0))

        _row(2, "Plot title",    self._font_title_size,  self._font_title_bold)
        _row(3, "X-axis label",  self._font_xlabel_size, self._font_xlabel_bold)
        _row(4, "Y-axis labels", self._font_ylabel_size, self._font_ylabel_bold)
        _row(5, "Tick labels",   self._font_tick_size)
        _row(6, "Legend",        self._font_legend_size)

        ttk.Separator(frm, orient=tk.HORIZONTAL).grid(
            row=7, column=0, columnspan=4, sticky="ew", pady=(10, 6))

        def _reset_all():
            self._font_title_size.set(11);  self._font_title_bold.set(True)
            self._font_xlabel_size.set(11); self._font_xlabel_bold.set(False)
            self._font_ylabel_size.set(11); self._font_ylabel_bold.set(False)
            self._font_tick_size.set(9)
            self._font_legend_size.set(9)
            self._replot()

        btn_row = tk.Frame(frm)
        btn_row.grid(row=8, column=0, columnspan=4, pady=(0, 2))
        tk.Button(btn_row, text="Reset Defaults", command=_reset_all,
                  font=("", 8)).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="Close", command=win.destroy,
                  font=("", 9)).pack(side=tk.LEFT, padx=4)

        tk.Label(frm,
                 text="Changes apply live. Size range 5 – 36 pt.",
                 font=("", 8), fg="gray").grid(
            row=9, column=0, columnspan=4, pady=(4, 0))

        _centre_window(win, self)

    # ══════════════════════════════════════════════════════════════════════════
    #  Style dialogs
    # ══════════════════════════════════════════════════════════════════════════
    def _open_tddft_style_dialog(self):
        """Global TDDFT plot style: envelope and sticks."""
        win = tk.Toplevel(self)
        win.title("TDDFT Plot Style")
        win.resizable(False, False)
        win.grab_set()

        st = self._tddft_style

        # ── Envelope ────────────────────────────────────────────────────────
        env_frame = tk.LabelFrame(win, text="Envelope", padx=10, pady=6)
        env_frame.pack(fill=tk.X, padx=12, pady=(10, 4))

        env_lw  = tk.DoubleVar(value=st["env_linewidth"])
        env_fill = tk.BooleanVar(value=st["env_fill"])
        env_fa  = tk.DoubleVar(value=st["env_fill_alpha"])

        _slider_row(env_frame, "Line width:", env_lw, 0.5, 5.0, 0.1, row=0)

        fill_row = tk.Frame(env_frame)
        fill_row.grid(row=1, column=0, columnspan=3, sticky="w", pady=2)
        tk.Checkbutton(fill_row, text="Fill area under curve",
                       variable=env_fill).pack(side=tk.LEFT)
        _slider_row(env_frame, "Fill opacity:", env_fa, 0.0, 0.50, 0.01, row=2)

        # ── Sticks ──────────────────────────────────────────────────────────
        stk_frame = tk.LabelFrame(win, text="Sticks (stems)", padx=10, pady=6)
        stk_frame.pack(fill=tk.X, padx=12, pady=(4, 10))

        stk_lw   = tk.DoubleVar(value=st["stick_linewidth"])
        stk_alph = tk.DoubleVar(value=st["stick_alpha"])
        stk_ms   = tk.DoubleVar(value=st["stick_markersize"])
        stk_mkr  = tk.BooleanVar(value=st.get("stick_markers", True))

        _slider_row(stk_frame, "Line width:",   stk_lw,   0.5, 4.0, 0.1, row=0)
        _slider_row(stk_frame, "Opacity:",      stk_alph, 0.1, 1.0, 0.05, row=1)

        mkr_row = tk.Frame(stk_frame)
        mkr_row.grid(row=2, column=0, columnspan=3, sticky="w", pady=2)
        tk.Checkbutton(mkr_row, text="Show tip markers (dots)",
                       variable=stk_mkr).pack(side=tk.LEFT)

        _slider_row(stk_frame, "Marker size:",  stk_ms,   1,   12,  1,    row=3)

        # ── Buttons ─────────────────────────────────────────────────────────
        def _read_dialog():
            return {
                "env_linewidth":   env_lw.get(),
                "env_fill":        env_fill.get(),
                "env_fill_alpha":  env_fa.get(),
                "stick_linewidth": stk_lw.get(),
                "stick_alpha":     stk_alph.get(),
                "stick_markersize": int(stk_ms.get()),
                "stick_markers":   stk_mkr.get(),
            }

        def apply():
            st.update(_read_dialog())
            self._replot()
            win.destroy()

        def set_as_default():
            vals = _read_dialog()
            _TDDFT_STYLE_DEFAULTS.update(vals)
            st.update(vals)
            _save_style_config()    # persist to disk for next restart
            self._replot()
            messagebox.showinfo("Default saved",
                "TDDFT style saved as default.\n"
                "All new computational spectra will use these settings.",
                parent=win)

        def reset():
            # Restore the hard-coded factory defaults (not the saved ones)
            factory = {
                "env_linewidth": 2.0, "env_fill": True, "env_fill_alpha": 0.10,
                "stick_linewidth": 1.2, "stick_alpha": 0.75,
                "stick_markersize": 4, "stick_markers": True,
            }
            env_lw.set(factory["env_linewidth"])
            env_fill.set(factory["env_fill"])
            env_fa.set(factory["env_fill_alpha"])
            stk_lw.set(factory["stick_linewidth"])
            stk_alph.set(factory["stick_alpha"])
            stk_ms.set(factory["stick_markersize"])
            stk_mkr.set(factory["stick_markers"])

        btn = tk.Frame(win)
        btn.pack(pady=(0, 10))
        tk.Button(btn, text="Apply",          width=12, command=apply).pack(side=tk.LEFT, padx=4)
        tk.Button(btn, text="Set as Default", width=14, bg="#003366", fg="white",
                  command=set_as_default).pack(side=tk.LEFT, padx=4)
        tk.Button(btn, text="Factory Reset",  width=12, command=reset).pack(side=tk.LEFT, padx=4)
        tk.Button(btn, text="Cancel",         width=10, command=win.destroy).pack(side=tk.LEFT, padx=4)

        _centre_window(win, self)

    def _pick_exp_colour(self, idx: int):
        """Open the system colour-wheel directly from the panel swatch."""
        from tkinter import colorchooser
        label, scan, var, style = self._exp_scans[idx]
        auto_col  = EXP_COLOURS[idx % len(EXP_COLOURS)]
        init_col  = style.get("color") or auto_col
        result    = colorchooser.askcolor(
            color=init_col,
            title=f"Colour — {label[:40]}",
            parent=self,
        )
        if result and result[1]:                    # user confirmed a colour
            style["color"] = result[1]
            self._refresh_panel_content()
            self._replot()

    def _pick_overlay_colour(self, idx: int):
        """Open the system colour-wheel for a TDDFT overlay swatch."""
        from tkinter import colorchooser
        lbl, sp, var, cur_col = self._overlay_spectra[idx]
        default_col = OVERLAY_COLOURS[(idx + 1) % len(OVERLAY_COLOURS)]
        init_col    = cur_col or default_col
        result = colorchooser.askcolor(
            color=init_col,
            title=f"Colour — {lbl[:40]}",
            parent=self,
        )
        if result and result[1]:
            self._overlay_spectra[idx] = (lbl, sp, var, result[1])
            self._refresh_panel_content()
            self._replot()

    def _open_exp_style_dialog(self, idx: int):
        """Per-scan experimental plot style."""
        label, scan, var, style = self._exp_scans[idx]

        win = tk.Toplevel(self)
        win.title(f"Exp. Scan Style")
        win.resizable(False, False)
        win.grab_set()

        # Header
        hdr = tk.Frame(win, bg="#6B0000", padx=10, pady=6)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Experimental Scan Style",
                 bg="#6B0000", fg="white", font=("", 10, "bold")).pack(anchor="w")
        tk.Label(hdr, text=label[:60] + ("…" if len(label) > 60 else ""),
                 bg="#6B0000", fg="#ffaaaa", font=("", 8)).pack(anchor="w")

        body = tk.Frame(win, padx=12, pady=8)
        body.pack(fill=tk.BOTH)

        # ── Line style ──────────────────────────────────────────────────────
        tk.Label(body, text="Line style:", font=("", 9, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 4))
        ls_var = tk.StringVar(value=style["linestyle"])
        ls_frame = tk.Frame(body)
        ls_frame.grid(row=0, column=1, columnspan=3, sticky="w")
        for display, value in LS_OPTIONS:
            tk.Radiobutton(ls_frame, text=display, variable=ls_var,
                           value=value).pack(side=tk.LEFT, padx=4)

        # ── Line width ──────────────────────────────────────────────────────
        lw_var = tk.DoubleVar(value=style["linewidth"])
        _slider_row(body, "Line width:", lw_var, 0.5, 5.0, 0.1, row=1)

        # ── Color ───────────────────────────────────────────────────────────
        auto_col = EXP_COLOURS[idx % len(EXP_COLOURS)]
        tk.Label(body, text="Color:", font=("", 9, "bold")).grid(
            row=2, column=0, sticky="w", pady=4)
        col_var = tk.StringVar(value=style.get("color", ""))

        # Swatch button — shows current colour; click opens colour-wheel
        _swatch_init = col_var.get() or auto_col
        col_swatch = tk.Button(body, bg=_swatch_init, width=4, relief=tk.RAISED,
                               cursor="hand2")
        col_swatch.grid(row=2, column=1, sticky="w", padx=(4, 4))

        def _pick_color_dialog():
            from tkinter import colorchooser
            init = col_var.get().strip() or auto_col
            result = colorchooser.askcolor(color=init,
                                           title="Choose scan colour",
                                           parent=win)
            if result and result[1]:
                col_var.set(result[1])
                col_swatch.config(bg=result[1], activebackground=result[1])

        col_swatch.config(command=_pick_color_dialog)

        def _reset_to_auto():
            col_var.set("")
            col_swatch.config(bg=auto_col, activebackground=auto_col)

        tk.Label(body, text=f"(auto: {auto_col})",
                 fg="gray", font=("", 8)).grid(row=2, column=2, sticky="w")
        tk.Button(body, text="Reset to auto", font=("", 8),
                  command=_reset_to_auto).grid(row=2, column=3, sticky="w")

        # ── Fill ────────────────────────────────────────────────────────────
        ttk.Separator(body, orient=tk.HORIZONTAL).grid(
            row=3, column=0, columnspan=4, sticky="ew", pady=6)
        tk.Label(body, text="Fill area under curve:",
                 font=("", 9, "bold")).grid(row=4, column=0, sticky="w")
        fill_var  = tk.BooleanVar(value=style["fill"])
        alpha_var = tk.DoubleVar(value=style["fill_alpha"])

        fill_frame = tk.Frame(body)
        fill_frame.grid(row=4, column=1, columnspan=3, sticky="w")
        tk.Checkbutton(fill_frame, text="Show fill", variable=fill_var).pack(side=tk.LEFT)

        _slider_row(body, "Fill opacity:", alpha_var, 0.0, 0.5, 0.01, row=5)

        # ── Buttons ─────────────────────────────────────────────────────────
        def _validate_col():
            # Color comes from the colour-wheel so it's always valid hex or "".
            return col_var.get().strip()

        def _read_dialog():
            return {
                "linestyle":  ls_var.get(),
                "linewidth":  lw_var.get(),
                "color":      col_var.get().strip(),
                "fill":       fill_var.get(),
                "fill_alpha": alpha_var.get(),
            }

        def apply():
            raw_col = _validate_col()
            if raw_col is None:
                return
            vals = _read_dialog()
            style.update(vals)
            self._refresh_panel_content()
            self._replot()
            win.destroy()

        def apply_to_all():
            """Push line/fill style to every exp scan — NEVER changes colours."""
            vals = _read_dialog()
            for _lbl, _sc, _var, _st in self._exp_scans:
                _st["linestyle"]  = vals["linestyle"]
                _st["linewidth"]  = vals["linewidth"]
                _st["fill"]       = vals["fill"]
                _st["fill_alpha"] = vals["fill_alpha"]
                # colour is intentionally NOT touched — every scan keeps its own
            self._refresh_panel_content()
            self._replot()
            win.destroy()

        def set_as_default():
            vals = _read_dialog()
            # Don't save the per-scan colour as the default colour —
            # new scans should still get auto-assigned colours.
            saved = dict(vals)
            saved["color"] = ""
            _EXP_STYLE_DEFAULTS.update(saved)
            style.update(vals)      # apply colour to this scan only
            _save_style_config()    # persist to disk for next restart
            self._refresh_panel_content()
            self._replot()
            messagebox.showinfo("Default saved",
                "Experimental style saved as default.\n"
                "All new scans loaded from now on will use these settings.",
                parent=win)

        def reset_defaults():
            # Restore hard-coded factory defaults
            factory = {"linestyle": "solid", "linewidth": 1.8,
                       "fill": True, "fill_alpha": 0.06, "color": ""}
            ls_var.set(factory["linestyle"])
            lw_var.set(factory["linewidth"])
            col_var.set(factory["color"])
            fill_var.set(factory["fill"])
            alpha_var.set(factory["fill_alpha"])

        btn = tk.Frame(win)
        btn.pack(pady=(0, 10))
        tk.Button(btn, text="Apply",            width=12,
                  command=apply).pack(side=tk.LEFT, padx=3)
        tk.Button(btn, text="Apply to ALL Exp.", width=16,
                  bg="#004400", fg="white",
                  command=apply_to_all).pack(side=tk.LEFT, padx=3)
        tk.Button(btn, text="Set as Default",   width=14,
                  bg="#003366", fg="white",
                  command=set_as_default).pack(side=tk.LEFT, padx=3)
        tk.Button(btn, text="Factory Reset",    width=12,
                  command=reset_defaults).pack(side=tk.LEFT, padx=3)
        tk.Button(btn, text="Cancel",           width=10,
                  command=win.destroy).pack(side=tk.LEFT, padx=3)

        _centre_window(win, self)

    # ══════════════════════════════════════════════════════════════════════════
    #  Legend editor dialog
    # ══════════════════════════════════════════════════════════════════════════
    def _open_legend_editor(self):
        win = tk.Toplevel(self)
        win.title("Edit Legend")
        win.resizable(True, False)
        win.grab_set()

        tk.Label(win, text="Edit legend labels below. Leave blank to hide that entry.",
                 font=("", 9), fg="gray").pack(anchor="w", padx=10, pady=(8, 2))

        entries = []

        # Primary spectrum
        _local_show_primary = tk.BooleanVar(value=self._show_primary_in_legend.get())
        if self.spectrum:
            pf = tk.Frame(win, padx=8)
            pf.pack(fill=tk.X, pady=2)
            tk.Label(pf, text="Primary:", width=10, anchor="e",
                     font=("", 8, "bold")).pack(side=tk.LEFT)
            pe = tk.Entry(pf, width=46, font=("", 9))
            pe.insert(0, getattr(self.spectrum, "_custom_label", None) or
                      self.spectrum.display_name())
            pe.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
            tk.Checkbutton(pf, text="Show in legend", variable=_local_show_primary,
                           font=("", 8)).pack(side=tk.LEFT, padx=(6, 0))
            entries.append(("primary", pe))

        # TDDFT overlays
        if self._overlay_spectra:
            ttk.Separator(win, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=4)
            tk.Label(win, text="TDDFT Overlays:", font=("", 8, "bold")).pack(anchor="w", padx=10)
            for i, (label, sp, var, _ov_col) in enumerate(self._overlay_spectra):
                col = _ov_col or OVERLAY_COLOURS[(i + 1) % len(OVERLAY_COLOURS)]
                of = tk.Frame(win, padx=8)
                of.pack(fill=tk.X, pady=1)
                tk.Label(of, bg=col, width=2, relief=tk.RAISED).pack(side=tk.LEFT)
                tk.Label(of, text=f"  #{i+1}", width=4, anchor="e",
                         font=("", 8)).pack(side=tk.LEFT)
                oe = tk.Entry(of, width=55, font=("", 9))
                oe.insert(0, label)
                oe.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
                entries.append(("overlay", i, oe))

        # Experimental scans
        if self._exp_scans:
            ttk.Separator(win, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=4)
            tk.Label(win, text="Experimental Scans:", font=("", 8, "bold"),
                     fg="darkred").pack(anchor="w", padx=10)
            for i, (label, scan, var, style) in enumerate(self._exp_scans):
                col = style.get("color") or EXP_COLOURS[i % len(EXP_COLOURS)]
                ef = tk.Frame(win, padx=8)
                ef.pack(fill=tk.X, pady=1)
                tk.Label(ef, bg=col, width=2, relief=tk.RAISED).pack(side=tk.LEFT)
                tk.Label(ef, text=f"  E{i+1}", width=4, anchor="e",
                         font=("", 8)).pack(side=tk.LEFT)
                ee = tk.Entry(ef, width=55, font=("", 9))
                ee.insert(0, label)
                ee.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
                entries.append(("exp", i, ee))

        ttk.Separator(win, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # Use a LOCAL copy of _show_legend so the checkbox in this dialog
        # cannot accidentally flip the live state while the user is typing.
        # The value is only written back to self._show_legend on Apply.
        _local_show_legend = tk.BooleanVar(value=self._show_legend.get())
        vis_frame = tk.Frame(win, padx=10)
        vis_frame.pack(fill=tk.X)
        tk.Checkbutton(vis_frame, text="Show legend on plot",
                       variable=_local_show_legend).pack(side=tk.LEFT)

        def apply():
            for entry_info in entries:
                kind = entry_info[0]
                if kind == "primary":
                    new = entry_info[1].get().strip()
                    self.spectrum._custom_label = new if new else None
                elif kind == "overlay":
                    _, idx, e = entry_info
                    new = e.get().strip()
                    lbl, sp, var, _oc = self._overlay_spectra[idx]
                    self._overlay_spectra[idx] = (new if new else lbl, sp, var, _oc)
                elif kind == "exp":
                    _, idx, e = entry_info
                    new = e.get().strip()
                    lbl, scan, var, style = self._exp_scans[idx]
                    self._exp_scans[idx] = (new if new else lbl, scan, var, style)
            # Push legend-visibility choices back to the real variables
            self._show_legend.set(_local_show_legend.get())
            self._show_primary_in_legend.set(_local_show_primary.get())
            # Destroy dialog BEFORE replot so the modal grab is released first
            win.destroy()
            self._refresh_panel_content()
            self._replot()

        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=8)
        tk.Button(btn_frame, text="Apply",  command=apply,      width=10).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Cancel", command=win.destroy, width=10).pack(side=tk.LEFT, padx=4)
        _centre_window(win, self)

    # ══════════════════════════════════════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════════════════════════════════════
    def load_spectrum(self, spectrum: TDDFTSpectrum):
        self.spectrum = spectrum
        if not self._custom_title.get().strip():
            self._custom_title.set(spectrum.display_name())
        if spectrum.is_xas:
            self._fwhm_slider.config(from_=0.05, to=20.0, resolution=0.05)
            self._fwhm_unit_label.config(text="eV")
            if self._x_unit.get() == "nm":
                self._x_unit.set("eV")
            if self._fwhm.get() > 20:
                self._fwhm.set(1.0)
        else:
            self._fwhm_slider.config(from_=100, to=8000, resolution=100)
            self._fwhm_unit_label.config(text="cm\u207b\u00b9")
            self._x_unit.set("nm")
            if self._fwhm.get() < 50:
                self._fwhm.set(3000)
        self._sync_fwhm_entry()
        self._replot()

    def add_overlay(self, label: str, spectrum: TDDFTSpectrum):
        var = tk.BooleanVar(value=True)
        self._overlay_spectra.append((label, spectrum, var, ""))
        if not self._overlay_mode.get():
            self._overlay_mode.set(True)
        self._refresh_panel_content()
        self._replot()

    def add_exp_scan(self, label: str, scan: ExperimentalScan):
        self._ensure_exp_scan_original_backup(scan)
        var   = tk.BooleanVar(value=True)
        style = _default_exp_style()
        self._exp_scans.append((label, scan, var, style))
        self._refresh_panel_content()
        self._replot()

    def _next_exp_merge_label(self, base: str) -> str:
        existing = {lbl for lbl, *_ in self._exp_scans}
        if base not in existing:
            return base
        idx = 2
        while f"{base} #{idx}" in existing:
            idx += 1
        return f"{base} #{idx}"

    def _exp_merge_source_arrays(self, scan: ExperimentalScan, use_original: bool):
        meta = getattr(scan, "metadata", {}) or {}
        if use_original and "_binah_original_energy" in meta and "_binah_original_mu" in meta:
            energy = np.array(meta["_binah_original_energy"], dtype=float).copy()
            mu = np.array(meta["_binah_original_mu"], dtype=float).copy()
            is_norm = bool(meta.get("_binah_original_norm", scan.is_normalized))
            e0 = float(meta.get("_binah_original_e0", scan.e0))
            scan_type = str(meta.get("_binah_original_scan_type", getattr(scan, "scan_type", "")))
        else:
            energy = np.array(scan.energy_ev, dtype=float).copy()
            mu = np.array(scan.mu, dtype=float).copy()
            is_norm = bool(scan.is_normalized)
            e0 = float(scan.e0)
            scan_type = str(getattr(scan, "scan_type", ""))

        order = np.argsort(energy)
        energy = energy[order]
        mu = mu[order]
        energy, unique_idx = np.unique(energy, return_index=True)
        mu = mu[unique_idx]
        return energy, mu, is_norm, e0, scan_type

    def _merge_exp_scans(self, use_original: bool = False):
        selected = [(lbl, scan) for lbl, scan, var, _style in self._exp_scans if var.get()]
        if len(selected) < 2:
            messagebox.showinfo(
                "Merge Experimental Scans",
                "Enable at least two experimental scans to merge."
            )
            return

        series = []
        norm_flags = []
        for lbl, scan in selected:
            self._ensure_exp_scan_original_backup(scan)
            energy, mu, is_norm, e0, scan_type = self._exp_merge_source_arrays(
                scan, use_original=use_original)
            if len(energy) < 2:
                continue
            series.append((lbl, energy, mu, e0, scan_type))
            norm_flags.append(is_norm)

        if len(series) < 2:
            messagebox.showerror(
                "Merge Experimental Scans",
                "At least two scans need usable energy data to merge."
            )
            return

        if any(flag != norm_flags[0] for flag in norm_flags):
            source_txt = "original loaded" if use_original else "current"
            messagebox.showerror(
                "Merge Experimental Scans",
                "Cannot merge a mixture of raw and normalized scans from the same source.\n\n"
                f"Choose scans with the same normalization state, or switch to the other merge mode.\n"
                f"Current mode: {source_txt} data."
            )
            return

        overlap_lo = max(float(en.min()) for _, en, *_ in series)
        overlap_hi = min(float(en.max()) for _, en, *_ in series)
        if overlap_hi <= overlap_lo:
            messagebox.showerror(
                "Merge Experimental Scans",
                "The selected scans do not share an overlapping energy range."
            )
            return

        ref_idx = max(
            range(len(series)),
            key=lambda i: int(np.count_nonzero(
                (series[i][1] >= overlap_lo) & (series[i][1] <= overlap_hi)))
        )
        ref_energy = series[ref_idx][1]
        grid = ref_energy[(ref_energy >= overlap_lo) & (ref_energy <= overlap_hi)]
        if len(grid) < 2:
            grid = np.linspace(overlap_lo, overlap_hi, 400)

        merged_rows = []
        for _lbl, energy, mu, _e0, _stype in series:
            merged_rows.append(np.interp(grid, energy, mu))
        merged_arr = np.vstack(merged_rows)
        merged_mu = np.mean(merged_arr, axis=0)

        valid_e0 = [e0 for _, _, _, e0, _ in series if e0 > 0]
        merged_e0 = float(np.mean(valid_e0)) if valid_e0 else 0.0
        merged_is_norm = bool(norm_flags[0])
        merge_kind = "normalized" if merged_is_norm else "raw"
        label = self._next_exp_merge_label(
            f"Merged {merge_kind} ({len(series)} scans, {'original' if use_original else 'current'})")

        merged_scan = ExperimentalScan(
            label=label,
            source_file="merged://experimental",
            energy_ev=grid,
            mu=merged_mu,
            e0=merged_e0,
            is_normalized=merged_is_norm,
            scan_type=f"merged {merge_kind}",
            metadata={
                "merged_from_labels": [lbl for lbl, *_ in series],
                "merge_source": "original" if use_original else "current",
                "merge_count": len(series),
                "merge_overlap_ev": [overlap_lo, overlap_hi],
            },
        )
        self.add_exp_scan(label, merged_scan)
        source_txt = "original loaded" if use_original else "current"
        messagebox.showinfo(
            "Merge Experimental Scans",
            f"Created merged scan:\n{label}\n\n"
            f"Source: {source_txt}\n"
            f"Scans merged: {len(series)}\n"
            f"Overlap: {overlap_lo:.2f} to {overlap_hi:.2f} eV"
        )

    def _remove_overlay_idx(self, idx: int):
        if 0 <= idx < len(self._overlay_spectra):
            self._overlay_spectra.pop(idx)
            self._refresh_panel_content()
            self._replot()

    def _remove_exp_scan_idx(self, idx: int):
        if 0 <= idx < len(self._exp_scans):
            self._exp_scans.pop(idx)
            self._refresh_panel_content()
            self._replot()

    def _clear_overlays(self):
        self._overlay_spectra.clear()
        self._refresh_panel_content()
        self._replot()

    def _clear_exp_scans(self):
        self._exp_scans.clear()
        self._refresh_panel_content()
        self._replot()

    # ══════════════════════════════════════════════════════════════════════════
    #  Unit helpers
    # ══════════════════════════════════════════════════════════════════════════
    def _ev_to_unit(self, ev_arr):
        unit = self._x_unit.get()
        if unit == "nm":
            with np.errstate(divide="ignore"):
                return np.where(ev_arr > 0, 1239.84 / ev_arr, 0.0)
        elif unit == "cm\u207b\u00b9":
            return ev_arr * 8065.54
        return ev_arr

    def _fwhm_in_ev(self) -> float:
        unit = self._x_unit.get()
        fwhm = self._fwhm.get()
        if unit in ("cm\u207b\u00b9", "nm"):
            return fwhm / 8065.54
        return fwhm

    def _xlabel(self) -> str:
        unit = self._x_unit.get()
        if unit == "nm":    return "Wavelength (nm)"
        if unit == "eV":    return "Energy (eV)"
        return "Wavenumber (cm\u207b\u00b9)"

    # ══════════════════════════════════════════════════════════════════════════
    #  Main plot routine — uses cla() so axes objects are never destroyed
    # ══════════════════════════════════════════════════════════════════════════
    def _replot(self, *_):
        # ── Determine what to draw ────────────────────────────────────────────
        tddft_to_draw: List[Tuple[str, TDDFTSpectrum, str]] = []
        if self.spectrum:
            name = (getattr(self.spectrum, "_custom_label", None) or
                    self.spectrum.display_name())
            tddft_to_draw.append((name, self.spectrum, OVERLAY_COLOURS[0]))
        if self._overlay_mode.get():
            for k, (lbl, sp, var, _ov_col) in enumerate(self._overlay_spectra):
                if var.get():
                    col = _ov_col or OVERLAY_COLOURS[(k + 1) % len(OVERLAY_COLOURS)]
                    tddft_to_draw.append((lbl, sp, col))

        active_exp = [
            (lbl, sc, style.get("color") or EXP_COLOURS[i % len(EXP_COLOURS)], style)
            for i, (lbl, sc, var, style) in enumerate(self._exp_scans)
            if var.get()
        ]

        # ── Rebuild axes from scratch every redraw ────────────────────────────
        # fig.clear() + add_subplot() is the standard embedded-matplotlib pattern.
        # toolbar.update() (called at the end) re-registers the fresh axes so that
        # zoom, pan, home, back, forward all work correctly after every redraw.
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        # Create twin axis only when experimental scans are active
        if active_exp:
            self.ax2 = self.ax.twinx()
            # Tell the toolbar to navigate ax only — never ax2.
            # Without this the toolbar tries to zoom both axes simultaneously,
            # which with the shared x-axis creates conflicting updates → no zoom.
            self.ax2.set_navigate(False)
            # Raise ax above ax2 so its hover annotation is always visible.
            # Hide ax's background patch so it doesn't blank out ax2's data.
            self.ax.set_zorder(self.ax2.get_zorder() + 1)
            self.ax.patch.set_visible(False)
        else:
            self.ax2 = None
            self.ax.patch.set_visible(True)

        if not tddft_to_draw and not active_exp:
            self.toolbar.update()
            self.canvas.draw_idle()
            self._notify_popouts()
            return

        multi_tddft = len(tddft_to_draw) > 1
        first_hover_set = False
        ylabel = "Oscillator Strength (f)"
        delta_e = self._delta_e.get()
        scale   = self._tddft_scale.get()
        st = self._tddft_style

        # ── Collect fresh inset data each replot ─────────────────────────────
        self._inset_plot_data = []
        self._inset_indicator = None   # fig.clear() already destroyed old indicator

        # ── Auto-normalise TDDFT to 0-1 when experimental overlay is present ──
        # Individual oscillator strengths are tiny (≈1e-3) while experimental
        # XAS is normalized to 1.  Pre-scan all spectra so the tallest feature
        # (stick OR envelope peak) reaches 1.0 and the comparison is visual.
        show_tddft = self._show_tddft.get() and bool(tddft_to_draw)
        if active_exp and show_tddft and not self._normalise.get():
            _peak = 0.0
            for _, _sp, _ in tddft_to_draw:
                _y = np.array(_sp.fosc if not _sp.is_cd() else _sp.rotatory_strength)
                if len(_y):
                    _peak = max(_peak, float(np.abs(_y).max()))
            _auto_norm = (1.0 / (_peak * scale)) if _peak * scale > 0 else 1.0
        else:
            _auto_norm = 1.0

        # ── Draw TDDFT spectra (left axis) ────────────────────────────────────
        for name, sp, colour in (tddft_to_draw if show_tddft else []):
            ev_arr = np.array(sp.energies_ev) + delta_e
            is_cd  = sp.is_cd()
            is_comb = sp.is_combined() and not multi_tddft  # component mode only in single view
            if len(ev_arr) == 0:
                continue
            x_arr = self._ev_to_unit(ev_arr)

            import matplotlib.pyplot as _plt

            if is_comb:
                # ── Combined spectrum: draw each enabled component separately ──────
                # Colours fixed so D2/m2/Q2 are always the same colour regardless of
                # which overlay slot this spectrum occupies.
                COMB_COLS = {
                    "total": colour,         # uses normal spectrum colour
                    "d2":    "#1f77b4",      # blue
                    "m2":    "#2ca02c",      # green
                    "q2":    "#d62728",      # red
                }
                comp_defs = []
                if self._comb_total.get():
                    comp_defs.append(("total", np.array(sp.fosc),    COMB_COLS["total"], "Total"))
                if self._comb_d2.get() and sp.fosc_d2:
                    comp_defs.append(("d2",    np.array(sp.fosc_d2), COMB_COLS["d2"],   "D\u00b2 (elec. dip.)"))
                if self._comb_m2.get() and sp.fosc_m2:
                    comp_defs.append(("m2",    np.array(sp.fosc_m2), COMB_COLS["m2"],   "m\u00b2 (mag. dip.)"))
                if self._comb_q2.get() and sp.fosc_q2:
                    comp_defs.append(("q2",    np.array(sp.fosc_q2), COMB_COLS["q2"],   "Q\u00b2 (elec. quad.)"))

                if not comp_defs:
                    continue   # nothing selected — skip this spectrum silently

                for k, (key, y_c, col_c, comp_lbl) in enumerate(comp_defs):
                    if self._normalise.get() and np.abs(y_c).max() != 0:
                        yp = (y_c / np.abs(y_c).max()) * scale
                        ylabel = "Normalised Intensity"
                    else:
                        yp = y_c * scale * _auto_norm
                        ylabel = ("Rel. Intensity" if active_exp else "Oscillator Strength (f)")

                    if self._show_sticks.get():
                        ml, sl, bl = self.ax.stem(
                            x_arr, yp, linefmt=col_c, markerfmt="o", basefmt=" ",
                            label=comp_lbl
                        )
                        ml.set_markersize(st["stick_markersize"] if st.get("stick_markers", True) else 0)
                        ml.set_color(col_c)
                        _plt.setp(sl, linewidth=st["stick_linewidth"],
                                  alpha=st["stick_alpha"], color=col_c)
                        self._inset_plot_data.append(
                            ("sticks", x_arr.copy(), yp.copy(), col_c,
                             st["stick_linewidth"], st["stick_alpha"]))

                    if self._show_env.get():
                        x_env, y_env = self._draw_envelope(ev_arr, yp, col_c, True, comp_lbl)
                        if x_env is not None:
                            self._inset_plot_data.append(
                                ("line", x_env, y_env, col_c,
                                 st["env_linewidth"], "solid", 0.9))
                            if st["env_fill"]:
                                self._inset_plot_data.append(
                                    ("fill", x_env, y_env, col_c, st["env_fill_alpha"]))

                    if self._show_trans.get() and sp.excited_states and k == 0:
                        self._draw_transition_labels(x_arr, yp, sp)

                    # Hover data from the first/primary component
                    if k == 0 and not first_hover_set:
                        self._hover_x      = x_arr
                        self._hover_y      = yp
                        self._hover_ev     = ev_arr
                        self._hover_cm     = np.array(sp.energies_cm)
                        self._hover_states = list(sp.states)
                        self._hover_labels = list(sp.transition_labels)
                        first_hover_set = True

            else:
                # ── Normal (non-combined) spectrum ────────────────────────────────
                y_arr = np.array(sp.rotatory_strength if is_cd else sp.fosc)

                if self._normalise.get() and np.abs(y_arr).max() != 0:
                    y_plot = (y_arr / np.abs(y_arr).max()) * scale
                    ylabel = ("Normalised \u00d7 scale" if abs(scale - 1) > 1e-6
                              else "Normalised Intensity")
                else:
                    y_plot = y_arr * scale * _auto_norm
                    if active_exp:
                        ylabel = "Rel. Intensity"
                    else:
                        ylabel = "Rotatory Strength" if is_cd else "Oscillator Strength (f)"

                if self._show_sticks.get():
                    ml, sl, bl = self.ax.stem(
                        x_arr, y_plot, linefmt=colour, markerfmt="o", basefmt=" ",
                        label=name if multi_tddft else None
                    )
                    ml.set_markersize(st["stick_markersize"] if st.get("stick_markers", True) else 0)
                    ml.set_color(colour)
                    _plt.setp(sl, linewidth=st["stick_linewidth"],
                              alpha=st["stick_alpha"], color=colour)
                    self._inset_plot_data.append(
                        ("sticks", x_arr.copy(), y_plot.copy(), colour,
                         st["stick_linewidth"], st["stick_alpha"]))

                if self._show_env.get():
                    x_env, y_env = self._draw_envelope(ev_arr, y_plot, colour, multi_tddft, name)
                    if x_env is not None:
                        self._inset_plot_data.append(
                            ("line", x_env, y_env, colour,
                             st["env_linewidth"], "solid", 0.9))
                        if st["env_fill"]:
                            self._inset_plot_data.append(
                                ("fill", x_env, y_env, colour, st["env_fill_alpha"]))

                if self._show_trans.get() and sp.excited_states and not multi_tddft:
                    self._draw_transition_labels(x_arr, y_plot, sp)

                if not first_hover_set:
                    self._hover_x      = x_arr
                    self._hover_y      = y_plot
                    self._hover_ev     = ev_arr
                    self._hover_cm     = np.array(sp.energies_cm)
                    self._hover_states = list(sp.states)
                    self._hover_labels = list(sp.transition_labels)
                    first_hover_set = True

        # ── Draw experimental scans (right twin axis) ─────────────────────────
        if active_exp and self.ax2 is not None:
            for lbl, scan, colour, style in active_exp:
                ev_exp = scan.energy_ev
                mu_exp = scan.mu
                if len(ev_exp) == 0:
                    continue
                x_exp = self._ev_to_unit(ev_exp)
                ls    = style.get("linestyle", "solid")
                lw    = style.get("linewidth", 1.8)
                self.ax2.plot(x_exp, mu_exp, color=colour, linewidth=lw,
                              linestyle=ls, alpha=0.9, label=lbl)
                if style.get("fill", True):
                    fa = style.get("fill_alpha", 0.06)
                    self.ax2.fill_between(x_exp, 0, mu_exp, alpha=fa, color=colour)
                self._inset_plot_data.append(
                    ("exp", x_exp.copy(), mu_exp.copy(), colour, lw, ls, 0.9,
                     style.get("fill", True), style.get("fill_alpha", 0.06)))

            _left_lbl = (self._custom_left_ylabel.get().strip()
                         or "\u03bc(E) \u2014 normalized XAS")
            _show_left = self._show_left_ylabel.get()
            self.ax2.set_ylabel(
                _left_lbl if _show_left else "",
                fontsize=self._font_ylabel_size.get(),
                fontweight="bold" if self._font_ylabel_bold.get() else "normal",
                color="darkred")
            self.ax2.tick_params(
                axis="y",
                labelcolor="darkred" if _show_left else "none",
                labelleft=_show_left,
                labelright=False,
            )
            self.ax2.axhline(0, color="darkred", linewidth=0.4, alpha=0.3)

        # ── Align y=0 of both axes when experimental overlay is active ───────
        # Ensure both axes share the same zero position regardless of scale,
        # so TDDFT sticks and XAS curve both start from the same baseline.
        if active_exp and self.ax2 is not None:
            # Get the natural top of each axis after plotting
            _ax1_bot, _ax1_top = self.ax.get_ylim()
            _ax2_bot, _ax2_top = self.ax2.get_ylim()
            # Force bottom to 0 on both (sticks and XAS data never go below 0)
            _ax1_top = max(_ax1_top, 1e-6)
            _ax2_top = max(_ax2_top, 1e-6)
            self.ax.set_ylim(bottom=0, top=_ax1_top)
            self.ax2.set_ylim(bottom=0, top=_ax2_top)

        # ── Axes decoration ───────────────────────────────────────────────────
        self.ax.set_xlabel(
            self._xlabel(),
            fontsize=self._font_xlabel_size.get(),
            fontweight="bold" if self._font_xlabel_bold.get() else "normal")
        _right_lbl = self._custom_right_ylabel.get().strip() or ylabel
        self.ax.set_ylabel(
            _right_lbl if self._show_right_ylabel.get() else "",
            fontsize=self._font_ylabel_size.get(),
            fontweight="bold" if self._font_ylabel_bold.get() else "normal")

        # Visual axis placement: experimental labels on the left, TDDFT on the right.
        self.ax.yaxis.set_label_position("right")
        self.ax.yaxis.tick_right()
        self.ax.spines["right"].set_visible(True)
        self.ax.spines["left"].set_visible(False)
        if self.ax2 is not None:
            self.ax2.yaxis.set_label_position("left")
            self.ax2.yaxis.tick_left()
            self.ax2.spines["left"].set_visible(True)
            self.ax2.spines["right"].set_visible(False)

        title = self._custom_title.get().strip() or self._auto_title()
        self.ax.set_title(
            title,
            fontsize=self._font_title_size.get(),
            fontweight="bold" if self._font_title_bold.get() else "normal")

        # Tick label size (both axes) — visual right axis is TDDFT, left axis is experimental
        _show_right = self._show_right_ylabel.get()
        self.ax.tick_params(axis="both", labelsize=self._font_tick_size.get())
        self.ax.tick_params(
            axis="y",
            labelleft=False,
            labelright=_show_right,
            labelcolor="navy" if _show_right else "none",
        )
        if self.ax2 is not None:
            _show_left = self._show_left_ylabel.get()
            self.ax2.tick_params(
                axis="y",
                labelsize=self._font_tick_size.get(),
                labelcolor="darkred" if _show_left else "none",
                labelleft=_show_left,
                labelright=False,
            )
        self.ax.set_facecolor(self._bg_colour)
        self.fig.patch.set_facecolor(self._bg_colour)
        self.ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        if self._show_grid.get():
            self.ax.grid(True, alpha=0.25, linestyle=":", color="gray")
        else:
            self.ax.grid(False)

        if self._x_unit.get() == "nm" and len(self._hover_x):
            lo, hi = self._hover_x.min(), self._hover_x.max()
            if lo < hi:
                self.ax.set_xlim(hi * 1.02, lo * 0.98)

        # ── Apply manual axis-range overrides ─────────────────────────────────
        def _parse_float(s):
            try: return float(s.get().strip())
            except (ValueError, AttributeError): return None
        x_lo = _parse_float(self._xlim_lo)
        x_hi = _parse_float(self._xlim_hi)
        y_lo = _parse_float(self._ylim_lo)
        y_hi = _parse_float(self._ylim_hi)
        y_left_lo = _parse_float(self._yleft_lo)
        y_left_hi = _parse_float(self._yleft_hi)
        if x_lo is not None or x_hi is not None:
            cur = self.ax.get_xlim()
            self.ax.set_xlim(x_lo if x_lo is not None else cur[0],
                             x_hi if x_hi is not None else cur[1])
        if y_lo is not None or y_hi is not None:
            cur = self.ax.get_ylim()
            self.ax.set_ylim(y_lo if y_lo is not None else cur[0],
                             y_hi if y_hi is not None else cur[1])
        if self.ax2 is not None and (y_left_lo is not None or y_left_hi is not None):
            cur = self.ax2.get_ylim()
            self.ax2.set_ylim(y_left_lo if y_left_lo is not None else cur[0],
                              y_left_hi if y_left_hi is not None else cur[1])

        # Corner annotations for any active adjustments
        ann_lines = []
        if abs(delta_e) > 1e-4:
            ann_lines.append(f"\u0394E = {delta_e:+.2f} eV")
        if abs(scale - 1.0) > 1e-6:
            ann_lines.append(f"scale \u00d7{scale:.3g}")
        if ann_lines:
            self.ax.annotate(
                "\n".join(ann_lines),
                xy=(0.01, 0.97), xycoords="axes fraction",
                fontsize=8, color="gray", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
            )

        # ── Layout first so legend bbox coords are stable ─────────────────────
        self.fig.tight_layout(pad=2.5)

        # ── Legend (after tight_layout so position is correct) ────────────────
        # Disconnect all stale drag callbacks from previous draw
        for _attr in ("_legend_drag_cid", "_legend_drag_cid2", "_legend_drag_cid3"):
            _cid = getattr(self, _attr, None)
            if _cid is not None:
                try:
                    self.canvas.mpl_disconnect(_cid)
                except Exception:
                    pass
                setattr(self, _attr, None)
        self._legend_ref      = None
        self._legend_dragging = False

        if self._show_legend.get():
            # h1/l1 = TDDFT handles (left axis); h2/l2 = exp handles (right axis)
            h1, l1 = self.ax.get_legend_handles_labels()
            h2, l2 = (self.ax2.get_legend_handles_labels()
                      if active_exp and self.ax2 is not None else ([], []))
            # Filter primary TDDFT from h1/l1 when its toggle is off
            if self.spectrum and not self._show_primary_in_legend.get():
                _primary_lbl = (getattr(self.spectrum, "_custom_label", None)
                                or self.spectrum.display_name())
                _pairs = [(h, l) for h, l in zip(h1, l1) if l != _primary_lbl]
                h1 = [p[0] for p in _pairs]
                l1 = [p[1] for p in _pairs]
            # Honour "TDDFT in legend" sub-toggle
            if self._show_tddft_in_legend.get():
                all_h, all_l = h1 + h2, l1 + l2
            else:
                all_h, all_l = h2, l2
            if all_h:
                _leg_fs = self._font_legend_size.get()
                if self._legend_bbox is not None:
                    # Restore using figure-fraction coordinates so the position
                    # is stable across tight_layout / axes size changes.
                    leg = self.ax.legend(all_h, all_l, fontsize=_leg_fs,
                                         loc="upper left",
                                         bbox_to_anchor=self._legend_bbox,
                                         bbox_transform=self.fig.transFigure)
                else:
                    leg = self.ax.legend(all_h, all_l, fontsize=_leg_fs, loc="best")
                leg.set_draggable(True, update="bbox")
                self._legend_ref = leg
                # Connect drag-end callback once per draw.
                # We use motion_notify_event + button_release to detect a real
                # drag (as opposed to a plain click which must not move the saved pos).
                self._legend_dragging = False
                self._legend_drag_cid = self.canvas.mpl_connect(
                    "button_press_event",   self._on_legend_press)
                self._legend_drag_cid2 = self.canvas.mpl_connect(
                    "motion_notify_event",  self._on_legend_motion)
                self._legend_drag_cid3 = self.canvas.mpl_connect(
                    "button_release_event", self._on_legend_release)

        # ── Inset zoomed sub-plot ─────────────────────────────────────────────
        if self._inset_active and None not in self._inset_xlim:
            self._draw_inset()

        # Re-register axes with toolbar so zoom/pan/home work after every redraw
        self.toolbar.update()
        self.canvas.draw_idle()
        self._setup_inset_drag()
        self._setup_hover()
        self._update_comb_ui()
        self._notify_popouts()

    # ─────────────────────────────────────────────────────────────────────────
    def _draw_envelope(self, ev_arr, y_arr, colour, multi, name):
        """Draw broadened envelope on self.ax; returns (x_grid, env) for inset re-use."""
        fwhm_ev = self._fwhm_in_ev()
        if fwhm_ev <= 0:
            return None, None
        ev_min  = max(1e-3, ev_arr.min() - 4 * fwhm_ev)
        ev_max  = ev_arr.max() + 4 * fwhm_ev
        ev_grid = np.linspace(ev_min, ev_max, 2000)
        fn      = gaussian if self._broadening.get() == "Gaussian" else lorentzian
        env     = sum(y * fn(ev_grid, c, fwhm_ev) for c, y in zip(ev_arr, y_arr))
        x_grid  = self._ev_to_unit(ev_grid)
        label   = f"{name} (env)" if multi else "Envelope"
        st      = self._tddft_style

        self.ax.plot(x_grid, env, color=colour, linewidth=st["env_linewidth"],
                     alpha=0.9, label=label if multi else None)
        if st["env_fill"]:
            self.ax.fill_between(x_grid, 0, env,
                                 alpha=st["env_fill_alpha"], color=colour)
        return x_grid, env

    # ══════════════════════════════════════════════════════════════════════════
    #  Inset zoomed sub-plot
    # ══════════════════════════════════════════════════════════════════════════

    def _open_inset_dialog(self):
        """Dialog to configure (or remove) the inset zoomed sub-plot."""
        win = tk.Toplevel(self)
        win.title("Inset zoom settings")
        win.resizable(False, False)
        win.grab_set()

        unit = self._x_unit.get()

        frm = tk.Frame(win, padx=14, pady=10)
        frm.pack(fill=tk.BOTH)

        tk.Label(frm, text="Zoom region — X axis", font=("", 9, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 4))

        def _lbl(r, c, txt): tk.Label(frm, text=txt).grid(row=r, column=c, sticky="e", padx=4)

        _lbl(1, 0, f"X min ({unit}):")
        xlo_v = tk.StringVar(value="" if self._inset_xlim[0] is None else str(self._inset_xlim[0]))
        tk.Entry(frm, textvariable=xlo_v, width=12).grid(row=1, column=1, sticky="w")
        _lbl(1, 2, f"X max ({unit}):")
        xhi_v = tk.StringVar(value="" if self._inset_xlim[1] is None else str(self._inset_xlim[1]))
        tk.Entry(frm, textvariable=xhi_v, width=12).grid(row=1, column=3, sticky="w")

        tk.Label(frm, text="Zoom region — Y axis  (leave blank to auto-scale)",
                 font=("", 9, "bold")).grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(8, 4))
        _lbl(3, 0, "Y min:")
        ylo_v = tk.StringVar(value="" if self._inset_ylim[0] is None else str(self._inset_ylim[0]))
        tk.Entry(frm, textvariable=ylo_v, width=12).grid(row=3, column=1, sticky="w")
        _lbl(3, 2, "Y max:")
        yhi_v = tk.StringVar(value="" if self._inset_ylim[1] is None else str(self._inset_ylim[1]))
        tk.Entry(frm, textvariable=yhi_v, width=12).grid(row=3, column=3, sticky="w")

        tk.Label(frm, text="Inset size (% of plot area  —  1 % … 100 %)",
                 font=("", 9, "bold")).grid(
            row=4, column=0, columnspan=4, sticky="w", pady=(8, 4))

        # Width — slider + live numeric entry
        _lbl(5, 0, "Width %:")
        wv = tk.DoubleVar(value=round(self._inset_pos[2] * 100))
        w_entry_var = tk.StringVar(value=str(int(wv.get())))

        def _w_slider_moved(*_):
            w_entry_var.set(str(int(wv.get())))

        def _w_entry_committed(*_):
            try:
                v = float(w_entry_var.get())
                v = max(1.0, min(100.0, v))
                wv.set(v)
                w_entry_var.set(str(int(v)))
            except ValueError:
                w_entry_var.set(str(int(wv.get())))

        w_frame = tk.Frame(frm)
        w_frame.grid(row=5, column=1, sticky="w")
        tk.Scale(w_frame, variable=wv, from_=1, to=100, resolution=1,
                 orient=tk.HORIZONTAL, length=160,
                 command=_w_slider_moved).pack(side=tk.LEFT)
        we = tk.Entry(w_frame, textvariable=w_entry_var, width=5,
                      font=("Courier", 9))
        we.pack(side=tk.LEFT, padx=(4, 0))
        we.bind("<Return>",   _w_entry_committed)
        we.bind("<FocusOut>", _w_entry_committed)

        # Height — slider + live numeric entry
        _lbl(5, 2, "Height %:")
        hv = tk.DoubleVar(value=round(self._inset_pos[3] * 100))
        h_entry_var = tk.StringVar(value=str(int(hv.get())))

        def _h_slider_moved(*_):
            h_entry_var.set(str(int(hv.get())))

        def _h_entry_committed(*_):
            try:
                v = float(h_entry_var.get())
                v = max(1.0, min(100.0, v))
                hv.set(v)
                h_entry_var.set(str(int(v)))
            except ValueError:
                h_entry_var.set(str(int(hv.get())))

        h_frame = tk.Frame(frm)
        h_frame.grid(row=5, column=3, sticky="w")
        tk.Scale(h_frame, variable=hv, from_=1, to=100, resolution=1,
                 orient=tk.HORIZONTAL, length=160,
                 command=_h_slider_moved).pack(side=tk.LEFT)
        he = tk.Entry(h_frame, textvariable=h_entry_var, width=5,
                      font=("Courier", 9))
        he.pack(side=tk.LEFT, padx=(4, 0))
        he.bind("<Return>",   _h_entry_committed)
        he.bind("<FocusOut>", _h_entry_committed)

        def _parse(sv):
            try: return float(sv.get().strip())
            except ValueError: return None

        def apply():
            xl, xh = _parse(xlo_v), _parse(xhi_v)
            if xl is None or xh is None:
                messagebox.showerror("Inset", "Please fill in both X range fields.",
                                     parent=win)
                return
            # Y range is optional — leave blank for auto-scale
            yl, yh = _parse(ylo_v), _parse(yhi_v)
            self._inset_xlim = [xl, xh]
            self._inset_ylim = [yl, yh]   # None = auto
            self._inset_pos[2] = wv.get() / 100.0
            self._inset_pos[3] = hv.get() / 100.0
            # Clamp position so inset stays within the axes (handles 1-100 %)
            self._inset_pos[0] = float(np.clip(
                self._inset_pos[0], 0.0, max(0.0, 1.0 - self._inset_pos[2])))
            self._inset_pos[1] = float(np.clip(
                self._inset_pos[1], 0.0, max(0.0, 1.0 - self._inset_pos[3])))
            self._inset_active = True
            self._inset_btn.config(text="✎ Inset\u2026", fg="darkgreen")
            win.destroy()
            self._replot()

        def remove():
            self._inset_active = False
            self._inset_xlim = [None, None]
            self._inset_ylim = [None, None]
            self._inset_ax   = None
            self._inset_ax2  = None
            self._inset_btn.config(text="+ Add Inset\u2026", fg="darkgreen")
            win.destroy()
            self._replot()

        btn_row = tk.Frame(win, pady=8)
        btn_row.pack()
        tk.Button(btn_row, text="Apply",        width=10, command=apply).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="Remove Inset", width=12, command=remove).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="Cancel",       width=10, command=win.destroy).pack(side=tk.LEFT, padx=4)

        _centre_window(win, self)

    def _draw_inset(self):
        """Create the inset axes, re-draw plot data at zoom limits, add indicator."""
        xl, xh = self._inset_xlim
        yl, yh = self._inset_ylim   # either floats or None (auto)

        inset_ax = self.ax.inset_axes(self._inset_pos)
        self._inset_ax  = inset_ax
        self._inset_ax2 = None

        has_tddft = any(d[0] in ("sticks", "line", "fill")
                        for d in self._inset_plot_data)
        has_exp   = any(d[0] == "exp" for d in self._inset_plot_data)

        if has_exp:
            inset_ax2 = inset_ax.twinx()
            inset_ax2.set_navigate(False)
            self._inset_ax2 = inset_ax2
            inset_ax2.yaxis.set_label_position("left")
            inset_ax2.yaxis.tick_left()
            inset_ax2.spines["left"].set_visible(True)
            inset_ax2.spines["right"].set_visible(False)

        inset_ax.yaxis.set_label_position("right")
        inset_ax.yaxis.tick_right()
        inset_ax.spines["right"].set_visible(True)
        inset_ax.spines["left"].set_visible(False)

        # ── Plot data ────────────────────────────────────────────────────────
        for d in self._inset_plot_data:
            kind = d[0]
            if kind == "sticks":
                _, x, y, col, lw, alpha = d
                inset_ax.vlines(x, 0, y, colors=col, linewidth=lw, alpha=alpha)
            elif kind == "line":
                _, x, y, col, lw, ls, alpha = d
                inset_ax.plot(x, y, color=col, linewidth=lw, linestyle=ls, alpha=alpha)
            elif kind == "fill":
                _, x, y, col, alpha = d
                inset_ax.fill_between(x, 0, y, color=col, alpha=alpha)
            elif kind == "exp" and self._inset_ax2 is not None:
                _, x, y, col, lw, ls, alpha, do_fill, fa = d
                self._inset_ax2.plot(x, y, color=col, linewidth=lw,
                                     linestyle=ls, alpha=alpha)
                if do_fill:
                    self._inset_ax2.fill_between(x, 0, y, color=col, alpha=fa)

        # ── Compute auto y-range from experimental data inside the x-zoom ────
        # Needed when TDDFT is absent so indicate_inset_zoom uses the right scale.
        if has_exp and not has_tddft:
            exp_ys = []
            for d in self._inset_plot_data:
                if d[0] == "exp":
                    x, y = d[1], d[2]
                    mask = (x >= xl) & (x <= xh)
                    if mask.sum():
                        exp_ys.extend(y[mask].tolist())
            if exp_ys:
                auto_ylo = min(0.0, float(np.min(exp_ys)))
                auto_yhi = float(np.max(exp_ys)) * 1.08
            else:
                auto_ylo, auto_yhi = 0.0, 1.1
            # Override user-supplied Y (which was for TDDFT) with the exp range
            yl, yh = auto_ylo, auto_yhi

        # ── Apply zoom limits ────────────────────────────────────────────────
        inset_ax.set_xlim(xl, xh)
        if yl is not None and yh is not None:
            inset_ax.set_ylim(yl, yh)
        # else: leave inset_ax y auto-scaled to whatever TDDFT data is there

        if self._inset_ax2 is not None:
            self._inset_ax2.set_xlim(xl, xh)
            if not has_tddft and yl is not None:
                # No TDDFT: sync right axis to the same exp scale
                self._inset_ax2.set_ylim(yl, yh)
            elif has_tddft:
                # ── Zero-alignment across twin axes ──────────────────────────
                # After the TDDFT ylim is settled, compute where y=0 sits as a
                # fraction of the left axis height, then force the right (exp)
                # axis to the same fraction so both zeros land on the same pixel.
                tddft_yl, tddft_yh = inset_ax.get_ylim()
                tddft_span = tddft_yh - tddft_yl
                # Fraction from bottom at which zero sits on the left axis
                zero_frac = ((-tddft_yl) / tddft_span) if tddft_span > 0 else 0.0
                zero_frac = float(np.clip(zero_frac, 0.0, 1.0))

                # Determine the natural top of the exp data in the x-window
                exp_ys_in = []
                for _d in self._inset_plot_data:
                    if _d[0] == "exp":
                        _x, _y = _d[1], _d[2]
                        _m = (_x >= xl) & (_x <= xh)
                        if _m.sum():
                            exp_ys_in.extend(_y[_m].tolist())

                if exp_ys_in:
                    exp_top = float(np.max(exp_ys_in)) * 1.08
                    exp_top = max(exp_top, 0.1)   # guard against flat-zero data
                else:
                    exp_top = 1.1

                # Solve: (0 - exp_bot) / (exp_top - exp_bot) = zero_frac
                #   => exp_bot = zero_frac * exp_top / (zero_frac - 1)
                if zero_frac < 1.0:
                    exp_bot = zero_frac * exp_top / (zero_frac - 1.0)
                else:
                    exp_bot = -exp_top  # degenerate: zero at very top
                self._inset_ax2.set_ylim(exp_bot, exp_top)

        # ── Label / tick visibility ──────────────────────────────────────────
        show_lbl = self._inset_show_labels.get()
        # Left axis: hide entirely when there's no TDDFT data on it
        if not has_tddft:
            inset_ax.yaxis.set_visible(False)
        else:
            inset_ax.tick_params(labelsize=7, labelleft=False, labelright=show_lbl)
        inset_ax.tick_params(labelsize=7, labelbottom=show_lbl)

        if self._inset_ax2 is not None:
            self._inset_ax2.tick_params(
                axis="y", labelsize=6, labelcolor="darkred",
                labelleft=show_lbl, labelright=False)

        inset_ax.set_facecolor(self._bg_colour)

        # ── Indicator rectangle on the main axes ─────────────────────────────
        # We draw this manually rather than using indicate_inset_zoom() because
        # that function only understands a single axis coordinate system.  When
        # exp data lives on ax2 (right axis) but TDDFT lives on ax (left axis),
        # indicate_inset_zoom(inset_ax) draws the box in LEFT-axis (oscillator-
        # strength) coordinates — so the box appears at totally the wrong y
        # position (e.g. a 0–0.04 sliver at the bottom instead of the
        # 0.4–1.1 XAS range).
        #
        # Rule:
        #   • exp data present  → draw on self.ax2 using the XAS y-range of the
        #                          zoom window so the box overlaps the actual curves
        #   • TDDFT only        → draw on self.ax using the TDDFT y-range
        from matplotlib.patches import Rectangle as _Rect, ConnectionPatch as _ConPatch

        if has_exp and self.ax2 is not None:
            _ref_ax = self.ax2
            # Y-range: actual exp values inside [xl, xh] on the main plot ax2
            _ind_ys = []
            for _d in self._inset_plot_data:
                if _d[0] == "exp":
                    _dx, _dy = _d[1], _d[2]
                    _m = (_dx >= xl) & (_dx <= xh)
                    if _m.sum():
                        _ind_ys.extend(_dy[_m].tolist())
            if _ind_ys:
                _rect_yl = min(0.0, float(np.min(_ind_ys)))
                _rect_yh = float(np.max(_ind_ys)) * 1.08
                _rect_yh = max(_rect_yh, _rect_yl + 0.05)
            else:
                _rect_yl, _rect_yh = 0.0, 1.1
        else:
            _ref_ax  = self.ax
            _rect_yl = yl  if yl is not None else self.ax.get_ylim()[0]
            _rect_yh = yh  if yh is not None else self.ax.get_ylim()[1]

        _rect = _Rect(
            (xl, _rect_yl), xh - xl, _rect_yh - _rect_yl,
            linewidth=1.2, edgecolor="black", facecolor="none",
            linestyle="--", alpha=0.8, transform=_ref_ax.transData, zorder=5,
        )
        _ref_ax.add_patch(_rect)

        # Connector lines: two ConnectionPatch objects linking facing corners of
        # the indicator rectangle (in _ref_ax data coords) to the inset axes
        # corners (in inset_ax axes-fraction coords).
        #
        # ConnectionPatch stores coordinates in axis-relative terms and
        # recomputes screen positions on every render — so it works correctly
        # after figure resize AND after pickle-copy to the pop-out window.
        _connectors = []
        try:
            # Determine whether the inset is to the left or right of the
            # indicator rectangle by comparing their centres in axes-fraction.
            _xl_lim, _xr_lim = _ref_ax.get_xlim()
            _x_span = (_xr_lim - _xl_lim) if _xr_lim != _xl_lim else 1.0
            _rect_cx_frac = ((xl + xh) / 2.0 - _xl_lim) / _x_span
            _ins_cx_frac  = self._inset_pos[0] + self._inset_pos[2] / 2.0

            if _ins_cx_frac >= _rect_cx_frac:
                # Inset to the RIGHT → connect right edge of rect to left edge of inset
                _pairs = [
                    ((xh, _rect_yh), (0.0, 1.0)),   # rect top-right    → inset top-left
                    ((xh, _rect_yl), (0.0, 0.0)),   # rect bottom-right → inset bottom-left
                ]
            else:
                # Inset to the LEFT → connect left edge of rect to right edge of inset
                _pairs = [
                    ((xl, _rect_yh), (1.0, 1.0)),   # rect top-left    → inset top-right
                    ((xl, _rect_yl), (1.0, 0.0)),   # rect bottom-left → inset bottom-right
                ]

            for _xyA, _xyB in _pairs:
                _con = _ConPatch(
                    xyA=_xyA, coordsA="data",           axesA=_ref_ax,
                    xyB=_xyB, coordsB="axes fraction",  axesB=inset_ax,
                    color="black", linewidth=0.8,
                    alpha=0.6, linestyle="--",
                    arrowstyle="-",
                )
                self.fig.add_artist(_con)
                _connectors.append(_con)
        except Exception:
            pass   # connectors are cosmetic — skip if anything goes wrong

        self._inset_indicator = (_rect, _connectors)

    # ── Inset drag support ────────────────────────────────────────────────────

    def _setup_inset_drag(self):
        """Connect mouse events for dragging the inset."""
        for cid in self._inset_drag_cids:
            try: self.canvas.mpl_disconnect(cid)
            except Exception: pass
        self._inset_drag_cids = []
        if not self._inset_active:
            return
        self._inset_drag_cids = [
            self.canvas.mpl_connect("button_press_event",   self._inset_on_press),
            self.canvas.mpl_connect("motion_notify_event",  self._inset_on_motion),
            self.canvas.mpl_connect("button_release_event", self._inset_on_release),
        ]

    def _inset_on_press(self, event):
        # Use pixel bbox hit-testing — event.inaxes is unreliable for inset axes
        # in the TkAgg backend because the axes locator isn't registered the same way.
        if self._inset_ax is None or event.x is None or event.y is None:
            return
        try:
            bbox = self._inset_ax.get_window_extent()
        except Exception:
            return
        if not bbox.contains(event.x, event.y):
            return
        self._inset_drag = {
            "x0":   event.x,
            "y0":   event.y,
            "pos0": list(self._inset_pos),
        }

    def _inset_on_motion(self, event):
        if self._inset_drag is None or event.x is None:
            return
        # Skip if toolbar is active (zoom/pan mode)
        if self.toolbar.mode != "":
            return
        ax_bb = self.ax.get_window_extent()
        if ax_bb.width == 0 or ax_bb.height == 0:
            return
        dx = (event.x - self._inset_drag["x0"]) / ax_bb.width
        dy = (event.y - self._inset_drag["y0"]) / ax_bb.height
        p0 = self._inset_drag["pos0"]
        new_l = float(np.clip(p0[0] + dx, 0.0, 1.0 - p0[2]))
        new_b = float(np.clip(p0[1] + dy, 0.0, 1.0 - p0[3]))
        # Only redraw if position changed by at least 0.5% to avoid thrashing
        if (abs(new_l - self._inset_pos[0]) > 0.005 or
                abs(new_b - self._inset_pos[1]) > 0.005):
            self._inset_pos[0] = new_l
            self._inset_pos[1] = new_b
            self._quick_move_inset()

    def _inset_on_release(self, event):
        self._inset_drag = None   # position already saved; full replot not needed

    def _quick_move_inset(self):
        """Remove old inset artists and re-draw at updated _inset_pos (no full replot)."""
        # Remove old inset axes
        for ax in [self._inset_ax, self._inset_ax2]:
            if ax is not None:
                try: ax.remove()
                except Exception: pass
        self._inset_ax  = None
        self._inset_ax2 = None
        # Remove the stored indicator (rectangle + connector arrows) by direct reference.
        # This is the only reliable way — attribute-scanning misses FancyArrowPatch objects.
        if self._inset_indicator is not None:
            rect, connectors = self._inset_indicator
            try: rect.remove()
            except Exception: pass
            for c in (connectors or []):
                try: c.remove()
                except Exception: pass
            self._inset_indicator = None
        self._draw_inset()
        self.canvas.draw_idle()

    def _draw_transition_labels(self, x_arr, y_arr, sp: TDDFTSpectrum):
        for i, state in enumerate(sp.excited_states):
            if i >= len(x_arr):
                break
            if not state.transitions:
                continue
            dom = max(state.transitions, key=lambda t: t[2])
            self.ax.annotate(
                f"{dom[0]}\u2192{dom[1]} ({dom[2]:.2f})",
                xy=(x_arr[i], y_arr[i]),
                xytext=(0, 8), textcoords="offset points",
                fontsize=6, ha="center", color="gray", rotation=70
            )

    # ══════════════════════════════════════════════════════════════════════════
    #  Hover tooltip
    # ══════════════════════════════════════════════════════════════════════════
    def _setup_hover(self):
        if self._hover_cid is not None:
            try:
                self.canvas.mpl_disconnect(self._hover_cid)
            except Exception:
                pass
        if self._annot is not None:
            try:
                self._annot.remove()
            except Exception:
                pass
        self._annot = self.ax.annotate(
            "", xy=(0, 0), xytext=(18, 60), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.92),
            arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.2,
                            connectionstyle="arc3,rad=0.15"),
            fontsize=8, visible=False
        )
        self._hover_cid = self.canvas.mpl_connect("motion_notify_event", self._on_hover)

    def _on_hover(self, event):
        # Never interfere while the toolbar is in zoom-rect or pan mode.
        # Our draw_idle() call corrupts the rubber-band overlay if we fire during a drag.
        if self.toolbar.mode != "":
            return
        valid = {self.ax} | ({self.ax2} if self.ax2 is not None else set())
        if event.inaxes not in valid or len(self._hover_x) == 0:
            if self._annot and self._annot.get_visible():
                self._annot.set_visible(False)
                self.canvas.draw_idle()
            return

        dists = np.abs(self._hover_x - event.xdata)
        idx   = int(np.argmin(dists))
        rng   = (self._hover_x.max() - self._hover_x.min()) if len(self._hover_x) > 1 else 1
        if rng > 0 and dists[idx] > rng * 0.05:
            self._annot.set_visible(False)
            self.canvas.draw_idle()
            return

        ev  = self._hover_ev[idx]
        cm  = self._hover_cm[idx] if idx < len(self._hover_cm) else ev * 8065.54
        nm  = 1239.84 / ev if ev > 0 else 0
        fy  = self._hover_y[idx]
        lbl = "R" if (self.spectrum and self.spectrum.is_cd()) else "f"

        delta_e = self._delta_e.get()

        # State identifier: use actual root number + transition label if available
        if self._hover_states and idx < len(self._hover_states):
            root = self._hover_states[idx]
            if self._hover_labels and idx < len(self._hover_labels):
                tip = [f"State {root}  \u2014  {self._hover_labels[idx]}"]
            else:
                tip = [f"State {root}"]
        else:
            tip = [f"State {idx + 1}"]

        if abs(delta_e) > 1e-4:
            tip.append(f"Shifted: {ev:.4f} eV  (orig. {ev - delta_e:.4f} eV)")
        else:
            tip.append(f"{ev:.4f} eV  |  {nm:.2f} nm  |  {cm:.0f} cm\u207b\u00b9")
        tip.append(f"{lbl} = {fy:.6f}")

        if (self.spectrum and self.spectrum.excited_states and
                idx < len(self.spectrum.excited_states)):
            es = self.spectrum.excited_states[idx]
            if not es.transitions:
                tip.append("\u2500" * 28)
                tip.append("(no CI vectors in output file)")
            if es.transitions:
                # Sort all contributions by weight (c²) descending.
                # Show every transition ≥ 1%, or at minimum the top 1.
                # c² is stored directly in the tuple (captured before "(c=" in the regex).
                sorted_t = sorted(es.transitions, key=lambda t: t[2], reverse=True)
                threshold = 0.01   # 1 %
                shown = [t for t in sorted_t if t[2] >= threshold]
                if not shown:
                    shown = sorted_t[:1]   # always show at least the dominant one
                shown = shown[:8]          # cap at 8 lines to keep tooltip readable
                tip.append("\u2500" * 28)
                tip.append("Acceptor orbitals (MO\u1d47 \u2192 MO\u1d43):")
                for fr, to, w2 in shown:
                    bar = "\u2588" * int(w2 * 20 + 0.5)   # mini bar up to 20 chars
                    # Display MOs counting from 0 (ORCA stores 1-based indices)
                    tip.append(f"  {fr-1:3d} \u2192 {to-1:3d}   {w2*100:5.1f}%  {bar}")
                if len(sorted_t) > len(shown):
                    rest_w = sum(t[2] for t in sorted_t[len(shown):])
                    tip.append(f"  ... {len(sorted_t)-len(shown)} more  ({rest_w*100:.1f}% total)")

        self._annot.set_text("\n".join(tip))
        self._annot.xy = (self._hover_x[idx], self._hover_y[idx])
        self._annot.set_visible(True)
        self.canvas.draw_idle()

    def _toggle_legend(self):
        """Called when the Legend checkbox is clicked.
        Resets the saved position so the legend returns to 'best' on next show."""
        self._legend_bbox = None
        self._replot()

    def _on_legend_press(self, event):
        """Record that a mouse button was pressed (drag detection start)."""
        self._legend_dragging = False   # reset; motion will set to True

    def _on_legend_motion(self, event):
        """Mark that a drag is in progress (mouse moved while button held)."""
        if event.button is not None:
            self._legend_dragging = True

    def _on_legend_release(self, event):
        """Save the legend's upper-left corner in figure-fraction coordinates
        — but ONLY if a real drag happened (not a plain click)."""
        if not self._legend_dragging:
            return
        self._legend_dragging = False
        if self._legend_ref is None:
            return
        try:
            # get_window_extent() → display-pixel Bbox of the legend frame
            bb  = self._legend_ref.get_window_extent()
            # Convert the UPPER-LEFT corner to figure-fraction.
            # We consistently restore with loc="upper left" so we always
            # anchor the top-left — no drift across successive drags.
            inv       = self.fig.transFigure.inverted()
            x_fig, y_fig = inv.transform((bb.x0, bb.y1))
            self._legend_bbox = (x_fig, y_fig)
        except Exception:
            pass   # silently ignore if legend was removed mid-drag

    def _update_comb_ui(self):
        """Show the component checkboxes only when a combined spectrum is loaded
        in single-spectrum view (not overlay mode)."""
        if self._comb_frame is None:
            return
        is_comb = (
            self.spectrum is not None
            and self.spectrum.is_combined()
            and not self._overlay_mode.get()
        )
        if is_comb:
            self._comb_frame.pack(side=tk.LEFT)
        else:
            self._comb_frame.pack_forget()

    # ══════════════════════════════════════════════════════════════════════════
    #  Export
    # ══════════════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════════════
    #  Pop-out graph window
    # ══════════════════════════════════════════════════════════════════════════
    def _notify_popouts(self):
        """Call every registered pop-out refresh callback (skip dead windows)."""
        if not self._popout_callbacks:
            return
        alive = []
        for cb in self._popout_callbacks:
            try:
                cb()
                alive.append(cb)
            except Exception:
                pass  # window was already destroyed — silently drop
        self._popout_callbacks = alive

    def _pop_out_graph(self):
        """Open the current graph in a standalone, resizable window.

        The pop-out is a deep copy of the live figure, so it is independent of
        the embedded panel — you can zoom/pan/save it without affecting the
        main view.  The figure auto-resizes as you drag the window border.
        """
        import pickle
        import io as _io

        # Make sure the figure is fully up to date before copying
        self._replot()

        # ── Helper: deep-copy the current live figure ─────────────────────────
        def _copy_fig():
            buf = _io.BytesIO()
            pickle.dump(self.fig, buf)
            buf.seek(0)
            f = pickle.load(buf)
            f.set_size_inches(12, 7)
            try:
                f.set_layout_engine("tight")
            except AttributeError:
                f.tight_layout()
            return f

        try:
            fig_copy = _copy_fig()
        except Exception as exc:
            messagebox.showerror(
                "Pop Out — Error",
                f"Could not copy the figure:\n{exc}\n\n"
                "Try saving to SVG/PDF as a workaround.",
            )
            return

        # Mutable reference so save always uses the latest figure object
        fig_ref = [fig_copy]

        # ── Build the Toplevel ────────────────────────────────────────────────
        win = tk.Toplevel(self)
        win.title("Graph — Pop Out  (resize freely)")
        win.geometry("1200x760")
        win.resizable(True, True)
        win.minsize(600, 400)

        # ── Top status / control bar ──────────────────────────────────────────
        top_bar = tk.Frame(win, bd=1, relief=tk.SUNKEN,
                           padx=6, pady=3, bg="#e8eef8")
        top_bar.pack(side=tk.TOP, fill=tk.X)

        auto_update_var = tk.BooleanVar(value=True)

        tk.Label(top_bar, text="Pop-Out Controls:", font=("", 9, "bold"),
                 bg="#e8eef8").pack(side=tk.LEFT, padx=(0, 8))

        tk.Checkbutton(
            top_bar, text="Auto Update  (syncs with main panel on every replot)",
            variable=auto_update_var, bg="#e8eef8", font=("", 9),
            fg="#003399",
        ).pack(side=tk.LEFT, padx=4)

        def _do_refresh():
            """Re-copy the live figure into this pop-out canvas."""
            try:
                new_fig = _copy_fig()
                canvas_po.figure = new_fig
                fig_ref[0] = new_fig
                toolbar_po.update()
                canvas_po.draw_idle()
            except Exception as exc:
                messagebox.showerror("Refresh Error", str(exc), parent=win)

        tk.Button(top_bar, text="\u21ba Refresh Now", command=_do_refresh,
                  font=("", 9, "bold"), bg="#cce0ff",
                  relief=tk.RAISED).pack(side=tk.LEFT, padx=8)

        tk.Label(top_bar,
                 text="Toolbar below: zoom \u00b7 pan \u00b7 home \u00b7 save",
                 font=("", 8), fg="gray", bg="#e8eef8").pack(side=tk.RIGHT, padx=6)

        # Embed the copied figure
        canvas_po = FigureCanvasTkAgg(fig_copy, master=win)
        canvas_po.draw()
        cw = canvas_po.get_tk_widget()
        cw.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Matplotlib navigation toolbar (zoom, pan, home, back/forward, save)
        tb_frame = tk.Frame(win, bd=1, relief=tk.FLAT)
        tb_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar_po = NavigationToolbar2Tk(canvas_po, tb_frame)
        toolbar_po.update()

        # ── Bottom button row ─────────────────────────────────────────────────
        btn_row = tk.Frame(win, bd=1, relief=tk.SUNKEN, padx=6, pady=4)
        btn_row.pack(side=tk.BOTTOM, fill=tk.X)

        def _save_popout():
            path = filedialog.asksaveasfilename(
                defaultextension=".svg",
                filetypes=[
                    ("SVG — vector, best for Word/Illustrator", "*.svg"),
                    ("PDF — vector, best for LaTeX / Acrobat",  "*.pdf"),
                    ("EPS — vector, journals / Illustrator",    "*.eps"),
                    ("PNG — raster 600 dpi, for presentations", "*.png"),
                    ("All files", "*.*"),
                ],
                title="Save Pop-Out Figure",
                parent=win,
            )
            if not path:
                return
            ext  = _os.path.splitext(path)[1].lower()
            fig_cur = fig_ref[0]
            skw  = dict(bbox_inches="tight", facecolor=fig_cur.get_facecolor())
            if ext == ".png":
                skw["dpi"] = 600
            elif ext in (".pdf", ".eps"):
                skw["backend"] = "pdf" if ext == ".pdf" else "ps"
            try:
                fig_cur.savefig(path, **skw)
                messagebox.showinfo(
                    "Saved", f"Figure saved to:\n{path}", parent=win)
            except Exception as exc:
                messagebox.showerror(
                    "Save Error", f"Could not save:\n{exc}", parent=win)

        tk.Button(btn_row, text="Save Figure\u2026", command=_save_popout,
                  font=("", 9), relief=tk.RAISED).pack(side=tk.LEFT, padx=4)

        tk.Button(btn_row, text="Close", command=lambda: _on_close(),
                  font=("", 9)).pack(side=tk.RIGHT, padx=4)

        # ── Auto-update callback registered with the main widget ──────────────
        def _auto_refresh():
            """Called by _notify_popouts() after every main _replot()."""
            if auto_update_var.get():
                _do_refresh()

        self._popout_callbacks.append(_auto_refresh)

        def _on_close():
            try:
                self._popout_callbacks.remove(_auto_refresh)
            except ValueError:
                pass
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)

        # ── Auto-tight on window resize ───────────────────────────────────────
        def _on_win_resize(event):
            if event.widget is win:
                try:
                    fig_ref[0].tight_layout()
                    canvas_po.draw_idle()
                except Exception:
                    pass

        win.bind("<Configure>", _on_win_resize)

        # Focus the new window so it's on top
        win.focus_force()

    def _save_figure(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".svg",
            filetypes=[
                ("SVG — vector, best for Word/Illustrator", "*.svg"),
                ("PDF — vector, best for LaTeX / Acrobat",  "*.pdf"),
                ("EPS — vector, journals / Illustrator",    "*.eps"),
                ("PNG — raster 600 dpi, for presentations", "*.png"),
                ("All files", "*.*"),
            ],
            title="Save Figure  (SVG / PDF / EPS = vector; PNG = high-res raster)",
        )
        if not path:
            return

        ext = _os.path.splitext(path)[1].lower()

        # Vector formats: no dpi needed — matplotlib writes true vector paths.
        # PNG: use 600 dpi so it's still sharp when resized in a document.
        save_kw = dict(bbox_inches="tight", facecolor=self.fig.get_facecolor())
        if ext == ".png":
            save_kw["dpi"] = 600
        elif ext in (".pdf", ".eps"):
            # Embed fonts so the file is self-contained
            save_kw["backend"] = "pdf" if ext == ".pdf" else "ps"

        try:
            self.fig.savefig(path, **save_kw)
        except Exception as exc:
            messagebox.showerror("Save Error", f"Could not save figure:\n{exc}")
            return

        fmt_hint = {
            ".svg": "SVG vector — insert into Word via Insert → Pictures → SVG.",
            ".pdf": "PDF vector — drag into Illustrator / LaTeX \\includegraphics.",
            ".eps": "EPS vector — use with Illustrator or LaTeX.",
            ".png": "PNG 600 dpi — paste directly into Word or PowerPoint.",
        }.get(ext, "")
        messagebox.showinfo("Figure Saved",
                            f"Saved to:\n{path}\n\n{fmt_hint}")

    def _export_csv(self):
        sp = self.spectrum
        if not sp:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("All","*.*")],
            title="Export CSV"
        )
        if not path:
            return
        is_cd   = sp.is_cd()
        is_comb = bool(sp.fosc_d2)
        delta_e = self._delta_e.get()
        with open(path, "w") as f:
            if is_cd:
                hdr = "State,Label,Energy_eV,Energy_eV_shifted,Energy_cm,Wavelength_nm,RotatoryStrength"
            elif is_comb:
                hdr = "State,Label,Energy_eV,Energy_eV_shifted,Energy_cm,Wavelength_nm,Fosc_Total,Fosc_D2,Fosc_m2,Fosc_Q2"
            else:
                hdr = "State,Label,Energy_eV,Energy_eV_shifted,Energy_cm,Wavelength_nm,Fosc"
            f.write(hdr + "\n")
            n = len(sp.states)
            for i in range(n):
                lbl  = sp.transition_labels[i] if i < len(sp.transition_labels) else str(sp.states[i])
                ev   = sp.energies_ev[i]    if i < len(sp.energies_ev)    else 0
                cm   = sp.energies_cm[i]    if i < len(sp.energies_cm)    else 0
                nm   = sp.wavelengths_nm[i] if i < len(sp.wavelengths_nm) else 0
                ev_s = ev + delta_e
                if is_cd:
                    val = sp.rotatory_strength[i] if i < len(sp.rotatory_strength) else 0
                    row = f"{sp.states[i]},{lbl},{ev:.6f},{ev_s:.6f},{cm:.2f},{nm:.4f},{val:.8f}"
                elif is_comb:
                    ft  = sp.fosc[i]    if i < len(sp.fosc)    else 0
                    fd2 = sp.fosc_d2[i] if i < len(sp.fosc_d2) else 0
                    fm2 = sp.fosc_m2[i] if i < len(sp.fosc_m2) else 0
                    fq2 = sp.fosc_q2[i] if i < len(sp.fosc_q2) else 0
                    row = f"{sp.states[i]},{lbl},{ev:.6f},{ev_s:.6f},{cm:.2f},{nm:.4f},{ft:.8f},{fd2:.8f},{fm2:.8e},{fq2:.8e}"
                else:
                    val = sp.fosc[i] if i < len(sp.fosc) else 0
                    row = f"{sp.states[i]},{lbl},{ev:.6f},{ev_s:.6f},{cm:.2f},{nm:.4f},{val:.8f}"
                f.write(row + "\n")
        messagebox.showinfo("Exported", f"Data saved to:\n{path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sep(parent):
    ttk.Separator(parent, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)


def _slider_row(parent, label: str, var: tk.DoubleVar,
                lo: float, hi: float, res: float, row: int):
    """One label + slider + live readout, laid out in a grid."""
    tk.Label(parent, text=label, font=("", 9)).grid(
        row=row, column=0, sticky="e", padx=(0, 4), pady=2)
    sl = tk.Scale(parent, from_=lo, to=hi, resolution=res,
                  orient=tk.HORIZONTAL, length=180,
                  variable=var, showvalue=False)
    sl.grid(row=row, column=1, sticky="w")
    # Live numeric label
    lbl = tk.Label(parent, textvariable=tk.StringVar(), width=6,
                   font=("Courier", 9), anchor="w")
    lbl.grid(row=row, column=2, sticky="w", padx=(4, 0))

    def _update_lbl(*_):
        v = var.get()
        lbl.config(text=f"{v:.2f}" if res < 1 else f"{int(v)}")
    var.trace_add("write", _update_lbl)
    _update_lbl()


def _centre_window(win: tk.Toplevel, parent: tk.Widget):
    win.update_idletasks()
    px = parent.winfo_rootx() + (parent.winfo_width()  - win.winfo_width())  // 2
    py = parent.winfo_rooty() + (parent.winfo_height() - win.winfo_height()) // 2
    win.geometry(f"+{px}+{py}")
