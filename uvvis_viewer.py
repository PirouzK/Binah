"""
UV-Vis TDDFT Viewer
Standalone viewer for ORCA TDDFT UV-Vis spectra.

Features:
  - Singlet excited states from the Electric Dipole absorption table
    (with real oscillator strengths and Gaussian/Lorentzian broadening)
  - Triplet state positions shown as sticks at user-adjustable height
  - Dual x-axis: wavelength (nm) AND wavenumber (cm⁻¹) simultaneously
    via matplotlib secondary_xaxis
  - Flexible excitation labels: state number, nm, cm⁻¹, eV, f-value,
    dominant MO transition, or combinations thereof
  - Export figure (PNG/PDF/SVG) and data (CSV)
  - Inspired by Binah (ORCA TDDFT XAS Viewer)

Supported input: ORCA 4/5 UV/Vis output format
  (TD-DFT/TDA EXCITED STATES, Electric Dipole table)

Run with:  python uvvis_viewer.py
"""

import os
import re
import sys

import numpy as np

try:
    import tkinter as tk
except ImportError:
    print(
        "\nERROR: tkinter is not installed.\n"
        "  Windows : Reinstall Python → Custom → check 'tcl/tk and IDLE'\n"
        "  Linux   : sudo apt install python3-tk\n",
        file=sys.stderr,
    )
    sys.exit(1)

from tkinter import ttk, filedialog, messagebox, colorchooser
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as _plt


# ══════════════════════════════════════════════════════════════════════════════
#  Physical constants
# ══════════════════════════════════════════════════════════════════════════════

CM_TO_EV = 1.23984e-4    # 1 cm⁻¹ in eV
EV_TO_CM = 8065.54       # 1 eV in cm⁻¹
EV_TO_NM = 1239.84       # hc in eV·nm  (E[eV] × λ[nm] = 1239.84)

# ── Excitation label modes ────────────────────────────────────────────────────
LBL_NONE       = "None"
LBL_STATE      = "State # (S1, T1 …)"
LBL_NM         = "Wavelength (nm)"
LBL_CM         = "Wavenumber (cm⁻¹)"
LBL_EV         = "Energy (eV)"
LBL_FOSC       = "Osc. strength (f)"
LBL_MO         = "Dominant MO"
LBL_STATE_NM   = "State + nm"
LBL_STATE_FOSC = "State + f"
LBL_STATE_MO   = "State + MO"
LBL_FULL       = "Full (state, nm, f)"

ALL_LABEL_MODES = [
    LBL_NONE, LBL_STATE, LBL_NM, LBL_CM, LBL_EV,
    LBL_FOSC, LBL_MO, LBL_STATE_NM, LBL_STATE_FOSC,
    LBL_STATE_MO, LBL_FULL,
]

# ── Default colours ───────────────────────────────────────────────────────────
SINGLET_COL = "#1f77b4"   # matplotlib default blue
TRIPLET_COL = "#d62728"   # matplotlib default red


# ══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MoTransition:
    from_mo: int
    to_mo:   int
    weight:  float   # c² coefficient


@dataclass
class ExcState:
    """One TDDFT excited state with MO transition character."""
    index:     int
    energy_ev: float
    energy_cm: float
    spin:      str    # "S" (singlet) or "T" (triplet)
    transitions: List[MoTransition] = field(default_factory=list)

    @property
    def wavelength_nm(self) -> float:
        return EV_TO_NM / self.energy_ev if self.energy_ev > 0 else 0.0

    def dominant(self) -> Optional[MoTransition]:
        if not self.transitions:
            return None
        return max(self.transitions, key=lambda t: t.weight)


@dataclass
class UVVisData:
    """Complete parsed UV-Vis TDDFT dataset from one ORCA output file."""
    filename: str = ""

    # Singlet states — from the Electric Dipole absorption table
    singlet_idx:  List[int]   = field(default_factory=list)
    singlet_cm:   List[float] = field(default_factory=list)
    singlet_nm:   List[float] = field(default_factory=list)
    singlet_ev:   List[float] = field(default_factory=list)
    singlet_fosc: List[float] = field(default_factory=list)

    # Triplet states — spin-forbidden from S₀, no oscillator strength
    triplet_idx:  List[int]   = field(default_factory=list)
    triplet_cm:   List[float] = field(default_factory=list)
    triplet_nm:   List[float] = field(default_factory=list)
    triplet_ev:   List[float] = field(default_factory=list)

    # Detailed MO transition data (for labels)
    singlet_xs: List[ExcState] = field(default_factory=list)
    triplet_xs: List[ExcState] = field(default_factory=list)

    @property
    def has_singlets(self) -> bool:
        return bool(self.singlet_cm)

    @property
    def has_triplets(self) -> bool:
        return bool(self.triplet_cm)

    @property
    def n_singlets(self) -> int:
        return len(self.singlet_cm)

    @property
    def n_triplets(self) -> int:
        return len(self.triplet_cm)


# ══════════════════════════════════════════════════════════════════════════════
#  Parser
# ══════════════════════════════════════════════════════════════════════════════

class UVVisParser:
    """
    Parse ORCA UV-Vis TDDFT output: singlet absorption table + triplet states.

    Singlet energies and fosc come from the Electric Dipole absorption table.
    Triplet energies come from the TD-DFT/TDA EXCITED STATES (TRIPLETS) block.
    MO transition details come from both excited-state blocks.
    """

    # Electric Dipole absorption table header
    _EDIP  = re.compile(
        r"ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS", re.I)
    # Separator line (10+ dashes)
    _SEP   = re.compile(r"^\s*-{10,}")
    # UV-style data row: State  cm-1  nm  fosc  [extra cols ignored]
    _UVROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([-\d.eE+]+)")
    # Excited-states block headers
    _SHDR  = re.compile(r"TD-DFT.*EXCITED\s+STATES\s*\(SINGLETS\)", re.I)
    _THDR  = re.compile(r"TD-DFT.*EXCITED\s+STATES\s*\(TRIPLETS\)", re.I)
    # Individual state header line
    _STHDR = re.compile(
        r"STATE\s+(\d+):\s+E=\s*[\d.]+\s+au\s+([\d.]+)\s+eV\s+([\d.]+)\s+cm\*\*-1"
    )
    # MO transition line (handles optional spin suffix a/b)
    _TRANS = re.compile(
        r"^\s+(\d+)\w*\s*->\s*(\d+)\w*\s*:\s*([\d.]+)\s+\(c="
    )
    # Stop tokens that mark the end of an excited-states block
    _STOP  = re.compile(
        r"(TD-DFT.*EXCITED\s+STATES"
        r"|ABSORPTION SPECTRUM"
        r"|CD SPECTRUM"
        r"|COMBINED.*QUADRUPOLE"
        r"|ORCA TERMINATED"
        r"|TOTAL RUN TIME"
        r"|Timings for individual modules)",
        re.I,
    )

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, filepath: str) -> UVVisData:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        data = UVVisData(filename=os.path.basename(filepath))
        self._parse_absorption_table(lines, data)
        self._parse_excited_state_blocks(lines, data)
        return data

    # ── Absorption table ──────────────────────────────────────────────────────

    def _parse_absorption_table(self, lines: list, data: UVVisData):
        """Find the first Electric Dipole table and extract singlet fosc."""
        n = len(lines)
        i = 0
        while i < n:
            if self._EDIP.search(lines[i]):
                sep_count = 0
                in_data   = False
                i += 1
                while i < n:
                    ln = lines[i]
                    if self._SEP.match(ln):
                        sep_count += 1
                        if sep_count == 2:
                            in_data = True
                        elif sep_count == 3:
                            break
                        i += 1
                        continue
                    if not in_data or not ln.strip():
                        i += 1
                        continue
                    m = self._UVROW.match(ln)
                    if m:
                        data.singlet_idx.append(int(m.group(1)))
                        cm   = float(m.group(2))
                        nm   = float(m.group(3))
                        fosc = float(m.group(4))
                        data.singlet_cm.append(cm)
                        data.singlet_nm.append(nm)
                        data.singlet_ev.append(cm * CM_TO_EV)
                        data.singlet_fosc.append(fosc)
                    i += 1
                break  # only the first Electric Dipole section
            i += 1

    # ── Excited-state blocks ──────────────────────────────────────────────────

    def _parse_excited_state_blocks(self, lines: list, data: UVVisData):
        """Parse SINGLETS and TRIPLETS blocks for MO character."""
        n = len(lines)
        i = 0
        while i < n:
            ln = lines[i]
            if self._SHDR.search(ln):
                xs, i = self._read_states(lines, i + 1, "S")
                data.singlet_xs = xs
                # Fallback: populate singlet energy list if no absorption table
                if not data.singlet_idx:
                    for s in xs:
                        data.singlet_idx.append(s.index)
                        data.singlet_cm.append(s.energy_cm)
                        data.singlet_nm.append(s.wavelength_nm)
                        data.singlet_ev.append(s.energy_ev)
                        data.singlet_fosc.append(0.0)
            elif self._THDR.search(ln):
                xs, i = self._read_states(lines, i + 1, "T")
                data.triplet_xs  = xs
                data.triplet_idx = [s.index for s in xs]
                data.triplet_cm  = [s.energy_cm for s in xs]
                data.triplet_nm  = [s.wavelength_nm for s in xs]
                data.triplet_ev  = [s.energy_ev for s in xs]
            else:
                i += 1

    def _read_states(
        self, lines: list, start: int, spin: str
    ) -> Tuple[List[ExcState], int]:
        result:  List[ExcState]       = []
        current: Optional[ExcState]   = None
        n = len(lines)
        i = start
        while i < n:
            ln = lines[i]
            # Stop when we hit the next major section header (allow a few intro lines)
            if self._STOP.search(ln) and i > start + 8:
                break
            m = self._STHDR.search(ln)
            if m:
                current = ExcState(
                    index=int(m.group(1)),
                    energy_ev=float(m.group(2)),
                    energy_cm=float(m.group(3)),
                    spin=spin,
                )
                result.append(current)
            elif current:
                t = self._TRANS.match(ln)
                if t:
                    current.transitions.append(
                        MoTransition(
                            from_mo=int(t.group(1)),
                            to_mo=int(t.group(2)),
                            weight=float(t.group(3)),
                        )
                    )
            i += 1
        return result, i


# ══════════════════════════════════════════════════════════════════════════════
#  Broadening functions
# ══════════════════════════════════════════════════════════════════════════════

def gaussian(x, center, fwhm):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def lorentzian(x, center, fwhm):
    gamma = fwhm / 2.0
    return gamma ** 2 / ((x - center) ** 2 + gamma ** 2)


# ══════════════════════════════════════════════════════════════════════════════
#  Small helpers
# ══════════════════════════════════════════════════════════════════════════════

def _vsep(bar):
    """Pack a vertical separator into a horizontal control bar."""
    ttk.Separator(bar, orient=tk.VERTICAL).pack(
        side=tk.LEFT, fill=tk.Y, padx=6)


# ══════════════════════════════════════════════════════════════════════════════
#  Plot widget
# ══════════════════════════════════════════════════════════════════════════════

class UVVisPlotWidget(tk.Frame):
    """
    Embedded matplotlib figure with control strips.

    Rows:
      Row 1 — X-axis unit | Dual axis | Broadening type | FWHM | display toggles | tools
      Row 2 — Singlet/Triplet visibility & colour | triplet height | label options | grid/legend
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.data: Optional[UVVisData] = None

        # ── Control variables ────────────────────────────────────────────────
        self._x_unit       = tk.StringVar(value="nm")
        self._dual_axis    = tk.BooleanVar(value=True)
        self._broadening   = tk.StringVar(value="Gaussian")
        self._fwhm         = tk.DoubleVar(value=2000.0)   # cm⁻¹
        self._fwhm_str     = tk.StringVar(value="2000")
        self._show_s_sticks = tk.BooleanVar(value=True)
        self._show_s_env    = tk.BooleanVar(value=True)
        self._show_t_sticks = tk.BooleanVar(value=True)
        self._show_singlets = tk.BooleanVar(value=True)
        self._show_triplets = tk.BooleanVar(value=True)
        self._normalise     = tk.BooleanVar(value=False)
        self._invert_x      = tk.BooleanVar(value=False)
        self._show_legend   = tk.BooleanVar(value=True)
        self._show_grid     = tk.BooleanVar(value=False)
        self._title_str     = tk.StringVar(value="")

        # Colours
        self._s_col = SINGLET_COL
        self._t_col = TRIPLET_COL

        # Triplet display height as a fraction of max singlet fosc
        self._t_frac = tk.DoubleVar(value=0.20)

        # Label controls
        self._lbl_mode     = tk.StringVar(value=LBL_NONE)
        self._lbl_min_fosc = tk.StringVar(value="0.0")
        self._lbl_top_n    = tk.IntVar(value=30)
        self._lbl_rotation = tk.IntVar(value=90)
        self._lbl_fontsize = tk.IntVar(value=7)

        # Font sizes
        self._font_xlabel  = tk.IntVar(value=13)
        self._font_ylabel  = tk.IntVar(value=13)
        self._font_tick    = tk.IntVar(value=11)
        self._font_legend  = tk.IntVar(value=10)

        # Manual axis limits (empty = auto)
        self._xlim_lo = tk.StringVar(value="")
        self._xlim_hi = tk.StringVar(value="")
        self._ylim_lo = tk.StringVar(value="")
        self._ylim_hi = tk.StringVar(value="")

        self._build_controls()
        self._build_figure()

    # ── Scrollable control-bar factory ───────────────────────────────────────

    def _scrollable_bar(self, **kw) -> tk.Frame:
        outer = tk.Frame(self)
        outer.pack(side=tk.TOP, fill=tk.X)
        bg  = kw.get("bg", None)
        cnv = tk.Canvas(outer, highlightthickness=0,
                        **({"bg": bg} if bg else {}))
        hbar = ttk.Scrollbar(outer, orient="horizontal", command=cnv.xview)
        cnv.configure(xscrollcommand=hbar.set)
        cnv.pack(side=tk.TOP, fill=tk.X, expand=True)
        inner = tk.Frame(cnv, **kw)
        cnv.create_window((0, 0), window=inner, anchor="nw")

        def _resize(_=None):
            cnv.configure(scrollregion=cnv.bbox("all"))
            h = inner.winfo_reqheight()
            if h > 1:
                cnv.configure(height=h)
            try:
                wide = inner.winfo_reqwidth() > outer.winfo_width() > 1
            except Exception:
                wide = False
            if wide:
                hbar.pack(side=tk.BOTTOM, fill=tk.X, before=cnv)
            else:
                hbar.pack_forget()

        inner.bind("<Configure>", lambda e: outer.after_idle(_resize))
        outer.bind("<Configure>", lambda e: outer.after_idle(_resize))
        return inner

    # ── Build control strips ─────────────────────────────────────────────────

    def _build_controls(self):
        # ── Row 1: axis / broadening / display ────────────────────────────────
        r1 = self._scrollable_bar(bd=1, relief=tk.SUNKEN, padx=4, pady=3)

        tk.Label(r1, text="X axis:").pack(side=tk.LEFT)
        for unit in ("nm", "cm\u207b\u00b9", "eV"):
            tk.Radiobutton(r1, text=unit, variable=self._x_unit,
                           value=unit,
                           command=self._on_unit_change).pack(side=tk.LEFT, padx=1)

        _vsep(r1)
        tk.Checkbutton(r1, text="Dual Axis", variable=self._dual_axis,
                       command=self._replot, fg="darkblue",
                       font=("", 9, "bold")).pack(side=tk.LEFT, padx=2)

        _vsep(r1)
        tk.Label(r1, text="Broadening:").pack(side=tk.LEFT)
        for b in ("Gaussian", "Lorentzian"):
            tk.Radiobutton(r1, text=b, variable=self._broadening,
                           value=b, command=self._replot).pack(side=tk.LEFT, padx=1)

        _vsep(r1)
        tk.Label(r1, text="FWHM:").pack(side=tk.LEFT)
        fwhm_e = tk.Entry(r1, textvariable=self._fwhm_str, width=7,
                          font=("Courier", 9))
        fwhm_e.pack(side=tk.LEFT, padx=(2, 0))
        fwhm_e.bind("<Return>",   self._on_fwhm_entry)
        fwhm_e.bind("<FocusOut>", self._on_fwhm_entry)
        self._fwhm_unit_lbl = tk.Label(r1, text="cm\u207b\u00b9", width=5, anchor="w")
        self._fwhm_unit_lbl.pack(side=tk.LEFT)
        self._fwhm_slider = tk.Scale(
            r1, from_=50, to=8000, resolution=50,
            orient=tk.HORIZONTAL, length=140,
            variable=self._fwhm, showvalue=False,
            command=self._on_fwhm_slider,
        )
        self._fwhm_slider.pack(side=tk.LEFT, padx=2)

        _vsep(r1)
        tk.Checkbutton(r1, text="Sticks",    variable=self._show_s_sticks,
                       command=self._replot).pack(side=tk.LEFT)
        tk.Checkbutton(r1, text="Envelope",  variable=self._show_s_env,
                       command=self._replot).pack(side=tk.LEFT)
        tk.Checkbutton(r1, text="Normalise", variable=self._normalise,
                       command=self._replot).pack(side=tk.LEFT)
        tk.Checkbutton(r1, text="Invert X",  variable=self._invert_x,
                       command=self._replot).pack(side=tk.LEFT)

        _vsep(r1)
        tk.Button(r1, text="Save Fig",    command=self._save_fig,
                  font=("", 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(r1, text="Export CSV",  command=self._export_csv,
                  font=("", 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(r1, text="\u29c9 Pop Out", command=self._pop_out,
                  font=("", 8, "bold"), fg="#003399",
                  relief=tk.RAISED).pack(side=tk.LEFT, padx=2)
        tk.Button(r1, text="Axes\u2026",  command=self._open_axis_dialog,
                  font=("", 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(r1, text="Fonts\u2026", command=self._open_font_dialog,
                  font=("", 8)).pack(side=tk.LEFT, padx=2)

        # ── Row 2: species + labels ────────────────────────────────────────────
        r2 = self._scrollable_bar(bd=1, relief=tk.SUNKEN, padx=4, pady=3)

        # Singlets
        tk.Checkbutton(r2, text="Singlets", variable=self._show_singlets,
                       command=self._replot, fg="navy",
                       font=("", 9, "bold")).pack(side=tk.LEFT)
        self._s_col_btn = tk.Button(r2, bg=self._s_col, width=2,
                                    relief=tk.RAISED,
                                    command=lambda: self._pick_col("s"))
        self._s_col_btn.pack(side=tk.LEFT, padx=(0, 8))

        # Triplets
        tk.Checkbutton(r2, text="Triplets", variable=self._show_triplets,
                       command=self._replot, fg="#8B0000",
                       font=("", 9, "bold")).pack(side=tk.LEFT)
        self._t_col_btn = tk.Button(r2, bg=self._t_col, width=2,
                                    relief=tk.RAISED,
                                    command=lambda: self._pick_col("t"))
        self._t_col_btn.pack(side=tk.LEFT, padx=(0, 2))
        tk.Checkbutton(r2, text="T-sticks", variable=self._show_t_sticks,
                       command=self._replot, font=("", 8)).pack(side=tk.LEFT)
        tk.Label(r2, text="Height:").pack(side=tk.LEFT, padx=(6, 0))
        tk.Scale(r2, from_=0.01, to=1.00, resolution=0.01,
                 orient=tk.HORIZONTAL, length=90,
                 variable=self._t_frac, showvalue=False,
                 command=lambda _: self._replot()).pack(side=tk.LEFT)
        self._t_frac_lbl = tk.Label(r2, text="20%", width=5)
        self._t_frac_lbl.pack(side=tk.LEFT)
        self._t_frac.trace_add("write", self._update_t_frac_label)

        _vsep(r2)

        # Labels
        tk.Label(r2, text="Labels:").pack(side=tk.LEFT)
        ttk.Combobox(r2, textvariable=self._lbl_mode, values=ALL_LABEL_MODES,
                     state="readonly", width=22).pack(side=tk.LEFT, padx=2)
        self._lbl_mode.trace_add("write", lambda *_: self._replot())

        tk.Label(r2, text="Min f:").pack(side=tk.LEFT, padx=(6, 0))
        tk.Entry(r2, textvariable=self._lbl_min_fosc, width=7,
                 font=("Courier", 9)).pack(side=tk.LEFT)

        tk.Label(r2, text="Top N:").pack(side=tk.LEFT, padx=(6, 0))
        tk.Spinbox(r2, from_=1, to=999, textvariable=self._lbl_top_n,
                   width=5, command=self._replot).pack(side=tk.LEFT)

        tk.Label(r2, text="Rot:").pack(side=tk.LEFT, padx=(6, 0))
        tk.Spinbox(r2, from_=0, to=90, textvariable=self._lbl_rotation,
                   width=5, command=self._replot).pack(side=tk.LEFT)

        tk.Label(r2, text="Size:").pack(side=tk.LEFT, padx=(6, 0))
        tk.Spinbox(r2, from_=4, to=20, textvariable=self._lbl_fontsize,
                   width=5, command=self._replot).pack(side=tk.LEFT)

        _vsep(r2)
        tk.Checkbutton(r2, text="Legend", variable=self._show_legend,
                       command=self._replot).pack(side=tk.LEFT)
        tk.Checkbutton(r2, text="Grid", variable=self._show_grid,
                       command=self._replot).pack(side=tk.LEFT)
        tk.Label(r2, text="Title:").pack(side=tk.LEFT, padx=(6, 0))
        tk.Entry(r2, textvariable=self._title_str, width=24).pack(side=tk.LEFT)
        self._title_str.trace_add("write", lambda *_: self._replot())

    # ── Figure canvas ─────────────────────────────────────────────────────────

    def _build_figure(self):
        fig_frame = tk.Frame(self)
        fig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(9, 5.5), dpi=100)
        self.ax  = self.fig.add_subplot(111)

        self.canvas  = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, fig_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_unit_change(self):
        unit = self._x_unit.get()
        if unit == "eV":
            self._fwhm_unit_lbl.config(text="eV")
            self._fwhm_slider.config(from_=0.05, to=5.0, resolution=0.05)
            # Convert displayed value cm⁻¹ → eV
            try:
                cm_val = float(self._fwhm_str.get())
                ev_val = round(cm_val * CM_TO_EV, 3)
                self._fwhm.set(ev_val)
                self._fwhm_str.set(f"{ev_val:.2f}")
            except ValueError:
                pass
        else:
            self._fwhm_unit_lbl.config(text="cm\u207b\u00b9")
            self._fwhm_slider.config(from_=50, to=8000, resolution=50)
            # Convert displayed value eV → cm⁻¹ if currently small (eV range)
            try:
                v = float(self._fwhm_str.get())
                if v < 50:   # likely was in eV
                    cm_val = round(v * EV_TO_CM)
                    self._fwhm.set(cm_val)
                    self._fwhm_str.set(str(int(cm_val)))
            except ValueError:
                pass
        self._replot()

    def _on_fwhm_slider(self, val):
        try:
            v = float(val)
            self._fwhm_str.set(f"{v:.2f}" if self._x_unit.get() == "eV"
                               else str(int(v)))
        except ValueError:
            pass
        self._replot()

    def _on_fwhm_entry(self, _=None):
        try:
            self._fwhm.set(float(self._fwhm_str.get()))
        except ValueError:
            unit = self._x_unit.get()
            self._fwhm_str.set(
                f"{self._fwhm.get():.2f}" if unit == "eV"
                else str(int(self._fwhm.get()))
            )
        self._replot()

    def _update_t_frac_label(self, *_):
        self._t_frac_lbl.config(text=f"{self._t_frac.get()*100:.0f}%")

    def _pick_col(self, which: str):
        init = self._s_col if which == "s" else self._t_col
        _, hex_col = colorchooser.askcolor(color=init, title="Choose colour",
                                           parent=self)
        if hex_col:
            if which == "s":
                self._s_col = hex_col
                self._s_col_btn.config(bg=hex_col)
            else:
                self._t_col = hex_col
                self._t_col_btn.config(bg=hex_col)
            self._replot()

    # ── Unit/FWHM helpers ─────────────────────────────────────────────────────

    def _ev_to_unit(self, ev_arr: np.ndarray) -> np.ndarray:
        unit = self._x_unit.get()
        if unit == "nm":
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(ev_arr > 0, EV_TO_NM / ev_arr, 0.0)
        elif unit == "cm\u207b\u00b9":
            return ev_arr * EV_TO_CM
        return ev_arr  # eV

    def _fwhm_ev(self) -> float:
        """Return FWHM in eV regardless of display unit."""
        unit = self._x_unit.get()
        fwhm = self._fwhm.get()
        if unit == "eV":
            return max(fwhm, 1e-6)
        return max(fwhm * CM_TO_EV, 1e-6)   # cm⁻¹ → eV

    def _xlabel(self) -> str:
        unit = self._x_unit.get()
        if unit == "nm":         return "Wavelength (nm)"
        if unit == "cm\u207b\u00b9": return "Wavenumber (cm\u207b\u00b9)"
        return "Energy (eV)"

    # ── Data loader ───────────────────────────────────────────────────────────

    def load_data(self, data: UVVisData):
        self.data = data
        self._replot()

    # ── Main replot ───────────────────────────────────────────────────────────

    def _replot(self, *_):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        if self.data is None:
            self.canvas.draw_idle()
            return

        d        = self.data
        col_s    = self._s_col
        col_t    = self._t_col
        fwhm_ev  = self._fwhm_ev()

        # ── Prepare singlet arrays ────────────────────────────────────────
        if self._show_singlets.get() and d.has_singlets:
            ev_s  = np.array(d.singlet_ev,   dtype=float)
            fosc  = np.array(d.singlet_fosc, dtype=float)
            x_s   = self._ev_to_unit(ev_s)
            if self._normalise.get() and fosc.max() > 0:
                y_s    = fosc / fosc.max()
                ylabel = "Normalised Intensity"
            else:
                y_s    = fosc
                ylabel = "Oscillator Strength  f"
            max_y = float(y_s.max()) if len(y_s) > 0 else 1.0
        else:
            ev_s = x_s = y_s = np.array([])
            max_y  = 1.0
            ylabel = "Oscillator Strength  f"

        # Triplet display height
        t_ht = self._t_frac.get() * max_y

        # ── Singlet sticks ────────────────────────────────────────────────
        if self._show_singlets.get() and len(x_s) > 0 and self._show_s_sticks.get():
            stk_lbl = "Singlets" if not self._show_s_env.get() else None
            ml, sl, _ = self.ax.stem(
                x_s, y_s, linefmt=col_s, markerfmt="o", basefmt=" ",
                label=stk_lbl,
            )
            ml.set_markersize(4)
            ml.set_color(col_s)
            _plt.setp(sl, linewidth=1.0, alpha=0.65, color=col_s)

        # ── Singlet broadened envelope ────────────────────────────────────
        if self._show_singlets.get() and len(ev_s) > 0 and self._show_s_env.get():
            self._draw_envelope(ev_s, y_s, col_s, fwhm_ev, "Singlets")

        # ── Triplet sticks ────────────────────────────────────────────────
        if self._show_triplets.get() and d.has_triplets and self._show_t_sticks.get():
            ev_t = np.array(d.triplet_ev, dtype=float)
            x_t  = self._ev_to_unit(ev_t)
            y_t  = np.full(len(x_t), t_ht)
            ml2, sl2, _ = self.ax.stem(
                x_t, y_t, linefmt=col_t, markerfmt="D", basefmt=" ",
                label="Triplets",
            )
            ml2.set_markersize(4)
            ml2.set_color(col_t)
            _plt.setp(sl2, linewidth=1.0, alpha=0.70, color=col_t)

        # ── Excitation labels ─────────────────────────────────────────────
        lmode = self._lbl_mode.get()
        if lmode != LBL_NONE:
            if self._show_singlets.get() and len(x_s) > 0:
                self._draw_labels(x_s, y_s, d, "S", lmode, max_y)
            if self._show_triplets.get() and d.has_triplets and self._show_t_sticks.get():
                ev_t2 = np.array(d.triplet_ev, dtype=float)
                x_t2  = self._ev_to_unit(ev_t2)
                y_t2  = np.full(len(x_t2), t_ht)
                self._draw_labels(x_t2, y_t2, d, "T", lmode, max_y)

        # ── Axis decorations ──────────────────────────────────────────────
        fs_x = self._font_xlabel.get()
        fs_y = self._font_ylabel.get()
        fs_t = self._font_tick.get()

        self.ax.set_xlabel(self._xlabel(), fontsize=fs_x)
        self.ax.set_ylabel(
            "Normalised Intensity" if self._normalise.get() else ylabel,
            fontsize=fs_y,
        )
        self.ax.tick_params(labelsize=fs_t)
        self.ax.set_ylim(bottom=0)

        # Manual axis limits
        def _f(s):
            try:    return float(s.strip())
            except: return None
        xl, xh = _f(self._xlim_lo.get()), _f(self._xlim_hi.get())
        yl, yh = _f(self._ylim_lo.get()), _f(self._ylim_hi.get())
        if xl is not None or xh is not None:
            cur = self.ax.get_xlim()
            self.ax.set_xlim(xl if xl is not None else cur[0],
                             xh if xh is not None else cur[1])
        if yl is not None or yh is not None:
            cur = self.ax.get_ylim()
            self.ax.set_ylim(yl if yl is not None else cur[0],
                             yh if yh is not None else cur[1])

        if self._invert_x.get():
            self.ax.invert_xaxis()

        # ── Dual secondary x-axis ─────────────────────────────────────────
        if self._dual_axis.get():
            self._add_secondary_axis()

        # ── Title / legend / grid ─────────────────────────────────────────
        title = self._title_str.get().strip() or (d.filename if d else "")
        if title:
            self.ax.set_title(title, fontsize=11)

        if self._show_legend.get():
            h, l = self.ax.get_legend_handles_labels()
            if h:
                self.ax.legend(h, l, fontsize=self._font_legend.get(),
                               framealpha=0.85)

        if self._show_grid.get():
            self.ax.grid(True, linestyle="--", alpha=0.35)

        # Adjust margins so the top secondary-axis label isn't clipped
        top_pad = 0.88 if self._dual_axis.get() else 0.94
        self.fig.tight_layout(rect=[0, 0, 1, top_pad])

        self.toolbar.update()
        self.canvas.draw_idle()

    # ── Envelope drawing ──────────────────────────────────────────────────────

    def _draw_envelope(self, ev_arr, y_arr, colour, fwhm_ev, label):
        ev_min  = max(1e-4, float(ev_arr.min()) - 4.0 * fwhm_ev)
        ev_max  = float(ev_arr.max()) + 4.0 * fwhm_ev
        ev_grid = np.linspace(ev_min, ev_max, 2000)
        fn  = gaussian if self._broadening.get() == "Gaussian" else lorentzian
        env = sum(y * fn(ev_grid, c, fwhm_ev) for c, y in zip(ev_arr, y_arr))
        x_grid = self._ev_to_unit(ev_grid)
        self.ax.plot(x_grid, env, color=colour, linewidth=2.0,
                     alpha=0.9, label=label)
        self.ax.fill_between(x_grid, 0, env, alpha=0.10, color=colour)

    # ── Secondary (dual) x-axis ───────────────────────────────────────────────

    def _add_secondary_axis(self):
        """Add a secondary x-axis on top: nm ↔ cm⁻¹ (or eV ↔ nm)."""
        unit = self._x_unit.get()

        # For nm↔cm⁻¹ the transform is self-inverse: f(x) = 1e7/x
        def _inv_fwd(x):
            a = np.asarray(x, dtype=float)
            return np.where(a > 0, 1e7 / a, np.nan)

        # For eV↔nm the transform is also self-inverse: f(x) = 1239.84/x
        def _ev_nm(x):
            a = np.asarray(x, dtype=float)
            return np.where(a > 0, EV_TO_NM / a, np.nan)

        try:
            if unit == "nm":
                ax2 = self.ax.secondary_xaxis("top",
                                              functions=(_inv_fwd, _inv_fwd))
                ax2.set_xlabel("Wavenumber (cm\u207b\u00b9)",
                               fontsize=self._font_xlabel.get())
                ax2.tick_params(labelsize=self._font_tick.get())

            elif unit == "cm\u207b\u00b9":
                ax2 = self.ax.secondary_xaxis("top",
                                              functions=(_inv_fwd, _inv_fwd))
                ax2.set_xlabel("Wavelength (nm)",
                               fontsize=self._font_xlabel.get())
                ax2.tick_params(labelsize=self._font_tick.get())

            elif unit == "eV":
                ax2 = self.ax.secondary_xaxis("top",
                                              functions=(_ev_nm, _ev_nm))
                ax2.set_xlabel("Wavelength (nm)",
                               fontsize=self._font_xlabel.get())
                ax2.tick_params(labelsize=self._font_tick.get())

        except Exception:
            pass   # secondary_xaxis can occasionally fail on edge cases

    # ── Label drawing ─────────────────────────────────────────────────────────

    def _draw_labels(self, x_arr, y_arr, d: UVVisData, spin: str,
                     mode: str, max_y: float):
        """Annotate excitation sticks with the selected label mode."""
        if len(x_arr) == 0:
            return

        idxs  = d.singlet_idx  if spin == "S" else d.triplet_idx
        foscs = d.singlet_fosc if spin == "S" else [0.0] * len(d.triplet_idx)
        evs   = d.singlet_ev   if spin == "S" else d.triplet_ev

        if len(x_arr) != len(idxs):
            return

        # Min-fosc filter (singlets only)
        try:
            min_f = float(self._lbl_min_fosc.get())
        except ValueError:
            min_f = 0.0
        top_n = max(1, self._lbl_top_n.get())

        # Build candidate list
        if spin == "S" and min_f > 0:
            candidates = [(i, foscs[i]) for i in range(len(idxs))
                          if foscs[i] >= min_f]
        else:
            candidates = [(i, foscs[i]) for i in range(len(idxs))]

        # Sort by fosc desc, keep top N
        candidates.sort(key=lambda t: t[1], reverse=True)
        candidates = candidates[:top_n]
        cand_set   = {i for i, _ in candidates}

        rot = self._lbl_rotation.get()
        fs  = self._lbl_fontsize.get()
        col = self._s_col if spin == "S" else self._t_col
        y_off = max_y * 0.025

        for i, (xi, yi) in enumerate(zip(x_arr, y_arr)):
            if i not in cand_set:
                continue
            text = self._make_label(
                state=idxs[i], spin=spin,
                ev=evs[i], fosc=foscs[i],
                mode=mode, d=d,
            )
            if not text:
                continue
            self.ax.annotate(
                text,
                xy=(xi, yi + y_off),
                xytext=(0, 2),
                textcoords="offset points",
                fontsize=fs, ha="center", va="bottom",
                color=col, rotation=rot,
                annotation_clip=True,
            )

    def _make_label(self, state: int, spin: str, ev: float, fosc: float,
                    mode: str, d: UVVisData) -> str:
        """Construct the label string for one excitation."""
        nm  = EV_TO_NM / ev if ev > 0 else 0.0
        cm  = ev * EV_TO_CM
        pfx = f"{spin}{state}"

        def _mo_str() -> str:
            xs_list = d.singlet_xs if spin == "S" else d.triplet_xs
            for xs in xs_list:
                if xs.index == state:
                    dom = xs.dominant()
                    return f"{dom.from_mo}\u2192{dom.to_mo}" if dom else ""
            return ""

        if mode == LBL_STATE:       return pfx
        if mode == LBL_NM:          return f"{nm:.1f}"
        if mode == LBL_CM:          return f"{cm:.0f}"
        if mode == LBL_EV:          return f"{ev:.3f}"
        if mode == LBL_FOSC:        return f"f={fosc:.4f}" if spin == "S" else ""
        if mode == LBL_MO:          return _mo_str()
        if mode == LBL_STATE_NM:    return f"{pfx}: {nm:.1f} nm"
        if mode == LBL_STATE_FOSC:
            return f"{pfx}: f={fosc:.4f}" if spin == "S" else pfx
        if mode == LBL_STATE_MO:
            mo = _mo_str()
            return f"{pfx}: {mo}" if mo else pfx
        if mode == LBL_FULL:
            if spin == "S":
                return f"{pfx}: {nm:.1f} nm, f={fosc:.4f}"
            return f"{pfx}: {nm:.1f} nm"
        return pfx

    # ── Dialogs ───────────────────────────────────────────────────────────────

    def _open_axis_dialog(self):
        win = tk.Toplevel(self)
        win.title("Axis Limits")
        win.resizable(False, False)
        win.grab_set()

        unit = self._x_unit.get()
        frm = tk.Frame(win, padx=14, pady=10)
        frm.pack(fill=tk.BOTH)

        def _lbl(r, c, t):
            tk.Label(frm, text=t).grid(row=r, column=c, sticky="e",
                                       padx=4, pady=3)

        _lbl(0, 0, f"X min ({unit}):")
        tk.Entry(frm, textvariable=self._xlim_lo, width=12).grid(
            row=0, column=1, sticky="w")
        _lbl(0, 2, f"X max ({unit}):")
        tk.Entry(frm, textvariable=self._xlim_hi, width=12).grid(
            row=0, column=3, sticky="w")
        _lbl(1, 0, "Y min:")
        tk.Entry(frm, textvariable=self._ylim_lo, width=12).grid(
            row=1, column=1, sticky="w")
        _lbl(1, 2, "Y max:")
        tk.Entry(frm, textvariable=self._ylim_hi, width=12).grid(
            row=1, column=3, sticky="w")
        tk.Label(frm, text="Leave blank for auto-scale",
                 font=("", 8), fg="gray").grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))

        btn = tk.Frame(win, pady=8)
        btn.pack()
        tk.Button(btn, text="Apply",  width=10,
                  command=lambda: (win.destroy(), self._replot())).pack(
            side=tk.LEFT, padx=4)
        tk.Button(btn, text="Reset",  width=10,
                  command=self._reset_axis_limits).pack(side=tk.LEFT, padx=4)
        tk.Button(btn, text="Cancel", width=10,
                  command=win.destroy).pack(side=tk.LEFT, padx=4)

    def _reset_axis_limits(self):
        for v in (self._xlim_lo, self._xlim_hi, self._ylim_lo, self._ylim_hi):
            v.set("")
        self._replot()

    def _open_font_dialog(self):
        win = tk.Toplevel(self)
        win.title("Font Sizes")
        win.resizable(False, False)
        win.grab_set()

        frm = tk.Frame(win, padx=14, pady=10)
        frm.pack(fill=tk.BOTH)

        rows = [
            ("X-axis label:", self._font_xlabel),
            ("Y-axis label:", self._font_ylabel),
            ("Tick labels:",  self._font_tick),
            ("Legend:",       self._font_legend),
        ]
        for r, (lbl, var) in enumerate(rows):
            tk.Label(frm, text=lbl).grid(row=r, column=0, sticky="e",
                                         padx=4, pady=3)
            tk.Spinbox(frm, from_=6, to=24, textvariable=var,
                       width=6).grid(row=r, column=1, sticky="w")

        tk.Button(win, text="Apply & Close", width=14,
                  command=lambda: (win.destroy(), self._replot())).pack(pady=8)

    # ── Export helpers ────────────────────────────────────────────────────────

    def _save_fig(self):
        path = filedialog.asksaveasfilename(
            title="Save Figure",
            defaultextension=".png",
            filetypes=[("PNG",  "*.png"), ("PDF", "*.pdf"),
                       ("SVG",  "*.svg"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Figure saved:\n{path}", parent=self)
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc), parent=self)

    def _export_csv(self):
        if self.data is None:
            messagebox.showinfo("No Data", "Load a file first.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            title="Export CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if not path:
            return
        d = self.data
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# UV-Vis TDDFT data — UV-Vis Viewer\n")
                fh.write(f"# Source: {d.filename}\n")
                fh.write("# Singlet states (Electric Dipole absorption table)\n")
                fh.write("spin,state,energy_eV,energy_cm-1,wavelength_nm,fosc\n")
                for i in range(d.n_singlets):
                    fh.write(
                        f"S,{d.singlet_idx[i]},"
                        f"{d.singlet_ev[i]:.6f},"
                        f"{d.singlet_cm[i]:.2f},"
                        f"{d.singlet_nm[i]:.3f},"
                        f"{d.singlet_fosc[i]:.8f}\n"
                    )
                fh.write("# Triplet states (no oscillator strength)\n")
                for i in range(d.n_triplets):
                    fh.write(
                        f"T,{d.triplet_idx[i]},"
                        f"{d.triplet_ev[i]:.6f},"
                        f"{d.triplet_cm[i]:.2f},"
                        f"{d.triplet_nm[i]:.3f},"
                        f"0.0\n"
                    )
            messagebox.showinfo("Exported", f"Data exported:\n{path}",
                                parent=self)
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc), parent=self)

    def _pop_out(self):
        """Open an independent matplotlib window with the current spectrum."""
        if self.data is None:
            return
        d        = self.data
        col_s    = self._s_col
        col_t    = self._t_col
        fwhm_ev  = self._fwhm_ev()
        unit     = self._x_unit.get()
        norm     = self._normalise.get()

        def ev2u(ev):
            ev = np.asarray(ev, dtype=float)
            if unit == "nm":
                return np.where(ev > 0, EV_TO_NM / ev, 0.0)
            elif unit == "cm\u207b\u00b9":
                return ev * EV_TO_CM
            return ev

        fig2, ax2 = _plt.subplots(figsize=(10, 6))

        y_s = np.array([])
        if self._show_singlets.get() and d.has_singlets:
            ev_s = np.array(d.singlet_ev)
            fosc = np.array(d.singlet_fosc)
            x_s  = ev2u(ev_s)
            y_s  = fosc / fosc.max() if norm and fosc.max() > 0 else fosc
            if self._show_s_sticks.get():
                ml, sl, _ = ax2.stem(x_s, y_s, linefmt=col_s, markerfmt="o",
                                     basefmt=" ")
                ml.set_markersize(4); ml.set_color(col_s)
                _plt.setp(sl, linewidth=1.0, alpha=0.65, color=col_s)
            if self._show_s_env.get() and fwhm_ev > 0:
                ev_min  = max(1e-4, ev_s.min() - 4 * fwhm_ev)
                ev_grid = np.linspace(ev_min, ev_s.max() + 4 * fwhm_ev, 2000)
                fn  = gaussian if self._broadening.get() == "Gaussian" else lorentzian
                env = sum(y * fn(ev_grid, c, fwhm_ev) for c, y in zip(ev_s, y_s))
                ax2.plot(ev2u(ev_grid), env, color=col_s, lw=2, alpha=0.9,
                         label="Singlets")
                ax2.fill_between(ev2u(ev_grid), 0, env, alpha=0.10, color=col_s)

        max_y = float(y_s.max()) if len(y_s) > 0 else 1.0
        if self._show_triplets.get() and d.has_triplets and self._show_t_sticks.get():
            ev_t = np.array(d.triplet_ev)
            x_t  = ev2u(ev_t)
            y_t  = np.full(len(x_t), self._t_frac.get() * max_y)
            ml2, sl2, _ = ax2.stem(x_t, y_t, linefmt=col_t, markerfmt="D",
                                   basefmt=" ", label="Triplets")
            ml2.set_markersize(4); ml2.set_color(col_t)
            _plt.setp(sl2, linewidth=1.0, alpha=0.70, color=col_t)

        if unit == "nm":
            xlabel, sec_lbl = "Wavelength (nm)", "Wavenumber (cm\u207b\u00b9)"
        elif unit == "cm\u207b\u00b9":
            xlabel, sec_lbl = "Wavenumber (cm\u207b\u00b9)", "Wavelength (nm)"
        else:
            xlabel, sec_lbl = "Energy (eV)", "Wavelength (nm)"

        ax2.set_xlabel(xlabel, fontsize=13)
        ax2.set_ylabel(
            "Normalised Intensity" if norm else "Oscillator Strength  f",
            fontsize=13)
        ax2.set_ylim(bottom=0)
        if self._invert_x.get():
            ax2.invert_xaxis()

        if self._dual_axis.get():
            def _sinv(x):
                a = np.asarray(x, dtype=float)
                return np.where(a > 0, 1e7 / a, np.nan)
            def _ev_nm_f(x):
                a = np.asarray(x, dtype=float)
                return np.where(a > 0, EV_TO_NM / a, np.nan)
            try:
                fn2 = _sinv if unit in ("nm", "cm\u207b\u00b9") else _ev_nm_f
                ax_top = ax2.secondary_xaxis("top", functions=(fn2, fn2))
                ax_top.set_xlabel(sec_lbl, fontsize=13)
            except Exception:
                pass

        title = self._title_str.get().strip() or d.filename
        if title:
            ax2.set_title(title, fontsize=12)
        if self._show_legend.get():
            h, lb = ax2.get_legend_handles_labels()
            if h:
                ax2.legend(h, lb)
        if self._show_grid.get():
            ax2.grid(True, linestyle="--", alpha=0.35)

        fig2.tight_layout(rect=[0, 0, 1, 0.90])
        _plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  Info sidebar panel
# ══════════════════════════════════════════════════════════════════════════════

class _InfoPanel(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._txt = tk.Text(self, height=24, width=28, state=tk.DISABLED,
                            font=("Courier", 8), wrap=tk.WORD, bd=0,
                            bg=self.cget("bg"))
        self._txt.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

    def update_data(self, data: Optional[UVVisData]):
        self._txt.config(state=tk.NORMAL)
        self._txt.delete("1.0", tk.END)
        if data is None:
            self._txt.insert(tk.END, "No data loaded.")
            self._txt.config(state=tk.DISABLED)
            return

        lines = [
            f"File: {data.filename}",
            "",
            f"Singlets: {data.n_singlets}",
        ]
        if data.has_singlets:
            fosc = data.singlet_fosc
            lines += [
                f"  Max f  : {max(fosc):.6f}",
                f"  Sum f  : {sum(fosc):.4f}",
                "",
                "  Top 5 by f:",
            ]
            top5 = sorted(range(data.n_singlets),
                          key=lambda i: fosc[i], reverse=True)[:5]
            for k in top5:
                lines.append(
                    f"  S{data.singlet_idx[k]:>3}: "
                    f"{data.singlet_nm[k]:>7.1f} nm  "
                    f"f={fosc[k]:.5f}"
                )
        lines += ["", f"Triplets: {data.n_triplets}"]
        if data.has_triplets:
            lines += [
                f"  T1: {data.triplet_nm[0]:.1f} nm",
                f"      {data.triplet_cm[0]:.1f} cm\u207b\u00b9",
                f"      {data.triplet_ev[0]:.4f} eV",
            ]
            if data.n_triplets >= 2:
                lines.append(f"  T2: {data.triplet_nm[1]:.1f} nm")
            if data.n_triplets >= 3:
                lines.append(f"  T3: {data.triplet_nm[2]:.1f} nm")
        lines += [
            "",
            "MO detail:",
            f"  Singlets: {len(data.singlet_xs)} states",
            f"  Triplets: {len(data.triplet_xs)} states",
        ]
        self._txt.insert(tk.END, "\n".join(lines))
        self._txt.config(state=tk.DISABLED)


# ══════════════════════════════════════════════════════════════════════════════
#  Main application window
# ══════════════════════════════════════════════════════════════════════════════

class UVVisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("UV-Vis TDDFT Viewer")
        self.geometry("1200x780")
        self.minsize(900, 600)

        self._parser       = UVVisParser()
        self._data: Optional[UVVisData] = None
        self._current_file = ""

        self._build_menu()
        self._build_top_bar()
        self._build_main_area()
        self._build_status_bar()

    # ── Menu ──────────────────────────────────────────────────────────────────

    def _build_menu(self):
        mbar = tk.Menu(self)
        fm   = tk.Menu(mbar, tearoff=0)
        fm.add_command(label="Open ORCA .out File\u2026",
                       accelerator="Ctrl+O", command=self._open_file)
        fm.add_command(label="Reload",
                       accelerator="Ctrl+R", command=self._reload_file)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.destroy)
        mbar.add_cascade(label="File", menu=fm)

        hm = tk.Menu(mbar, tearoff=0)
        hm.add_command(label="About", command=self._about)
        mbar.add_cascade(label="Help", menu=hm)

        self.config(menu=mbar)
        self.bind_all("<Control-o>", lambda _: self._open_file())
        self.bind_all("<Control-r>", lambda _: self._reload_file())

    # ── Top toolbar ───────────────────────────────────────────────────────────

    def _build_top_bar(self):
        bar = tk.Frame(self, bd=1, relief=tk.RAISED, padx=6, pady=4)
        bar.pack(side=tk.TOP, fill=tk.X)
        tk.Button(bar, text="Open File", width=10,
                  command=self._open_file).pack(side=tk.LEFT, padx=2)
        tk.Button(bar, text="Reload", width=8,
                  command=self._reload_file).pack(side=tk.LEFT, padx=2)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8)
        self._file_lbl = tk.Label(bar, text="No file loaded",
                                  fg="gray", anchor="w")
        self._file_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

    # ── Main area ─────────────────────────────────────────────────────────────

    def _build_main_area(self):
        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL,
                              sashwidth=5, sashrelief=tk.RAISED)
        pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(pane, width=220, bd=1, relief=tk.SUNKEN)
        pane.add(sidebar, minsize=180)
        tk.Label(sidebar, text="Spectrum Info",
                 font=("", 9, "bold")).pack(anchor="w", padx=4, pady=2)
        self._info = _InfoPanel(sidebar)
        self._info.pack(fill=tk.BOTH, expand=True)

        plot_frame = tk.Frame(pane)
        pane.add(plot_frame, minsize=600)
        self._plot = UVVisPlotWidget(plot_frame)
        self._plot.pack(fill=tk.BOTH, expand=True)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_status_bar(self):
        self._status = tk.StringVar(
            value="Ready. Open an ORCA .out file (Ctrl+O).")
        tk.Label(self, textvariable=self._status, bd=1, relief=tk.SUNKEN,
                 anchor="w", padx=6, font=("", 8)).pack(
            side=tk.BOTTOM, fill=tk.X)

    # ── File operations ───────────────────────────────────────────────────────

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open ORCA Output File",
            filetypes=[("ORCA Output", "*.out"), ("All files", "*.*")],
        )
        if path:
            self._load_file(path)

    def _reload_file(self):
        if self._current_file:
            self._load_file(self._current_file)

    def _load_file(self, path: str):
        self._status.set(f"Parsing: {os.path.basename(path)}\u2026")
        self.update_idletasks()
        try:
            data = self._parser.parse(path)
        except Exception as exc:
            messagebox.showerror("Parse Error",
                                 f"Failed to parse file:\n{exc}", parent=self)
            self._status.set("Parse error.")
            return

        if not data.has_singlets and not data.has_triplets:
            messagebox.showwarning(
                "No Data Found",
                "No UV-Vis TDDFT singlet or triplet data found.\n\n"
                "• Check the file is an ORCA TDDFT output (not XAS)\n"
                "• The calculation must have completed normally\n"
                "• Ensure at least one Electric Dipole absorption table exists",
                parent=self,
            )
            self._status.set("No UV-Vis data found.")
            return

        self._data         = data
        self._current_file = path
        self._file_lbl.config(text=os.path.basename(path), fg="black")
        self._plot.load_data(data)
        self._info.update_data(data)
        self._status.set(
            f"Loaded: {os.path.basename(path)}  —  "
            f"{data.n_singlets} singlet(s),  {data.n_triplets} triplet(s)  |  "
            f"Max f = {max(data.singlet_fosc):.5f}"
            if data.has_singlets else
            f"Loaded: {os.path.basename(path)}  —  "
            f"{data.n_singlets} singlet(s),  {data.n_triplets} triplet(s)"
        )

    # ── About ─────────────────────────────────────────────────────────────────

    def _about(self):
        messagebox.showinfo(
            "About UV-Vis TDDFT Viewer",
            "UV-Vis TDDFT Viewer\n"
            "Visualise ORCA TDDFT singlet and triplet excited states.\n\n"
            "Key features:\n"
            "  \u2022 Singlet states with real oscillator strengths\n"
            "  \u2022 Gaussian or Lorentzian broadening\n"
            "  \u2022 Triplet state positions at user-adjustable height\n"
            "  \u2022 Dual x-axis: wavelength (nm) + wavenumber (cm\u207b\u00b9)\n"
            "    shown simultaneously via secondary_xaxis\n"
            "  \u2022 Excitation labels: state #, nm, cm\u207b\u00b9, eV,\n"
            "    oscillator strength, dominant MO, or combinations\n"
            "  \u2022 Export figure (PNG/PDF/SVG) and data (CSV)\n\n"
            "Input: ORCA 4/5 UV-Vis TDDFT output\n"
            "(Electric Dipole absorption table + TD-DFT excited states)\n\n"
            "Inspired by Binah (ORCA TDDFT XAS Viewer).",
            parent=self,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    try:
        import numpy      # noqa: F401
        import matplotlib # noqa: F401
    except ImportError:
        print("Missing dependencies.  Run:  pip install numpy matplotlib",
              file=sys.stderr)
        sys.exit(1)
    app = UVVisApp()
    app.mainloop()


if __name__ == "__main__":
    main()
