"""
L-Edge XAS Normalizer
Simultaneous BlueprintXAS-style fit and normalization for transition-metal
L2,3-edge X-ray absorption spectra.

Procedure (ref: BlueprintXAS, Delgado-Jaime & DeBeer, 2012):
  1.  Load raw XAS scan(s) from two-column text files (.csv / .txt / .dat).
  2.  Inspect individual scans, select good ones, sum them.
  3.  Fit the summed raw data *simultaneously* with:
        B(x)    -- four-domain background function
        eL3/eL2 -- cumulative pseudo-Voigt edge steps  (2:1 branching ratio)
        peaks   -- any number of L3/L2 white-line and edge-resolved
                   MLCT/LMCT pseudo-Voigt peaks
  4.  Normalize:  mu_norm(E) = ( mu_raw(E) - B(E) ) / ( 3/2 * I_L3,Edge )
  5.  Export normalized spectrum + fit parameters as CSV.

Usage  :  python ledge_normalizer.py
Requires: numpy  scipy  matplotlib   (pip install numpy scipy matplotlib)
"""

import os, sys, csv, threading
import numpy as np
from scipy.optimize import least_squares, differential_evolution
from scipy.special import erf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from quanty_seed_export import SUPPORTED_SITE_SYMMETRIES, write_quanty_seed_bundle

# Optional: SGMPython integration for loading directly from SGM stacks.
# If sgmanalysis is installed (pip install sgmanalysis or local install from
# github.com/Beamlines-CanadianLightSource/SGMPython), StackScan is used as
# the data-loading backend when a stack directory is opened.
try:
    from sgmanalysis.scans import StackScan as _SGMStackScan
    _SGM_AVAILABLE = True
except ImportError:
    _SGM_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  Physical constants
# ══════════════════════════════════════════════════════════════════════════════
SOC = {                        # 2p spin-orbit coupling constants  zeta_2p  (eV)
    "Ni": 11.507,              # Ni L3 ~852 eV   L2 ~869 eV
    "Cu": 13.498,              # Cu L3 ~930 eV   L2 ~951 eV
    "Co":  9.755,              # Co L3 ~778 eV
    "Fe":  8.194,              # Fe L3 ~707 eV
    "Mn":  6.748,              # Mn L3 ~638 eV
    "V":   5.572,              # V  L3 ~512 eV
}

_FDG = 0.05   # Fermi-Dirac transition width (eV)

# ══════════════════════════════════════════════════════════════════════════════
#  Core spectral functions
# ══════════════════════════════════════════════════════════════════════════════

def _on(x, a):
    return 1.0 / (1.0 + np.exp(np.clip(-(x - a) / _FDG, -500, 500)))

def _off(x, a):
    return 1.0 / (1.0 + np.exp(np.clip( (x - a) / _FDG, -500, 500)))

def pv(x, c, w, I, g):
    """Pseudo-Voigt peak. g=1 -> pure Gaussian, g=0 -> pure Lorentzian."""
    w = max(float(w), 1e-6)
    g = float(np.clip(g, 0.0, 1.0))
    s  = w / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gm = w / 2.0
    G  = np.exp(-((x - c)**2) / (2.0 * s**2))
    L  = gm**2 / ((x - c)**2 + gm**2)
    return float(I) * (g * G + (1.0 - g) * L)

def cpv(x, c, w, I, g):
    """Cumulative pseudo-Voigt (edge step). g=1 -> Gaussian CDF, g=0 -> arctan."""
    w = max(float(w), 1e-6)
    g = float(np.clip(g, 0.0, 1.0))
    s  = w / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gm = w / 2.0
    Gc = 0.5 * (1.0 + erf((x - c) / (s * np.sqrt(2.0))))
    Lc = 0.5 + np.arctan((x - c) / gm) / np.pi
    return float(I) * (g * Gc + (1.0 - g) * Lc)

def bg4(x, Ek, E0, Tk, Et, C, m1, m2, a_log, c_log, b, m4):
    """Four-domain background function B(x) (BlueprintXAS L-edge)."""
    Q  = 10.0**a_log - 10.0**c_log
    v0 = m1 * (E0 - Ek) + C
    vk = m2 * (Tk - E0) + v0
    ve = Q * (Et**2 - Tk**2) + b * (Et - Tk) + vk
    f1 = m1 * (x - Ek) + C
    f2 = m2 * (x - E0) + v0
    f3 = Q  * (x**2 - Tk**2) + b * (x - Tk) + vk
    f4 = m4 * (x - Et) + ve
    return (f1 * _off(x, E0) +
            f2 * _on(x, E0)  * _off(x, Tk) +
            f3 * _on(x, Tk)  * _off(x, Et) +
            f4 * _on(x, Et))

# ══════════════════════════════════════════════════════════════════════════════
#  Static parameter table  (Background + Edge only)
# ══════════════════════════════════════════════════════════════════════════════
#  Each row: (key, section, label, default, lo, hi, step, fixed)
PARAMS_STATIC = [
    # ── Background ──────────────────────────────────────────────────────────
    ("E0",      "Background", "E0   L3 onset (eV)",         852.0,  800.0, 1000.0, 0.1,   False),
    ("Ek",      "Background", "Ek   data start (eV)",        820.0,  400.0,  995.0, 0.5,   True ),
    ("Et",      "Background", "Et   quad to lin pt (eV)",    895.0,  820.0, 1200.0, 0.5,   False),
    ("C",       "Background", "C    pre-edge offset",          0.5,  -10.0,  100.0, 0.001, False),
    ("m1",      "Background", "m1   pre-edge slope",          0.0,   -2.0,    2.0, 1e-4,  False),
    ("m2",      "Background", "m2   L-region slope",         0.001,  -1.0,    1.0, 1e-4,  False),
    ("a_log",   "Background", "a    log10 quad coeff",        -9.0,  -20.0,   -1.0, 0.1,   False),
    ("c_log",   "Background", "c    log10 quad coeff",        -7.0,  -20.0,   -1.0, 0.1,   False),
    ("b",       "Background", "b    linear in quad domain",   0.0,   -2.0,    2.0, 1e-4,  False),
    ("m4",      "Background", "m4   far post-edge slope",    0.001,  -1.0,    1.0, 1e-4,  False),
    # ── Edge step functions ──────────────────────────────────────────────────
    ("Eshift",  "Calibration","A    energy shift (eV)",       0.0,    -5.0,    5.0, 0.001, False),
    ("Escale",  "Calibration","B    energy scale",            1.0,     0.97,   1.03, 1e-4, False),
    ("IL3_Edge","Edge",       "I L3,Edge  step height",       0.5,    0.001,  200.0, 0.01, False),
    ("WL3_Edge","Edge",       "W L3,Edge  FWHM (eV)",         2.0,    0.01,   20.0, 0.1,  False),
    ("GL3_Edge","Edge",       "G L3,Edge  Gauss frac [0-1]",  0.0,    0.0,     1.0, 0.05, False),
    ("WL2_Edge","Edge",       "W L2,Edge  FWHM (eV)",         3.0,    0.01,   20.0, 0.1,  False),
    ("GL2_Edge","Edge",       "G L2,Edge  Gauss frac [0-1]",  0.0,    0.0,     1.0, 0.05, False),
]
_KEYS_STATIC = [p[0] for p in PARAMS_STATIC]

# ══════════════════════════════════════════════════════════════════════════════
#  Dynamic peak infrastructure
# ══════════════════════════════════════════════════════════════════════════════
PEAK_FIELDS = ['o', 'W', 'I', 'G']


def _is_l3_kind(kind):
    return kind in ('L3', 'L3_MLCT', 'L3_LMCT')


def _is_l2_kind(kind):
    return kind in ('L2', 'L2_MLCT', 'L2_LMCT')

# Static fallback bounds (used by nudge buttons and when E0 is unavailable).
# Model.bounds() overrides energy positions with dynamic E0/Tk-relative constraints.
PEAK_BOUNDS = {
    'L3_MLCT': {'o': (790., 875.), 'W': (0.05, 30.), 'I': (0., 200.), 'G': (0., 1.)},
    'L3':      {'o': (800., 880.), 'W': (0.05, 20.), 'I': (0., 200.), 'G': (0., 1.)},
    'L3_LMCT': {'o': (800., 900.), 'W': (0.05, 30.), 'I': (0., 200.), 'G': (0., 1.)},
    'L2_MLCT': {'o': (845., 920.), 'W': (0.05, 30.), 'I': (0., 200.), 'G': (0., 1.)},
    'L2':      {'o': (855., 915.), 'W': (0.05, 20.), 'I': (0., 200.), 'G': (0., 1.)},
    'L2_LMCT': {'o': (855., 930.), 'W': (0.05, 30.), 'I': (0., 200.), 'G': (0., 1.)},
    # Legacy generic CT buckets kept for compatibility with older exports.
    'MLCT':    {'o': (790., 920.), 'W': (0.05, 30.), 'I': (0., 200.), 'G': (0., 1.)},
    'LMCT':    {'o': (845., 915.), 'W': (0.05, 30.), 'I': (0., 200.), 'G': (0., 1.)},
}
# Physical energy ordering low→high:
#   L3 MLCT → L3 → L3 LMCT → L2 MLCT → L2 → L2 LMCT

PEAK_PLOT_COLORS = {
    'L3_MLCT': ['#8e6ad8', '#a17ce5', '#b497ec', '#c8b2f2', '#dccdf8'],
    'L3':      ['#ff8844', '#ffaa66', '#ffcc88', '#ffdd99', '#ffe8bb'],
    'L3_LMCT': ['#4d8cff', '#6aa1ff', '#88b7ff', '#a8ccff', '#c7e0ff'],
    'L2_MLCT': ['#7f59c8', '#9571d6', '#ac8ae2', '#c3a4ed', '#dabff6'],
    'L2':      ['#ee44aa', '#ff66bb', '#ff88cc', '#ffaadd', '#ffccee'],
    'L2_LMCT': ['#2fa8c8', '#4fb9d6', '#72c9e2', '#99d8eb', '#c2e7f4'],
    'MLCT':    ['#bb99ff', '#cc88ee', '#ddaaff', '#eeccff', '#f0ddff'],
    'LMCT':    ['#44aaff', '#66bbff', '#88ccff', '#aaddff', '#cceeFF'],
}

PEAK_SECTION_BG = {
    'L3_MLCT': '#ede7f6',
    'L3':      '#fff3e0',
    'L3_LMCT': '#e3f2fd',
    'L2_MLCT': '#f3ebfb',
    'L2':      '#fce4ec',
    'L2_LMCT': '#e0f4f8',
    'MLCT':    '#ede7f6',
    'LMCT':    '#e3f2fd',
}

# Display order matches physical energy ordering low→high:
# L3 MLCT · L3 · L3 LMCT · L2 MLCT · L2 · L2 LMCT
PEAK_KIND_ORDER = ['L3_MLCT', 'L3', 'L3_LMCT', 'L2_MLCT', 'L2', 'L2_LMCT']

def make_peak(kind, o=854.0, W=1.0, I=0.3, G=0.0, enabled=True):
    """Create a peak parameter dict."""
    constraints = {}
    for field in PEAK_FIELDS:
        lo, hi = PEAK_BOUNDS[kind][field]
        constraints[field] = {'lock': False, 'lo': float(lo), 'hi': float(hi)}
    return {
        'kind': kind,
        'o':    float(o),
        'W':    float(W),
        'I':    float(I),
        'G':    float(G),
        'enabled': bool(enabled),
        'constraints': constraints,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Model class
# ══════════════════════════════════════════════════════════════════════════════

class Model:
    """
    Full L-edge spectral model with dynamic peak list.

    peaks is a list of dicts created by make_peak().
    Only peaks with enabled=True are included in the optimization.
    """

    def __init__(self, zeta=11.507):
        self.zeta = float(zeta)

    def Tk(self, E0):
        """L2 edge onset = E0 + 3/2 * zeta_2p."""
        return float(E0) + 1.5 * self.zeta

    def default(self):
        return {p[0]: p[3] for p in PARAMS_STATIC}

    # ── Pack / unpack ──────────────────────────────────────────────────────

    def pack(self, d, peaks):
        """Pack static params + enabled peak params into 1-D array."""
        v = [d[k] for k in _KEYS_STATIC]
        for pk in peaks:
            if pk['enabled']:
                v += [pk['o'], pk['W'], pk['I'], pk['G']]
        return np.array(v, dtype=float)

    def unpack(self, v, peaks):
        """Unpack 1-D array back to (d_dict, new_peaks_list)."""
        d   = {k: float(v[i]) for i, k in enumerate(_KEYS_STATIC)}
        idx = len(_KEYS_STATIC)
        new_peaks = []
        for pk in peaks:
            npk = dict(pk)
            if pk['enabled']:
                npk['o'] = float(v[idx]);     npk['W'] = float(v[idx + 1])
                npk['I'] = float(v[idx + 2]); npk['G'] = float(v[idx + 3])
                idx += 4
            new_peaks.append(npk)
        return d, new_peaks

    def _calibrated_x(self, x, d):
        return d.get('Eshift', 0.0) + d.get('Escale', 1.0) * x

    def _default_static_constraints(self):
        return {
            p[0]: {'lock': bool(p[7]), 'lo': float(p[4]), 'hi': float(p[5])}
            for p in PARAMS_STATIC
        }

    def _default_peak_constraints(self, kind):
        return {
            field: {'lock': False,
                    'lo': float(PEAK_BOUNDS[kind][field][0]),
                    'hi': float(PEAK_BOUNDS[kind][field][1])}
            for field in PEAK_FIELDS
        }

    def _merge_static_constraints(self, static_constraints=None):
        merged = self._default_static_constraints()
        for key, spec in (static_constraints or {}).items():
            if key in merged:
                merged[key].update(spec or {})
        return merged

    def _peak_constraints(self, pk):
        merged = self._default_peak_constraints(pk['kind'])
        for field, spec in (pk.get('constraints') or {}).items():
            if field in merged:
                merged[field].update(spec or {})
        return merged

    def _apply_constraint(self, lo, hi, value, spec):
        eps = 1e-12
        lo = float(spec.get('lo', lo))
        hi = float(spec.get('hi', hi))
        if spec.get('lock'):
            center = float(value)
            return center - eps, center + eps
        if lo >= hi:
            center = float(value)
            lo = min(lo, hi, center) - eps
            hi = max(lo + eps, hi, center + eps)
        return lo, hi

    def _dynamic_peak_bounds(self, kind, d):
        pb = dict(PEAK_BOUNDS[kind])
        E0 = d['E0'] if d is not None else None
        if E0 is None:
            return pb
        Tk = self.Tk(E0)
        if kind == 'L3_MLCT':
            pb['o'] = (E0 - 15.0, E0 + 4.0)
        elif kind == 'L3':
            pb['o'] = (E0 - 5.0, Tk - 3.0)
        elif kind == 'L3_LMCT':
            pb['o'] = (E0 + 0.2, Tk - 1.5)
        elif kind == 'L2_MLCT':
            pb['o'] = (Tk - 6.0, Tk + 2.0)
        elif kind == 'L2':
            pb['o'] = (Tk - 2.0, Tk + 15.0)
        elif kind == 'L2_LMCT':
            pb['o'] = (Tk + 0.2, Tk + 15.0)
        elif kind == 'MLCT':
            pb['o'] = (E0 - 15.0, Tk - 0.5)
        elif kind == 'LMCT':
            pb['o'] = (E0 + 0.5, Tk - 0.5)
        lo, hi = pb['o']
        if lo >= hi:
            pb['o'] = (min(lo, hi) - 0.5, max(lo, hi) + 0.5)
        return pb

    def bounds(self, peaks, d=None, static_constraints=None):
        """Build bounds list matching pack(), honoring user locks and ranges."""
        merged_static = self._merge_static_constraints(static_constraints)
        cur = dict(self.default())
        if d is not None:
            cur.update(d)
        bds = []
        for key in _KEYS_STATIC:
            lo, hi = next((p[4], p[5]) for p in PARAMS_STATIC if p[0] == key)
            bds.append(self._apply_constraint(lo, hi, cur[key], merged_static[key]))

        for pk in peaks:
            if not pk['enabled']:
                continue
            pb = self._dynamic_peak_bounds(pk['kind'], cur)
            cons = self._peak_constraints(pk)
            for field in PEAK_FIELDS:
                lo, hi = pb[field]
                bds.append(self._apply_constraint(lo, hi, pk[field], cons[field]))
        return bds

    # ── Spectral components ────────────────────────────────────────────────

    def get_bg(self, x, d):
        x_eval = self._calibrated_x(x, d)
        return bg4(x_eval, d['Ek'], d['E0'], self.Tk(d['E0']), d['Et'],
                   d['C'], d['m1'], d['m2'], d['a_log'], d['c_log'],
                   d['b'], d['m4'])

    # ── Branching ratio helper ─────────────────────────────────────────────

    def _l2_scale(self, peaks, br):
        """
        The branch ratio is now treated as a soft target in the fitter rather
        than an exact rescaling rule. Model evaluation therefore leaves the
        entered L2 intensities untouched.
        """
        return 1.0

    def branch_ratio(self, peaks):
        il3 = sum(pk['I'] for pk in peaks if pk['enabled'] and _is_l3_kind(pk['kind']))
        il2 = sum(pk['I'] for pk in peaks if pk['enabled'] and _is_l2_kind(pk['kind']))
        return il2 / il3 if il3 > 1e-12 else np.nan

    # ── Spectral components ────────────────────────────────────────────────

    def get_full(self, x, d, peaks, br=None):
        """Full model: B(x) + edge steps + all enabled peaks."""
        x_eval = self._calibrated_x(x, d)
        Tk     = self.Tk(d['E0'])
        bgv    = bg4(x_eval, d['Ek'], d['E0'], Tk, d['Et'],
                     d['C'], d['m1'], d['m2'], d['a_log'], d['c_log'],
                     d['b'], d['m4'])
        eL3    = cpv(x_eval, d['E0'], d['WL3_Edge'], d['IL3_Edge'],       d['GL3_Edge'])
        eL2    = cpv(x_eval, Tk,      d['WL2_Edge'], d['IL3_Edge'] / 2.0, d['GL2_Edge'])
        tot    = bgv + eL3 + eL2
        l2_sc  = self._l2_scale(peaks, br)
        for pk in peaks:
            if pk['enabled']:
                eff_I = pk['I'] * l2_sc if _is_l2_kind(pk['kind']) else pk['I']
                tot  += pv(x_eval, pk['o'], pk['W'], eff_I, pk['G'])
        return tot

    def get_norm(self, x, y, d):
        """Normalized spectrum: (raw - BG) / (3/2 * IL3_Edge)."""
        bgv = self.get_bg(x, d)
        f   = max(1.5 * d['IL3_Edge'], 1e-12)
        return (y - bgv) / f

    def get_norm_components(self, x, d, peaks, br=None):
        """Normalized spectral components for decomposition plot."""
        x_eval = self._calibrated_x(x, d)
        Tk    = self.Tk(d['E0'])
        f     = max(1.5 * d['IL3_Edge'], 1e-12)
        l2_sc = self._l2_scale(peaks, br)
        c = {
            'eL3': cpv(x_eval, d['E0'], d['WL3_Edge'], d['IL3_Edge'],       d['GL3_Edge']) / f,
            'eL2': cpv(x_eval, Tk,      d['WL2_Edge'], d['IL3_Edge'] / 2.0, d['GL2_Edge']) / f,
        }
        kind_count = {}
        for pk in peaks:
            if pk['enabled']:
                kind  = pk['kind']
                n     = kind_count.get(kind, 0)
                eff_I = pk['I'] * l2_sc if _is_l2_kind(kind) else pk['I']
                c[f'p{kind}_{n}'] = pv(x_eval, pk['o'], pk['W'], eff_I, pk['G']) / f
                kind_count[kind]  = n + 1
        return c

    def get_norm_full_model(self, x, d, peaks, br=None):
        """Normalized total model (no background)."""
        x_eval = self._calibrated_x(x, d)
        Tk    = self.Tk(d['E0'])
        f     = max(1.5 * d['IL3_Edge'], 1e-12)
        l2_sc = self._l2_scale(peaks, br)
        eL3   = cpv(x_eval, d['E0'], d['WL3_Edge'], d['IL3_Edge'],       d['GL3_Edge'])
        eL2   = cpv(x_eval, Tk,      d['WL2_Edge'], d['IL3_Edge'] / 2.0, d['GL2_Edge'])
        tot   = eL3 + eL2
        for pk in peaks:
            if pk['enabled']:
                eff_I = pk['I'] * l2_sc if _is_l2_kind(pk['kind']) else pk['I']
                tot  += pv(x_eval, pk['o'], pk['W'], eff_I, pk['G'])
        return tot / f

    def il3_plus_2il2_norm(self, d, peaks, br=None):
        """Normalized manifold intensity IL3_total + 2*IL2_total."""
        f     = max(1.5 * d['IL3_Edge'], 1e-12)
        l2_sc = self._l2_scale(peaks, br)
        IL3   = sum(pk['I']         for pk in peaks if pk['enabled'] and _is_l3_kind(pk['kind']))
        IL2   = sum(pk['I'] * l2_sc for pk in peaks if pk['enabled'] and _is_l2_kind(pk['kind']))
        return (IL3 + 2.0 * IL2) / f

    # ── Goodness of fit ────────────────────────────────────────────────────

    def chi2_vec(self, v, x, y, peaks, br=None):
        d, pks = self.unpack(v, peaks)
        return float(np.sum((y - self.get_full(x, d, pks, br=br))**2))

    def r2(self, d, peaks, x, y, br=None):
        res  = y - self.get_full(x, d, peaks, br=br)
        ss_r = np.sum(res**2)
        ss_t = np.sum((y - np.mean(y))**2)
        return 1.0 - ss_r / ss_t if ss_t > 0 else 0.0

    # ── Fitting ────────────────────────────────────────────────────────────

    def _stage_peaks(self, peaks, stage):
        out = [dict(pk) for pk in peaks]
        if stage == 'background':
            for pk in out:
                pk['enabled'] = False
        elif stage == 'manifold':
            for pk in out:
                pk['enabled'] = pk['enabled'] and pk['kind'] in ('L3', 'L2')
        return out

    def _weight_vector(self, x_eval, d, stage):
        E0 = d['E0']
        Tk = self.Tk(E0)
        Et = d['Et']
        w = np.ones_like(x_eval, dtype=float)

        pre = x_eval <= E0 - 3.0
        l3_peak = (x_eval >= E0 - 1.0) & (x_eval <= E0 + 3.5)
        inter = (x_eval >= E0 + 4.0) & (x_eval <= Tk - 2.0)
        l2_peak = (x_eval >= Tk - 1.0) & (x_eval <= Tk + 3.5)
        post = x_eval >= max(Tk + 6.0, Et)

        w[pre] = 2.8
        w[inter] = 1.8
        w[post] = 2.2
        w[l3_peak] = 0.9
        w[l2_peak] = 1.0

        if stage == 'background':
            w[l3_peak] *= 0.20
            w[l2_peak] *= 0.25
            w[inter] *= 0.70
        elif stage == 'manifold':
            w[l3_peak] *= 1.15
            w[l2_peak] *= 1.15
        elif stage == 'full':
            w[inter] *= 1.25

        mean_w = np.mean(w) if np.mean(w) > 1e-12 else 1.0
        return w / mean_w

    def _penalty_residuals(self, x, y, d, peaks, br_target=None, stage='full'):
        x_eval = self._calibrated_x(x, d)
        bg = self.get_bg(x, d)
        dx = max(float(np.mean(np.diff(x))), 1e-6)
        signal_scale = max(float(np.ptp(y)), 1e-6)
        slope_scale = max(signal_scale / max(float(x[-1] - x[0]), 1e-6), 1e-6)

        bg_grad = np.gradient(bg, dx)
        bg_curv = np.gradient(bg_grad, dx)
        penalties = []

        if br_target is not None:
            actual_br = self.branch_ratio(peaks)
            if np.isfinite(actual_br):
                penalties.append(4.0 * (actual_br - float(br_target)))

        pre_mask = x_eval <= d['E0'] - 4.0
        post_mask = x_eval >= max(self.Tk(d['E0']) + 6.0, d['Et'])
        for mask in (pre_mask, post_mask):
            if np.any(mask):
                slope = float(np.mean(bg_grad[mask]))
                tol = 0.35 * slope_scale
                over = max(0.0, abs(slope) - tol)
                penalties.append(2.0 * over / max(tol, 1e-9))

        smooth_mask = pre_mask | post_mask
        if np.any(smooth_mask):
            curv_rms = float(np.sqrt(np.mean(bg_curv[smooth_mask]**2)))
            tol = 0.50 * slope_scale / max(float(x[-1] - x[0]), 1e-6)
            over = max(0.0, curv_rms - tol)
            penalties.append(1.5 * over / max(tol, 1e-9))

        if stage == 'background':
            escale_dev = abs(d.get('Escale', 1.0) - 1.0)
            penalties.append(2.0 * escale_dev / 0.01)

        return np.asarray(penalties, dtype=float)

    def residual_vector(self, v, x, y, peaks, stage='full',
                        static_constraints=None, fixed=(), br_target=None):
        d, pks = self.unpack(v, peaks)
        x_eval = self._calibrated_x(x, d)
        y_model = self.get_full(x, d, pks, br=None)

        weights = self._weight_vector(x_eval, d, stage)
        signal_scale = max(float(np.ptp(y)), 1e-6)
        resid = weights * (y - y_model) / signal_scale

        deriv_weight = 0.00 if stage == 'background' else (0.10 if stage == 'manifold' else 0.18)
        if deriv_weight > 0:
            dx = max(float(np.mean(np.diff(x))), 1e-6)
            dy = np.gradient(y, dx)
            dm = np.gradient(y_model, dx)
            grad_scale = max(float(np.ptp(dy)), signal_scale / max(float(x[-1] - x[0]), 1e-6), 1e-6)
            resid_deriv = np.sqrt(weights) * deriv_weight * (dy - dm) / grad_scale
        else:
            resid_deriv = np.array([], dtype=float)

        penalties = self._penalty_residuals(x, y, d, pks, br_target=br_target, stage=stage)
        return np.concatenate([resid, resid_deriv, penalties])

    def _scalar_cost(self, v, x, y, peaks, stage, static_constraints, fixed, br_target):
        r = self.residual_vector(v, x, y, peaks, stage=stage,
                                 static_constraints=static_constraints,
                                 fixed=fixed, br_target=br_target)
        return float(np.dot(r, r))

    def _fit_stage(self, x, y, d0, peaks, fixed=(), br_target=None,
                   static_constraints=None, stage='full', global_init=False):
        v0 = self.pack(d0, peaks)
        bds = list(self.bounds(peaks, d=d0, static_constraints=static_constraints))
        for i, key in enumerate(_KEYS_STATIC):
            if key in fixed:
                center = v0[i]
                bds[i] = (center - 1e-12, center + 1e-12)
        lb = np.array([b[0] for b in bds], dtype=float)
        ub = np.array([b[1] for b in bds], dtype=float)

        if global_init and len(v0) <= 24:
            try:
                de = differential_evolution(
                    lambda vec: self._scalar_cost(vec, x, y, peaks, stage,
                                                  static_constraints, fixed, br_target),
                    bounds=bds, maxiter=18, popsize=8, polish=False,
                    updating='deferred', workers=1)
                v0 = np.clip(de.x, lb, ub)
            except Exception:
                pass

        result = least_squares(
            self.residual_vector, v0,
            args=(x, y, peaks, stage, static_constraints, fixed, br_target),
            bounds=(lb, ub),
            method='trf',
            loss='soft_l1',
            f_scale=0.25,
            x_scale='jac',
            max_nfev=6000)
        return self.unpack(result.x, peaks)

    def fit_once(self, x, y, d0, peaks, fixed=('Ek',), br=None,
                 static_constraints=None, global_init=False):
        """Staged weighted robust least-squares fit with optional global init."""
        d_cur = dict(d0)
        peaks_cur = [dict(pk) for pk in peaks]

        stage_specs = [
            ('background', False),
            ('manifold', False),
            ('full', bool(global_init)),
        ]
        for stage, use_global in stage_specs:
            stage_peaks = self._stage_peaks(peaks_cur, stage)
            d_next, fitted_stage_peaks = self._fit_stage(
                x, y, d_cur, stage_peaks,
                fixed=fixed,
                br_target=br,
                static_constraints=static_constraints,
                stage=stage,
                global_init=use_global)
            merged_peaks = []
            for base_pk, stage_pk in zip(peaks_cur, fitted_stage_peaks):
                merged = dict(base_pk)
                if stage_pk.get('enabled', False):
                    for field in PEAK_FIELDS:
                        merged[field] = stage_pk[field]
                merged_peaks.append(merged)
            d_cur = d_next
            peaks_cur = merged_peaks
        return d_cur, peaks_cur

    def mc_fit(self, x, y, d0, peaks, n=200, spread=0.10,
               fixed=('Ek',), br=None, cb=None, stop_event=None,
               live_cb=None, live_every=10, static_constraints=None,
               global_init=False):
        """
        Monte Carlo fitting: n fits from randomly perturbed starting points.
        Returns list of (d, peaks) tuples for the surviving fits.

        live_cb(d, peaks) is called with the current best fit every
        live_every iterations so the GUI can show intermediate progress.
        """
        v0       = self.pack(d0, peaks)
        n_static = len(_KEYS_STATIC)
        out      = []
        base_bounds = self.bounds(peaks, d=d0, static_constraints=static_constraints)

        for i in range(n):
            if stop_event and stop_event.is_set():
                break
            v = v0.copy()
            # Perturb static params
            for j, k in enumerate(_KEYS_STATIC):
                if k in fixed:
                    continue
                lo, hi = base_bounds[j]
                v[j] = np.clip(
                    v[j] + np.random.uniform(-(hi - lo) * spread,
                                              (hi - lo) * spread), lo, hi)
            # Perturb peak params (use dynamic bounds relative to current d)
            d_tmp, _ = self.unpack(v, peaks)
            dyn_bds  = self.bounds(peaks, d=d_tmp, static_constraints=static_constraints)
            idx      = n_static
            pk_bnd_idx = n_static  # parallel index into dyn_bds
            for pk in peaks:
                if pk['enabled']:
                    for fi, field in enumerate(PEAK_FIELDS):
                        lo, hi = dyn_bds[pk_bnd_idx + fi]
                        v[idx] = np.clip(
                            v[idx] + np.random.uniform(-(hi - lo) * spread,
                                                        (hi - lo) * spread), lo, hi)
                        idx += 1
                    pk_bnd_idx += 4
            try:
                d_try, pks_try = self.unpack(v, peaks)
                d_f, pks_f = self.fit_once(
                    x, y, d_try, pks_try,
                    fixed=fixed,
                    br=br,
                    static_constraints=static_constraints,
                    global_init=global_init and i == 0)
                c2 = self.chi2_vec(self.pack(d_f, pks_f), x, y, pks_f, br=br)
                out.append((c2, d_f, pks_f))
            except Exception:
                pass
            if cb:
                cb(i + 1, n)
            # ── Live plot callback every live_every iterations ──────────────
            if live_cb and out and (i + 1) % live_every == 0:
                # Send current best fit (lowest chi2 so far)
                best = min(out, key=lambda r: r[0])
                live_cb(best[1], best[2])

        # Discard outlier fits (chi2 > 5× median)
        if out:
            chi2s = np.array([r[0] for r in out])
            med   = np.median(chi2s)
            out   = [r for r in out if r[0] <= 5.0 * med]
        return [(r[1], r[2]) for r in out]

    def mc_stats(self, fits):
        """
        Compute mean and std-dev over MC ensemble.
        Returns (mu_d, sd_d, mu_peaks, sd_peaks).
        mu_peaks / sd_peaks have the same list structure as fits[0][1].
        """
        if not fits:
            return {}, {}, [], []
        ds   = [f[0] for f in fits]
        mu_d = {k: float(np.mean([d[k] for d in ds])) for k in _KEYS_STATIC}
        sd_d = {k: float(np.std( [d[k] for d in ds])) for k in _KEYS_STATIC}
        pks0 = fits[0][1]
        mu_pk, sd_pk = [], []
        for pi, pk in enumerate(pks0):
            mp = dict(pk)
            sp = dict(pk)
            for field in PEAK_FIELDS:
                vals     = [f[1][pi][field] for f in fits]
                mp[field] = float(np.mean(vals))
                sp[field] = float(np.std(vals))
            mu_pk.append(mp)
            sd_pk.append(sp)
        return mu_d, sd_d, mu_pk, sd_pk

    # ── Auto-estimate starting parameters ─────────────────────────────────

    def auto_guess(self, x, y, element='Ni'):
        """Heuristic starting parameters + default peak list from data."""
        self.zeta = SOC.get(element, self.zeta)
        d = self.default()
        d['Ek'] = float(x[0])

        # ── Find tallest peak in L3 region (lower 60% of energy range) ─────
        x_lo     = float(x[0])
        x_hi     = x_lo + (float(x[-1]) - x_lo) * 0.60
        mask_l3  = (x >= x_lo) & (x <= x_hi)
        if mask_l3.sum() > 0:
            peak_idx = int(np.argmax(y[mask_l3]))
            peak_en  = float(x[mask_l3][peak_idx])
            peak_val = float(y[mask_l3][peak_idx])
        else:
            peak_idx = int(np.argmax(y))
            peak_en  = float(x[peak_idx])
            peak_val = float(y[peak_idx])

        # ── E0 = max-gradient point just below peak ─────────────────────────
        onset_mask = (x >= peak_en - 8.0) & (x <= peak_en)
        if onset_mask.sum() > 2:
            grad = np.gradient(y[onset_mask], x[onset_mask])
            E0   = float(x[onset_mask][int(np.argmax(grad))])
        else:
            E0 = peak_en - 2.0

        d['E0'] = E0
        d['Et'] = E0 + 1.5 * self.zeta + 25.0

        # ── Pre-edge level ──────────────────────────────────────────────────
        pre   = y[x < E0 - 5.0]
        C_val = float(np.mean(pre[-min(10, max(len(pre), 1)):])) if len(pre) else float(y[0])
        d['C'] = C_val

        # ── Edge step height ────────────────────────────────────────────────
        post = y[x > E0 + 1.5 * self.zeta + 5.0]
        eh   = ((float(np.mean(post[:min(15, len(post))])) - C_val)
                if len(post) else peak_val - C_val)
        eh   = max(eh, 0.01)
        d['IL3_Edge'] = max(eh * 0.55, 0.001)

        # ── L2 peak position ────────────────────────────────────────────────
        Tk      = E0 + 1.5 * self.zeta
        mask_l2 = (x >= Tk - 2.0) & (x <= Tk + 8.0)
        oL2     = (float(x[mask_l2][int(np.argmax(y[mask_l2]))])
                   if mask_l2.sum() > 0 else Tk + 2.0)

        # ── Default peaks: 1 L3 + 1 L2 ─────────────────────────────────────
        peaks = [
            make_peak('L3', o=peak_en, W=1.0, I=(peak_val - C_val) * 0.55, G=0.0),
            make_peak('L2', o=oL2,     W=2.0, I=eh * 0.10,                 G=0.0),
        ]
        return d, peaks


# ══════════════════════════════════════════════════════════════════════════════
#  Scan data loader
# ══════════════════════════════════════════════════════════════════════════════

class Scan:
    """A single XAS scan loaded from a two-column text file."""

    def __init__(self, path, x, y):
        self.path = path
        self.name = os.path.basename(path)
        self.x    = np.asarray(x, dtype=float)
        self.y    = np.asarray(y, dtype=float)

    @classmethod
    def from_file(cls, path):
        xs, ys = [], []
        with open(path, 'r', errors='replace') as fh:
            for line in fh:
                line = line.strip()
                if not line or line[0] in '#%!;':
                    continue
                parts = line.replace(',', ' ').replace('\t', ' ').split()
                if len(parts) >= 2:
                    try:
                        xs.append(float(parts[0]))
                        ys.append(float(parts[1]))
                    except ValueError:
                        pass
        if not xs:
            raise ValueError(f"No numeric two-column data found in:\n{path}")
        x, y  = np.array(xs), np.array(ys)
        order = np.argsort(x)
        return cls(path, x[order], y[order])


def sum_scans(scans):
    """Interpolate all scans onto the first scan's energy grid and sum."""
    if not scans:
        return None, None
    x_ref = scans[0].x
    y_sum = scans[0].y.copy()
    for s in scans[1:]:
        y_sum += np.interp(x_ref, s.x, s.y, left=s.y[0], right=s.y[-1])
    return x_ref, y_sum


# ══════════════════════════════════════════════════════════════════════════════
#  Static parameter panel  (Background + Edge)
# ══════════════════════════════════════════════════════════════════════════════

class ParamPanel(tk.Frame):
    """
    Scrollable panel of Background + Edge parameter entry fields.
    Redraw fires on FocusOut / Return / Tab / nudge — NOT on every keystroke.
    """

    def __init__(self, master, on_change=None, **kw):
        super().__init__(master, **kw)
        self._on_change = on_change
        self._vars      = {}
        self._entries   = {}
        self._constraints = {
            p[0]: {'lock': bool(p[7]), 'lo': float(p[4]), 'hi': float(p[5])}
            for p in PARAMS_STATIC
        }

        canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        vsb    = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner  = tk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=inner, anchor='nw')

        inner.bind('<Configure>',
                   lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',
                    lambda e: canvas.itemconfig(win_id, width=e.width))
        canvas.bind_all('<MouseWheel>',
                        lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), 'units'))

        current_section = None
        row = 0
        for (key, section, label, default, lo, hi, step, fixed) in PARAMS_STATIC:
            if section != current_section:
                current_section = section
                hdr = tk.Label(inner, text=f'── {section} ──',
                               font=('TkDefaultFont', 9, 'bold'),
                               fg='#444', anchor='w')
                hdr.grid(row=row, column=0, columnspan=3, sticky='we',
                         padx=4, pady=(8, 2))
                row += 1

            var = tk.StringVar(value=f'{default:.6g}')
            self._vars[key] = var

            lbl = tk.Label(inner, text=label, anchor='w',
                           font=('TkFixedFont', 8),
                           fg='#555' if fixed else 'black')
            lbl.grid(row=row, column=0, sticky='w', padx=(6, 2), pady=1)

            ent = tk.Entry(inner, textvariable=var, width=11,
                           font=('TkFixedFont', 9),
                           bg='#e8e8e8' if fixed else 'white',
                           state='disabled' if fixed else 'normal')
            ent.grid(row=row, column=1, sticky='we', padx=2, pady=1)
            self._entries[key] = ent

            if not fixed:
                bf = tk.Frame(inner)
                bf.grid(row=row, column=2, sticky='w', padx=2)
                tk.Button(bf, text='▲', width=1, pady=0,
                          font=('TkFixedFont', 7),
                          command=lambda k=key, s=step: self._nudge(k, +s)
                          ).pack(side=tk.LEFT)
                tk.Button(bf, text='▼', width=1, pady=0,
                          font=('TkFixedFont', 7),
                          command=lambda k=key, s=step: self._nudge(k, -s)
                          ).pack(side=tk.LEFT)

            def _commit(event, k=key):
                self._changed()
            ent.bind('<FocusOut>', _commit)
            ent.bind('<Return>',   _commit)
            ent.bind('<Tab>',      _commit)
            row += 1

        inner.columnconfigure(1, weight=1)

    def _nudge(self, key, delta):
        try:
            val = float(self._vars[key].get()) + delta
            lo, hi = next((p[4], p[5]) for p in PARAMS_STATIC if p[0] == key)
            self._vars[key].set(f'{max(lo, min(hi, val)):.6g}')
            self._changed()
        except ValueError:
            pass

    def _changed(self):
        if self._on_change:
            self._on_change()

    def get_params(self):
        d = {}
        for p in PARAMS_STATIC:
            key, default = p[0], p[3]
            try:
                d[key] = float(self._vars[key].get())
            except ValueError:
                d[key] = default
        return d

    def set_params(self, d):
        for key in _KEYS_STATIC:
            if key in d:
                self._vars[key].set(f'{d[key]:.6g}')

    def get_constraints(self):
        return {k: dict(v) for k, v in self._constraints.items()}

    def set_constraints(self, constraints):
        for key, spec in (constraints or {}).items():
            if key in self._constraints:
                self._constraints[key].update(spec or {})

    def set_entry_bg(self, key, color):
        if key in self._entries:
            self._entries[key].config(bg=color)


# ══════════════════════════════════════════════════════════════════════════════
#  Dynamic peak panel
# ══════════════════════════════════════════════════════════════════════════════

class _PeakRow(tk.Frame):
    """One row in the DynamicPeakPanel representing a single peak."""

    FIELD_STEP = {'o': 0.1, 'W': 0.1, 'I': 0.01, 'G': 0.05}
    FIELD_LABEL = {'o': 'o', 'W': 'W', 'I': 'I', 'G': 'G'}

    def __init__(self, master, kind, index, peak_dict,
                 on_change=None, on_remove=None, **kw):
        super().__init__(master, bd=1, relief='groove',
                         bg=PEAK_SECTION_BG.get(kind, '#f5f5f5'), **kw)
        self._kind      = kind
        self._index     = index
        self._on_change = on_change
        self._on_remove = on_remove
        self._vars      = {}
        self._constraints = {}

        # ── Row 1: header (enable checkbox, label, remove button) ──────────
        hdr = tk.Frame(self, bg=self['bg'])
        hdr.pack(fill=tk.X, padx=2, pady=(2, 0))

        self._enabled_var = tk.BooleanVar(value=peak_dict.get('enabled', True))
        tk.Checkbutton(hdr, variable=self._enabled_var, bg=self['bg'],
                       command=self._changed).pack(side=tk.LEFT)
        tk.Label(hdr, text=f'{kind} #{index + 1}',
                 font=('TkDefaultFont', 8, 'bold'),
                 bg=self['bg']).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(hdr, text='✕', font=('TkFixedFont', 8), fg='#cc0000',
                  bd=0, padx=2, pady=0, bg=self['bg'],
                  command=self._remove).pack(side=tk.RIGHT)

        # ── Row 2: parameter entries ───────────────────────────────────────
        fld = tk.Frame(self, bg=self['bg'])
        fld.pack(fill=tk.X, padx=4, pady=(0, 2))

        for col, field in enumerate(PEAK_FIELDS):
            lo, hi = PEAK_BOUNDS[kind][field]
            spec = ((peak_dict.get('constraints') or {}).get(field) or
                    {'lock': False, 'lo': lo, 'hi': hi})
            self._constraints[field] = {
                'lock': bool(spec.get('lock', False)),
                'lo': float(spec.get('lo', lo)),
                'hi': float(spec.get('hi', hi)),
            }
            var = tk.StringVar(value=f'{peak_dict.get(field, 0.0):.5g}')
            self._vars[field] = var

            tk.Label(fld, text=self.FIELD_LABEL[field] + ':',
                     font=('TkFixedFont', 8), bg=self['bg']
                     ).grid(row=0, column=col * 3, sticky='e', padx=(4, 0))

            ent = tk.Entry(fld, textvariable=var, width=7,
                           font=('TkFixedFont', 8))
            ent.grid(row=0, column=col * 3 + 1, padx=1)

            bf = tk.Frame(fld, bg=self['bg'])
            bf.grid(row=0, column=col * 3 + 2)

            step = self.FIELD_STEP[field]
            tk.Button(bf, text='▲', width=1, pady=0, font=('TkFixedFont', 6),
                      command=lambda f=field, s=step: self._nudge(f, +s)
                      ).pack(side=tk.TOP)
            tk.Button(bf, text='▼', width=1, pady=0, font=('TkFixedFont', 6),
                      command=lambda f=field, s=step: self._nudge(f, -s)
                      ).pack(side=tk.TOP)

            def _commit(event, fi=field):
                self._changed()
            ent.bind('<FocusOut>', _commit)
            ent.bind('<Return>',   _commit)
            ent.bind('<Tab>',      _commit)

    def _nudge(self, field, delta):
        try:
            lo, hi = PEAK_BOUNDS[self._kind][field]
            val = float(self._vars[field].get()) + delta
            self._vars[field].set(f'{max(lo, min(hi, val)):.5g}')
            self._changed()
        except ValueError:
            pass

    def _changed(self):
        if self._on_change:
            self._on_change()

    def _remove(self):
        if self._on_remove:
            self._on_remove(self)

    def get_peak(self):
        d = {'kind': self._kind, 'enabled': self._enabled_var.get()}
        for field in PEAK_FIELDS:
            lo, hi = PEAK_BOUNDS[self._kind][field]
            try:
                d[field] = float(self._vars[field].get())
            except ValueError:
                d[field] = (lo + hi) / 2.0
        d['constraints'] = {k: dict(v) for k, v in self._constraints.items()}
        return d

    def set_peak(self, peak_dict):
        self._enabled_var.set(peak_dict.get('enabled', True))
        for field in PEAK_FIELDS:
            if field in peak_dict:
                self._vars[field].set(f'{peak_dict[field]:.5g}')
            lo, hi = PEAK_BOUNDS[self._kind][field]
            spec = ((peak_dict.get('constraints') or {}).get(field) or {})
            self._constraints[field] = {
                'lock': bool(spec.get('lock', False)),
                'lo': float(spec.get('lo', lo)),
                'hi': float(spec.get('hi', hi)),
            }


class DynamicPeakPanel(tk.Frame):
    """
        Scrollable panel with dynamically add-able / removable peaks.
        Peaks are grouped into edge-resolved white-line / CT sections.
    """

    def __init__(self, master, on_change=None, **kw):
        super().__init__(master, **kw)
        self._on_change  = on_change
        self._rows       = []    # list of _PeakRow widgets (in display order)
        self._section_frames = {}  # kind -> frame holding rows for that kind

        # ── Scrollable canvas ──────────────────────────────────────────────
        self._canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient='vertical', command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._inner = tk.Frame(self._canvas)
        self._win_id = self._canvas.create_window((0, 0), window=self._inner, anchor='nw')

        self._inner.bind('<Configure>',
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox('all')))
        self._canvas.bind('<Configure>',
            lambda e: self._canvas.itemconfig(self._win_id, width=e.width))
        self._canvas.bind_all('<MouseWheel>',
            lambda e: self._canvas.yview_scroll(int(-1*(e.delta/120)), 'units'))

        self._build_sections()

    def _build_sections(self):
        """Build one collapsible section per peak kind."""
        for kind in PEAK_KIND_ORDER:
            bg = PEAK_SECTION_BG[kind]

            hdr = tk.Frame(self._inner, bg=bg, bd=1, relief='raised')
            hdr.pack(fill=tk.X, pady=(6, 0), padx=2)

            tk.Label(hdr, text=f'── {kind} Peaks ──',
                     font=('TkDefaultFont', 9, 'bold'),
                     bg=bg, fg='#333').pack(side=tk.LEFT, padx=6, pady=2)

            tk.Button(hdr, text=f'+ Add {kind}',
                      font=('TkDefaultFont', 8),
                      bg=bg, bd=1, relief='raised',
                      command=lambda k=kind: self._add_peak_default(k)
                      ).pack(side=tk.RIGHT, padx=4, pady=2)

            sec = tk.Frame(self._inner)
            sec.pack(fill=tk.X, padx=4)
            self._section_frames[kind] = sec

    def _kind_rows(self, kind):
        return [r for r in self._rows if r._kind == kind]

    def _add_peak_default(self, kind):
        # Default positions reflect physical energy ordering for Ni L-edge:
        #   L3 MLCT (~851) → L3 white line (~854) → L3 LMCT (~857)
        #   → L2 MLCT (~868) → L2 white line (~871) → L2 LMCT (~874)
        defaults = {
            'L3_MLCT': make_peak('L3_MLCT', o=851.0, W=1.5, I=0.05, G=0.0),
            'L3':      make_peak('L3',      o=854.0, W=1.0, I=0.30, G=0.0),
            'L3_LMCT': make_peak('L3_LMCT', o=857.0, W=1.6, I=0.06, G=0.0),
            'L2_MLCT': make_peak('L2_MLCT', o=868.0, W=1.6, I=0.04, G=0.0),
            'L2':      make_peak('L2',      o=871.0, W=2.0, I=0.15, G=0.0),
            'L2_LMCT': make_peak('L2_LMCT', o=874.0, W=1.8, I=0.05, G=0.0),
        }
        self.add_peak(kind, defaults[kind])

    def add_peak(self, kind, peak_dict=None):
        """Add a peak row for the given kind."""
        if peak_dict is None:
            peak_dict = make_peak(kind)
        sec   = self._section_frames[kind]
        index = len(self._kind_rows(kind))
        row   = _PeakRow(sec, kind, index, peak_dict,
                         on_change=self._changed,
                         on_remove=self._remove_row)
        row.pack(fill=tk.X, pady=2)
        self._rows.append(row)
        self._changed()

    def _remove_row(self, row):
        self._rows.remove(row)
        row.destroy()
        # Re-index remaining rows of the same kind
        kind = row._kind
        for i, r in enumerate(self._kind_rows(kind)):
            r._index = i
        self._changed()

    def _changed(self):
        if self._on_change:
            self._on_change()

    def get_peaks(self):
        """Return ordered list of peak dicts from all rows."""
        # Return in PEAK_KIND_ORDER order for consistency
        result = []
        for kind in PEAK_KIND_ORDER:
            for r in self._kind_rows(kind):
                result.append(r.get_peak())
        return result

    def set_peaks(self, peaks_list):
        """Replace all rows with the given peaks list."""
        # Destroy existing rows
        for r in self._rows:
            r.destroy()
        self._rows.clear()
        # Add new rows
        for pk in peaks_list:
            self.add_peak(pk['kind'], pk)
        self._changed()

    def update_peak_values(self, peaks_list):
        """
        Update numeric values in existing rows without destroying/recreating them.
        Used for live MC plot updates.  Only updates rows whose kind matches;
        falls back to set_peaks() if the structure has changed.
        """
        # Build ordered list of existing rows (same order as get_peaks)
        ordered = []
        for kind in PEAK_KIND_ORDER:
            ordered.extend(self._kind_rows(kind))

        if len(ordered) != len(peaks_list):
            # Structure mismatch — do a full rebuild
            self.set_peaks(peaks_list)
            return

        for row, pk in zip(ordered, peaks_list):
            if row._kind != pk['kind']:
                self.set_peaks(peaks_list)
                return
            row.set_peak(pk)

    def clear(self):
        for r in self._rows:
            r.destroy()
        self._rows.clear()


class ConstraintDialog(tk.Toplevel):
    """Scrollable dialog for editing locks and numeric bounds."""

    def __init__(self, master, params, param_constraints, peaks):
        super().__init__(master)
        self.title('Fit Constraints')
        self.geometry('860x720')
        self.minsize(760, 520)
        self.transient(master)
        self.grab_set()

        self._result = None
        self._param_vars = {}
        self._peak_vars = []

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        params_tab = self._make_scrollable_tab(nb)
        peaks_tab = self._make_scrollable_tab(nb)
        nb.add(params_tab['outer'], text='Static Parameters')
        nb.add(peaks_tab['outer'], text='Peak Parameters')

        self._build_param_rows(params_tab['inner'], params, param_constraints)
        self._build_peak_rows(peaks_tab['inner'], peaks)

        btns = tk.Frame(self)
        btns.pack(fill=tk.X, padx=8, pady=(0, 8))
        tk.Button(btns, text='Cancel', command=self._cancel).pack(side=tk.RIGHT, padx=4)
        tk.Button(btns, text='Apply', command=self._apply, bg='#dff0d8').pack(side=tk.RIGHT, padx=4)

        self.bind('<Escape>', lambda e: self._cancel())

    def _make_scrollable_tab(self, notebook):
        outer = tk.Frame(notebook)
        canvas = tk.Canvas(outer, bd=0, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(win_id, width=e.width))
        return {'outer': outer, 'inner': inner}

    def _add_headers(self, parent):
        headers = [('Parameter', 0), ('Value', 1), ('Lock', 2), ('Min', 3), ('Max', 4)]
        for text, col in headers:
            tk.Label(parent, text=text, font=('TkDefaultFont', 8, 'bold')).grid(
                row=0, column=col, sticky='w', padx=4, pady=(4, 2))

    def _build_param_rows(self, parent, params, constraints):
        self._add_headers(parent)
        row = 1
        current_section = None
        for key, section, label, default, lo, hi, step, fixed in PARAMS_STATIC:
            if section != current_section:
                current_section = section
                tk.Label(parent, text=section, font=('TkDefaultFont', 8, 'bold'),
                         fg='#335').grid(row=row, column=0, columnspan=5, sticky='w',
                                         padx=4, pady=(8, 2))
                row += 1
            spec = dict(constraints.get(key, {'lock': fixed, 'lo': lo, 'hi': hi}))
            lock_var = tk.BooleanVar(value=bool(spec.get('lock', fixed)))
            lo_var = tk.StringVar(value=f"{float(spec.get('lo', lo)):.6g}")
            hi_var = tk.StringVar(value=f"{float(spec.get('hi', hi)):.6g}")
            self._param_vars[key] = {'lock': lock_var, 'lo': lo_var, 'hi': hi_var}
            tk.Label(parent, text=label, anchor='w').grid(row=row, column=0, sticky='w', padx=4, pady=1)
            tk.Label(parent, text=f"{params.get(key, default):.6g}",
                     font=('TkFixedFont', 8), fg='#555').grid(row=row, column=1, sticky='w', padx=4, pady=1)
            tk.Checkbutton(parent, variable=lock_var).grid(row=row, column=2, sticky='w', padx=4, pady=1)
            tk.Entry(parent, textvariable=lo_var, width=10, font=('TkFixedFont', 8)).grid(
                row=row, column=3, sticky='we', padx=4, pady=1)
            tk.Entry(parent, textvariable=hi_var, width=10, font=('TkFixedFont', 8)).grid(
                row=row, column=4, sticky='we', padx=4, pady=1)
            row += 1
        parent.columnconfigure(0, weight=1)

    def _build_peak_rows(self, parent, peaks):
        self._add_headers(parent)
        row = 1
        for pi, pk in enumerate(peaks):
            kind = pk['kind']
            title = f'{kind} #{pi + 1}'
            tk.Label(parent, text=title, font=('TkDefaultFont', 8, 'bold'),
                     fg='#335').grid(row=row, column=0, columnspan=5, sticky='w',
                                     padx=4, pady=(8, 2))
            row += 1
            peak_spec = {}
            for field in PEAK_FIELDS:
                default_lo, default_hi = PEAK_BOUNDS[kind][field]
                spec = ((pk.get('constraints') or {}).get(field) or
                        {'lock': False, 'lo': default_lo, 'hi': default_hi})
                lock_var = tk.BooleanVar(value=bool(spec.get('lock', False)))
                lo_var = tk.StringVar(value=f"{float(spec.get('lo', default_lo)):.6g}")
                hi_var = tk.StringVar(value=f"{float(spec.get('hi', default_hi)):.6g}")
                peak_spec[field] = {'lock': lock_var, 'lo': lo_var, 'hi': hi_var}
                tk.Label(parent, text=field, anchor='w').grid(row=row, column=0, sticky='w', padx=18, pady=1)
                tk.Label(parent, text=f"{pk.get(field, 0.0):.6g}",
                         font=('TkFixedFont', 8), fg='#555').grid(row=row, column=1, sticky='w', padx=4, pady=1)
                tk.Checkbutton(parent, variable=lock_var).grid(row=row, column=2, sticky='w', padx=4, pady=1)
                tk.Entry(parent, textvariable=lo_var, width=10, font=('TkFixedFont', 8)).grid(
                    row=row, column=3, sticky='we', padx=4, pady=1)
                tk.Entry(parent, textvariable=hi_var, width=10, font=('TkFixedFont', 8)).grid(
                    row=row, column=4, sticky='we', padx=4, pady=1)
                row += 1
            self._peak_vars.append(peak_spec)
        parent.columnconfigure(0, weight=1)

    def _collect_result(self):
        param_constraints = {}
        for key, spec in self._param_vars.items():
            try:
                lo = float(spec['lo'].get())
                hi = float(spec['hi'].get())
            except ValueError:
                raise ValueError(f'Invalid range for {key}.')
            if lo >= hi and not spec['lock'].get():
                raise ValueError(f'Lower bound must be smaller than upper bound for {key}.')
            param_constraints[key] = {'lock': bool(spec['lock'].get()), 'lo': lo, 'hi': hi}

        peak_constraints = []
        for pi, peak_spec in enumerate(self._peak_vars):
            cons = {}
            for field, spec in peak_spec.items():
                try:
                    lo = float(spec['lo'].get())
                    hi = float(spec['hi'].get())
                except ValueError:
                    raise ValueError(f'Invalid peak range for peak #{pi + 1} field {field}.')
                if lo >= hi and not spec['lock'].get():
                    raise ValueError(f'Lower bound must be smaller than upper bound for peak #{pi + 1} field {field}.')
                cons[field] = {'lock': bool(spec['lock'].get()), 'lo': lo, 'hi': hi}
            peak_constraints.append(cons)
        return param_constraints, peak_constraints

    def _apply(self):
        try:
            self._result = self._collect_result()
        except Exception as e:
            messagebox.showerror('Constraint error', str(e), parent=self)
            return
        self.destroy()

    def _cancel(self):
        self._result = None
        self.destroy()

    @property
    def result(self):
        return self._result


# ══════════════════════════════════════════════════════════════════════════════
#  Main application
# ══════════════════════════════════════════════════════════════════════════════

class LEdgeNormApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('L-Edge XAS Normalizer')
        self.geometry('1450x860')
        self.minsize(1100, 680)

        # ── State ──────────────────────────────────────────────────────────
        self._scans      = []
        self._x_sum      = None
        self._y_sum      = None       # always stored in SCALED units (÷ _y_scale)
        self._y_scale    = 1.0        # factor applied: y_physical = y_sum * _y_scale
        self._mc_results = []
        self._mc_mean    = None
        self._mc_std_d   = None
        self._mc_mean_pk = None
        self._mc_std_pk  = None
        self._best_d     = None
        self._best_peaks = None
        self._stop_event = threading.Event()

        # ── Model ──────────────────────────────────────────────────────────
        self._element_var  = tk.StringVar(value='Ni')
        self._symmetry_var = tk.StringVar(value='Oh')
        self._lock_e0_var  = tk.BooleanVar(value=True)
        self._lock_br_var  = tk.BooleanVar(value=True)   # L2/L3 peak ratio locked by default
        self._br_var       = tk.StringVar(value='0.5')   # statistical 2:1 manifold branching ratio
        self._mc_n_var     = tk.StringVar(value='200')
        self._spread_var   = tk.StringVar(value='0.10')
        self._global_init_var = tk.BooleanVar(value=True)
        self._model        = Model(zeta=SOC['Ni'])

        self._build_menu()
        self._build_toolbar()
        self._build_body()
        self._build_statusbar()

    # ══════════════════════════════════════════════════════════════════════
    #  UI construction
    # ══════════════════════════════════════════════════════════════════════

    def _build_menu(self):
        mb = tk.Menu(self)
        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label='Load Scan(s)…',        accelerator='Ctrl+O',
                       command=self._load_scans)
        fm.add_separator()
        fm.add_command(label='Save Normalized CSV…',  accelerator='Ctrl+S',
                       command=self._save_normalized)
        fm.add_command(label='Save Fit Parameters…',
                       command=self._save_params)
        fm.add_command(label='Save Quanty Seed Bundle…',
                       command=self._save_quanty_bundle)
        fm.add_separator()
        fm.add_command(label='Exit', command=self.destroy)
        mb.add_cascade(label='File', menu=fm)
        hm = tk.Menu(mb, tearoff=0)
        hm.add_command(label='About', command=self._show_about)
        mb.add_cascade(label='Help', menu=hm)
        self.config(menu=mb)
        self.bind_all('<Control-o>', lambda _: self._load_scans())
        self.bind_all('<Control-s>', lambda _: self._save_normalized())

    def _build_toolbar(self):
        bar = tk.Frame(self, bd=1, relief='raised', padx=6, pady=4)
        bar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(bar, text='Element:').pack(side=tk.LEFT, padx=(0, 2))
        elem_cb = ttk.Combobox(bar, textvariable=self._element_var,
                               values=list(SOC.keys()) + ['Custom'],
                               state='readonly', width=7)
        elem_cb.pack(side=tk.LEFT, padx=(0, 6))
        elem_cb.bind('<<ComboboxSelected>>', self._on_element_change)

        tk.Label(bar, text='zeta_2p (eV):').pack(side=tk.LEFT)
        self._zeta_var = tk.StringVar(value=f'{SOC["Ni"]:.3f}')
        zeta_ent = tk.Entry(bar, textvariable=self._zeta_var, width=7,
                            font=('TkFixedFont', 9))
        zeta_ent.pack(side=tk.LEFT, padx=(2, 4))
        zeta_ent.bind('<FocusOut>', lambda e: self._on_zeta_change())
        zeta_ent.bind('<Return>',   lambda e: self._on_zeta_change())

        ttk.Separator(bar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=6)

        tk.Label(bar, text='Symmetry:').pack(side=tk.LEFT, padx=(0, 2))
        ttk.Combobox(bar, textvariable=self._symmetry_var,
                     values=SUPPORTED_SITE_SYMMETRIES,
                     state='readonly', width=6
                     ).pack(side=tk.LEFT, padx=(0, 4))

        ttk.Separator(bar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=6)

        tk.Checkbutton(bar, text='Lock E0', variable=self._lock_e0_var
                       ).pack(side=tk.LEFT, padx=(4, 2))

        ttk.Separator(bar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=6)

        tk.Checkbutton(bar, text='Lock L2/L3 BR:', variable=self._lock_br_var
                       ).pack(side=tk.LEFT, padx=(4, 0))
        tk.Entry(bar, textvariable=self._br_var, width=5,
                 font=('TkFixedFont', 9)).pack(side=tk.LEFT, padx=(2, 2))
        tk.Label(bar, text='(L2÷L3)', font=('TkFixedFont', 8), fg='#555'
                 ).pack(side=tk.LEFT, padx=(0, 4))

        ttk.Separator(bar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=6)

        tk.Label(bar, text='MC fits:').pack(side=tk.LEFT)
        tk.Entry(bar, textvariable=self._mc_n_var, width=6,
                 font=('TkFixedFont', 9)).pack(side=tk.LEFT, padx=(2, 4))
        tk.Label(bar, text='Spread:').pack(side=tk.LEFT)
        tk.Entry(bar, textvariable=self._spread_var, width=5,
                 font=('TkFixedFont', 9)).pack(side=tk.LEFT, padx=(2, 8))
        tk.Checkbutton(bar, text='Global init', variable=self._global_init_var
                       ).pack(side=tk.LEFT, padx=(2, 2))

        ttk.Separator(bar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=6)

        tk.Button(bar, text='Load Scans',    command=self._load_scans,
                  bg='#d0e8ff').pack(side=tk.LEFT, padx=3)
        tk.Button(bar, text='Load from Stack…', command=self._load_from_stack,
                  bg='#c8f0ff').pack(side=tk.LEFT, padx=3)
        tk.Button(bar, text='Sum Selected',  command=self._sum_selected,
                  bg='#d0ffe8').pack(side=tk.LEFT, padx=3)

        ttk.Separator(bar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=6)

        tk.Button(bar, text='Auto-Estimate', command=self._auto_estimate,
                  bg='#fff3b0').pack(side=tk.LEFT, padx=3)
        tk.Button(bar, text='Constraints…',  command=self._edit_constraints,
                  bg='#f0e0ff').pack(side=tk.LEFT, padx=3)
        tk.Button(bar, text='Fit Once',      command=self._fit_once,
                  bg='#ffddb0').pack(side=tk.LEFT, padx=3)
        tk.Button(bar, text='Monte Carlo',   command=self._run_mc,
                  bg='#ffd0d0').pack(side=tk.LEFT, padx=3)
        tk.Button(bar, text='STOP',          command=self._stop_mc,
                  bg='#ff8888').pack(side=tk.LEFT, padx=3)

        ttk.Separator(bar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=6)

        tk.Button(bar, text='Export Norm.', command=self._save_normalized,
                  bg='#e8d0ff').pack(side=tk.LEFT, padx=3)
        tk.Button(bar, text='Export Quanty', command=self._save_quanty_bundle,
                  bg='#d8d8ff').pack(side=tk.LEFT, padx=3)

    def _build_body(self):
        pw = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief='raised', sashwidth=5)
        pw.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── Left: scan list ────────────────────────────────────────────────
        left = tk.Frame(pw, width=190)
        pw.add(left, minsize=160)

        tk.Label(left, text='Loaded Scans',
                 font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', padx=4, pady=(4, 2))
        lf  = tk.Frame(left)
        lf.pack(fill=tk.BOTH, expand=True, padx=4)
        vsb = ttk.Scrollbar(lf)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._scan_lb = tk.Listbox(lf, selectmode=tk.EXTENDED,
                                   yscrollcommand=vsb.set,
                                   font=('TkFixedFont', 8),
                                   exportselection=False, height=14)
        self._scan_lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.config(command=self._scan_lb.yview)
        self._scan_lb.bind('<<ListboxSelect>>', self._on_scan_select)

        bf = tk.Frame(left)
        bf.pack(fill=tk.X, padx=4, pady=4)
        tk.Button(bf, text='Load',   width=7, command=self._load_scans).pack(side=tk.LEFT)
        tk.Button(bf, text='Remove', width=7, command=self._remove_selected).pack(side=tk.LEFT, padx=2)

        tk.Label(left, text='Select scans then\n"Sum Selected" above.',
                 font=('TkDefaultFont', 8), fg='gray', justify='left'
                 ).pack(anchor='w', padx=6, pady=4)

        self._stats_lbl = tk.Label(left, text='No data', justify='left',
                                   font=('TkFixedFont', 8), fg='#333',
                                   wraplength=180, anchor='nw')
        self._stats_lbl.pack(anchor='w', padx=6, pady=2)

        tk.Label(left, text='── MC Results ──',
                 font=('TkDefaultFont', 8, 'bold'), fg='#333'
                 ).pack(anchor='w', padx=4, pady=(8, 0))
        self._mc_lbl = tk.Label(left, text='', justify='left',
                                font=('TkFixedFont', 8), fg='#336',
                                wraplength=180, anchor='nw')
        self._mc_lbl.pack(anchor='w', padx=6, pady=2)

        # ── Center: matplotlib plots ───────────────────────────────────────
        center = tk.Frame(pw)
        pw.add(center, minsize=450)

        self._fig     = Figure(figsize=(7, 10), dpi=96, tight_layout=True)
        gs = self._fig.add_gridspec(3, 1, height_ratios=[2.5, 2.5, 1.2],
                                    hspace=0.38)
        self._ax_raw  = self._fig.add_subplot(gs[0])
        self._ax_norm = self._fig.add_subplot(gs[1])
        self._ax_res  = self._fig.add_subplot(gs[2])   # residuals

        self._canvas = FigureCanvasTkAgg(self._fig, master=center)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        nav = NavigationToolbar2Tk(self._canvas, center)
        nav.update()
        self._setup_axes()

        # ── Right: param panel + peak panel ───────────────────────────────
        right = tk.Frame(pw, width=330)
        pw.add(right, minsize=280)

        # Background/Edge params (top ~40%)
        tk.Label(right, text='Background & Edge Parameters',
                 font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', padx=4, pady=(4, 0))
        self._param_panel = ParamPanel(right, on_change=self._on_param_change)
        self._param_panel.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        ttk.Separator(right, orient='horizontal').pack(fill=tk.X, pady=4)

        # Peaks (bottom ~60%)
        tk.Label(right, text='Peaks  (L3 MLCT → L3 → L3 LMCT → L2 MLCT → L2 → L2 LMCT)',
                 font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', padx=4, pady=(0, 2))
        self._peak_panel = DynamicPeakPanel(right, on_change=self._on_param_change)
        self._peak_panel.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    def _build_statusbar(self):
        self._status_var = tk.StringVar(value='Ready.')
        tk.Label(self, textvariable=self._status_var, bd=1,
                 relief='sunken', anchor='w', font=('TkFixedFont', 8)
                 ).pack(side=tk.BOTTOM, fill=tk.X)
        self._progress = ttk.Progressbar(self, mode='determinate', length=200)
        self._progress.pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_axes(self):
        self._ax_raw.set_xlabel('Energy (eV)')
        self._ax_raw.set_ylabel('Intensity (a.u.)')
        self._ax_raw.set_title('Raw data + model')
        self._ax_norm.set_xlabel('Energy (eV)')
        self._ax_norm.set_ylabel('Normalized μ(E)')
        self._ax_norm.set_title('Normalized spectrum + components')
        self._ax_res.set_xlabel('Energy (eV)')
        self._ax_res.set_ylabel('Residual')
        self._ax_res.set_title('Fit residuals  (data − model)')
        self._fig.tight_layout()

    # ══════════════════════════════════════════════════════════════════════
    #  Event handlers
    # ══════════════════════════════════════════════════════════════════════

    def _on_element_change(self, event=None):
        el = self._element_var.get()
        if el in SOC:
            self._zeta_var.set(f'{SOC[el]:.3f}')
        try:
            self._model.zeta = float(self._zeta_var.get())
        except ValueError:
            pass

    def _on_zeta_change(self):
        try:
            self._model.zeta = float(self._zeta_var.get())
            self._update_plot()
        except ValueError:
            pass

    def _on_param_change(self):
        self._update_plot()

    def _on_scan_select(self, event=None):
        idxs = self._scan_lb.curselection()
        if idxs:
            self._redraw_scans([self._scans[i] for i in idxs])

    def _get_fixed(self):
        """Return tuple of parameter keys to keep fixed during fitting."""
        fixed = ['Ek']
        if self._lock_e0_var.get():
            fixed.append('E0')
        return tuple(fixed)

    def _get_static_constraints(self):
        return self._param_panel.get_constraints()

    def _get_br(self):
        """
        Return the target L2/L3 manifold branching ratio if constrained, else None.
        The fitter treats this as a soft physical target rather than an exact
        algebraic rescaling.
        Statistical value for L-edges: 0.5  (2:1 branching ratio L3:L2).
        """
        if not self._lock_br_var.get():
            return None
        try:
            val = float(self._br_var.get())
            return max(0.01, min(2.0, val))   # sensible guard rails
        except ValueError:
            return 0.5

    def _edit_constraints(self):
        params = self._param_panel.get_params()
        peaks = self._peak_panel.get_peaks()
        dlg = ConstraintDialog(self, params, self._param_panel.get_constraints(), peaks)
        self.wait_window(dlg)
        if not dlg.result:
            return
        param_constraints, peak_constraints = dlg.result
        self._param_panel.set_constraints(param_constraints)
        updated_peaks = self._peak_panel.get_peaks()
        for pk, cons in zip(updated_peaks, peak_constraints):
            pk['constraints'] = cons
        self._peak_panel.set_peaks(updated_peaks)
        self._set_status('Updated fit locks and numeric ranges.')
        self._update_plot()

    # ══════════════════════════════════════════════════════════════════════
    #  Scan management
    # ══════════════════════════════════════════════════════════════════════

    def _load_from_stack(self):
        """
        Load a PFY or TFY spectrum directly from an SGM stack directory,
        bypassing the CSV export step in sgm_xas_loader.py.

        Uses the same binary-reading approach as SGMPython's StackScan
        (github.com/Beamlines-CanadianLightSource/SGMPython):
          • StackScan.get_sdd_data(detector, energy) → (n_pixels, n_ch) array
          • Sums spatially and integrates over a user-supplied channel ROI.

        If sgmanalysis is installed the dialog shows a note; otherwise the
        built-in _load_sdd_bin logic is used (identical algorithm).
        """
        stack_dir = filedialog.askdirectory(
            title='Select SGM Stack Directory  (contains energy subdirs + .bin files)')
        if not stack_dir:
            return

        # ── Ask for ROI and signal type ─────────────────────────────────────
        dlg = tk.Toplevel(self)
        dlg.title('Stack Load Options')
        dlg.resizable(False, False)
        dlg.grab_set()

        def _lbl(parent, text, **kw):
            tk.Label(parent, text=text, anchor='w', **kw).grid(**kw)

        frm = ttk.Frame(dlg, padding=10)
        frm.pack()

        # Signal type
        sig_var = tk.StringVar(value='PFY')
        ttk.Label(frm, text='Signal type:').grid(row=0, column=0, sticky='w')
        for ci, st in enumerate(['PFY', 'TFY']):
            ttk.Radiobutton(frm, text=st, variable=sig_var, value=st
                            ).grid(row=0, column=ci + 1, padx=4)

        # ROI (only for PFY)
        ttk.Label(frm, text='ROI lo ch:').grid(row=1, column=0, sticky='w', pady=4)
        lo_var = tk.IntVar(value=250)
        hi_var = tk.IntVar(value=550)
        ttk.Spinbox(frm, textvariable=lo_var, from_=0, to=4095,
                    width=6).grid(row=1, column=1)
        ttk.Label(frm, text='hi:').grid(row=1, column=2)
        ttk.Spinbox(frm, textvariable=hi_var, from_=1, to=4096,
                    width=6).grid(row=1, column=3)

        # Detectors
        ttk.Label(frm, text='Detectors:').grid(row=2, column=0, sticky='w')
        det_vars = [tk.BooleanVar(value=True) for _ in range(4)]
        for ci, v in enumerate(det_vars):
            ttk.Checkbutton(frm, text=f'SDD{ci+1}',
                            variable=v).grid(row=2, column=ci + 1)

        # Norm
        norm_var = tk.StringVar(value='ring_current')
        ttk.Label(frm, text='Norm by:').grid(row=3, column=0, sticky='w')
        for ci, (txt, val) in enumerate([('Ring current', 'ring_current'),
                                          ('None', 'none')]):
            ttk.Radiobutton(frm, text=txt, variable=norm_var, value=val
                            ).grid(row=3, column=ci + 1, padx=4)

        if _SGM_AVAILABLE:
            ttk.Label(frm, text='✓ sgmanalysis detected',
                      foreground='#226600',
                      font=('TkDefaultFont', 8)).grid(
                row=4, column=0, columnspan=4, pady=(8, 0))

        ok_var = tk.BooleanVar(value=False)

        def _ok():
            ok_var.set(True)
            dlg.destroy()

        ttk.Button(frm, text='Load', command=_ok).grid(
            row=5, column=0, columnspan=2, pady=8, padx=4)
        ttk.Button(frm, text='Cancel', command=dlg.destroy).grid(
            row=5, column=2, columnspan=2, pady=8)

        self.wait_window(dlg)
        if not ok_var.get():
            return

        sig_type = sig_var.get()
        roi_lo   = lo_var.get()
        roi_hi   = hi_var.get()
        dets     = [i + 1 for i, v in enumerate(det_vars) if v.get()] or [1, 2, 3, 4]
        norm     = norm_var.get()

        self._set_status('Loading stack…')
        self.update_idletasks()

        # ── Inner load function (uses SGMPython's algorithm) ────────────────
        def _load_stack_spectrum():
            """
            Replicate SGMPython StackScan.get_sdd_data() logic:
              for each energy subdir, load SDD .bin → sum pixels → integrate ROI.
            """
            import re
            sdd_files = [f'sdd{n}_0.bin' for n in dets]

            def _parse_energy_local(name):
                m = re.search(r'_([\d]+)_([\d]+)[Ee][Vv]$', name)
                if m:
                    return float(f'{m.group(1)}.{m.group(2)}')
                m2 = re.search(r'_([\d]+(?:\.\d+)?)[Ee][Vv]$', name)
                return float(m2.group(1)) if m2 else None

            subdirs = sorted(
                [(d, _parse_energy_local(d))
                 for d in os.listdir(stack_dir)
                 if os.path.isdir(os.path.join(stack_dir, d))
                 and _parse_energy_local(d) is not None],
                key=lambda t: t[1])

            # Try to get ring-current from H5 (same as sgm_xas_loader.py)
            h5files = [f for f in os.listdir(stack_dir) if f.endswith('.h5')]
            h5_en   = np.array([])
            h5_rc   = np.array([])
            if h5files:
                try:
                    import h5py
                    with h5py.File(os.path.join(stack_dir, h5files[0]), 'r') as hf:
                        md = hf.get('map_data')
                        if md is not None:
                            h5_en = md['energy'][()]
                            h5_rc = md['ring_current'][()]
                except Exception:
                    pass

            def _rc_at(e):
                if h5_rc.size == 0:
                    return 1.0
                return max(float(h5_rc[int(np.argmin(np.abs(h5_en - e)))]), 1e-6)

            energy_list, signal_list = [], []
            for dname, e_ev in subdirs:
                sp_path = os.path.join(stack_dir, dname)
                sdd_sum = None
                for sn in sdd_files:
                    bp = os.path.join(sp_path, sn)
                    if not os.path.isfile(bp):
                        continue
                    raw = np.fromfile(bp, dtype=np.uint32)
                    if raw.size == 0:
                        continue
                    n_ch = raw.size // 81          # auto-detect channels (SGMPython pattern)
                    if n_ch < 1:
                        continue
                    mat = raw[:81 * n_ch].reshape(81, n_ch).astype(np.float64)
                    row = mat.sum(axis=0)
                    if sdd_sum is None:
                        sdd_sum = row
                    elif sdd_sum.size == row.size:
                        sdd_sum += row
                if sdd_sum is None:
                    continue
                if sig_type == 'TFY':
                    val = float(sdd_sum.sum())
                else:
                    lo = max(0, min(roi_lo, sdd_sum.size - 1))
                    hi = max(lo + 1, min(roi_hi, sdd_sum.size))
                    val = float(sdd_sum[lo:hi].sum())
                if norm == 'ring_current':
                    val /= _rc_at(e_ev)
                energy_list.append(e_ev)
                signal_list.append(val)

            if not energy_list:
                raise ValueError('No SDD data found in the selected stack.')
            en  = np.array(energy_list)
            sig = np.array(signal_list)
            idx = np.argsort(en)
            return en[idx], sig[idx], os.path.basename(stack_dir)

        try:
            en, sig, label = _load_stack_spectrum()
        except Exception as e:
            messagebox.showerror('Stack load error', str(e))
            self._set_status('Stack load failed.')
            return

        s = Scan(stack_dir, en, sig)
        s.name = f'{label} [{sig_type}]'
        self._scans.append(s)
        self._scan_lb.insert(tk.END, s.name)
        self._set_status(
            f'Loaded stack: {s.name}  ({len(en)} pts, '
            f'E={en[0]:.1f}–{en[-1]:.1f} eV)')

    def _load_scans(self):
        paths = filedialog.askopenfilenames(
            title='Load XAS scan file(s)',
            filetypes=[('Data files', '*.txt *.csv *.dat *.xy'),
                       ('All files', '*.*')])
        if not paths:
            return
        n_ok = n_fail = 0
        for p in paths:
            try:
                s = Scan.from_file(p)
                self._scans.append(s)
                self._scan_lb.insert(tk.END, s.name)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                self._set_status(f'Error loading {os.path.basename(p)}: {e}')
        self._set_status(f'Loaded {n_ok} scan(s).' +
                         (f'  {n_fail} failed.' if n_fail else ''))

    def _remove_selected(self):
        idxs = sorted(self._scan_lb.curselection(), reverse=True)
        for i in idxs:
            self._scans.pop(i)
            self._scan_lb.delete(i)
        self._set_status(f'Removed {len(idxs)} scan(s).')

    def _sum_selected(self):
        idxs = self._scan_lb.curselection()
        if not idxs:
            messagebox.showinfo('Sum Scans', 'Select at least one scan first.')
            return
        selected = [self._scans[i] for i in idxs]
        x, y = sum_scans(selected)
        if x is None:
            return

        # ── Auto-scale if data is in raw counts (values >> 100) ────────────
        # The parameter bounds are designed for I0-normalised data (~0-10).
        # Raw TFY/TEY counts can be 10^3–10^6; we rescale to keep the
        # optimiser well-behaved.  The scale factor is stored so exports
        # can report it, and the normalised output is scale-independent.
        y_max = float(np.max(np.abs(y)))
        if y_max > 50.0:
            self._y_scale = y_max / 10.0   # bring peak to ~10
            y = y / self._y_scale
        else:
            self._y_scale = 1.0

        self._x_sum = x
        self._y_sum = y
        scale_msg = (f'  [auto-scaled ÷{self._y_scale:.3g}]'
                     if self._y_scale != 1.0 else '')
        self._set_status(f'Summed {len(selected)} scan(s) → {len(x)} points, '
                         f'E = {x[0]:.1f} – {x[-1]:.1f} eV.{scale_msg}')
        self._update_stats()
        self._update_plot()

    def _update_stats(self):
        if self._x_sum is None:
            self._stats_lbl.config(text='No data')
            return
        x, y = self._x_sum, self._y_sum
        scale_line = (f'\nScale  : ÷{self._y_scale:.4g} (raw→fit)'
                      if self._y_scale != 1.0 else '')
        self._stats_lbl.config(
            text=f'Points : {len(x)}\n'
                 f'E range: {x[0]:.1f}–{x[-1]:.1f} eV\n'
                 f'I range: {y.min():.3g}–{y.max():.3g}'
                 f'{scale_line}')

    def _redraw_scans(self, scans):
        ax = self._ax_raw
        ax.cla()
        self._setup_axes()
        colors = matplotlib.cm.tab10.colors
        for i, s in enumerate(scans):
            ax.plot(s.x, s.y, lw=0.8, alpha=0.7,
                    color=colors[i % len(colors)], label=s.name)
        if len(scans) > 1:
            ax.legend(fontsize=7, loc='upper left')
        self._canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════════════
    #  Fitting
    # ══════════════════════════════════════════════════════════════════════

    def _require_data(self):
        if self._x_sum is None:
            messagebox.showwarning('No data',
                                   'Load scans and click "Sum Selected" first.')
            return False
        return True

    def _auto_estimate(self):
        if not self._require_data():
            return
        el   = self._element_var.get()
        d, peaks = self._model.auto_guess(self._x_sum, self._y_sum, element=el)
        d['Ek'] = float(self._x_sum[0])
        self._param_panel.set_params(d)
        self._peak_panel.set_peaks(peaks)
        self._zeta_var.set(f'{self._model.zeta:.3f}')
        self._set_status('Auto-estimated starting parameters.')
        self._update_plot()

    def _fit_once(self):
        if not self._require_data():
            return
        d0    = self._param_panel.get_params()
        peaks = self._peak_panel.get_peaks()
        if not any(pk['enabled'] for pk in peaks):
            messagebox.showwarning('No peaks',
                                   'Add at least one enabled peak before fitting.')
            return
        self._set_status('Fitting… please wait.')
        self.update_idletasks()
        try:
            br = self._get_br()
            d_fit, pks_fit = self._model.fit_once(
                self._x_sum, self._y_sum, d0, peaks,
                fixed=self._get_fixed(),
                br=br,
                static_constraints=self._get_static_constraints(),
                global_init=self._global_init_var.get())
            self._best_d     = d_fit
            self._best_peaks = pks_fit
            self._param_panel.set_params(d_fit)
            self._peak_panel.set_peaks(pks_fit)
            r2  = self._model.r2(d_fit, pks_fit, self._x_sum, self._y_sum, br=br)
            c2  = self._model.chi2_vec(
                self._model.pack(d_fit, pks_fit), self._x_sum, self._y_sum, pks_fit, br=br)
            val = self._model.il3_plus_2il2_norm(d_fit, pks_fit, br=br)
            br_txt = ''
            if br is not None:
                br_now = self._model.branch_ratio(pks_fit)
                if np.isfinite(br_now):
                    br_txt = f'  BR={br_now:.4f} (target {br:.4f})'
            self._set_status(f'Fit done.  chi2={c2:.5g}  r²={r2:.6f}  '
                             f'IL3+2IL2(norm)={val:.4f}{br_txt}')
            self._update_plot()
        except Exception as e:
            messagebox.showerror('Fit error', str(e))
            self._set_status('Fit failed.')

    def _run_mc(self):
        if not self._require_data():
            return
        try:
            n_mc = int(self._mc_n_var.get())
            spr  = float(self._spread_var.get())
        except ValueError:
            messagebox.showerror('Input error',
                                 'MC fits must be integer, spread a float.')
            return
        d0    = self._param_panel.get_params()
        peaks = self._peak_panel.get_peaks()
        if not any(pk['enabled'] for pk in peaks):
            messagebox.showwarning('No peaks',
                                   'Add at least one enabled peak before fitting.')
            return
        self._stop_event.clear()
        self._mc_results = []
        self._progress['value'] = 0
        self._set_status(f'Running {n_mc} Monte Carlo fits…')

        fixed = self._get_fixed()
        static_constraints = self._get_static_constraints()
        global_init = self._global_init_var.get()

        def _worker():
            def _cb(done, total):
                frac = int(100 * done / total)
                self.after(0, lambda: self._progress.configure(value=frac))
                self.after(0, lambda: self._set_status(
                    f'MC fit {done}/{total}…'))

            def _live(d_live, pks_live):
                # Snapshot to avoid closure capture issues with mutable objects
                d_snap   = dict(d_live)
                pks_snap = [dict(p) for p in pks_live]
                self.after(0, lambda: self._live_mc_update(d_snap, pks_snap))

            fits = self._model.mc_fit(
                self._x_sum, self._y_sum, d0, peaks,
                n=n_mc, spread=spr, fixed=fixed, br=self._get_br(),
                cb=_cb, stop_event=self._stop_event,
                live_cb=_live, live_every=10,
                static_constraints=static_constraints,
                global_init=global_init)
            self.after(0, lambda: self._on_mc_done(fits))

        threading.Thread(target=_worker, daemon=True).start()

    def _live_mc_update(self, d, peaks):
        """
        Update the parameter panel, peak panel, and plot with an intermediate
        MC result.  Fires every 10 fits during Monte Carlo so you can watch
        the fit converge in real time.

        on_change callbacks are suppressed during the update to avoid
        triggering a redundant replot from the param/peak panels themselves.
        """
        # Temporarily silence callbacks so set_params / update_peak_values
        # don't each fire another _update_plot on top of ours.
        old_pp = self._param_panel._on_change
        old_pk = self._peak_panel._on_change
        self._param_panel._on_change = None
        self._peak_panel._on_change  = None
        try:
            self._param_panel.set_params(d)
            self._peak_panel.update_peak_values(peaks)
        finally:
            self._param_panel._on_change = old_pp
            self._peak_panel._on_change  = old_pk
        self._update_plot()

    def _stop_mc(self):
        self._stop_event.set()
        self._set_status('MC stopped by user.')

    def _on_mc_done(self, fits):
        self._mc_results = fits
        self._progress['value'] = 100
        if not fits:
            self._set_status('MC complete — no successful fits.')
            return

        mu_d, sd_d, mu_pk, sd_pk = self._model.mc_stats(fits)
        self._mc_mean    = mu_d
        self._mc_std_d   = sd_d
        self._mc_mean_pk = mu_pk
        self._mc_std_pk  = sd_pk
        self._best_d     = mu_d
        self._best_peaks = mu_pk

        self._param_panel.set_params(mu_d)
        self._peak_panel.set_peaks(mu_pk)

        br     = self._get_br()
        val    = self._model.il3_plus_2il2_norm(mu_d, mu_pk, br=br)
        r2     = self._model.r2(mu_d, mu_pk, self._x_sum, self._y_sum, br=br)
        c2     = self._model.chi2_vec(
            self._model.pack(mu_d, mu_pk), self._x_sum, self._y_sum, mu_pk, br=br)
        br_now = self._model.branch_ratio(mu_pk)

        # Build MC result text
        lines = [f'n fits : {len(fits)}',
                 f'chi2   : {c2:.4g}',
                 f'r²     : {r2:.5f}',
                 f'IL3+2IL2(norm): {val:.4f}',
                 (f'BR     : {br_now:.4f}  (target {br:.4f})'
                  if br is not None and np.isfinite(br_now)
                  else ''),
                 '',
                 'Peak means ± sd:']
        kind_count = {}
        for pi, (mp, sp) in enumerate(zip(mu_pk, sd_pk)):
            kind  = mp['kind']
            n     = kind_count.get(kind, 0)
            kind_count[kind] = n + 1
            enb   = '✓' if mp.get('enabled', True) else '✗'
            lines.append(f'{enb}{kind}#{n+1} o={mp["o"]:.2f}±{sp["o"]:.2f}'
                         f'  I={mp["I"]:.3f}±{sp["I"]:.3f}')

        self._mc_lbl.config(text='\n'.join(lines))
        self._set_status(
            f'MC complete: {len(fits)} fits.  '
            f'IL3+2IL2(norm)={val:.4f}  r²={r2:.5f}')
        self._update_plot()

    # ══════════════════════════════════════════════════════════════════════
    #  Plotting
    # ══════════════════════════════════════════════════════════════════════

    def _update_plot(self):
        if self._x_sum is None:
            return
        x, y  = self._x_sum, self._y_sum
        d     = self._param_panel.get_params()
        peaks = self._peak_panel.get_peaks()

        ax1 = self._ax_raw
        ax2 = self._ax_norm
        ax1.cla()
        ax2.cla()
        self._setup_axes()

        # ── Top: raw data + full model + background ────────────────────────
        ax1.plot(x, y, 'b-', lw=1.2, alpha=0.8, label='Raw data', zorder=5)
        try:
            y_bg    = self._model.get_bg(x, d)
            y_model = self._model.get_full(x, d, peaks, br=self._get_br())
            ax1.plot(x, y_bg,    'g--', lw=1.0, alpha=0.9,
                     label='Background B(x)', zorder=4)
            ax1.plot(x, y_model, 'r-',  lw=1.5, alpha=0.85,
                     label='Total fit', zorder=6)
            E0 = d['E0']
            Tk = self._model.Tk(E0)
            Et = d['Et']
            for val, clr, lbl in [(E0, '#888', 'E0'),
                                   (Tk, '#aaa', 'Tk'),
                                   (Et, '#ccc', 'Et')]:
                ax1.axvline(val, color=clr, lw=0.8, ls=':', alpha=0.9)
                ymax = y.max()
                ax1.text(val, ymax, lbl, fontsize=7, color=clr,
                         ha='center', va='bottom')
        except Exception:
            pass
        ax1.legend(fontsize=8, loc='upper left')
        ax1.set_xlim(x[0], x[-1])

        # ── Bottom: normalized ─────────────────────────────────────────────
        try:
            y_norm     = self._model.get_norm(x, y, d)
            br         = self._get_br()
            y_fit_norm = self._model.get_norm_full_model(x, d, peaks, br=br)
            comp       = self._model.get_norm_components(x, d, peaks, br=br)

            ax2.plot(x, y_norm,     'b-', lw=1.3, alpha=0.85,
                     label='Normalized data', zorder=5)
            ax2.plot(x, y_fit_norm, 'r-', lw=1.5, alpha=0.80,
                     label='Total fit (norm.)', zorder=6)

            # Edge steps
            edge_colors = {'eL3': ('#aad4ff', 'L3 edge'),
                           'eL2': ('#aaffcc', 'L2 edge')}
            for key, (clr, lbl) in edge_colors.items():
                if key in comp:
                    ax2.fill_between(x, comp[key], alpha=0.35,
                                     color=clr, label=lbl, zorder=3)

            # Peaks — colour by kind + index
            kind_seen = {}
            for comp_key, arr in comp.items():
                if not comp_key.startswith('p'):
                    continue
                # key format: pL3_0, pL3_MLCT_0, pL2_LMCT_1
                # strip leading 'p'
                rest = comp_key[1:]           # e.g. "L3_0", "MLCT_0"
                parts = rest.rsplit('_', 1)
                if len(parts) != 2:
                    continue
                kind, idx_s = parts[0], int(parts[1])
                n = kind_seen.get(kind, 0)
                kind_seen[kind] = n + 1
                colors_list = PEAK_PLOT_COLORS.get(kind, ['#aaaaaa'])
                clr = colors_list[idx_s % len(colors_list)]
                ax2.fill_between(x, arr, alpha=0.45,
                                 color=clr,
                                 label=f'{kind} #{idx_s + 1}',
                                 zorder=3)

            ax2.axhline(0, color='k',    lw=0.5, ls='--')
            ax2.axhline(1, color='gray', lw=0.5, ls='--', alpha=0.5)
        except Exception:
            pass

        ax2.legend(fontsize=8, loc='upper left', ncol=2)
        ax2.set_xlim(x[0], x[-1])

        # ── Bottom: fit residuals (inspired by SGMPython's component decomp) ──
        ax3 = self._ax_res
        ax3.cla()
        self._setup_axes()
        try:
            y_model = self._model.get_full(x, d, peaks, br=self._get_br())
            resid   = y - y_model
            r2_val  = self._model.r2(d, peaks, x, y, br=self._get_br())
            ax3.plot(x, resid, color='#5588cc', lw=0.9, alpha=0.85,
                     label=f'data − model   r² = {r2_val:.5f}')
            ax3.fill_between(x, resid, alpha=0.20, color='#5588cc')
            ax3.axhline(0, color='k', lw=0.7, ls='--')
            # ±1σ envelope from MC if available
            if self._mc_std_d is not None and self._mc_mean_pk is not None:
                try:
                    y_mc   = self._model.get_full(x, self._mc_mean,
                                                  self._mc_mean_pk,
                                                  br=self._get_br())
                    r_mc   = y - y_mc
                    ax3.plot(x, r_mc, color='#cc4444', lw=0.8, ls='--',
                             alpha=0.7, label='data − MC mean')
                except Exception:
                    pass
            ax3.legend(fontsize=7, loc='upper right')
        except Exception:
            pass
        ax3.set_xlim(x[0], x[-1])

        self._fig.tight_layout()
        self._canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════════════
    #  Export
    # ══════════════════════════════════════════════════════════════════════

    def _save_normalized(self):
        if self._x_sum is None:
            messagebox.showwarning('No data', 'Nothing to save yet.')
            return
        d     = self._param_panel.get_params()
        peaks = self._peak_panel.get_peaks()
        path  = filedialog.asksaveasfilename(
            title='Save normalized spectrum',
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv'), ('Text', '*.txt'), ('All', '*.*')])
        if not path:
            return
        try:
            x, y   = self._x_sum, self._y_sum
            br         = self._get_br()
            y_bg       = self._model.get_bg(x, d)
            y_norm     = self._model.get_norm(x, y, d)
            y_fit_norm = self._model.get_norm_full_model(x, d, peaks, br=br)
            comp       = self._model.get_norm_components(x, d, peaks, br=br)

            # Restore physical (unscaled) raw intensity for the CSV output
            y_raw_physical = y    * self._y_scale
            y_bg_physical  = y_bg * self._y_scale

            header = (['energy_eV', 'raw_intensity', 'background',
                       'normalized', 'fit_normalized'] + list(comp.keys()))
            rows   = list(zip(x, y_raw_physical, y_bg_physical,
                              y_norm, y_fit_norm,
                              *[comp[k] for k in comp]))

            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['# L-Edge XAS Normalized Spectrum'])
                w.writerow([f'# Element : {self._element_var.get()}'])
                w.writerow([f'# zeta_2p : {self._model.zeta:.4f} eV'])
                w.writerow([f'# Tk=E0+3/2*zeta : {self._model.Tk(d["E0"]):.4f} eV'])
                w.writerow([f'# IL3_Edge : {d["IL3_Edge"]:.6g}'])
                w.writerow([f'# Normalization factor (3/2*IL3_Edge): '
                            f'{1.5*d["IL3_Edge"]:.6g}'])
                w.writerow([f'# Lock E0 : {self._lock_e0_var.get()}'])
                if self._y_scale != 1.0:
                    w.writerow([f'# Auto-scale factor : {self._y_scale:.6g} '
                                f'(raw_intensity already restored to physical units)'])
                # Peak summary
                for i, pk in enumerate(peaks):
                    enb = 'enabled' if pk['enabled'] else 'disabled'
                    w.writerow([f'# Peak {i+1}: kind={pk["kind"]} {enb}  '
                                f'o={pk["o"]:.4f}  W={pk["W"]:.4f}  '
                                f'I={pk["I"]:.6g}  G={pk["G"]:.4f}'])
                br_val = self._get_br()
                w.writerow([f'# L2/L3 BR constraint: '
                            f'{"locked = " + str(br_val) if br_val else "free"}'])
                if self._mc_mean is not None and self._mc_mean_pk is not None:
                    val = self._model.il3_plus_2il2_norm(
                        self._mc_mean, self._mc_mean_pk, br=br_val)
                    w.writerow([f'# IL3+2*IL2 (norm, MC mean): {val:.5f}'])
                w.writerow(header)
                for row in rows:
                    w.writerow([f'{v:.8g}' for v in row])
            self._set_status(f'Saved: {os.path.basename(path)}')
        except Exception as e:
            messagebox.showerror('Save error', str(e))

    def _save_params(self):
        d     = self._param_panel.get_params()
        peaks = self._peak_panel.get_peaks()
        path  = filedialog.asksaveasfilename(
            title='Save fit parameters',
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv'), ('Text', '*.txt'), ('All', '*.*')])
        if not path:
            return
        try:
            static_constraints = self._param_panel.get_constraints()
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['# L-Edge XAS Fit Parameters'])
                w.writerow(['# Generated by ledge_normalizer.py'])
                w.writerow(['parameter', 'value', 'std_dev', 'section', 'label',
                            'lock', 'lo', 'hi'])
                # Static params
                for p in PARAMS_STATIC:
                    key, sec, lbl = p[0], p[1], p[2]
                    val  = d[key]
                    sd   = (self._mc_std_d[key]
                            if self._mc_std_d and key in self._mc_std_d else '')
                    spec = static_constraints.get(key, {'lock': p[7], 'lo': p[4], 'hi': p[5]})
                    w.writerow([key, f'{val:.8g}',
                                f'{sd:.6g}' if sd != '' else '',
                                sec, lbl,
                                int(bool(spec.get('lock', False))),
                                f'{float(spec.get("lo", p[4])):.8g}',
                                f'{float(spec.get("hi", p[5])):.8g}'])
                # Peaks
                for i, pk in enumerate(peaks):
                    kind  = pk['kind']
                    enb   = 'enabled' if pk['enabled'] else 'disabled'
                    sd_pk = (self._mc_std_pk[i]
                             if self._mc_std_pk and i < len(self._mc_std_pk)
                             else {})
                    for field in PEAK_FIELDS:
                        key = f'peak{i+1}_{kind}_{field}'
                        val = pk[field]
                        sd  = sd_pk.get(field, '')
                        field_cons = ((pk.get('constraints') or {}).get(field) or
                                      {'lock': False,
                                       'lo': PEAK_BOUNDS[kind][field][0],
                                       'hi': PEAK_BOUNDS[kind][field][1]})
                        w.writerow([key, f'{val:.8g}',
                                    f'{sd:.6g}' if sd != '' else '',
                                    f'{kind} Peak #{i+1}', f'{enb}  {field}',
                                    int(bool(field_cons.get('lock', False))),
                                    f'{float(field_cons.get("lo", PEAK_BOUNDS[kind][field][0])):.8g}',
                                    f'{float(field_cons.get("hi", PEAK_BOUNDS[kind][field][1])):.8g}'])
            self._set_status(f'Parameters saved: {os.path.basename(path)}')
        except Exception as e:
            messagebox.showerror('Save error', str(e))

    def _save_quanty_bundle(self):
        if self._x_sum is None:
            messagebox.showwarning('No data', 'Load and sum data before exporting.')
            return

        d = self._param_panel.get_params()
        peaks = self._peak_panel.get_peaks()
        path = filedialog.asksaveasfilename(
            title='Save Quanty seed bundle',
            defaultextension='.json',
            filetypes=[('JSON', '*.json'), ('All', '*.*')])
        if not path:
            return

        try:
            br = self._get_br()
            fit_metrics = self._collect_fit_metrics(d, peaks, br=br)
            written = write_quanty_seed_bundle(
                path,
                element=self._element_var.get(),
                zeta_2p=self._model.zeta,
                params=d,
                peaks=peaks,
                symmetry=self._symmetry_var.get(),
                br=br,
                mc_std_d=self._mc_std_d,
                mc_std_pk=self._mc_std_pk,
                fit_metrics=fit_metrics,
            )
            names = ', '.join(os.path.basename(p) for p in written)
            self._set_status(f'Quanty seed bundle saved: {names}')
        except Exception as e:
            messagebox.showerror('Save error', str(e))

    # ══════════════════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════════════════

    def _collect_fit_metrics(self, d, peaks, br=None):
        metrics = {}
        try:
            metrics['r2'] = float(self._model.r2(d, peaks, self._x_sum, self._y_sum, br=br))
        except Exception:
            pass
        try:
            metrics['sse'] = float(self._model.chi2_vec(
                self._model.pack(d, peaks), self._x_sum, self._y_sum, peaks, br=br))
        except Exception:
            pass
        try:
            metrics['normalized_peak_area'] = float(
                self._model.il3_plus_2il2_norm(d, peaks, br=br))
        except Exception:
            pass
        try:
            metrics['y_scale_factor'] = float(self._y_scale)
        except Exception:
            pass
        return metrics

    def _set_status(self, msg):
        self._status_var.set(msg)
        self.update_idletasks()

    def _show_about(self):
        messagebox.showinfo(
            'About L-Edge XAS Normalizer',
            'L-Edge XAS Normalizer\n\n'
            'BlueprintXAS simultaneous fit-and-normalize for\n'
            'transition-metal L2,3-edge XAS.\n\n'
            'Four-domain background B(x), cumulative pseudo-Voigt edge\n'
            'steps (2:1 L3:L2 branching ratio), dynamic pseudo-Voigt\n'
            'peaks (L3 / L2 plus edge-resolved MLCT / LMCT),\n'
            'Monte Carlo error estimation,\n'
            'and Quanty seed export for crystal-field / ligand-field work,\n'
            'including lower-symmetry scaffolds such as C2v.\n\n'
            'Normalization:  μ_norm = (μ_raw − B) / (3/2 · IL3,Edge)\n\n'
            'Lock E0 is ON by default to prevent catastrophic drift\n'
            'during Monte Carlo fitting.\n\n'
            'Reference: Delgado-Jaime & DeBeer, BlueprintXAS (2012)\n'
            'Pollock & DeBeer, JACS 133, 5594–5601 (2011)')


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app = LEdgeNormApp()
    # Pre-load CSV files passed as command-line arguments
    if len(sys.argv) > 1:
        n_ok = 0
        for p in sys.argv[1:]:
            if os.path.isfile(p):
                try:
                    s = Scan.from_file(p)
                    app._scans.append(s)
                    app._scan_lb.insert(tk.END, s.name)
                    n_ok += 1
                except Exception:
                    pass
        if n_ok:
            app._set_status(f'Pre-loaded {n_ok} scan(s) from SGM XAS Loader.')
    app.mainloop()
