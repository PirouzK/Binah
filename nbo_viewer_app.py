"""
NBO Analysis Viewer v2
Tkinter + matplotlib-based UI for ORCA/NBO7 output files.

Run with:  python nbo_viewer_app.py
Requires:  pip install numpy matplotlib
Optional:  pip install scikit-image   (isosurface marching-cubes)

Orbital rendering sources
─────────────────────────
  .cube files        : any orbital type (NBO, LP, BD*, MO) — generate with orca_plot
  .out (LargePrint)  : canonical MOs — parsed directly from ORCA output text
                       (reads basis set + MO coefficients, evaluates on 3D grid)
                       Run ORCA with  ! LargePrint  to include MO coefficients.

MO Composition
──────────────
  When loading MOs from .out files, the viewer computes % contributions
  from each atom and angular momentum type (s/p/d/f) using |C_μi|² analysis.
  This shows which atoms and orbital types (e.g. Ni dxy, P pz) contribute
  to each molecular orbital.

Element colours follow the Jmol/Avogadro colour scheme.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

try:
    from skimage.measure import marching_cubes as _mc
    _HAS_SKIMAGE = True
except ImportError:
    try:
        from skimage.measure import marching_cubes_lewiner as _mc  # type: ignore
        _HAS_SKIMAGE = True
    except ImportError:
        _HAS_SKIMAGE = False

_HAS_CCLIB = False  # cclib not used (crashes on NBO7 ORCA files); using custom parser

BOHR_TO_ANG = 0.529177

# ─────────────────────────────────────────────────────────────────────────────
#  Default directory
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_DIR = Path(
    r"F:\OneDrive\OneDrive - University of Calgary"
    r"\Science Work\Nickel Thesis Project\Synthesized complexes\Josiphos"
)

# ─────────────────────────────────────────────────────────────────────────────
#  Element styling
# ─────────────────────────────────────────────────────────────────────────────

ELEM_COLOR = {
    'H':  '#F0F0F0', 'C':  '#7F7F7F', 'N':  '#3050F8', 'O':  '#FF0D0D',
    'F':  '#B2FFFF', 'P':  '#FF8000', 'S':  '#FFFF30',
    'Cl': '#1FF01F', 'Br': '#A62929', 'I':  '#940094',
    'Sc': '#E6E6E6', 'Ti': '#BFC2C7', 'V':  '#A6A6AB', 'Cr': '#8A99C7',
    'Mn': '#9C7AC7', 'Fe': '#E06633', 'Co': '#F090A0', 'Ni': '#50D050',
    'Cu': '#C88033', 'Zn': '#7D80B0',
    'Ru': '#248F8F', 'Rh': '#0A7D8C', 'Pd': '#006985',
    'Os': '#266696', 'Ir': '#175487', 'Pt': '#D0D0E0',
    'As': '#BD80E3', 'Se': '#FFA100', 'Sb': '#9E63B5',
    'B':  '#FFB5B5', 'Si': '#F0C8A0', 'Au': '#FFD123',
}

# ── Covalent radii (Å) ───────────────────────────────────────────────────────
ELEM_RADIUS = {
    'H':  0.31, 'C':  0.77, 'N':  0.75, 'O':  0.73, 'F':  0.64,
    'P':  1.07, 'S':  1.02, 'Cl': 0.99, 'Br': 1.14, 'I':  1.33,
    'B':  0.82, 'Si': 1.11, 'As': 1.19, 'Se': 1.20, 'Sb': 1.39,
    'Sc': 1.44, 'Ti': 1.36, 'V':  1.25, 'Cr': 1.27, 'Mn': 1.39,
    'Fe': 1.26, 'Co': 1.25, 'Ni': 1.24, 'Cu': 1.28, 'Zn': 1.22,
    'Ru': 1.25, 'Rh': 1.25, 'Pd': 1.20,
    'Os': 1.29, 'Ir': 1.22, 'Pt': 1.23,
}

# ── Atomic numbers (used to reverse-lookup element from cube file Z) ─────────
ATOMIC_NUM: Dict[str, int] = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7,
    'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14,
    'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'As': 33, 'Se': 34,
    'Br': 35, 'Pd': 46, 'Sb': 51, 'I': 53, 'Pt': 78, 'Au': 79,
}
_ATOMIC_NUM_REV = {v: k for k, v in ATOMIC_NUM.items()}

_DEFAULT_COLOR  = '#888888'
_DEFAULT_RADIUS = 0.80

# Atoms highlighted in the NPA bar chart and sidebar colour-coding
KEY_METALS = {'Ni', 'Fe', 'Co', 'Cu', 'Pd', 'Pt', 'Ru', 'Rh',
              'P', 'As', 'Cl', 'Br', 'I', 'F', 'S', 'N', 'O'}


def _ec(el: str, overrides: Optional[Dict[str, str]] = None) -> str:
    """Return element colour, checking runtime overrides first."""
    if overrides and el in overrides:
        return overrides[el]
    return ELEM_COLOR.get(el, _DEFAULT_COLOR)


def _charge_rgba(q: float, vmin: float, vmax: float):
    span = vmax - vmin
    t = max(0.0, min(1.0, (q - vmin) / span if span > 0 else 0.5))
    return cm.RdBu_r(t)


# ─────────────────────────────────────────────────────────────────────────────
#  Parsers
# ─────────────────────────────────────────────────────────────────────────────

_RE_NPA_START = re.compile(r"^\s*Summary of Natural Population Analysis\s*:\s*$", re.I)
_RE_NPA_ROW   = re.compile(
    r"^\s*([A-Z][a-z]?)\s+(\d+)"
    r"\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)"
    r"\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*$"
)
_RE_NEC_START = re.compile(r"^\s*Atom No\s+Natural Electron Configuration\s*$", re.I)
_RE_NEC_ROW   = re.compile(r"^\s*([A-Z][a-z]?)\s+(\d+)\s+(.+)$")
_RE_DASH      = re.compile(r"^\s*-{5,}\s*$")


def parse_out_file(path: str) -> dict:
    """Return {'npa': [...], 'nec': [...]} from the LAST NBO block in an ORCA .out file."""
    npa: List[dict] = []
    nec: List[dict] = []
    in_npa = in_nec = npa_hdr = nec_hdr = False

    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if _RE_NPA_START.match(line):
                npa = []; in_npa = True; npa_hdr = False; continue
            if in_npa:
                if not npa_hdr:
                    if _RE_DASH.match(line): npa_hdr = True
                    continue
                if not line.strip(): in_npa = False; continue
                m = _RE_NPA_ROW.match(line)
                if m:
                    npa.append({"Element": m.group(1), "Atom#": int(m.group(2)),
                                "Charge": float(m.group(3)), "Core": float(m.group(4)),
                                "Valence": float(m.group(5)), "Rydberg": float(m.group(6)),
                                "Total": float(m.group(7))})

            if _RE_NEC_START.match(line):
                nec = []; in_nec = True; nec_hdr = False; continue
            if in_nec:
                if not nec_hdr:
                    if _RE_DASH.match(line): nec_hdr = True
                    continue
                if not line.strip(): in_nec = False; continue
                m = _RE_NEC_ROW.match(line)
                if m:
                    nec.append({"Element": m.group(1), "Atom#": int(m.group(2)),
                                "Config": m.group(3).strip()})
    return {"npa": npa, "nec": nec}


# ─────────────────────────────────────────────────────────────────────────────
#  Löwdin / Mulliken reduced orbital population parser
# ─────────────────────────────────────────────────────────────────────────────
#
#  ORCA prints this section in multi-column blocks (like the MO coefficients):
#
#    LOEWDIN REDUCED ORBITAL POPULATIONS PER MO
#    -------------------------------------------
#    THRESHOLD FOR PRINTING IS 0.1%
#                          0         1         2    ...
#                     -299.787  -256.046  -101.396  ...   (energies)
#                       2.000     2.000     2.000  ...   (occupancies)
#                      ------   ------   ------   ...
#     0 Ni s            100.0      0.0      0.0   ...
#     1 Cl s              0.0      0.0    100.0   ...
#     0 Ni pz             1.4     45.8     52.7   ...
#
#  Each atom row has:  <atom_idx>  <elem>  <ang_label>   <val1>  <val2>  ...
#  Values are percentages (0-100).

_RE_LORB_SEC = re.compile(
    r"(LOEWDIN|MULLIKEN)\s+REDUCED\s+ORBITAL\s+POPULATIONS\s+PER\s+MO", re.I)


def parse_loewdin_mo_pops(path: str) -> dict:
    """
    Parse the last LOEWDIN (or MULLIKEN) REDUCED ORBITAL POPULATIONS PER MO
    section from an ORCA .out file.

    Handles the multi-column block format produced by ORCA 5.x / 6.x.

    Returns
    -------
    dict :  mo_idx (0-based int) ->
                {"label": str,
                 "atoms": {
                     "Ni1": {"s": f, "p": f, "d": f, "f_orb": f, "total": f},
                     ...
                 }}
    Empty dict if section is absent.
    """
    # Collect all lines of the last matching section
    sec_lines: List[str] = []
    in_sec = False

    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if _RE_LORB_SEC.search(line):
                sec_lines = []
                in_sec = True
                continue
            if in_sec:
                stripped = line.strip()
                # Stop when we hit a new major section header
                if stripped and not stripped.startswith("-") and \
                   stripped.isupper() and len(stripped) > 10 and \
                   "POPULATION" not in stripped and "THRESHOLD" not in stripped:
                    # Check if this looks like a section boundary
                    if any(kw in stripped for kw in (
                        "NATURAL BOND", "NBO ANALYSIS", "MAYER POPULATION",
                        "MULLIKEN ATOMIC", "TIMINGS", "TOTAL RUN TIME",
                        "DENSITY MATRIX", "FINAL SINGLE POINT")):
                        in_sec = False
                        continue
                sec_lines.append(line.rstrip())

    if not sec_lines:
        return {}

    # Parse the multi-column blocks
    cur: dict = {}
    pos = 0

    while pos < len(sec_lines):
        line = sec_lines[pos].strip()

        # Skip blank, dashes, threshold lines
        if not line or line.startswith("-") or "THRESHOLD" in line.upper():
            pos += 1
            continue

        # Try to read a MO-index header row (all integers)
        parts = line.split()
        try:
            mo_indices = [int(p) for p in parts]
        except ValueError:
            pos += 1
            continue

        # ── Energy row ────────────────────────────────────────────────
        pos += 1
        if pos >= len(sec_lines):
            break
        eparts = sec_lines[pos].split()
        try:
            block_ene = [float(e) for e in eparts]
        except ValueError:
            continue

        # ── Occupancy row ─────────────────────────────────────────────
        pos += 1
        if pos >= len(sec_lines):
            break
        oparts = sec_lines[pos].split()
        try:
            block_occ = [float(o) for o in oparts]
        except ValueError:
            continue

        # ── Separator row (--------) ──────────────────────────────────
        pos += 1
        if pos < len(sec_lines) and "---" in sec_lines[pos]:
            pos += 1

        n_mos = len(mo_indices)

        # Initialise MO entries
        for k, mo_i in enumerate(mo_indices):
            if mo_i not in cur:
                cur[mo_i] = {"label": "", "atoms": {}}

        # ── Data rows: atom populations ───────────────────────────────
        while pos < len(sec_lines):
            dline = sec_lines[pos].strip()
            if not dline:
                pos += 1
                break
            # Check for next block header (all integers)
            dparts = dline.split()
            try:
                _ = [int(x) for x in dparts]
                break  # next block
            except ValueError:
                pass

            # Parse atom row:  "0 Ni s   100.0  0.0  0.0"
            #              or: "0 Ni pz    1.4 45.8 52.7"
            #              or: "98 C  2pz   0.0  0.0  0.2"
            # Format: atom_idx elem ang_label  val1 val2 ...
            m = re.match(r"\s*(\d+)\s+([A-Z][a-z]?)\s+(\S+)\s+(.*)", dline)
            if not m:
                pos += 1
                continue

            a_idx    = int(m.group(1))
            a_el     = m.group(2)
            ang_lbl  = m.group(3).lower()
            vals_str = m.group(4).split()

            try:
                vals = [float(v) for v in vals_str]
            except ValueError:
                pos += 1
                continue

            # Determine angular momentum type from label
            if ang_lbl.startswith("d"):
                ang_type = "d"
            elif ang_lbl.startswith("f"):
                ang_type = "f_orb"
            elif ang_lbl.startswith("p"):
                ang_type = "p"
            else:
                ang_type = "s"

            # atom_key uses 1-based index to match our convention
            atom_key = f"{a_el}{a_idx + 1}"

            for k in range(min(len(vals), n_mos)):
                mo_i = mo_indices[k]
                pct  = vals[k] / 100.0   # convert percentage to fraction

                if atom_key not in cur[mo_i]["atoms"]:
                    cur[mo_i]["atoms"][atom_key] = {
                        "s": 0.0, "p": 0.0, "d": 0.0, "f_orb": 0.0, "total": 0.0
                    }
                cur[mo_i]["atoms"][atom_key][ang_type] += pct
                cur[mo_i]["atoms"][atom_key]["total"]  += pct

            pos += 1

    return cur


def parse_ni_summary(path: str) -> dict:
    """Parse NiSummary txt: LP d-orbitals + LP(Ni)→acceptor E₂."""
    lp: List[dict] = []
    bd: List[dict] = []
    ry: List[dict] = []
    RE_LP  = re.compile(r"Ni LP\(NBO\) d-type orbitals", re.I)
    RE_BD  = re.compile(r"BD\* acceptors in LP\(Ni\)")
    RE_RY  = re.compile(r"RY acceptors in LP\(Ni\)")
    RE_LPR = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
    RE_ACC = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+(.+)$")
    section = None; hdr_ok = False
    try:
        with open(path, "r", errors="ignore") as fh:
            for line in fh:
                if RE_LP.search(line):  section = "lp"; hdr_ok = False; continue
                if RE_BD.search(line):  section = "bd"; hdr_ok = False; continue
                if RE_RY.search(line):  section = "ry"; hdr_ok = False; continue
                if not section:         continue
                if "NBO#" in line or "AccNBO#" in line: hdr_ok = True; continue
                if _RE_DASH.match(line) or not line.strip(): continue
                if not hdr_ok: continue
                if section == "lp":
                    m = RE_LPR.match(line)
                    if m:
                        lp.append({"NBO#": int(m.group(1)), "LP#": int(m.group(2)),
                                   "Occ": float(m.group(3)), "s%": float(m.group(4)),
                                   "p%": float(m.group(5)), "d%": float(m.group(6))})
                else:
                    m = RE_ACC.match(line)
                    if m:
                        e = {"AccNBO#": int(m.group(1)), "E2sum": float(m.group(2)),
                             "Occ": float(m.group(3)), "Label": m.group(4).strip()}
                        (bd if section == "bd" else ry).append(e)
    except FileNotFoundError:
        pass
    return {"lp": lp, "bd": bd, "ry": ry}


def read_xyz(path: str) -> List[Tuple]:
    atoms: List[Tuple] = []
    try:
        lines = Path(path).read_text(errors="ignore").splitlines()
        n = int(lines[0].strip())
        for ln in lines[2 : 2 + n]:
            p = ln.split()
            if len(p) >= 4:
                atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    except Exception:
        pass
    return atoms


def parse_cube_file(path: str) -> dict:
    """
    Parse a Gaussian/ORCA cube file.
    All coordinates converted to Ångström.
    Returns:
        comment, atoms [(Z, q, x, y, z)], origin [3], axes [3×3],
        n [3], data ndarray(nx,ny,nz)
    """
    lines = Path(path).read_text(errors="ignore").splitlines()
    comment = lines[0].strip() + "  |  " + lines[1].strip()

    p = lines[2].split()
    natoms = abs(int(p[0]))
    origin = np.array([float(p[1]), float(p[2]), float(p[3])]) * BOHR_TO_ANG

    axes = np.zeros((3, 3))
    n    = np.zeros(3, dtype=int)
    for i in range(3):
        p = lines[3 + i].split()
        n[i] = int(p[0])
        axes[i] = [float(p[1]) * BOHR_TO_ANG,
                   float(p[2]) * BOHR_TO_ANG,
                   float(p[3]) * BOHR_TO_ANG]

    atoms = []
    for i in range(natoms):
        p = lines[6 + i].split()
        atoms.append((int(p[0]), float(p[1]),
                      float(p[2]) * BOHR_TO_ANG,
                      float(p[3]) * BOHR_TO_ANG,
                      float(p[4]) * BOHR_TO_ANG))

    data_start = 6 + natoms
    vals = []
    for ln in lines[data_start:]:
        vals.extend(float(v) for v in ln.split())
    data = np.array(vals).reshape(n[0], n[1], n[2])

    return {"comment": comment, "atoms": atoms, "origin": origin,
            "axes": axes, "n": n, "data": data}


# ─────────────────────────────────────────────────────────────────────────────
#  ORCA LargePrint parser  +  GTO evaluator
#  (reads basis set & MO coefficients directly from the .out text)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_orca_sections(filepath: str) -> Tuple[str, str]:
    """
    Stream through a (potentially huge) ORCA .out file and extract ONLY
    the 'BASIS SET IN INPUT FORMAT' and 'MOLECULAR ORBITALS' sections.

    Returns (basis_text, mo_text) — small strings ready for the parsers.
    This avoids loading a 400+ MB file entirely into memory.
    """
    basis_lines: List[str] = []
    mo_lines:    List[str] = []
    section = None  # None, "basis", "mo"
    basis_done = False
    mo_done    = False
    blank_run  = 0          # consecutive blank lines in MO section
    last_was_end = False    # for basis termination ("end" or "end;")
    basis_end_count = 0     # count of "end;" seen — stops after all elements

    # Headers that signal end of the MO block
    _MO_STOP = frozenset([
        "LOEWDIN REDUCED ORBITAL POPULATIONS",
        "LOEWDIN ORBITAL POPULATIONS",
        "NATURAL BOND ORBITAL ANALYSIS",
        "NBO ANALYSIS",
        "MULLIKEN ATOMIC CHARGES",
        "MAYER POPULATION ANALYSIS",
        "DENSITY",
        "TIMINGS",
        "TOTAL RUN TIME",
    ])

    with open(filepath, "r", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n\r")   # strip line endings only
            stripped = line.strip()

            # ── Detect section starts ──────────────────────────────
            if not basis_done and "BASIS SET IN INPUT FORMAT" in line:
                if "AUXILIARY" in line:
                    continue
                section = "basis"
                basis_lines = ["BASIS SET IN INPUT FORMAT"]
                last_was_end = False
                continue

            if not mo_done and stripped.startswith("MOLECULAR ORBITALS"):
                section = "mo"
                mo_lines = ["MOLECULAR ORBITALS"]
                blank_run = 0
                continue

            # ── Collect lines for active section ──────────────────
            if section == "basis":
                if stripped.startswith("=") and len(basis_lines) > 3:
                    # Section separator — end of basis block
                    basis_done = True
                    section = None
                    continue
                # Detect "Atom  NNN  basis set group" lines that precede AUXILIARY section
                if "basis set group" in line and basis_end_count > 0:
                    # We've collected all NewGTO blocks; this is atom-group listing
                    # before the next section — stop collecting
                    basis_done = True
                    section = None
                    continue
                basis_lines.append(line)
                # Track "end;" closings
                if stripped in ("end", "end;"):
                    basis_end_count += 1
                    last_was_end = True
                else:
                    last_was_end = False

            elif section == "mo":
                # Stop conditions
                if any(hdr in line for hdr in _MO_STOP):
                    mo_done = True
                    section = None
                    continue
                if not stripped:
                    blank_run += 1
                    if blank_run >= 3:
                        mo_done = True
                        section = None
                        continue
                else:
                    blank_run = 0
                mo_lines.append(line)

            # Early exit once we have both
            if basis_done and mo_done:
                break

    return "\n".join(basis_lines), "\n".join(mo_lines)


def _parse_orca_basis(text: str) -> Dict[str, List[dict]]:
    """
    Parse the 'BASIS SET IN INPUT FORMAT' block from an ORCA .out file.

    Returns {element: [shell, ...]}
    where shell = {'type': 'S'|'P'|'D'|'F', 'prims': [(exponent, coeff), ...]}
    """
    result: Dict[str, List[dict]] = {}
    in_block = False
    elem: Optional[str] = None
    shell: Optional[dict] = None

    for line in text.splitlines():
        s = line.strip()
        if "BASIS SET IN INPUT FORMAT" in s:
            in_block = True
            continue
        if not in_block:
            continue
        if not s or s.startswith("#"):
            continue
        if s.startswith("NewGTO"):
            elem = s.split()[1]
            result[elem] = []
            shell = None
            continue
        if s in ("end", "end;"):
            elem = None
            shell = None
            continue
        parts = s.split()
        if len(parts) == 2 and parts[0] in ("S", "P", "D", "F", "G"):
            shell = {"type": parts[0], "prims": []}
            if elem is not None:
                result[elem].append(shell)
            continue
        if shell is not None and len(parts) >= 3:
            try:
                _idx = int(parts[0])
                exp_val = float(parts[1])
                coeff   = float(parts[2])
                shell["prims"].append((exp_val, coeff))
            except ValueError:
                pass
    return result


def _parse_orca_mos(text: str) -> Optional[dict]:
    """
    Parse the 'MOLECULAR ORBITALS' block printed by ORCA's LargePrint.

    Returns:
      {'energies': ndarray (nmo,),
       'occs':     ndarray (nmo,),
       'coeffs':   ndarray (nao, nmo),
       'ao_labels': [(atom_idx, elem, shell_tag, ang_label), ...]}
    or None if the section is not found.
    """
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if "MOLECULAR ORBITALS" in ln:
            start = i + 1
    if start is None:
        return None

    # Skip separator dashes
    while start < len(lines) and lines[start].strip().startswith("-"):
        start += 1

    # Read blocks (each block: indices, energies, occupancies, separator, coefficients)
    energies: List[float] = []
    occs:     List[float] = []
    coeff_rows: Dict[int, List[float]] = {}  # ao_index → list of coeffs across all MOs
    ao_labels: List[tuple] = []
    labels_done = False
    ao_idx = 0
    pos = start

    while pos < len(lines):
        line = lines[pos].strip()
        if not line:
            pos += 1
            continue

        # ── Try to read an MO-index header row (integers) ────────────
        parts = line.split()
        try:
            mo_indices = [int(p) for p in parts]
        except ValueError:
            pos += 1
            # If we already have some data and hit a totally different section, stop
            if energies and not any(c in line for c in (".", ":", "-")):
                break
            continue

        # ── Energy row ────────────────────────────────────────────────
        pos += 1
        if pos >= len(lines):
            break
        eparts = lines[pos].split()
        try:
            block_ene = [float(e) for e in eparts]
            energies.extend(block_ene)
        except ValueError:
            continue

        # ── Occupancy row ─────────────────────────────────────────────
        pos += 1
        if pos >= len(lines):
            break
        oparts = lines[pos].split()
        try:
            block_occ = [float(o) for o in oparts]
            occs.extend(block_occ)
        except ValueError:
            continue

        # ── Separator row (--------) ──────────────────────────────────
        pos += 1
        if pos < len(lines) and "---" in lines[pos]:
            pos += 1

        # ── Coefficient rows ──────────────────────────────────────────
        n_mos_in_block = len(mo_indices)
        ao_idx = 0
        while pos < len(lines):
            cline = lines[pos]
            if not cline.strip():
                pos += 1
                break
            # Check if this is the start of a new block (all integers)
            test = cline.split()
            try:
                _ = [int(t) for t in test]
                break  # next block header
            except ValueError:
                pass

            # Parse coefficient line — two formats:
            #   With colon:    "  0Ni 1s    :  0.99912  -0.26543"
            #   Without colon: "  0Ni  1s        -0.997663  0.000000"
            #   Also:          "  0Ni  1f+1       0.001234 -0.005678"

            # Split the label from the coefficients
            has_colon = ":" in cline
            if has_colon:
                label_part, coeff_part = cline.split(":", 1)
                coeffs_str = coeff_part.split()
            else:
                # No colon — use regex to find where numbers start
                # Pattern: label tokens then floats like "-0.997663"
                m_num = re.search(r"(?<!\S)(-?\d+\.\d+)", cline)
                if not m_num:
                    pos += 1
                    continue
                split_pos = m_num.start()
                label_part = cline[:split_pos]
                coeff_part = cline[split_pos:]
                coeffs_str = coeff_part.split()

            try:
                coeffs = [float(c) for c in coeffs_str]
            except ValueError:
                pos += 1
                continue

            if len(coeffs) != n_mos_in_block:
                # Wrong number of coefficients — skip
                pos += 1
                continue

            if True:  # always enter this block (replaces old "if ':' in cline")
                if not labels_done:
                    # Parse label: "  0Ni 1s" or "  23C  2px"
                    lp = label_part.split()
                    if len(lp) >= 2:
                        # atom token like "0Ni" or "23C"
                        atom_tok = lp[0]
                        ang_tok  = lp[1]          # e.g. "1s", "2px", "3dz2"
                        # split atom_tok into index + element
                        m = re.match(r"(\d+)([A-Z][a-z]?)", atom_tok)
                        if m:
                            a_idx = int(m.group(1))
                            a_el  = m.group(2)
                        else:
                            a_idx, a_el = 0, atom_tok
                        # split ang_tok: e.g. "3dz2" → shell_n="3", ang="dz2"
                        m2 = re.match(r"(\d+)(.*)", ang_tok)
                        if m2:
                            shell_n = m2.group(1)
                            ang_lbl = m2.group(2)  # "s", "px", "py", "pz", "dz2", etc.
                        else:
                            shell_n, ang_lbl = "1", ang_tok
                        ao_labels.append((a_idx, a_el, shell_n, ang_lbl))

                # Store coefficients
                if ao_idx not in coeff_rows:
                    coeff_rows[ao_idx] = []
                coeff_rows[ao_idx].extend(coeffs)
                ao_idx += 1
            pos += 1

        labels_done = True  # only parse labels from the first block

    if not energies or not coeff_rows:
        return None

    nao = len(coeff_rows)
    nmo = len(energies)
    C = np.zeros((nao, nmo))
    for i in range(nao):
        row = coeff_rows.get(i, [])
        C[i, : len(row)] = row

    return {
        "energies":  np.array(energies),
        "occs":      np.array(occs),
        "coeffs":    C,
        "ao_labels": ao_labels,
    }


# ── GTO primitive normalisation constants ─────────────────────────────────────

def _norm_s(alpha: float) -> float:
    return (2.0 * alpha / np.pi) ** 0.75


def _norm_p(alpha: float) -> float:
    return (128.0 * alpha ** 5 / np.pi ** 3) ** 0.25


def _norm_d(alpha: float) -> float:
    """Common normalisation for all 5 spherical-d angular functions (xy convention)."""
    return (512.0 * alpha ** 7 / np.pi ** 3) ** 0.25


# ── Angular function look-up (Cartesian expressions for real solid harmonics) ─
#    With factors chosen so all 5 d-components share the SAME radial norm N_d.
#    dz2  angular = (2z² - x² - y²)/(2√3)
#    dxz  angular = xz
#    dyz  angular = yz
#    dx2y2 angular = (x² - y²)/2
#    dxy  angular = xy

_ANG_FUNC = {
    "s":     lambda dx, dy, dz: np.ones_like(dx),
    "px":    lambda dx, dy, dz: dx,
    "py":    lambda dx, dy, dz: dy,
    "pz":    lambda dx, dy, dz: dz,
    "dz2":   lambda dx, dy, dz: (2.0 * dz * dz - dx * dx - dy * dy) / (2.0 * np.sqrt(3.0)),
    "dxz":   lambda dx, dy, dz: dx * dz,
    "dyz":   lambda dx, dy, dz: dy * dz,
    "dx2y2": lambda dx, dy, dz: 0.5 * (dx * dx - dy * dy),
    "dxy":   lambda dx, dy, dz: dx * dy,
    # f-shell (7 spherical) — placeholder shapes, relative norms may be approximate
    "fz3":      lambda dx, dy, dz: dz * (2*dz*dz - 3*dx*dx - 3*dy*dy),
    "fxz2":     lambda dx, dy, dz: dx * (4*dz*dz - dx*dx - dy*dy),
    "fyz2":     lambda dx, dy, dz: dy * (4*dz*dz - dx*dx - dy*dy),
    "fxyz":     lambda dx, dy, dz: dx * dy * dz,
    "fz(x2-y2)": lambda dx, dy, dz: dz * (dx*dx - dy*dy),
    "fx(x2-3y2)": lambda dx, dy, dz: dx * (dx*dx - 3*dy*dy),
    "fy(3x2-y2)": lambda dx, dy, dz: dy * (3*dx*dx - dy*dy),
}

# Alternative ORCA angular labels we might encounter in the MO printout
# ORCA uses spherical harmonics for f: f0, f+1, f-1, f+2, f-2, f+3, f-3
_ANG_ALIASES = {
    "x2-y2": "dx2y2", "x2y2": "dx2y2",
    "z2": "dz2", "xz": "dxz", "yz": "dyz", "xy": "dxy",
    "x": "px", "y": "py", "z": "pz",
    # ORCA spherical-f labels → our _ANG_FUNC keys
    "f0":  "fz3",
    "f+1": "fxz2", "f1":  "fxz2",
    "f-1": "fyz2", "f1":  "fyz2",
    "f+2": "fz(x2-y2)", "f2": "fz(x2-y2)",
    "f-2": "fxyz",
    "f+3": "fx(x2-3y2)", "f3": "fx(x2-3y2)",
    "f-3": "fy(3x2-y2)",
}


def _resolve_ang(label: str) -> str:
    """Map an angular label from the ORCA MO printout to our _ANG_FUNC key."""
    lo = label.lower().strip()
    # First try exact match (preserving +/- for f labels)
    if lo in _ANG_FUNC:
        return lo
    if lo in _ANG_ALIASES:
        return _ANG_ALIASES[lo]
    # Try without dashes (for d labels like "x2-y2" → "x2y2")
    lo_nodash = lo.replace("-", "").replace(" ", "")
    if lo_nodash in _ANG_FUNC:
        return lo_nodash
    if lo_nodash in _ANG_ALIASES:
        return _ANG_ALIASES[lo_nodash]
    # Try with "d" or "f" prefix
    for prefix in ("d", "f"):
        key = prefix + lo_nodash
        if key in _ANG_FUNC:
            return key
    return lo  # best-effort


def _norm_for_ang(ang_key: str, alpha: float) -> float:
    """Return the full primitive normalisation N(alpha) for the given angular type."""
    if ang_key == "s":
        return _norm_s(alpha)
    if ang_key.startswith("p"):
        return _norm_p(alpha)
    if ang_key.startswith("d"):
        return _norm_d(alpha)
    if ang_key.startswith("f"):
        # Approximate — f normalisation varies by component; use the "fxyz" form
        return (2.0 * alpha / np.pi) ** 0.75 * (4.0 * alpha) ** 1.5 / np.sqrt(15.0)
    return 1.0  # fallback (g-shell etc.)


def _build_ao_info(basis: Dict[str, List[dict]],
                   ao_labels: List[tuple],
                   atom_coords: List[Tuple]) -> List[dict]:
    """
    Combine the parsed basis set with the AO labels from the MO printout
    to produce one record per AO:
      {'center': (x,y,z), 'ang': str, 'prims': [(alpha,coeff),...]}
    """
    # Build a map: (element, shell_type, shell_ordinal) → shell dict
    # shell_ordinal is the 1-based index of that shell type on the element.
    elem_shell_map: Dict[Tuple[str, str, int], dict] = {}
    for el, shells in basis.items():
        counters: Dict[str, int] = {}
        for sh in shells:
            st = sh["type"]
            counters[st] = counters.get(st, 0) + 1
            elem_shell_map[(el, st, counters[st])] = sh

    ao_info: List[dict] = []
    # Track which shell ordinal each (atom, shell_type) is up to
    seen_shell: Dict[Tuple[int, str], int] = {}  # (atom_idx, shell_type) → ordinal

    for a_idx, a_el, shell_n_str, ang_lbl in ao_labels:
        ang_key = _resolve_ang(ang_lbl)
        if ang_key.startswith("d"):
            sh_type = "D"
        elif ang_key.startswith("f"):
            sh_type = "F"
        elif ang_key.startswith("p"):
            sh_type = "P"
        else:
            sh_type = "S"

        shell_ord = int(shell_n_str)
        sh = elem_shell_map.get((a_el, sh_type, shell_ord))

        center = atom_coords[a_idx] if a_idx < len(atom_coords) else (0.0, 0.0, 0.0)
        ao_info.append({
            "center": (center[1], center[2], center[3]),  # xyz from read_xyz tuple
            "ang":    ang_key,
            "prims":  sh["prims"] if sh else [],
        })
    return ao_info


def evaluate_mo_on_grid(ao_info: List[dict],
                        mo_coeffs: np.ndarray,
                        spacing: float = 0.30,
                        padding: float = 4.5,
                        reorient_R: Optional[np.ndarray] = None,
                        reorient_T: Optional[np.ndarray] = None,
                        ao_labels: Optional[List[tuple]] = None) -> dict:
    """
    Evaluate a single MO on a regular 3D grid using the AO definitions.

    Parameters
    ----------
    ao_info    : list of AO dicts from _build_ao_info
    mo_coeffs  : 1-D array of MO expansion coefficients (length = nao)
    spacing    : grid spacing in Angstrom
    padding    : extra space around the molecule in Angstrom
    reorient_R : 3x3 rotation matrix for re-orientation (optional)
    reorient_T : 3-vector translation for re-orientation (optional)
    ao_labels  : AO label list (needed for coefficient rotation)

    Returns
    -------
    cube-format dict (compatible with _render_iso).
    """
    # If re-orientation is requested, rotate MO coefficients and transform centres
    if reorient_R is not None and reorient_T is not None:
        # Rotate the MO coefficients so the angular functions are expressed
        # in the new coordinate frame
        if ao_labels is not None:
            mo_coeffs = _rotate_mo_coeffs(mo_coeffs, ao_labels, reorient_R)
        # Transform atom centres to new frame
        ao_info = [
            {**ao, "center": tuple(reorient_R @ (np.array(ao["center"]) - reorient_T))}
            for ao in ao_info
        ]

    # Determine grid extents from atom centres
    centres = np.array([ao["center"] for ao in ao_info])
    lo = centres.min(axis=0) - padding
    hi = centres.max(axis=0) + padding
    xi = np.arange(lo[0], hi[0] + spacing, spacing)
    yi = np.arange(lo[1], hi[1] + spacing, spacing)
    zi = np.arange(lo[2], hi[2] + spacing, spacing)
    NX, NY, NZ = len(xi), len(yi), len(zi)

    # Pre-build full 3D coordinate arrays (memory-efficient broadcasting)
    # Use (NX,1,1), (1,NY,1), (1,1,NZ) shapes for broadcasting
    val = np.zeros((NX, NY, NZ), dtype=np.float64)

    # Group AOs by unique atom centre to reuse dx/dy/dz/r2
    centre_groups: Dict[tuple, List[int]] = {}
    for i, ao in enumerate(ao_info):
        key = ao["center"]
        centre_groups.setdefault(key, []).append(i)

    for centre, ao_indices in centre_groups.items():
        cx, cy, cz = centre
        dx = (xi - cx).reshape(-1, 1, 1)     # (NX,1,1)
        dy = (yi - cy).reshape(1, -1, 1)     # (1,NY,1)
        dz = (zi - cz).reshape(1, 1, -1)     # (1,1,NZ)
        r2 = dx * dx + dy * dy + dz * dz     # (NX,NY,NZ) via broadcasting

        for i_ao in ao_indices:
            c_mo = mo_coeffs[i_ao]
            if abs(c_mo) < 1e-12:
                continue  # skip negligible contributions

            ao = ao_info[i_ao]
            ang_key = ao["ang"]
            prims   = ao["prims"]
            if not prims:
                continue

            ang_func = _ANG_FUNC.get(ang_key)
            if ang_func is None:
                continue

            # Contracted radial part (with primitive normalisation)
            radial = np.zeros_like(r2)
            for alpha, coeff in prims:
                N = _norm_for_ang(ang_key, alpha)
                radial += coeff * N * np.exp(-alpha * r2)

            val += c_mo * ang_func(dx, dy, dz) * radial

    # Build a cube-format dict
    # Unique atoms for the molecular structure
    seen = {}
    atom_list = []
    for ao in ao_info:
        c = ao["center"]
        if c not in seen:
            seen[c] = True
            atom_list.append(c)

    atoms_cube = []
    for c in atom_list:
        atoms_cube.append((6, 6.0, c[0], c[1], c[2]))  # default to C (we fix below)

    return {
        "comment": "MO from ORCA LargePrint",
        "atoms":   atoms_cube,
        "origin":  np.array([lo[0], lo[1], lo[2]]),
        "axes":    np.diag([spacing, spacing, spacing]),
        "n":       np.array([NX, NY, NZ], dtype=int),
        "data":    val,
        "source":  "orca_largeprint",
    }


def build_cube_from_orca_output(out_path: str,
                                mo_index: int,
                                spacing: float = 0.30,
                                padding: float = 4.5,
                                reorient_R: Optional[np.ndarray] = None,
                                reorient_T: Optional[np.ndarray] = None) -> dict:
    """
    High-level: read an ORCA .out file that was run with LargePrint,
    extract basis set + MO coefficients, and evaluate the requested MO
    on a 3D grid.

    Parameters
    ----------
    out_path   : path to the ORCA .out file
    mo_index   : 0-based MO index
    spacing    : grid resolution (Angstrom) — smaller = finer (0.20 is fine, 0.40 coarse)
    padding    : padding around molecule (Angstrom)
    reorient_R : 3x3 rotation matrix for re-orientation (optional)
    reorient_T : 3-vector translation (optional)

    Returns
    -------
    cube-format dict ready for _render_iso().

    Raises RuntimeError with a human-readable explanation on failure.
    """
    # Use streaming extractor (handles 400+ MB files efficiently)
    basis_text, mo_text = _extract_orca_sections(out_path)

    # 1. Parse basis set
    basis = _parse_orca_basis(basis_text) if basis_text else {}
    if not basis:
        raise RuntimeError(
            "Could not find 'BASIS SET IN INPUT FORMAT' in the output file.\n"
            "Make sure ORCA printed the basis set (this is default)."
        )

    # 2. Parse MO coefficients
    mo_data = _parse_orca_mos(mo_text) if mo_text else None
    if mo_data is None:
        raise RuntimeError(
            "Could not find the MOLECULAR ORBITALS section.\n\n"
            "Add one of these to your ORCA input and re-run:\n"
            "  ! LargePrint\n"
            "or:\n"
            "  %output Print[P_MOs] 1 end"
        )

    nmo = mo_data["coeffs"].shape[1]
    if mo_index < 0 or mo_index >= nmo:
        raise RuntimeError(f"MO index {mo_index} out of range [0, {nmo - 1}].")

    # 3. Read atom coordinates from the same .out (re-use our xyz reader on the
    #    companion .xyz, or parse from CARTESIAN COORDINATES in the .out)
    xyz_path = Path(out_path).with_suffix(".xyz")
    atoms = read_xyz(str(xyz_path))
    if not atoms:
        # Try parsing coordinates from the .out text
        with open(out_path, "r", errors="ignore") as fh:
            out_text = fh.read()
        atoms = _parse_orca_coords(out_text)
    if not atoms:
        raise RuntimeError(
            f"Could not find atom coordinates.\n"
            f"Place a .xyz file next to the .out: {xyz_path.name}"
        )

    # 4. Build AO info (maps each row of the coefficient matrix to a GTO)
    ao_info = _build_ao_info(basis, mo_data["ao_labels"], atoms)

    # 5. Fix atom list in the cube for proper rendering
    mo_coeffs = mo_data["coeffs"][:, mo_index]

    cube = evaluate_mo_on_grid(ao_info, mo_coeffs, spacing, padding,
                               reorient_R=reorient_R, reorient_T=reorient_T,
                               ao_labels=mo_data["ao_labels"])

    # Overwrite dummy atoms with real atom info
    unique_atoms = []
    seen_idx = set()
    for lbl in mo_data["ao_labels"]:
        a_idx, a_el = lbl[0], lbl[1]
        if a_idx not in seen_idx:
            seen_idx.add(a_idx)
            z = ATOMIC_NUM.get(a_el, 6)
            x, y, zc = atoms[a_idx][1], atoms[a_idx][2], atoms[a_idx][3]
            if reorient_R is not None and reorient_T is not None:
                r = reorient_R @ (np.array([x, y, zc]) - reorient_T)
                x, y, zc = float(r[0]), float(r[1]), float(r[2])
            unique_atoms.append((z, float(z), x, y, zc))
    cube["atoms"] = unique_atoms

    # Enrich the comment
    ene = mo_data["energies"][mo_index]
    occ = mo_data["occs"][mo_index]
    homo_idx = int(np.max(np.where(mo_data["occs"] > 0.5)))
    delta = mo_index - homo_idx
    if   delta == 0:  tag = "HOMO"
    elif delta == 1:  tag = "LUMO"
    elif delta < 0:   tag = f"HOMO{delta}"
    else:             tag = f"LUMO+{delta - 1}"
    cube["comment"] = (f"MO {mo_index}  ({tag})  "
                       f"E = {ene:.5f} Eh  occ = {occ:.2f}  |  "
                       f"{Path(out_path).stem}")

    return cube


def compute_mo_composition(mo_data: dict, mo_index: int,
                           reorient_R: Optional[np.ndarray] = None) -> dict:
    """
    Compute % contribution of each atom and each angular momentum type
    to a given MO, using Mulliken-style |C_μi|² analysis.

    If reorient_R is provided, MO coefficients are first rotated into the
    new coordinate frame so that d-orbital labels (dxy, dz2, etc.) reflect
    the re-oriented axes.

    Returns
    -------
    dict with keys:
      'atom_contribs': {atom_key: {'s': %, 'p': %, 'd': %, 'f': %, 'total': %}, ...}
      'total_s', 'total_p', 'total_d', 'total_f': overall %
      'top_atoms': [(atom_key, total%), ...] sorted descending
      'top_ao': [(atom_key, ang_label, %), ...] sorted descending
    """
    coeffs = mo_data["coeffs"][:, mo_index]
    ao_labels = mo_data["ao_labels"]

    # If re-oriented, rotate the coefficients first
    if reorient_R is not None:
        coeffs = _rotate_mo_coeffs(coeffs, ao_labels, reorient_R)

    # Mulliken-style: contribution of AO mu = C_mu^2
    # (This is approximate — true Mulliken uses overlap matrix S,
    #  but C^2 gives a reasonable picture for orthogonal-ish AOs)
    c2 = coeffs ** 2
    total_c2 = c2.sum()
    if total_c2 < 1e-30:
        return {'atom_contribs': {}, 'total_s': 0, 'total_p': 0, 'total_d': 0, 'total_f': 0,
                'top_atoms': [], 'top_ao': []}

    atom_contribs = {}  # atom_key -> {s, p, d, f, total}
    ao_detail = []      # (atom_key, ang_label, pct)
    # Per-atom, per-resolved-AO detail: atom_key -> {resolved_ang -> pct}
    atom_ao_detail = {}   # atom_key -> {ang_resolved: pct}  (summed across shells)
    # Per-atom, per-shell AO detail: atom_key -> [(shell_n, ang_resolved, pct), ...]
    atom_ao_shells = {}   # atom_key -> list of (shell_n, ang_resolved, pct)
    # Global summed AO types: ang_resolved -> total pct (across all atoms/shells)
    summed_ao_types = {}  # e.g. {"dx2y2": 12.5, "dz2": 45.0, "px": 3.2, ...}

    for i, (a_idx, a_el, shell_n, ang_lbl) in enumerate(ao_labels):
        atom_key = f"{a_el}{a_idx + 1}"
        pct = 100.0 * c2[i] / total_c2

        ang_resolved = _resolve_ang(ang_lbl)
        if ang_resolved.startswith('d'):
            ang_type = 'd'
        elif ang_resolved.startswith('f'):
            ang_type = 'f'
        elif ang_resolved.startswith('p'):
            ang_type = 'p'
        else:
            ang_type = 's'

        if atom_key not in atom_contribs:
            atom_contribs[atom_key] = {'s': 0.0, 'p': 0.0, 'd': 0.0, 'f': 0.0, 'total': 0.0,
                                       'element': a_el, 'index': a_idx}
        atom_contribs[atom_key][ang_type] += pct
        atom_contribs[atom_key]['total'] += pct

        if pct > 0.1:
            ao_detail.append((atom_key, f"{shell_n}{ang_lbl}", pct))

        # Sum by resolved angular type across shells
        summed_ao_types[ang_resolved] = summed_ao_types.get(ang_resolved, 0.0) + pct

        # Per-atom resolved AO (summed across shells, e.g. 1dz2 + 2dz2 → dz2)
        if atom_key not in atom_ao_detail:
            atom_ao_detail[atom_key] = {}
        atom_ao_detail[atom_key][ang_resolved] = atom_ao_detail[atom_key].get(ang_resolved, 0.0) + pct

        # Per-atom per-shell AO detail
        if atom_key not in atom_ao_shells:
            atom_ao_shells[atom_key] = []
        atom_ao_shells[atom_key].append((shell_n, ang_resolved, pct))

    total_s = sum(v['s'] for v in atom_contribs.values())
    total_p = sum(v['p'] for v in atom_contribs.values())
    total_d = sum(v['d'] for v in atom_contribs.values())
    total_f = sum(v['f'] for v in atom_contribs.values())

    top_atoms = sorted(atom_contribs.items(), key=lambda x: x[1]['total'], reverse=True)
    top_ao = sorted(ao_detail, key=lambda x: x[2], reverse=True)

    # Sort summed AO types by contribution
    sorted_ao_types = sorted(summed_ao_types.items(), key=lambda x: x[1], reverse=True)

    return {
        'atom_contribs': atom_contribs,
        'total_s': total_s,
        'total_p': total_p,
        'total_d': total_d,
        'total_f': total_f,
        'top_atoms': [(k, v['total']) for k, v in top_atoms],
        'top_ao': top_ao[:20],  # top 20 AOs
        'summed_ao_types': sorted_ao_types,     # [(ang_resolved, total_pct), ...]
        'atom_ao_detail': atom_ao_detail,        # {atom_key: {ang_resolved: pct}}
        'atom_ao_shells': atom_ao_shells,        # {atom_key: [(shell_n, ang, pct), ...]}
    }


_RE_ATOM_KEY = re.compile(r"^([A-Z][a-z]?)(\d+)$")
_RE_MO_INDEX = re.compile(r"^(?:MO)?\s*(\d+)$", re.I)
_RE_MO_RANGE = re.compile(r"^(?:MO)?\s*(\d+)\s*-\s*(?:MO)?\s*(\d+)$", re.I)
_RE_MO_SPAN = re.compile(r"^(.+?)(?::|\.\.)(.+)$")
_RE_HOMO_TOKEN = re.compile(r"^H(?:OMO)?(?:([+-]\d+))?$", re.I)
_RE_LUMO_TOKEN = re.compile(r"^L(?:UMO)?(?:([+-]\d+))?$", re.I)


def _mo_frontier_label(mo_index: int, homo_idx: int) -> str:
    """Return a HOMO/LUMO style label for a molecular orbital index."""
    delta = mo_index - homo_idx
    if delta == 0:
        return "HOMO"
    if delta == 1:
        return "LUMO"
    if delta < 0:
        return f"HOMO{delta}"
    return f"LUMO+{delta - 1}"


def _parse_single_mo_token(token: str, homo_idx: int) -> Optional[int]:
    """Parse one MO token like 118, MO118, HOMO, or LUMO+2."""
    tok = token.strip().upper().replace(" ", "")
    if not tok:
        return None

    m = _RE_MO_INDEX.match(tok)
    if m:
        return int(m.group(1))

    m = _RE_HOMO_TOKEN.match(tok)
    if m:
        offset = int(m.group(1)) if m.group(1) else 0
        return homo_idx + offset

    m = _RE_LUMO_TOKEN.match(tok)
    if m:
        offset = int(m.group(1)) if m.group(1) else 0
        return homo_idx + 1 + offset

    return None


def parse_mo_selection_list(text: str, nmo: int, homo_idx: int) -> Tuple[List[int], List[str]]:
    """
    Parse a comma/semicolon separated MO list.

    Examples
    --------
    118-124, 126, HOMO, LUMO+2
    HOMO-5:LUMO+3
    """
    chosen = set()
    errors: List[str] = []

    tokens = [tok.strip() for tok in re.split(r"[,\n;]+", text) if tok.strip()]
    for tok in tokens:
        m = _RE_MO_RANGE.match(tok)
        if m:
            lo = int(m.group(1))
            hi = int(m.group(2))
            if lo > hi:
                lo, hi = hi, lo
            for mo_i in range(lo, hi + 1):
                if 0 <= mo_i < nmo:
                    chosen.add(mo_i)
                else:
                    errors.append(f"{tok} (out of range)")
                    break
            continue

        m = _RE_MO_SPAN.match(tok)
        if m:
            lo = _parse_single_mo_token(m.group(1), homo_idx)
            hi = _parse_single_mo_token(m.group(2), homo_idx)
            if lo is None or hi is None:
                errors.append(tok)
                continue
            if lo > hi:
                lo, hi = hi, lo
            if lo < 0 or hi >= nmo:
                errors.append(f"{tok} (out of range)")
                continue
            for mo_i in range(lo, hi + 1):
                chosen.add(mo_i)
            continue

        mo_i = _parse_single_mo_token(tok, homo_idx)
        if mo_i is None:
            errors.append(tok)
            continue
        if not (0 <= mo_i < nmo):
            errors.append(f"{tok} (out of range)")
            continue
        chosen.add(mo_i)

    return sorted(chosen), errors


def _parse_orca_coords(text: str) -> List[Tuple]:
    """Fallback: parse CARTESIAN COORDINATES (ANGSTROEM) from ORCA output."""
    atoms: List[Tuple] = []
    in_block = False
    for line in text.splitlines():
        if "CARTESIAN COORDINATES (ANGSTROEM)" in line:
            atoms = []
            in_block = True
            continue
        if in_block:
            s = line.strip()
            if not s or s.startswith("-"):
                if atoms:
                    break
                continue
            parts = s.split()
            if len(parts) >= 4:
                try:
                    atoms.append((parts[0], float(parts[1]),
                                  float(parts[2]), float(parts[3])))
                except ValueError:
                    pass
    return atoms


# ─────────────────────────────────────────────────────────────────────────────
#  Bond detection
# ─────────────────────────────────────────────────────────────────────────────

_METALS = {'Ni', 'Fe', 'Co', 'Cu', 'Pd', 'Pt', 'Ru', 'Rh',
           'Mn', 'Cr', 'Ti', 'V', 'Sc', 'Zn', 'Os', 'Ir', 'Au'}


def _bond_cutoff(el1: str, el2: str) -> float:
    if el1 == "H" and el2 == "H": return 0.0          # no H–H bonds shown
    if "H" in (el1, el2):         return 1.20
    if el1 in _METALS or el2 in _METALS: return 2.85   # metal–ligand bond
    # Sum of covalent radii + 20 % tolerance
    r1 = ELEM_RADIUS.get(el1, _DEFAULT_RADIUS)
    r2 = ELEM_RADIUS.get(el2, _DEFAULT_RADIUS)
    return (r1 + r2) * 1.20


def _detect_bonds(atoms: List[Tuple]) -> List[Tuple[int, int]]:
    bonds = []
    n = len(atoms)
    coords = np.array([[a[1], a[2], a[3]] for a in atoms])
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < _bond_cutoff(atoms[i][0], atoms[j][0]):
                bonds.append((i, j))
    return bonds


# ─────────────────────────────────────────────────────────────────────────────
#  Isosurface helper
# ─────────────────────────────────────────────────────────────────────────────

def _dark_3d_ax(ax) -> None:
    """Apply dark background to a 3D axes including the wall panes and grid."""
    ax.set_facecolor("#1c1c1e")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = True
        pane.set_facecolor("#1c1c1e")
        pane.set_edgecolor("#3a3a3a")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:                                      # guard against API changes
            axis._axinfo["grid"]["color"]     = "#333333"
            axis._axinfo["grid"]["linewidth"] = 0.5
        except Exception:
            pass


def _clean_3d_ax(ax, bg_color: str = "#1c1c1e") -> None:
    """
    Avogadro-style clean 3D axes: no ticks, no grid, no labels, no pane edges.
    Just a solid background color with the molecule floating in space.
    """
    ax.set_facecolor(bg_color)
    # Hide all pane edges and fill with background
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = True
        pane.set_facecolor(bg_color)
        pane.set_edgecolor(bg_color)        # hide edges
        pane.set_alpha(1.0)
    # Hide axes, ticks, labels, and grid
    ax.set_axis_off()
    # Equal aspect-like: set same scale on all axes
    ax.grid(False)


def _apply_reorient(atoms: List[Tuple], R: np.ndarray, T: np.ndarray) -> List[Tuple]:
    """Apply translation T then rotation R to atom coordinates."""
    result = []
    for a in atoms:
        el = a[0]
        xyz = np.array([a[1], a[2], a[3]]) - T
        xyz_rot = R @ xyz
        result.append((el, float(xyz_rot[0]), float(xyz_rot[1]), float(xyz_rot[2])))
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Rotation matrices for real solid harmonics (for orbital re-orientation)
# ─────────────────────────────────────────────────────────────────────────────
#
#  When we rotate the coordinate frame by R (3x3 orthogonal matrix), the
#  Cartesian real solid harmonics transform among themselves:
#    S_lm(R⁻¹ r) = Σ_{m'} D^l_{mm'} S_{lm'}(r)
#
#  For l=1 (p-orbitals): D¹ = R itself (px,py,pz transform as a vector)
#  For l=2 (d-orbitals): D² is a 5x5 matrix computed from R
#
#  The d-orbital order used throughout is: [dz2, dxz, dyz, dx2y2, dxy]
#  matching the _ANG_FUNC keys.

_D_ORBITAL_ORDER = ["dz2", "dxz", "dyz", "dx2y2", "dxy"]
_P_ORBITAL_ORDER = ["px", "py", "pz"]


def _p_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """
    Return the 3x3 rotation matrix for real p-orbitals.
    p-orbitals (px, py, pz) transform as the Cartesian vector (x, y, z),
    so D¹ = R.
    """
    return R.copy()


def _d_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """
    Compute the 5x5 rotation matrix D² for real d-orbitals from the 3x3
    rotation matrix R.

    The d-orbitals in Cartesian form are (matching our _ANG_FUNC):
      dz2   = (2z² - x² - y²) / (2√3)
      dxz   = xz
      dyz   = yz
      dx2y2 = (x² - y²) / 2
      dxy   = xy

    Under rotation r = R·r', the old-frame polynomial evaluated at a new-frame
    point r' equals a linear combination of new-frame polynomials.

    We compute D² by substituting r = R·r' into each polynomial and collecting
    coefficients in terms of the new-frame polynomials.
    """
    # R elements: R[i,j] gives old_i = Σ_j R[i,j] new_j
    # Notation: a,b,c = R[0,:]; d,e,f = R[1,:]; g,h,k = R[2,:]
    a, b, c = R[0, 0], R[0, 1], R[0, 2]
    d, e, f = R[1, 0], R[1, 1], R[1, 2]
    g, h, k = R[2, 0], R[2, 1], R[2, 2]

    # We need to express each OLD d-function as a linear combo of NEW d-functions.
    # x_old = a*x' + b*y' + c*z', etc.
    #
    # The 5 independent quadratic forms in NEW coords are:
    # z'², x'z', y'z', x'², y'², x'y'
    # But we must express in terms of our 5 basis polynomials:
    #   dz2'   = (2z'² - x'² - y'²)/(2√3)
    #   dxz'   = x'z'
    #   dyz'   = y'z'
    #   dx2y2' = (x'² - y'²)/2
    #   dxy'   = x'y'
    #
    # Note: x'² = (1/√3)*dz2' + dx2y2' + const*(x'²+y'²+z'²) ...
    # Actually, the 6 quadratic monomials span a 6D space, but
    # x²+y²+z² = r² is rotationally invariant and decouples.
    # The 5 traceless combinations are our d-functions.

    sq3 = np.sqrt(3.0)
    D = np.zeros((5, 5), dtype=np.float64)

    # Row 0: old dz2 = (2*z_old² - x_old² - y_old²) / (2√3)
    # z_old² = (gx'+hy'+kz')² = g²x'² + h²y'² + k²z'² + 2ghx'y' + 2gkx'z' + 2hky'z'
    # x_old² = a²x'² + b²y'² + c²z'² + 2abx'y' + 2acx'z' + 2bcy'z'
    # y_old² = d²x'² + e²y'² + f²z'² + 2dex'y' + 2dfx'z' + 2efy'z'
    #
    # 2z² - x² - y² gives coefficients of each monomial:
    # x'²:  2g²-a²-d²,  y'²: 2h²-b²-e²,  z'²: 2k²-c²-f²
    # x'y': 2(2gh-ab-de), x'z': 2(2gk-ac-df), y'z': 2(2hk-bc-ef)
    #
    # Divide by (2√3):
    coeff_x2  = (2*g*g - a*a - d*d) / (2*sq3)
    coeff_y2  = (2*h*h - b*b - e*e) / (2*sq3)
    coeff_z2  = (2*k*k - c*c - f*f) / (2*sq3)
    coeff_xy  = 2*(2*g*h - a*b - d*e) / (2*sq3)
    coeff_xz  = 2*(2*g*k - a*c - d*f) / (2*sq3)
    coeff_yz  = 2*(2*h*k - b*c - e*f) / (2*sq3)

    # Now express in terms of d-basis:
    # x'² = (2/√3)*dz2' + dx2y2' + (x'²+y'²+z'²)/3  ... wait, let me be more careful.
    #
    # The d-basis polynomials form an orthogonal basis for traceless rank-2
    # symmetric tensors. We can invert:
    #   x'² = -(1/√3)*dz2' + dx2y2' + r'²/3
    #   y'² = -(1/√3)*dz2' - dx2y2' + r'²/3
    #   z'² = (2/√3)*dz2' + r'²/3
    #   x'y' = dxy'
    #   x'z' = dxz'
    #   y'z' = dyz'
    #
    # So a general traceless quadratic A*x'² + B*y'² + C*z'² + D*x'y' + E*x'z' + F*y'z'
    # with A+B+C = 0 (traceless — guaranteed for rotated d-functions) becomes:
    #   dz2  coeff: -(A+B)/√3 + 2C/√3 = (-A-B+2C)/√3
    #   But A+B+C=0 → A+B=-C → (-(-C)+2C)/√3 = 3C/√3 = C√3
    #   Hmm, let me redo this properly.
    #
    # dz2'  = (2z'²-x'²-y'²)/(2√3)  →  z'² = √3·dz2' + (x'²+y'²+z'²)/3
    # Actually let me just convert monomial coefficients to d-basis coefficients directly.
    #
    # From the d-basis definitions:
    #   dz2'   = (2z'² - x'² - y'²)/(2√3)
    #   dx2y2' = (x'² - y'²)/2
    #   dxy'   = x'y'
    #   dxz'   = x'z'
    #   dyz'   = y'z'
    #
    # Inverse (for traceless part, i.e., coeff_x2+coeff_y2+coeff_z2=0):
    #   D[i, dz2]   = coeff_z2 * √3  (since dz2=(2z²-x²-y²)/(2√3)
    #                  and 2*coeff_z2 - coeff_x2 - coeff_y2 = 2*coeff_z2 + coeff_z2 = 3*coeff_z2
    #                  so: 3*coeff_z2/(2√3) = coeff_z2 * √3/2... )
    # Let me just do this numerically to avoid mistakes.

    # Express the traceless quadratic as d-basis coefficients.
    # We have 6 monomial coefficients but the traceless constraint removes 1 DOF → 5 DOF = 5 d-functions.
    # Key: x'z', y'z', x'y' map directly to dxz', dyz', dxy'.
    # For the diagonal terms, use:
    #   x'² = -(1/√3)*dz2'(2√3) ... let me just solve the 3×2 system.
    #
    # dz2'   = (2*z'² - x'² - y'²)/(2√3)
    # dx2y2' = (x'² - y'²)/2
    #
    # Given traceless: x'²_coeff + y'²_coeff + z'²_coeff = 0
    # Let A = coeff_x2, B = coeff_y2, C = coeff_z2 (A+B+C=0)
    #
    # coeff of dz2'   in terms of A,B,C:
    #   original = A*x'² + B*y'² + C*z'² = ...
    #   x'² = -(1/√3)*P + Q + r²/3  where P = dz2'*(2√3), Q = dx2y2' ...
    #
    # OK I'm overcomplicating this. Let me just directly compute each row of D
    # using a numerical approach: evaluate old d-function at 5 test points in the
    # new frame.

    # NUMERICAL APPROACH: D[old_m, new_m'] = value of old_d_func(R @ r'_test)
    # where r'_test is chosen to isolate new_m'.
    #
    # But actually the cleanest way is: for each old d-function, substitute
    # x_old = R[0,:]·r', y_old = R[1,:]·r', z_old = R[2,:]·r' and collect
    # monomial coefficients, then convert to d-basis.
    #
    # Final clean approach:
    # For a general traceless quadratic Q = A*x'² + B*y'² + C*z'² + D*x'y' + E*x'z' + F*y'z'
    # with A+B+C=0:
    #   Q = A*(x'²-z'²) + B*(y'²-z'²) + D*x'y' + E*x'z' + F*y'z'
    #     (using C = -A-B)
    #
    # dz2  = (2z'²-x'²-y'²)/(2√3) = -(x'²+y'²-2z'²)/(2√3)
    #       = -(  (x'²-z'²) + (y'²-z'²)  )/(2√3)
    #
    # dx2y2 = (x'²-y'²)/2 = ((x'²-z'²) - (y'²-z'²))/2
    #
    # So:  x'²-z'² = -√3*dz2 + dx2y2    ... wait...
    # Let u = x'²-z'², v = y'²-z'²
    #   dz2   = -(u+v)/(2√3)
    #   dx2y2 = (u-v)/2
    # Invert:
    #   u+v = -2√3*dz2
    #   u-v = 2*dx2y2
    #   u = -√3*dz2 + dx2y2
    #   v = -√3*dz2 - dx2y2
    #
    # So: Q = A*u + B*v + D*dxy + E*dxz + F*dyz
    #       = A*(-√3*dz2+dx2y2) + B*(-√3*dz2-dx2y2) + D*dxy + E*dxz + F*dyz
    #       = -(A+B)*√3*dz2 + (A-B)*dx2y2 + D*dxy + E*dxz + F*dyz
    #       = C*√3*dz2 + (A-B)*dx2y2 + D*dxy + E*dxz + F*dyz
    #       (since A+B = -C)
    #
    # So: d-basis coefficients for a traceless quadratic with monomial coeffs A,B,C,D,E,F:
    #   dz2   coeff = C*√3     (= coeff_z2 * √3)
    #   dxz   coeff = E        (= coeff_xz)
    #   dyz   coeff = F        (= coeff_yz)
    #   dx2y2 coeff = A - B    (= coeff_x2 - coeff_y2)
    #   dxy   coeff = D        (= coeff_xy)

    # But we need to account for the normalisation factors in the d-basis definitions!
    # Our d-functions have specific numerical prefactors:
    #   dz2   = (2z²-x²-y²)/(2√3)     → monomial z² has factor 2/(2√3) = 1/√3
    #   dx2y2 = (x²-y²)/2             → monomial x² has factor 1/2
    #   dxy   = xy                     → monomial xy has factor 1
    #   dxz   = xz                     → monomial xz has factor 1
    #   dyz   = yz                     → monomial yz has factor 1
    #
    # So if the rotated d-function gives monomials with coefficients A,B,C,D,E,F:
    # Q = A*x'² + B*y'² + C*z'² + D*x'y' + E*x'z' + F*y'z'
    #
    # We want: Q = α*dz2' + β*dxz' + γ*dyz' + δ*dx2y2' + ε*dxy'
    # Expanding the RHS:
    #   α*(2z'²-x'²-y'²)/(2√3) + δ*(x'²-y'²)/2 + ε*x'y' + β*x'z' + γ*y'z'
    #
    # Matching x'²: -α/(2√3) + δ/2 = A
    # Matching y'²: -α/(2√3) - δ/2 = B
    # Matching z'²: α*2/(2√3) = α/√3 = C
    # Matching x'y': ε = D
    # Matching x'z': β = E
    # Matching y'z': γ = F
    #
    # From z'²: α = C*√3
    # From x'²: -C*√3/(2√3) + δ/2 = A → -C/2 + δ/2 = A → δ = 2A + C = 2A + C
    # But A+B+C=0 → C = -A-B → δ = 2A - A - B = A - B ✓
    #
    # So the conversion is confirmed:
    #   α (dz2)   = C * √3           = coeff_z2 * √3
    #   β (dxz)   = E                = coeff_xz
    #   γ (dyz)   = F                = coeff_yz
    #   δ (dx2y2) = A - B            = coeff_x2 - coeff_y2
    #   ε (dxy)   = D                = coeff_xy

    def _rotated_d_coeffs(old_func_monomial_coeffs):
        """Convert 6 monomial coefficients (A,B,C for x²,y²,z² and D,E,F for xy,xz,yz)
        of a traceless quadratic to d-basis coefficients [dz2, dxz, dyz, dx2y2, dxy]."""
        A, B, C, D, E, F = old_func_monomial_coeffs
        return np.array([C * sq3, E, F, A - B, D])

    # For each old d-function, compute monomial coefficients after rotation.
    # old x = a*x' + b*y' + c*z',  old y = d*x' + e*y' + f*z',  old z = g*x' + h*y' + k*z'

    # Row 0: old dz2 = (2*z_old² - x_old² - y_old²) / (2√3)
    # The monomial coefficients of the raw quadratic (2z²-x²-y²) are:
    # x'²: 2g²-a²-d²,   y'²: 2h²-b²-e²,   z'²: 2k²-c²-f²
    # x'y': 2(2gh-ab-de), x'z': 2(2gk-ac-df), y'z': 2(2hk-bc-ef)
    # Divided by (2√3) to get the actual d-function value.
    # But we need the monomial coefficients of the UNNORMALIZED quadratic (the full expression
    # as a quadratic in x',y',z'), then convert to d-basis.
    # Since dz2_old = (2z²-x²-y²)/(2√3), and we compute raw = 2z²-x²-y² monomials,
    # the actual monomials are raw/(2√3). Then we convert those to d-basis.

    # Actually, let me reconsider. The d-functions as defined in _ANG_FUNC have specific
    # numerical prefactors. We want D such that:
    #   d_old_m(R·r') = Σ_{m'} D[m, m'] d_new_m'(r')
    #
    # So we need to evaluate d_old_m at r = R·r' and express the result in terms of d_new_m'(r').

    # old dz2(R·r') = (2*(g*x'+h*y'+k*z')² - (a*x'+b*y'+c*z')² - (d*x'+e*y'+f*z')²) / (2√3)
    # Expand and collect monomials of x',y',z', then express in d-basis.

    # Helper: compute monomial coefficients [x'², y'², z'², x'y', x'z', y'z']
    # for product of two vectors: (p·r')(q·r') where p=[p1,p2,p3], q=[q1,q2,q3]
    def _quad_coeffs(p, q):
        """Monomial coefficients of (p·r')(q·r') = Σ p_i q_j x_i x_j"""
        return np.array([
            p[0]*q[0],                     # x'²
            p[1]*q[1],                     # y'²
            p[2]*q[2],                     # z'²
            p[0]*q[1] + p[1]*q[0],         # x'y'
            p[0]*q[2] + p[2]*q[0],         # x'z'
            p[1]*q[2] + p[2]*q[1],         # y'z'
        ])

    Rx, Ry, Rz = R[0, :], R[1, :], R[2, :]  # rows of R

    xx = _quad_coeffs(Rx, Rx)  # x_old²
    yy = _quad_coeffs(Ry, Ry)  # y_old²
    zz = _quad_coeffs(Rz, Rz)  # z_old²
    xy_q = _quad_coeffs(Rx, Ry)  # x_old * y_old
    xz_q = _quad_coeffs(Rx, Rz)  # x_old * z_old
    yz_q = _quad_coeffs(Ry, Rz)  # y_old * z_old

    # Row 0: dz2_old = (2*zz - xx - yy) / (2√3)
    raw = (2*zz - xx - yy) / (2*sq3)
    D[0, :] = _rotated_d_coeffs(raw)

    # Row 1: dxz_old = x_old * z_old = xz_q (already the monomial coeffs)
    D[1, :] = _rotated_d_coeffs(xz_q)

    # Row 2: dyz_old = y_old * z_old
    D[2, :] = _rotated_d_coeffs(yz_q)

    # Row 3: dx2y2_old = (x_old² - y_old²) / 2
    raw = (xx - yy) / 2.0
    D[3, :] = _rotated_d_coeffs(raw)

    # Row 4: dxy_old = x_old * y_old
    D[4, :] = _rotated_d_coeffs(xy_q)

    return D


def _rotate_mo_coeffs(mo_coeffs: np.ndarray, ao_labels: List[tuple],
                       R: np.ndarray) -> np.ndarray:
    """
    Rotate MO coefficients to account for a coordinate frame rotation R.

    For each angular momentum shell on each atom, apply the appropriate
    rotation matrix (identity for s, R for p, D² for d) to the MO coefficients.

    Parameters
    ----------
    mo_coeffs : 1-D array of MO expansion coefficients
    ao_labels : list of (atom_idx, element, shell_n, ang_lbl) tuples
    R         : 3x3 rotation matrix (new = R · old)

    Returns
    -------
    Rotated 1-D MO coefficient array.
    """
    rotated = mo_coeffs.copy()

    # Group AOs into shells (same atom, same shell_n, same l-type)
    # Each shell is a contiguous block of AOs
    i = 0
    n = len(ao_labels)
    D_p = _p_rotation_matrix(R)
    D_d = _d_rotation_matrix(R)

    while i < n:
        a_idx, a_el, shell_n, ang_lbl = ao_labels[i]
        ang = _resolve_ang(ang_lbl)

        if ang == "s":
            # s-orbital: no rotation needed
            i += 1
            continue

        if ang.startswith("p"):
            # Collect the 3 p-orbitals of this shell
            shell_indices = []
            shell_angs = []
            j = i
            while j < n and ao_labels[j][0] == a_idx and ao_labels[j][2] == shell_n:
                a_j = _resolve_ang(ao_labels[j][3])
                if a_j.startswith("p"):
                    shell_indices.append(j)
                    shell_angs.append(a_j)
                    j += 1
                else:
                    break
            if len(shell_indices) == 3:
                # Map to canonical order [px, py, pz]
                order_map = {a: idx for idx, a in enumerate(_P_ORBITAL_ORDER)}
                perm = [order_map.get(a, k) for k, a in enumerate(shell_angs)]
                old_c = np.array([mo_coeffs[shell_indices[p]] for p in perm])
                new_c = D_p @ old_c
                for k, p in enumerate(perm):
                    rotated[shell_indices[p]] = new_c[k]
            i = j if j > i else i + 1
            continue

        if ang.startswith("d"):
            # Collect the 5 d-orbitals of this shell
            shell_indices = []
            shell_angs = []
            j = i
            while j < n and ao_labels[j][0] == a_idx and ao_labels[j][2] == shell_n:
                a_j = _resolve_ang(ao_labels[j][3])
                if a_j.startswith("d"):
                    shell_indices.append(j)
                    shell_angs.append(a_j)
                    j += 1
                else:
                    break
            if len(shell_indices) == 5:
                # Map to canonical order [dz2, dxz, dyz, dx2y2, dxy]
                order_map = {a: idx for idx, a in enumerate(_D_ORBITAL_ORDER)}
                perm = [order_map.get(a, k) for k, a in enumerate(shell_angs)]
                old_c = np.array([mo_coeffs[shell_indices[p]] for p in perm])
                new_c = D_d @ old_c
                for k, p in enumerate(perm):
                    rotated[shell_indices[p]] = new_c[k]
            i = j if j > i else i + 1
            continue

        # f-orbitals: skip rotation for now (not yet implemented)
        i += 1

    return rotated


def _draw_axis_indicator(ax, xs, ys, zs, length_frac: float = 0.12):
    """
    Draw an Avogadro-style XYZ axis indicator: three small colored arrows
    in the bottom-left corner of the 3D view.
    X = red, Y = green, Z = blue.
    """
    if len(xs) == 0:
        return
    # Place origin at the bottom-left of the bounding box
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    zmin, zmax = float(zs.min()), float(zs.max())
    span = max(xmax - xmin, ymax - ymin, zmax - zmin, 1.0)
    arrow_len = span * length_frac

    # Place the arrows origin offset from the molecule (bottom-left-back corner)
    ox = xmin - span * 0.25
    oy = ymin - span * 0.25
    oz = zmin - span * 0.25

    arrow_cfg = dict(arrow_length_ratio=0.2, linewidth=2.0)
    ax.quiver(ox, oy, oz, arrow_len, 0, 0, color="#FF4444", **arrow_cfg)  # X red
    ax.quiver(ox, oy, oz, 0, arrow_len, 0, color="#44FF44", **arrow_cfg)  # Y green
    ax.quiver(ox, oy, oz, 0, 0, arrow_len, color="#4488FF", **arrow_cfg)  # Z blue

    lbl_off = arrow_len * 1.15
    ax.text(ox + lbl_off, oy, oz, "X", color="#FF4444", fontsize=9, fontweight="bold",
            ha="center", va="center")
    ax.text(ox, oy + lbl_off, oz, "Y", color="#44FF44", fontsize=9, fontweight="bold",
            ha="center", va="center")
    ax.text(ox, oy, oz + lbl_off, "Z", color="#4488FF", fontsize=9, fontweight="bold",
            ha="center", va="center")


_ISO_QUALITY_STRIDE = {
    "Preview": 3,
    "Balanced": 2,
    "High": 1,
}


def _blend_rgb(col_a, col_b, frac: float):
    """Blend two colours in RGB space."""
    a = np.array(mcolors.to_rgb(col_a), dtype=float)
    b = np.array(mcolors.to_rgb(col_b), dtype=float)
    frac = max(0.0, min(1.0, float(frac)))
    return tuple((1.0 - frac) * a + frac * b)


def _surface_edge_color(surface_color, bg_color, on_dark: bool) -> tuple:
    """Choose a subtle edge colour inspired by Avogadro's mesh/surface contrast."""
    blend_target = "#ffffff" if on_dark else "#000000"
    return _blend_rgb(surface_color, blend_target, 0.45 if on_dark else 0.35)


def _extract_isosurface_mesh(cube: dict, isovalue: float, stride: int = 1) -> Optional[dict]:
    """Return cached-ready vertices/faces for an isosurface level."""
    if not _HAS_SKIMAGE:
        return None

    data = cube["data"]
    if stride > 1:
        data = data[::stride, ::stride, ::stride]
        axes = np.asarray(cube["axes"]) * stride
    else:
        axes = np.asarray(cube["axes"])

    if data.size == 0:
        return None
    if isovalue >= 0 and float(data.max()) < isovalue:
        return None
    if isovalue < 0 and float(data.min()) > isovalue:
        return None

    try:
        verts, faces, _, _ = _mc(data, level=isovalue, allow_degenerate=False)
    except TypeError:
        try:
            verts, faces, _, _ = _mc(data, level=isovalue)
        except Exception:
            return None
    except Exception:
        return None

    real = np.asarray(cube["origin"]) + verts @ axes
    return {"verts": real, "faces": faces}


def _draw_isosurface_mesh(ax, mesh: dict, color, alpha: float,
                          style: str = "Glass",
                          edge_color=None,
                          edge_width: float = 0.0) -> bool:
    """Draw an already extracted isosurface mesh."""
    verts = mesh.get("verts")
    faces = mesh.get("faces")
    if verts is None or faces is None or len(faces) == 0:
        return False

    tris = verts[faces]
    coll = Poly3DCollection(tris, linewidths=edge_width)

    face_alpha = alpha
    st = (style or "Glass").lower()
    if st == "solid":
        face_alpha = max(alpha, 0.72)
    elif st == "mesh":
        face_alpha = min(alpha, 0.18)

    coll.set_facecolor(mcolors.to_rgba(color, face_alpha))
    coll.set_edgecolor(edge_color if edge_color is not None else "none")
    coll.set_alpha(face_alpha)
    coll.set_antialiaseds(True)
    ax.add_collection3d(coll)
    return True


def _render_iso(ax, cube: dict, isovalue: float, color, alpha: float) -> bool:
    """Render one isosurface lobe; returns True on success."""
    if not _HAS_SKIMAGE:
        return False
    data = cube["data"]
    if isovalue >= 0 and data.max() < isovalue:
        return False
    if isovalue < 0 and data.min() > isovalue:
        return False
    try:
        verts, faces, _, _ = _mc(data, level=isovalue, allow_degenerate=False)
    except Exception:
        return False
    # voxel → Cartesian: coords = origin + verts @ axes
    real = cube["origin"] + verts @ cube["axes"]
    ax.plot_trisurf(real[:, 0], real[:, 1], real[:, 2],
                    triangles=faces, color=color, alpha=alpha,
                    linewidth=0, antialiased=False)
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  cclib MO → cube grid
# ─────────────────────────────────────────────────────────────────────────────

def _cclib_build_cube(out_path: str, spin: int, iorb: int,
                      spacing: float = 0.35, pad: float = 4.5) -> dict:
    """
    Use cclib + scikit-image to evaluate canonical MO *iorb* on a grid.
    Returns a cube-format dict compatible with _render_iso().

    Raises RuntimeError with a human-readable message on any failure.
    """
    if not _HAS_CCLIB:
        raise RuntimeError("cclib not installed.  Run:  pip install cclib")
    if not _HAS_SKIMAGE:
        raise RuntimeError("scikit-image not installed.  Run:  pip install scikit-image")

    import cclib                          # noqa: PLC0415  (already checked above)
    from cclib.method.volume import Volume

    ccdata = cclib.io.ccread(out_path)
    if ccdata is None:
        raise RuntimeError(f"cclib could not parse: {out_path}")
    if not hasattr(ccdata, "mocoeffs") or not hasattr(ccdata, "gbasis"):
        raise RuntimeError(
            "cclib parsed the file but found no MO coefficients or basis set.\n"
            "Make sure your ORCA input prints the basis set and MOs."
        )

    nspin = len(ccdata.mocoeffs)
    if spin >= nspin:
        raise RuntimeError(f"Spin index {spin} out of range (file has {nspin} spin sets).")
    nmo = len(ccdata.mocoeffs[spin])
    if iorb < 0 or iorb >= nmo:
        raise RuntimeError(f"Orbital index {iorb} out of range [0, {nmo - 1}].")

    coords = ccdata.atomcoords[-1]          # (natoms, 3) Angstroms
    origin    = coords.min(axis=0) - pad
    topcorner = coords.max(axis=0) + pad

    vol = Volume(origin, topcorner, spacing)

    # cclib Volume API has varied across versions — try each variant
    success = False
    for attempt in (
        lambda: vol.fill_with_mo(ccdata, iorb, spin),
        lambda: vol.fill_with_mo(ccdata, iorb),
        lambda: vol.fill(ccdata, method=1),          # electron density fallback
    ):
        try:
            attempt()
            success = True
            break
        except Exception:
            continue

    if not success:
        raise RuntimeError(
            "cclib.method.volume.Volume could not evaluate the orbital.\n"
            "Your cclib version may not support direct MO evaluation.\n"
            "Generate a cube file with orca_plot instead."
        )

    data3d = np.array(vol.data)

    # Build atom list in the same format as parse_cube_file
    atoms = []
    for i, coord in enumerate(coords):
        anum = int(ccdata.atomnos[i])
        atoms.append((anum, float(anum), float(coord[0]),
                      float(coord[1]), float(coord[2])))

    try:
        ene   = ccdata.moenergies[spin][iorb]
        label = (f"MO {iorb} ({'alpha' if spin == 0 else 'beta'})  "
                 f"E = {ene:.4f} eV  |  {Path(out_path).stem}")
    except Exception:
        label = f"MO {iorb} ({'alpha' if spin == 0 else 'beta'})  |  {Path(out_path).stem}"

    n = np.array(data3d.shape, dtype=int)
    return {
        "comment": label,
        "atoms":   atoms,
        "origin":  origin,
        "axes":    np.diag([spacing, spacing, spacing]),
        "n":       n,
        "data":    data3d,
        "source":  "cclib",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Scrollable frame helper (for small-screen resilience)
# ─────────────────────────────────────────────────────────────────────────────

class _ScrollableFrame(tk.Frame):
    """
    A tk.Frame whose content can scroll vertically (and optionally
    horizontally).  Place child widgets inside ``self.interior``.

    Usage::

        sf = _ScrollableFrame(parent)
        sf.pack(fill=tk.BOTH, expand=True)
        tk.Label(sf.interior, text="hello").pack()
    """

    def __init__(self, parent, hscroll: bool = False, **kw):
        super().__init__(parent, **kw)
        self._canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self._vsb = ttk.Scrollbar(self, orient=tk.VERTICAL,
                                   command=self._canvas.yview)
        self._canvas.config(yscrollcommand=self._vsb.set)

        if hscroll:
            self._hsb = ttk.Scrollbar(self, orient=tk.HORIZONTAL,
                                       command=self._canvas.xview)
            self._canvas.config(xscrollcommand=self._hsb.set)
            self._hsb.pack(side=tk.BOTTOM, fill=tk.X)
        else:
            self._hsb = None

        self._vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.interior = tk.Frame(self._canvas)
        self._win_id = self._canvas.create_window(
            (0, 0), window=self.interior, anchor="nw")

        # Keep the interior width in sync with the canvas width so that
        # pack(fill=tk.X) inside the interior works properly.
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self.interior.bind("<Configure>", self._on_interior_configure)

        # Mousewheel scrolling (Windows + macOS + Linux)
        self._canvas.bind("<Enter>", self._bind_wheel)
        self._canvas.bind("<Leave>", self._unbind_wheel)

    def _on_canvas_configure(self, event):
        self._canvas.itemconfigure(self._win_id, width=event.width)

    def _on_interior_configure(self, event):
        self._canvas.config(scrollregion=self._canvas.bbox("all"))

    def _bind_wheel(self, _event):
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self._canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self._canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_wheel(self, _event):
        self._canvas.unbind_all("<MouseWheel>")
        self._canvas.unbind_all("<Button-4>")
        self._canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self._canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self._canvas.yview_scroll(1, "units")


def _clamp_geometry(win, default_w, default_h, min_w=400, min_h=300):
    """
    Set a window's geometry to *default_w × default_h* but never bigger than
    the user's screen, and enforce a minsize.
    """
    try:
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
    except Exception:
        sw, sh = 1920, 1080
    w = min(default_w, sw - 40)
    h = min(default_h, sh - 80)
    win.geometry(f"{w}x{h}")
    win.minsize(min(min_w, w), min(min_h, h))


# ─────────────────────────────────────────────────────────────────────────────
#  Application
# ─────────────────────────────────────────────────────────────────────────────

class NBOViewerApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("NBO Analysis Viewer v2")
        _clamp_geometry(self, 1350, 860, 800, 500)

        # State
        self._files:  Dict[str, Path] = {}       # display_name → .out path
        self._cache:  Dict[str, dict] = {}        # display_name → parsed data
        self._current: Optional[str]  = None
        self._highlighted: Optional[int] = None   # 1-based Atom# to highlight
        self._cubes: Dict[str, dict]  = {}        # label → cube dict (files + cclib)
        self._cclib_spin = tk.IntVar(value=0)
        self._cclib_iorb = tk.IntVar(value=0)

        # View options
        self._color_by     = tk.StringVar(value="Element")
        self._style_3d     = tk.StringVar(value="Ball & Stick")
        self._show_H       = tk.BooleanVar(value=False)
        self._show_labels  = tk.BooleanVar(value=False)
        self._show_axes    = tk.BooleanVar(value=True)   # Avogadro-style XYZ arrows
        self._clean_bg     = tk.BooleanVar(value=True)   # clean bg (no grid/panes)
        self._iso_quality  = tk.StringVar(value="Balanced")
        self._iso_surface_style = tk.StringVar(value="Glass")
        self._iso_surface_edges = tk.BooleanVar(value=False)
        self._iso_mol_alpha = tk.DoubleVar(value=0.92)
        # Re-orientation: rotation matrix (3x3) and translation (3,)
        self._reorient_R: Optional[np.ndarray] = None     # 3x3 rotation
        self._reorient_T: Optional[np.ndarray] = None     # 3-vector translation
        # Per-atom H visibility overrides: set of 0-based atom indices to force-show
        self._force_show_H: set = set()
        # Runtime element colour overrides (persists during session)
        self._elem_color_overrides: Dict[str, str] = {}
        self._iso_pos_on   = tk.BooleanVar(value=True)
        self._iso_neg_on   = tk.BooleanVar(value=True)
        self._iso_mol_on   = tk.BooleanVar(value=True)
        self._iso_mesh_cache: Dict[Tuple[int, float, int], Optional[dict]] = {}
        self._iso_mesh_cache_order: List[Tuple[int, float, int]] = []
        self._iso_mesh_cache_token_counter = 0
        self._last_mo_grid_key = None
        self._last_mo_grid_cube: Optional[dict] = None
        self._pending_mo_diagram_state: Optional[dict] = None

        # Orbital decomp state
        self._loewdin_data: dict = {}   # mo_idx (0-based) -> {label, atoms}
        self._decomp_rows:  list = []   # list of (name_var, atoms_var, row_frame)
        self._decomp_pending_after: Optional[str] = None

        self._build_menu()
        self._build_layout()
        self._auto_load_dir(_DEFAULT_DIR)

    # ─── Menu ────────────────────────────────────────────────────────────────

    def _build_menu(self):
        mb = tk.Menu(self)

        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label="Open File(s)…",       accelerator="Ctrl+O", command=self._open_files)
        fm.add_command(label="Load from Directory…", command=self._open_dir)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.destroy)
        mb.add_cascade(label="File", menu=fm)

        vm = tk.Menu(mb, tearoff=0)
        for val in ("Element", "Charge"):
            vm.add_radiobutton(label=f"Color by {val}", variable=self._color_by,
                               value=val, command=self._redraw_3d)
        vm.add_separator()
        for val in ("Ball & Stick", "Stick", "Spacefill"):
            vm.add_radiobutton(label=val, variable=self._style_3d,
                               value=val, command=self._redraw_3d)
        vm.add_separator()
        vm.add_checkbutton(label="Show H atoms",    variable=self._show_H,      command=self._redraw_3d)
        vm.add_checkbutton(label="Show atom labels",variable=self._show_labels,  command=self._redraw_3d)
        vm.add_checkbutton(label="Show XYZ axes",  variable=self._show_axes,    command=self._redraw_3d)
        vm.add_checkbutton(label="Clean background",variable=self._clean_bg,    command=self._redraw_3d)
        vm.add_separator()
        vm.add_command(label="Manage individual H atoms...", command=self._open_h_manager)
        vm.add_command(label="Element Colours...",           command=self._open_color_editor)
        vm.add_command(label="Re-orient molecule...",        command=self._open_reorient_tool)
        mb.add_cascade(label="View", menu=vm)

        self.config(menu=mb)
        self.bind_all("<Control-o>", lambda _: self._open_files())

    # ─── Layout ──────────────────────────────────────────────────────────────

    def _build_layout(self):
        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=5, sashrelief=tk.RAISED)
        pane.pack(fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(pane, width=210, bd=1, relief=tk.SUNKEN)
        pane.add(sidebar, minsize=170)
        self._build_sidebar(sidebar)

        right = tk.Frame(pane)
        pane.add(right, minsize=850)
        self._nb = ttk.Notebook(right)
        self._nb.pack(fill=tk.BOTH, expand=True)

        for title, builder in [
            ("🔬 3D Structure",   self._build_tab_3d),
            ("🧲 NBO Orbitals",   self._build_tab_nbo),
            ("🔄 Interactions",   self._build_tab_e2),
            ("🌐 Isosurface",     self._build_tab_iso),
            ("🔬 Orbital Decomp", self._build_tab_decomp),
        ]:
            frame = tk.Frame(self._nb)
            self._nb.add(frame, text=title)
            builder(frame)

        self._status_var = tk.StringVar(value="Open an NBO .out file to begin  (File → Open, or Ctrl+O).")
        tk.Label(self, textvariable=self._status_var, bd=1, relief=tk.SUNKEN,
                 anchor="w", padx=6, font=("", 8)).pack(side=tk.BOTTOM, fill=tk.X)

    def _build_sidebar(self, p):
        # File controls
        btn_frame = tk.Frame(p)
        btn_frame.pack(fill=tk.X, padx=4, pady=4)
        tk.Button(btn_frame, text="Open File(s)…", bg="#1a3c6e", fg="white",
                  activebackground="#2a5ca0", command=self._open_files).pack(fill=tk.X, pady=1)
        tk.Button(btn_frame, text="From Directory…", command=self._open_dir).pack(fill=tk.X, pady=1)
        tk.Button(btn_frame, text="Remove Selected", command=self._remove_selected).pack(fill=tk.X, pady=1)

        tk.Label(p, text="Loaded Files", font=("", 9, "bold")).pack(anchor="w", padx=6, pady=(4, 1))
        lf = tk.Frame(p)
        lf.pack(fill=tk.BOTH, expand=True, padx=4)
        sb = ttk.Scrollbar(lf)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._lb = tk.Listbox(lf, yscrollcommand=sb.set, selectmode=tk.SINGLE,
                               exportselection=False, activestyle="dotbox")
        self._lb.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self._lb.yview)
        self._lb.bind("<<ListboxSelect>>", self._on_select)

        ttk.Separator(p, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        tk.Label(p, text="Color", font=("", 8, "bold")).pack(anchor="w", padx=6)
        for v in ("Element", "Charge"):
            ttk.Radiobutton(p, text=v, variable=self._color_by, value=v,
                            command=self._redraw_3d).pack(anchor="w", padx=16)

        ttk.Separator(p, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        tk.Label(p, text="Style", font=("", 8, "bold")).pack(anchor="w", padx=6)
        for v in ("Ball & Stick", "Stick", "Spacefill"):
            ttk.Radiobutton(p, text=v, variable=self._style_3d, value=v,
                            command=self._redraw_3d).pack(anchor="w", padx=16)

        ttk.Separator(p, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        ttk.Checkbutton(p, text="Show H atoms",     variable=self._show_H,     command=self._redraw_3d).pack(anchor="w", padx=6)
        ttk.Checkbutton(p, text="Show atom labels", variable=self._show_labels, command=self._redraw_3d).pack(anchor="w", padx=6)
        ttk.Checkbutton(p, text="Show XYZ axes",   variable=self._show_axes,   command=self._redraw_3d).pack(anchor="w", padx=6)
        ttk.Checkbutton(p, text="Clean background", variable=self._clean_bg,   command=self._redraw_3d).pack(anchor="w", padx=6)
        ttk.Separator(p, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        tk.Button(p, text="Re-orient molecule…", font=("", 8),
                  command=self._open_reorient_tool).pack(anchor="w", padx=6, pady=2)

    # ─── Tab builders ────────────────────────────────────────────────────────

    def _build_tab_3d(self, p):
        self._fig_3d = Figure(figsize=(8, 6), dpi=96, facecolor="#1c1c1e")
        self._ax_3d  = self._fig_3d.add_subplot(111, projection="3d")
        _dark_3d_ax(self._ax_3d)
        self._ax_cbar = self._fig_3d.add_axes([0.02, 0.15, 0.025, 0.60])
        self._ax_cbar.set_visible(False)
        c3d = FigureCanvasTkAgg(self._fig_3d, master=p)
        c3d.draw()
        c3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(c3d, p)
        self._canvas_3d = c3d
        # Click-to-label state for 3D structure view
        self._3d_atom_annotations: Dict[int, object] = {}
        self._3d_atom_data: List[Tuple] = []  # (x, y, z, elem, serial)
        self._3d_pick_cid = self._fig_3d.canvas.mpl_connect(
            "pick_event", self._on_3d_atom_pick)

    def _build_tab_nbo(self, p):
        self._fig_nbo = Figure(figsize=(10, 4.2), dpi=96)
        self._ax_nbo1 = self._fig_nbo.add_subplot(121)
        self._ax_nbo2 = self._fig_nbo.add_subplot(122)
        cnbo = FigureCanvasTkAgg(self._fig_nbo, master=p)
        cnbo.draw()
        cnbo.get_tk_widget().pack(fill=tk.X)
        self._canvas_nbo = cnbo

        tk.Label(p, text="Natural Electron Configuration  (metals & donors)",
                 font=("", 9, "bold")).pack(anchor="w", padx=6, pady=(4, 0))
        nf = tk.Frame(p); nf.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        vsb = ttk.Scrollbar(nf, orient=tk.VERTICAL)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._nec_tree = ttk.Treeview(nf, columns=("Element", "Atom#", "Configuration"),
                                       show="headings", height=7, yscrollcommand=vsb.set)
        vsb.config(command=self._nec_tree.yview)
        self._nec_tree.heading("Element",       text="Element",  anchor="center")
        self._nec_tree.heading("Atom#",         text="Atom#",    anchor="center")
        self._nec_tree.heading("Configuration", text="Natural Electron Configuration")
        self._nec_tree.column("Element",       width=65,  anchor="center")
        self._nec_tree.column("Atom#",         width=55,  anchor="center")
        self._nec_tree.column("Configuration", width=700, anchor="w")
        self._nec_tree.pack(fill=tk.BOTH, expand=True)

    def _build_tab_e2(self, p):
        self._fig_e2 = Figure(figsize=(10, 5.5), dpi=96)
        self._ax_bd  = self._fig_e2.add_subplot(121)
        self._ax_ry  = self._fig_e2.add_subplot(122)
        ce2 = FigureCanvasTkAgg(self._fig_e2, master=p)
        ce2.draw()
        ce2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas_e2 = ce2
        self._e2_note = tk.Label(p, text="", fg="gray", font=("", 8))
        self._e2_note.pack(anchor="w", padx=6, pady=2)

    def _build_tab_iso(self, p):
        # ── source selector ───────────────────────────────────────────────
        top = tk.Frame(p); top.pack(fill=tk.X, padx=6, pady=4)
        tk.Label(top, text="Orbital source:", font=("", 9, "bold")).pack(side=tk.LEFT)
        self._iso_source = tk.StringVar(value="cube")
        ttk.Radiobutton(top, text="Cube files  (.cube from orca_plot)",
                        variable=self._iso_source, value="cube",
                        command=self._iso_switch_source).pack(side=tk.LEFT, padx=8)
        ttk.Radiobutton(top, text="MOs from .out  (LargePrint / Print[P_MOs])",
                        variable=self._iso_source, value="mo",
                        command=self._iso_switch_source).pack(side=tk.LEFT, padx=8)

        # ── middle: left list panel + right controls ───────────────────────
        mid = tk.Frame(p); mid.pack(fill=tk.X, padx=4, pady=2)

        left = tk.Frame(mid, bd=1, relief=tk.SUNKEN, width=310)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left.pack_propagate(False)

        self._iso_cube_pnl = tk.Frame(left)
        self._iso_cube_pnl.pack(fill=tk.BOTH, expand=True)
        self._build_iso_cube_subpanel(self._iso_cube_pnl)

        self._iso_mo_pnl = tk.Frame(left)           # hidden initially
        self._build_iso_mo_subpanel(self._iso_mo_pnl)

        right = tk.Frame(mid); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_iso_render_controls(right)

        # ── 3D canvas ─────────────────────────────────────────────────────
        self._fig_iso = Figure(figsize=(9, 5), dpi=96, facecolor="#1c1c1e")
        self._ax_iso  = self._fig_iso.add_subplot(111, projection="3d")
        _clean_3d_ax(self._ax_iso, "#1c1c1e")
        ciso = FigureCanvasTkAgg(self._fig_iso, master=p)
        ciso.draw()
        ciso.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(ciso, p)
        self._canvas_iso = ciso
        self._iso_pick_cid = None
        self._iso_atom_annotations = {}
        self._iso_atom_data = []

    def _build_iso_cube_subpanel(self, p):
        btn = tk.Frame(p); btn.pack(fill=tk.X, padx=4, pady=4)
        tk.Button(btn, text="Load Cube(s)...", bg="#1a3c6e", fg="white",
                  command=self._load_cube_files).pack(side=tk.LEFT, padx=2)
        tk.Button(btn, text="Remove", command=self._remove_cube).pack(side=tk.LEFT, padx=2)

        tk.Label(p, text="Loaded cubes:", font=("", 8, "bold")).pack(anchor="w", padx=4)
        lf = tk.Frame(p); lf.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        vsb = ttk.Scrollbar(lf); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._cube_lb = tk.Listbox(lf, yscrollcommand=vsb.set, height=7,
                                    selectmode=tk.SINGLE, exportselection=False)
        self._cube_lb.pack(fill=tk.BOTH, expand=True)
        vsb.config(command=self._cube_lb.yview)
        self._cube_lb.bind("<<ListboxSelect>>", self._on_cube_select)

        self._cube_info_lbl = tk.Label(p, text="", fg="#006600",
                                        font=("Courier", 7), justify=tk.LEFT, anchor="w")
        self._cube_info_lbl.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(p,
                 text=("Generate cube files on your cluster:\n"
                       "  orca_plot <name>.gbw -i\n"
                       "  -> pick orbital type (NBO/LP/BD*/MO)\n"
                       "  -> pick index, resolution, save .cube"),
                 fg="#777", font=("Courier", 7), justify=tk.LEFT).pack(
                 fill=tk.X, padx=4, pady=(6, 4))

    def _build_iso_mo_subpanel(self, p):
        # Info banner
        tk.Label(p,
                 text=("Reads MO coefficients + basis set directly from\n"
                       "the .out file.  Your ORCA job MUST include:\n"
                       "   ! LargePrint    or    %output Print[P_MOs] 1 end"),
                 bg="#e8f4fd", fg="#0c5460",
                 font=("", 8), justify=tk.LEFT, relief=tk.FLAT,
                 padx=6, pady=4).pack(fill=tk.X, padx=4, pady=(4, 6))

        btn = tk.Frame(p); btn.pack(fill=tk.X, padx=4, pady=2)
        tk.Button(btn, text="Try loading MOs from .out...", bg="#1a3c6e", fg="white",
                  command=self._load_mo_list).pack(side=tk.LEFT, padx=2)

        sf = tk.Frame(p); sf.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(sf, text="Spin:").pack(side=tk.LEFT)
        self._iso_mo_spin = tk.IntVar(value=0)
        ttk.Radiobutton(sf, text="Alpha", variable=self._iso_mo_spin,
                        value=0, command=self._repopulate_mo_tree).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(sf, text="Beta",  variable=self._iso_mo_spin,
                        value=1, command=self._repopulate_mo_tree).pack(side=tk.LEFT)

        tk.Label(p, text="Select MO to render:", font=("", 8, "bold")).pack(anchor="w", padx=4)
        tf = tk.Frame(p); tf.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        vsb = ttk.Scrollbar(tf); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        cols = ("#", "Label", "Energy (eV)", "Occ")
        self._mo_tree = ttk.Treeview(tf, columns=cols, show="headings",
                                      height=9, yscrollcommand=vsb.set,
                                      selectmode="browse")
        vsb.config(command=self._mo_tree.yview)
        for col, w in zip(cols, (42, 80, 95, 45)):
            self._mo_tree.heading(col, text=col, anchor="center")
            self._mo_tree.column(col, width=w, anchor="center", stretch=False)
        self._mo_tree.tag_configure("homo", background="#FFD700", foreground="#000")
        self._mo_tree.tag_configure("lumo", background="#90EE90", foreground="#000")
        self._mo_tree.tag_configure("occ",  background="#E8F0FF")
        self._mo_tree.tag_configure("virt", background="#F5F5F5")
        self._mo_tree.pack(fill=tk.BOTH, expand=True)
        self._mo_tree.bind("<<TreeviewSelect>>", self._on_mo_select)
        self._cclib_data  = None
        self._mo_homo_idx = 0

    def _build_iso_render_controls(self, p):
        _rc_sf = _ScrollableFrame(p)
        _rc_sf.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        gc = tk.LabelFrame(_rc_sf.interior, text="Render controls", padx=6, pady=4)
        gc.pack(fill=tk.X, padx=4, pady=4)

        r1 = tk.Frame(gc); r1.pack(fill=tk.X, pady=2)
        tk.Label(r1, text="Isovalue:").pack(side=tk.LEFT)
        self._iso_val_var = tk.DoubleVar(value=0.030)
        ttk.Spinbox(r1, textvariable=self._iso_val_var,
                    from_=0.0005, to=1.0, increment=0.005,
                    width=8, format="%.4f").pack(side=tk.LEFT, padx=4)
        # Live-update slider: drag to adjust, renders automatically on mouse release
        self._iso_val_slider = tk.Scale(
            r1, from_=0.0005, to=0.20, resolution=0.0001,
            orient=tk.HORIZONTAL, length=200,
            variable=self._iso_val_var, showvalue=False,
        )
        self._iso_val_slider.pack(side=tk.LEFT, padx=(0, 2))
        self._iso_val_slider.bind("<ButtonRelease-1>", self._rerender_last_iso)
        tk.Label(r1, text="(slide \u2192 auto-renders on release)",
                 font=("", 7), fg="gray").pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(r1, text="Opacity:").pack(side=tk.LEFT, padx=(0, 2))
        self._iso_alpha = tk.DoubleVar(value=0.38)
        self._iso_alpha_slider = ttk.Scale(r1, variable=self._iso_alpha, from_=0.05, to=0.90,
                                           orient=tk.HORIZONTAL, length=80)
        self._iso_alpha_slider.pack(side=tk.LEFT)
        self._iso_alpha_slider.bind("<ButtonRelease-1>", self._rerender_last_iso)

        r2 = tk.Frame(gc); r2.pack(fill=tk.X, pady=2)
        self._iso_pos_on = tk.BooleanVar(value=True)
        self._iso_neg_on = tk.BooleanVar(value=True)
        self._iso_mol_on = tk.BooleanVar(value=True)

        # + lobe checkbox + colour swatch
        ttk.Checkbutton(r2, text="+ lobe", variable=self._iso_pos_on,
                        command=self._rerender_last_iso).pack(side=tk.LEFT, padx=(4, 0))
        self._iso_pos_color = tk.StringVar(value="#4169E1")
        self._pos_swatch = tk.Button(r2, width=3, bg="#4169E1", relief=tk.RAISED,
                                      command=lambda: self._pick_lobe_color("+"))
        self._pos_swatch.pack(side=tk.LEFT, padx=(2, 8))

        # - lobe checkbox + colour swatch
        ttk.Checkbutton(r2, text="- lobe", variable=self._iso_neg_on,
                        command=self._rerender_last_iso).pack(side=tk.LEFT, padx=(4, 0))
        self._iso_neg_color = tk.StringVar(value="#DD2222")
        self._neg_swatch = tk.Button(r2, width=3, bg="#DD2222", relief=tk.RAISED,
                                      command=lambda: self._pick_lobe_color("-"))
        self._neg_swatch.pack(side=tk.LEFT, padx=(2, 8))

        ttk.Checkbutton(r2, text="Show molecule", variable=self._iso_mol_on,
                        command=self._rerender_last_iso).pack(side=tk.LEFT, padx=4)

        r3 = tk.Frame(gc); r3.pack(fill=tk.X, pady=2)
        tk.Label(r3, text="Grid spacing (MO only):").pack(side=tk.LEFT)
        self._iso_spacing = tk.DoubleVar(value=0.30)
        ttk.Spinbox(r3, textvariable=self._iso_spacing,
                    from_=0.10, to=0.80, increment=0.05,
                    width=6, format="%.2f").pack(side=tk.LEFT, padx=4)
        tk.Label(r3, text="Pad:").pack(side=tk.LEFT, padx=(8, 2))
        self._iso_pad = tk.DoubleVar(value=4.0)
        ttk.Spinbox(r3, textvariable=self._iso_pad,
                    from_=2.0, to=8.0, increment=0.5,
                    width=5, format="%.1f").pack(side=tk.LEFT)

        # ── Avogadro-style view options ───────────────────────────────────
        r_style = tk.Frame(gc); r_style.pack(fill=tk.X, pady=2)
        tk.Label(r_style, text="Surface:").pack(side=tk.LEFT)
        iso_style_cb = ttk.Combobox(
            r_style, textvariable=self._iso_surface_style,
            values=("Glass", "Solid", "Mesh"),
            width=8, state="readonly"
        )
        iso_style_cb.pack(side=tk.LEFT, padx=(4, 8))
        iso_style_cb.bind("<<ComboboxSelected>>", self._rerender_last_iso)

        ttk.Checkbutton(r_style, text="Surface edges",
                        variable=self._iso_surface_edges,
                        command=self._rerender_last_iso).pack(side=tk.LEFT)

        tk.Label(r_style, text="  Quality:").pack(side=tk.LEFT, padx=(8, 2))
        iso_quality_cb = ttk.Combobox(
            r_style, textvariable=self._iso_quality,
            values=("Preview", "Balanced", "High"),
            width=9, state="readonly"
        )
        iso_quality_cb.pack(side=tk.LEFT, padx=(2, 8))
        iso_quality_cb.bind("<<ComboboxSelected>>", self._rerender_last_iso)

        tk.Label(r_style, text="Atoms:").pack(side=tk.LEFT, padx=(4, 2))
        iso_atom_style_cb = ttk.Combobox(
            r_style, textvariable=self._style_3d,
            values=("Ball & Stick", "Stick", "Spacefill"),
            width=12, state="readonly"
        )
        iso_atom_style_cb.pack(side=tk.LEFT, padx=(2, 6))
        iso_atom_style_cb.bind("<<ComboboxSelected>>", self._rerender_last_iso)

        ttk.Checkbutton(r_style, text="XYZ axes", variable=self._show_axes,
                        command=self._rerender_last_iso).pack(side=tk.LEFT, padx=(4, 0))

        r_view = tk.Frame(gc); r_view.pack(fill=tk.X, pady=2)
        tk.Label(r_view, text="Background:").pack(side=tk.LEFT)
        self._iso_bg_color = tk.StringVar(value="#1c1c1e")
        self._bg_swatch = tk.Button(r_view, width=3, bg="#1c1c1e", relief=tk.RAISED,
                                     command=self._pick_bg_color)
        self._bg_swatch.pack(side=tk.LEFT, padx=(2, 4))
        # Preset background buttons
        for label, col in [("Black", "#000000"), ("Dark", "#1c1c1e"),
                           ("White", "#FFFFFF"), ("Grey", "#808080")]:
            tk.Button(r_view, text=label, width=5, font=("", 7),
                      command=lambda c=col: self._set_bg_color(c)).pack(side=tk.LEFT, padx=1)

        tk.Label(r_view, text="  Atom scale:").pack(side=tk.LEFT, padx=(8, 2))
        self._iso_atom_scale = tk.DoubleVar(value=1.0)
        ttk.Spinbox(r_view, textvariable=self._iso_atom_scale,
                    from_=0.3, to=3.0, increment=0.1,
                    width=4, format="%.1f").pack(side=tk.LEFT)

        tk.Label(r_view, text="  Mol opacity:").pack(side=tk.LEFT, padx=(8, 2))
        self._iso_mol_alpha_slider = ttk.Scale(r_view, variable=self._iso_mol_alpha,
                                               from_=0.15, to=1.0,
                                               orient=tk.HORIZONTAL, length=70)
        self._iso_mol_alpha_slider.pack(side=tk.LEFT)
        self._iso_mol_alpha_slider.bind("<ButtonRelease-1>", self._rerender_last_iso)

        tk.Label(r_view, text="  Bond width:").pack(side=tk.LEFT, padx=(8, 2))
        self._iso_bond_width = tk.DoubleVar(value=1.8)
        ttk.Spinbox(r_view, textvariable=self._iso_bond_width,
                    from_=0.5, to=5.0, increment=0.5,
                    width=4, format="%.1f").pack(side=tk.LEFT)

        r4 = tk.Frame(gc); r4.pack(fill=tk.X, pady=6)
        tk.Button(r4, text="  Render  ", bg="#2a4a00", fg="white",
                  activebackground="#3a6a00", font=("", 10, "bold"),
                  command=self._render_iso_tab).pack(side=tk.LEFT, padx=4)
        tk.Button(r4, text="Save as .cube...",
                  command=self._save_cube).pack(side=tk.LEFT, padx=4)
        tk.Button(r4, text="  Pop Out Window  ", bg="#4a0060", fg="white",
                  activebackground="#6a2080", font=("", 9, "bold"),
                  command=self._popout_render).pack(side=tk.LEFT, padx=8)

        self._iso_status = tk.Label(gc, text="", fg="gray", font=("", 8),
                                     anchor="w", justify=tk.LEFT)
        self._iso_status.pack(fill=tk.X)

        # ── MO Composition panel ──────────────────────────────────────────
        comp_frame = tk.LabelFrame(p, text="MO Composition (% character)", padx=6, pady=4)
        comp_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 2))

        # Top summary line
        self._comp_summary = tk.Label(comp_frame, text="Select and render an MO to see composition.",
                                       fg="gray", font=("", 8), anchor="w")
        self._comp_summary.pack(fill=tk.X)

        # ── Two-column layout: atom selector (left) + contribution table (right)
        comp_body = tk.Frame(comp_frame)
        comp_body.pack(fill=tk.BOTH, expand=True)

        # Left: Atom checklist
        atom_sel_frame = tk.LabelFrame(comp_body, text="Track atoms", padx=4, pady=2)
        atom_sel_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))

        atom_btn_row = tk.Frame(atom_sel_frame)
        atom_btn_row.pack(fill=tk.X)
        tk.Button(atom_btn_row, text="All", font=("", 7), width=4,
                  command=lambda: self._comp_select_all(True)).pack(side=tk.LEFT, padx=1)
        tk.Button(atom_btn_row, text="None", font=("", 7), width=4,
                  command=lambda: self._comp_select_all(False)).pack(side=tk.LEFT, padx=1)
        tk.Button(atom_btn_row, text="Metals", font=("", 7), width=5,
                  command=self._comp_select_metals).pack(side=tk.LEFT, padx=1)
        tk.Button(atom_btn_row, text="Donors", font=("", 7), width=5,
                  command=self._comp_select_donors).pack(side=tk.LEFT, padx=1)

        # Scrollable checkbox area
        atom_canvas = tk.Canvas(atom_sel_frame, width=150, height=140, highlightthickness=0)
        atom_sb = ttk.Scrollbar(atom_sel_frame, orient=tk.VERTICAL, command=atom_canvas.yview)
        self._comp_atom_inner = tk.Frame(atom_canvas)
        atom_canvas.create_window((0, 0), window=self._comp_atom_inner, anchor="nw")
        self._comp_atom_inner.bind("<Configure>",
            lambda e: atom_canvas.config(scrollregion=atom_canvas.bbox("all")))
        atom_canvas.config(yscrollcommand=atom_sb.set)
        atom_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        atom_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._comp_atom_vars: Dict[str, tk.BooleanVar] = {}  # atom_key → BooleanVar

        tk.Button(atom_sel_frame, text="Show tracked across MOs",
                  bg="#1a3c6e", fg="white", font=("", 8),
                  command=self._show_tracked_atom_table).pack(fill=tk.X, pady=(4, 0))
        tk.Button(atom_sel_frame, text="Atom AO Breakdown",
                  bg="#3c1a6e", fg="white", font=("", 8),
                  command=self._show_atom_ao_detail).pack(fill=tk.X, pady=(2, 0))
        tk.Button(atom_sel_frame, text="Summed AO Breakdown",
                  bg="#6e3c1a", fg="white", font=("", 8),
                  command=self._show_summed_atom_ao_detail).pack(fill=tk.X, pady=(2, 0))
        tk.Button(atom_sel_frame, text="MO Energy Diagram",
                  bg="#1a6e3c", fg="white", font=("", 8),
                  command=self._show_mo_diagram_picker).pack(fill=tk.X, pady=(2, 0))

        # Right: Composition treeview (current MO)
        right_comp = tk.Frame(comp_body)
        right_comp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        comp_cols = ("Atom", "Total %", "s %", "p %", "d %", "f %")
        tf = tk.Frame(right_comp)
        tf.pack(fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(tf, orient=tk.VERTICAL)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._comp_tree = ttk.Treeview(tf, columns=comp_cols, show="headings",
                                        height=8, yscrollcommand=vsb.set)
        vsb.config(command=self._comp_tree.yview)
        for col, w in zip(comp_cols, (80, 70, 60, 60, 60, 60)):
            self._comp_tree.heading(col, text=col, anchor="center")
            self._comp_tree.column(col, width=w, anchor="center", stretch=True)
        self._comp_tree.pack(fill=tk.BOTH, expand=True)
        self._comp_tree.tag_configure("major",  background="#E8FFE8")
        self._comp_tree.tag_configure("minor",  background="#FFFFFF")
        self._comp_tree.tag_configure("metal",  background="#D0FFD0", font=("", 9, "bold"))
        self._comp_tree.tag_configure("total",  background="#DDE8FF", font=("", 9, "bold"))

        # Top AOs detail line
        self._comp_ao_detail = tk.Label(comp_frame, text="", fg="#333", font=("Courier", 8),
                                         anchor="w", justify=tk.LEFT, wraplength=700)
        self._comp_ao_detail.pack(fill=tk.X, pady=(2, 0))

        # scikit-image status
        sk_ok = _HAS_SKIMAGE
        tk.Label(p,
                 text=f"scikit-image: {'ok' if sk_ok else 'MISSING  pip install scikit-image'}",
                 fg="#006600" if sk_ok else "#AA4444",
                 font=("Courier", 7)).pack(anchor="w", padx=4, pady=(8, 2))

    # ─── File loading ────────────────────────────────────────────────────────

    def _open_files(self):
        paths = filedialog.askopenfilenames(
            title="Open NBO .out File(s)",
            filetypes=[("ORCA Output", "*.out"), ("All files", "*.*")],
            initialdir=str(_DEFAULT_DIR) if _DEFAULT_DIR.exists() else ".",
        )
        for p in paths:
            self._add_file(Path(p))

    def _open_dir(self):
        d = filedialog.askdirectory(
            title="Load all .out files from directory",
            initialdir=str(_DEFAULT_DIR) if _DEFAULT_DIR.exists() else ".",
        )
        if d:
            self._auto_load_dir(Path(d))

    def _auto_load_dir(self, d: Path):
        if not d.exists():
            return
        for f in sorted(d.glob("*.out")):
            self._add_file(f, switch=False)
        if self._files and not self._current:
            self._lb.selection_set(0)
            self._on_select()

    def _add_file(self, path: Path, switch: bool = True):
        name = path.stem
        if name in self._files:
            return
        self._files[name] = path
        self._lb.insert(tk.END, name)
        if switch:
            idx = list(self._files.keys()).index(name)
            self._lb.selection_clear(0, tk.END)
            self._lb.selection_set(idx)
            self._on_select()

    def _remove_selected(self):
        sel = self._lb.curselection()
        if not sel:
            return
        name = self._lb.get(sel[0])
        self._lb.delete(sel[0])
        self._files.pop(name, None)
        self._cache.pop(name, None)
        if self._current == name:
            self._current = None

    # ─── Selection ───────────────────────────────────────────────────────────

    def _on_select(self, _ev=None):
        sel = self._lb.curselection()
        if not sel:
            return
        name = self._lb.get(sel[0])
        self._current = name
        self._highlighted = None
        self._status_var.set(f"Loading {name} …")
        self.update_idletasks()

        def _task():
            try:
                data = self._get_data(name)
                self.after(0, lambda: self._populate_all(name, data))
            except Exception as exc:
                self.after(0, lambda: (
                    messagebox.showerror("Error", f"Failed to load {name}:\n{exc}"),
                    self._status_var.set("Error during loading.")
                ))

        threading.Thread(target=_task, daemon=True).start()

    def _get_data(self, name: str) -> dict:
        if name not in self._cache:
            out_path = self._files[name]
            base     = out_path.parent / out_path.stem
            xyz = read_xyz(str(base.with_suffix(".xyz")))
            if not xyz:
                # Fallback: parse coordinates from the .out file itself
                try:
                    with open(str(out_path), "r", errors="ignore") as fh:
                        out_text = fh.read()
                    xyz = _parse_orca_coords(out_text)
                except Exception:
                    xyz = []
            self._cache[name] = {
                "out":     parse_out_file(str(out_path)),
                "xyz":     xyz,
                "summary": parse_ni_summary(str(base.parent / f"{base.name}_lastNBO_NiSummary.txt")),
            }
        return self._cache[name]

    # ─── Populate ────────────────────────────────────────────────────────────

    def _populate_all(self, name: str, data: dict):
        self._draw_3d(name, data)
        self._draw_nbo(name, data)
        self._draw_e2(name, data)
        if name in self._files:
            self._autoload_mos_for_path(str(self._files[name]))
        n_xyz = len(data["xyz"])
        has_s = bool(data["summary"]["lp"] or data["summary"]["bd"])
        self._status_var.set(
            f"{name}  |  {n_xyz} atoms"
            + ("  |  NiSummary ✓" if has_s else "")
        )

    # ─── 3D Drawing ──────────────────────────────────────────────────────────

    def _draw_3d(self, name: str, data: dict, highlighted: Optional[int] = None):
        atoms = data["xyz"]
        npa   = data["out"].get("npa", [])

        self._ax_3d.clear()
        self._ax_cbar.clear()
        if self._clean_bg.get():
            _clean_3d_ax(self._ax_3d, "#1c1c1e")
        else:
            _dark_3d_ax(self._ax_3d)
        self._ax_cbar.set_visible(False)

        if not atoms:
            self._ax_3d.set_title(f"{name}: no .xyz file", color="white")
            self._canvas_3d.draw()
            return

        charge_by_idx = {r["Atom#"]: r["Charge"] for r in npa}
        hi = highlighted if highlighted is not None else self._highlighted

        # Visible atom indices (0-based)
        # Show H if global toggle is on, OR if the atom is in the force-show set
        vis_idx = [i for i, (el, *_) in enumerate(atoms)
                   if self._show_H.get() or el != "H" or i in self._force_show_H]
        vis     = [atoms[i] for i in vis_idx]

        # Apply re-orientation transform if set
        if self._reorient_R is not None and self._reorient_T is not None:
            vis = _apply_reorient(vis, self._reorient_R, self._reorient_T)

        xs = np.array([a[1] for a in vis])
        ys = np.array([a[2] for a in vis])
        zs = np.array([a[3] for a in vis])

        ovr = self._elem_color_overrides
        mode = self._color_by.get()
        if mode == "Charge" and charge_by_idx:
            all_q = list(charge_by_idx.values())
            vmax  = max(abs(min(all_q)), abs(max(all_q)), 0.01)
            vmin  = -vmax
            colors = [_charge_rgba(charge_by_idx.get(vis_idx[i] + 1, 0.0), vmin, vmax)
                      for i in range(len(vis))]
        else:
            vmax  = None
            colors = [mcolors.to_rgba(_ec(a[0], ovr)) for a in vis]

        # Override highlighted atom color
        for i, orig_i in enumerate(vis_idx):
            if (orig_i + 1) == hi:
                colors[i] = (1.0, 0.85, 0.0, 1.0)  # gold

        style = self._style_3d.get()
        scale = 300 if style == "Spacefill" else (130 if style == "Ball & Stick" else 12)

        sizes = []
        for i, a in enumerate(vis):
            el = a[0]
            r  = ELEM_RADIUS.get(el, _DEFAULT_RADIUS)
            if style != "Stick":
                if el in _METALS:                       r *= 2.0
                elif el in ("P", "Cl", "Br", "I", "As"): r *= 1.5
                elif el == "S":                         r *= 1.3
            if (vis_idx[i] + 1) == hi:
                r *= 1.7
            sizes.append(r * scale)

        # ── Bonds drawn FIRST so atoms sit on top ────────────────────────────
        if style != "Spacefill":
            lw   = 2.0 if style == "Ball & Stick" else 1.2
            bclr = "#DDDDDD"          # light grey — clearly visible on dark panes
            for i, j in _detect_bonds(vis):
                self._ax_3d.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]],
                                 color=bclr, linewidth=lw, alpha=0.90, solid_capstyle="round")

        sc = self._ax_3d.scatter(xs, ys, zs, c=colors, s=sizes,
                                  depthshade=True, alpha=0.92, picker=True)
        sc.set_pickradius(8)

        # Store atom data for click-to-label (x, y, z, elem, serial)
        self._3d_atom_data = [(xs[i], ys[i], zs[i], vis[i][0], vis_idx[i] + 1)
                              for i in range(len(vis))]
        self._3d_atom_annotations = {}

        # Atom labels
        if self._show_labels.get():
            for i, a in enumerate(vis):
                serial = vis_idx[i] + 1
                el     = a[0]
                lbl    = f"{serial}"
                font_size = 6
                clr = "white"
                if (serial) == hi:
                    clr = "yellow"
                    font_size = 8
                self._ax_3d.text(a[1], a[2], a[3], lbl,
                                 fontsize=font_size, color=clr,
                                 ha="center", va="center", zorder=3)
        else:
            # Always label Ni and Fe
            for i, a in enumerate(vis):
                if a[0] in ("Ni", "Fe"):
                    self._ax_3d.text(a[1], a[2], a[3], a[0],
                                     fontsize=7, color="white",
                                     ha="center", va="center", zorder=3)

        # Colorbar
        if mode == "Charge" and vmax is not None:
            norm = mcolors.Normalize(-vmax, vmax)
            sm   = plt.cm.ScalarMappable(cmap=cm.RdBu_r, norm=norm)
            sm.set_array([])
            self._ax_cbar.set_visible(True)
            self._fig_3d.colorbar(sm, cax=self._ax_cbar, label="NPA charge (e)")
            self._ax_cbar.tick_params(labelsize=7, colors="white")
            self._ax_cbar.yaxis.label.set_color("white")

        if not self._clean_bg.get():
            for attr in ("xaxis", "yaxis", "zaxis"):
                ax = getattr(self._ax_3d, attr)
                ax.label.set_color("gray")
                ax.set_tick_params(labelsize=7, labelcolor="gray")
            self._ax_3d.set_xlabel("X (Å)", fontsize=8, color="gray")
            self._ax_3d.set_ylabel("Y (Å)", fontsize=8, color="gray")
            self._ax_3d.set_zlabel("Z (Å)", fontsize=8, color="gray")

        hi_txt = f"  |  atom {hi} highlighted" if hi else ""
        self._ax_3d.set_title(f"{name}{hi_txt}", color="white", fontsize=10)

        # ── Avogadro-style XYZ axis indicator (small arrows in corner) ───
        if self._show_axes.get():
            _draw_axis_indicator(self._ax_3d, xs, ys, zs)

        self._canvas_3d.draw()

    def _redraw_3d(self):
        if self._current and self._current in self._cache:
            self._draw_3d(self._current, self._cache[self._current])

    def _on_3d_atom_pick(self, event):
        """Toggle element label when an atom is clicked in 3D structure view."""
        if not hasattr(self, "_3d_atom_data") or not self._3d_atom_data:
            return
        ind = event.ind
        if ind is None or len(ind) == 0:
            return
        idx = int(ind[0])
        if idx >= len(self._3d_atom_data):
            return
        x, y, z, elem, serial = self._3d_atom_data[idx]

        if not hasattr(self, "_3d_atom_annotations"):
            self._3d_atom_annotations = {}

        if idx in self._3d_atom_annotations:
            # Already labelled — remove (toggle off)
            ann = self._3d_atom_annotations.pop(idx)
            ann.remove()
        else:
            # Add persistent label showing element + atom number
            ann = self._ax_3d.text(x, y, z, f"  {elem}{serial}",
                                   color="white", fontsize=9, fontweight="bold",
                                   ha="left", va="center",
                                   bbox=dict(boxstyle="round,pad=0.15",
                                             facecolor=_ec(elem, self._elem_color_overrides),
                                             alpha=0.85,
                                             edgecolor="white", linewidth=0.5),
                                   zorder=100)
            self._3d_atom_annotations[idx] = ann

        self._canvas_3d.draw_idle()

    # ─── H-atom manager ─────────────────────────────────────────────────────

    def _open_h_manager(self):
        """Window to individually show/hide hydrogen atoms."""
        if not self._current or self._current not in self._cache:
            messagebox.showinfo("Info", "Load a file first.")
            return
        atoms = self._cache[self._current].get("xyz", [])
        if not atoms:
            messagebox.showinfo("Info", "No atom data available.")
            return
        win = tk.Toplevel(self)
        win.title("Manage Hydrogen Atoms")
        _clamp_geometry(win, 380, 500, 300, 300)

        tk.Label(win, text="Check individual H atoms to show (even when 'Show H' is off):",
                 font=("", 9), wraplength=360, justify=tk.LEFT).pack(padx=8, pady=(8, 4))

        btn_row = tk.Frame(win)
        btn_row.pack(fill=tk.X, padx=8, pady=2)

        h_vars: Dict[int, tk.BooleanVar] = {}

        def _select_all():
            for v in h_vars.values():
                v.set(True)
        def _select_none():
            for v in h_vars.values():
                v.set(False)

        tk.Button(btn_row, text="Show All H", command=_select_all, width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Hide All H", command=_select_none, width=10).pack(side=tk.LEFT, padx=2)

        # Scrollable checkbox list
        sf = tk.Frame(win)
        sf.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        canvas = tk.Canvas(sf, highlightthickness=0)
        sb = ttk.Scrollbar(sf, orient=tk.VERTICAL, command=canvas.yview)
        inner = tk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.config(scrollregion=canvas.bbox("all")))
        canvas.config(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        for i, (el, x, y, z) in enumerate(atoms):
            if el != "H":
                continue
            var = tk.BooleanVar(value=(i in self._force_show_H))
            h_vars[i] = var
            # Find what this H is bonded to
            bonded_to = ""
            for j, (el2, x2, y2, z2) in enumerate(atoms):
                if j == i or el2 == "H":
                    continue
                dist = ((x-x2)**2 + (y-y2)**2 + (z-z2)**2) ** 0.5
                if dist < 1.25:
                    bonded_to = f" (bonded to {el2}{j+1})"
                    break
            tk.Checkbutton(inner, text=f"H{i+1}{bonded_to}",
                           variable=var, anchor="w",
                           font=("", 9)).pack(fill=tk.X, padx=4)

        def _apply():
            self._force_show_H = {idx for idx, var in h_vars.items() if var.get()}
            self._redraw_3d()
            # Also re-render iso if we have a cube
            cube = getattr(self, "_last_rendered_cube", None)
            if cube is not None:
                self._do_render(cube)

        def _apply_close():
            _apply()
            win.destroy()

        bottom = tk.Frame(win)
        bottom.pack(fill=tk.X, padx=8, pady=8)
        tk.Button(bottom, text="Apply", bg="#1a3c6e", fg="white",
                  command=_apply, width=10).pack(side=tk.LEFT, padx=4)
        tk.Button(bottom, text="Apply & Close", bg="#2a4a00", fg="white",
                  command=_apply_close, width=12).pack(side=tk.LEFT, padx=4)
        tk.Button(bottom, text="Cancel", command=win.destroy, width=8).pack(side=tk.LEFT, padx=4)

    # ─── Element colour editor ───────────────────────────────────────────────

    def _open_color_editor(self):
        """Open a window with a periodic-table-style grid to change element colours."""
        from tkinter import colorchooser

        win = tk.Toplevel(self)
        win.title("Element Colours")
        _clamp_geometry(win, 720, 480, 400, 300)

        tk.Label(win, text="Click any element to change its colour. Changes apply immediately.",
                 font=("", 10), fg="#333").pack(padx=8, pady=(8, 4))

        # Mini periodic table layout (rows of elements)
        _PT_ROWS = [
            ["H",  "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "", "He"],
            ["Li", "Be", "",   "",   "",   "",   "",   "",   "",   "",   "",   "", "B",  "C",  "N",  "O",  "F",  "Ne"],
            ["Na", "Mg", "",   "",   "",   "",   "",   "",   "",   "",   "",   "", "Al", "Si", "P",  "S",  "Cl", "Ar"],
            ["K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "",   "",  "As", "Se", "Br", ""],
            ["Rb", "Sr", "",   "",   "",   "Mo", "",   "Ru", "Rh", "Pd", "",   "",   "",   "Sb",  "",  "",   "I",  ""],
            ["Cs", "Ba", "",   "",   "",   "",   "",   "Os", "Ir", "Pt", "Au", "",   "",   "",   "",   "",   "",   ""],
        ]

        grid_frame = tk.Frame(win)
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self._color_swatches: Dict[str, tk.Button] = {}

        for r, row in enumerate(_PT_ROWS):
            for c, elem in enumerate(row):
                if not elem:
                    tk.Label(grid_frame, text="", width=4).grid(row=r, column=c, padx=1, pady=1)
                    continue
                cur_col = _ec(elem, self._elem_color_overrides)
                # Determine text color for contrast
                try:
                    rgb = tuple(int(cur_col.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                    fg = "white" if sum(rgb) < 384 else "black"
                except Exception:
                    fg = "black"

                btn = tk.Button(grid_frame, text=elem, width=4, font=("", 9, "bold"),
                                bg=cur_col, fg=fg, relief=tk.RAISED,
                                command=lambda e=elem: self._change_element_color(e))
                btn.grid(row=r, column=c, padx=1, pady=1, sticky="nsew")
                self._color_swatches[elem] = btn

        # Reset button
        bottom = tk.Frame(win)
        bottom.pack(fill=tk.X, padx=8, pady=8)
        tk.Button(bottom, text="Reset All to Defaults", bg="#aa2222", fg="white",
                  command=lambda: self._reset_all_colors(win)).pack(side=tk.LEFT, padx=4)
        tk.Label(bottom, text="Colours persist during this session",
                 fg="gray", font=("", 8)).pack(side=tk.LEFT, padx=12)

    def _change_element_color(self, elem: str):
        """Change the colour of a single element."""
        from tkinter import colorchooser
        cur = _ec(elem, self._elem_color_overrides)
        result = colorchooser.askcolor(color=cur, title=f"Choose colour for {elem}")
        if result and result[1]:
            new_col = result[1]
            self._elem_color_overrides[elem] = new_col
            # Update swatch button
            if elem in self._color_swatches:
                try:
                    rgb = tuple(int(new_col.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                    fg = "white" if sum(rgb) < 384 else "black"
                except Exception:
                    fg = "black"
                self._color_swatches[elem].config(bg=new_col, fg=fg)
            # Redraw
            self._redraw_3d()
            # Re-render iso if available
            cube = getattr(self, "_last_rendered_cube", None)
            if cube is not None:
                self._do_render(cube)

    def _reset_all_colors(self, win=None):
        """Reset all element colours to defaults."""
        self._elem_color_overrides.clear()
        # Update all swatches
        for elem, btn in self._color_swatches.items():
            col = ELEM_COLOR.get(elem, _DEFAULT_COLOR)
            try:
                rgb = tuple(int(col.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                fg = "white" if sum(rgb) < 384 else "black"
            except Exception:
                fg = "black"
            btn.config(bg=col, fg=fg)
        self._redraw_3d()
        cube = getattr(self, "_last_rendered_cube", None)
        if cube is not None:
            self._do_render(cube)

    # ─── NBO ─────────────────────────────────────────────────────────────────

    def _draw_nbo(self, name: str, data: dict):
        lp  = data["summary"]["lp"]
        nec = data["out"].get("nec", [])
        self._ax_nbo1.clear(); self._ax_nbo2.clear()

        if lp:
            labels = [f"LP({r['LP#']})" for r in lp]
            x = range(len(labels))
            occs = [r["Occ"] for r in lp]
            s_p  = [r["s%"] for r in lp]
            p_p  = [r["p%"] for r in lp]
            d_p  = [r["d%"] for r in lp]
            self._ax_nbo1.bar(x, occs, color="#4169E1", edgecolor="black", linewidth=0.6)
            self._ax_nbo1.set_xticks(list(x)); self._ax_nbo1.set_xticklabels(labels)
            self._ax_nbo1.set_ylabel("Occupancy (e)"); self._ax_nbo1.set_title("Ni LP d-Orbital Occupancies")
            self._ax_nbo1.set_ylim(max(1.90, min(occs) - 0.05), 2.01)
            self._ax_nbo1.grid(axis="y", linestyle="--", alpha=0.3)
            for xi, o in zip(x, occs):
                self._ax_nbo1.text(xi, o + 0.0003, f"{o:.4f}", ha="center", va="bottom", fontsize=7)
            self._ax_nbo2.bar(x, s_p, label="s%", color="#FF6B6B")
            self._ax_nbo2.bar(x, p_p, bottom=s_p, label="p%", color="#4ECDC4")
            self._ax_nbo2.bar(x, d_p, bottom=[s+p for s,p in zip(s_p, p_p)], label="d%", color="#45B7D1")
            self._ax_nbo2.set_xticks(list(x)); self._ax_nbo2.set_xticklabels(labels)
            self._ax_nbo2.set_ylabel("% Character"); self._ax_nbo2.set_title("LP Hybridisation Character")
            self._ax_nbo2.set_ylim(0, 105); self._ax_nbo2.legend(fontsize=8)
            self._ax_nbo2.grid(axis="y", linestyle="--", alpha=0.3)
        else:
            self._ax_nbo1.text(0.5, 0.5, f"No LP data\n({name}_lastNBO_NiSummary.txt not found)",
                               ha="center", va="center", transform=self._ax_nbo1.transAxes,
                               fontsize=8, color="gray", multialignment="center")
            self._ax_nbo2.axis("off")

        self._fig_nbo.tight_layout(); self._canvas_nbo.draw()

        for item in self._nec_tree.get_children():
            self._nec_tree.delete(item)
        for row in nec:
            if row["Element"] in KEY_METALS:
                self._nec_tree.insert("", tk.END, values=(row["Element"], row["Atom#"], row["Config"]))

    # ─── Interactions ────────────────────────────────────────────────────────

    def _draw_e2(self, name: str, data: dict):
        bd = data["summary"]["bd"]; ry = data["summary"]["ry"]
        self._ax_bd.clear(); self._ax_ry.clear()

        def _hbar(ax, rows, title, cmap):
            if not rows: ax.axis("off"); return
            labels = [r["Label"] for r in rows]
            vals   = [r["E2sum"] for r in rows]
            nrm    = [0.35 + 0.65 * v / max(vals) for v in vals]
            ax.barh(range(len(labels)), vals, color=cmap(nrm))
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel("E₂ (kcal/mol)", fontsize=8); ax.set_title(title, fontsize=9)
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            for yi, v in enumerate(vals):
                ax.text(v + 0.02, yi, f"{v:.2f}", va="center", fontsize=6)

        if not bd and not ry:
            self._ax_bd.text(0.5, 0.5, f"Requires {name}_lastNBO_NiSummary.txt",
                             ha="center", va="center", transform=self._ax_bd.transAxes,
                             fontsize=8, color="gray"); self._ax_ry.axis("off")
            self._e2_note.config(text="")
        else:
            _hbar(self._ax_bd, bd,       "LP(Ni) → BD* Acceptors", plt.cm.Reds)
            _hbar(self._ax_ry, ry[:12],  "LP(Ni) → RY Acceptors (top 12)", plt.cm.Blues)
            self._e2_note.config(
                text=f"Total BD* E₂: {sum(r['E2sum'] for r in bd):.2f} kcal/mol   |   "
                     f"Total RY E₂: {sum(r['E2sum'] for r in ry):.2f} kcal/mol", fg="#333")

        self._fig_e2.tight_layout(); self._canvas_e2.draw()

    # ─── Isosurface: source switching ────────────────────────────────────────

    def _iso_switch_source(self):
        if self._iso_source.get() == "cube":
            self._iso_mo_pnl.pack_forget()
            self._iso_cube_pnl.pack(fill=tk.BOTH, expand=True)
        else:
            self._iso_cube_pnl.pack_forget()
            self._iso_mo_pnl.pack(fill=tk.BOTH, expand=True)

    # ─── Cube file management ─────────────────────────────────────────────────

    def _load_cube_files(self):
        paths = filedialog.askopenfilenames(
            title="Load Cube File(s)",
            filetypes=[("Cube files", "*.cube *.cub"), ("All files", "*.*")],
            initialdir=str(_DEFAULT_DIR) if _DEFAULT_DIR.exists() else ".",
        )
        for path in paths:
            self._add_cube_file(Path(path))

    def _add_cube_file(self, path: Path):
        key = path.name
        suffix = 1
        while key in self._cubes:
            key = f"{path.stem}_{suffix}{path.suffix}"; suffix += 1
        try:
            cube = parse_cube_file(str(path))
            self._cubes[key] = cube
            self._cube_lb.insert(tk.END, key)
            self._cube_lb.selection_clear(0, tk.END)
            self._cube_lb.selection_set(tk.END)
            self._on_cube_select()
        except Exception as exc:
            messagebox.showerror("Cube Error", str(exc))

    def _remove_cube(self):
        sel = self._cube_lb.curselection()
        if not sel: return
        key = self._cube_lb.get(sel[0])
        self._cube_lb.delete(sel[0])
        self._cubes.pop(key, None)
        self._cube_info_lbl.config(text="")

    def _on_cube_select(self, _ev=None):
        sel = self._cube_lb.curselection()
        if not sel: return
        key  = self._cube_lb.get(sel[0])
        cube = self._cubes.get(key)
        if not cube: return
        grid = "x".join(str(int(n)) for n in cube["n"])
        vmin = float(cube["data"].min())
        vmax = float(cube["data"].max())
        self._cube_info_lbl.config(
            text=f"Grid: {grid}  range: [{vmin:.4f}, {vmax:.4f}]"
        )
        suggested = round(max(abs(vmin), abs(vmax)) * 0.05, 4)
        self._iso_val_var.set(max(0.001, suggested))

    # ─── MO list (cclib) ─────────────────────────────────────────────────────

    def _autoload_mos_for_path(self, path: str):
        """Auto-load MO data from an ORCA .out file without opening dialogs."""
        if not path or not str(path).lower().endswith(".out"):
            return
        if getattr(self, "_orca_mo_path", None) == path and getattr(self, "_orca_mo_data", None) is not None:
            return
        if getattr(self, "_mo_autoload_in_progress", None) == path:
            return
        self._load_mo_list_from_path(path, show_help_dialog=False, status_prefix="Auto-loading MOs")

    def _load_mo_list_from_path(self, path: str, show_help_dialog: bool = True,
                                status_prefix: str = "Parsing") -> None:
        """Load MOs from an ORCA .out file at a known path."""
        fname = Path(path).name
        self._mo_autoload_in_progress = path
        self._iso_status.config(text=f"{status_prefix} {fname} ...", fg="gray")
        self.update_idletasks()

        def _task():
            try:
                self.after(0, lambda: self._iso_status.config(
                    text=f"Scanning {fname} for MO sections (streaming)...", fg="gray"))

                basis_text, mo_text = _extract_orca_sections(path)
                basis = _parse_orca_basis(basis_text) if basis_text else {}
                mo_data = _parse_orca_mos(mo_text) if mo_text else None

                if mo_data is not None and basis:
                    self.after(0, lambda: self._on_mo_loaded_direct(path, basis, mo_data))
                    return

                if show_help_dialog:
                    self.after(0, lambda: self._show_largeprint_help(path))
                else:
                    self.after(0, lambda: self._clear_loaded_mos(
                        status_text=(f"{fname}: no LargePrint MO coefficients found.  "
                                     f"Rerun ORCA with ! LargePrint if you want MO rendering."),
                        status_color="#994400"))

            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                if show_help_dialog:
                    self.after(0, lambda: self._show_mo_load_error(msg))
                else:
                    self.after(0, lambda: self._clear_loaded_mos(
                        status_text=f"MO auto-load skipped for {fname}: {msg[:120]}",
                        status_color="#994400"))
            finally:
                self.after(0, lambda: setattr(self, "_mo_autoload_in_progress", None))

        threading.Thread(target=_task, daemon=True).start()

    def _load_mo_list(self):
        """Load MOs from an ORCA .out file using our custom LargePrint parser.

        Falls back to cclib if the LargePrint MO section is missing but
        cclib is available; shows a helpful error otherwise.
        """
        # Always show file dialog
        init_dir = str(self._files[self._current].parent) \
            if (self._current and self._current in self._files) \
            else (str(_DEFAULT_DIR) if _DEFAULT_DIR.exists() else ".")
        path = filedialog.askopenfilename(
            title="Select ORCA .out file (run with LargePrint for best results)",
            filetypes=[("ORCA Output", "*.out"), ("All files", "*.*")],
            initialdir=init_dir,
        )
        if not path:
            return
        self._load_mo_list_from_path(path, show_help_dialog=True, status_prefix="Parsing")
        return

        fname = Path(path).name
        self._iso_status.config(text=f"Parsing {fname} ...", fg="gray")
        self.update_idletasks()

        def _task():
            try:
                # Use streaming extractor for large files (avoids loading 400+ MB)
                self.after(0, lambda: self._iso_status.config(
                    text=f"Scanning {fname} for MO sections (streaming)...", fg="gray"))

                basis_text, mo_text = _extract_orca_sections(path)

                # 1. Try our custom LargePrint parser
                basis   = _parse_orca_basis(basis_text) if basis_text else {}
                mo_data = _parse_orca_mos(mo_text) if mo_text else None

                if mo_data is not None and basis:
                    # Success — we have both basis set and MO coefficients
                    self.after(0, lambda: self._on_mo_loaded_direct(path, basis, mo_data))
                    return

                # 2. MO section missing — file wasn't run with LargePrint
                self.after(0, lambda: self._show_largeprint_help(path))

            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                self.after(0, lambda: self._show_mo_load_error(msg))

        threading.Thread(target=_task, daemon=True).start()

    def _clear_loaded_mos(self, status_text: str = "No MO data loaded.",
                          status_color: str = "#994400") -> None:
        """Clear the currently displayed MO list and composition state."""
        self._cclib_data = None
        self._orca_mo_path = None
        self._orca_mo_basis = None
        self._orca_mo_data = None
        self._last_mo_grid_key = None
        self._last_mo_grid_cube = None
        for item in self._mo_tree.get_children():
            self._mo_tree.delete(item)
        self._mo_homo_idx = 0
        self._iso_status.config(text=status_text, fg=status_color)

    def _on_mo_loaded_direct(self, path: str, basis: dict, mo_data: dict):
        """Called when our custom parser successfully extracted MOs."""
        self._cclib_data  = None   # not using cclib
        self._orca_mo_path  = path
        self._orca_mo_basis = basis
        self._orca_mo_data  = mo_data
        self._last_mo_grid_key = None
        self._last_mo_grid_cube = None

        nmo = mo_data["coeffs"].shape[1]
        energies = mo_data["energies"]
        occs     = mo_data["occs"]

        # Find HOMO
        occ_mask = occs > 0.5
        homo_idx = int(np.max(np.where(occ_mask))) if np.any(occ_mask) else 0
        self._mo_homo_idx = homo_idx

        # Populate MO treeview
        for item in self._mo_tree.get_children():
            self._mo_tree.delete(item)

        for i in range(nmo):
            ene_str = f"{energies[i]:.5f}" if i < len(energies) else "N/A"
            occ_val = occs[i] if i < len(occs) else 0.0
            delta   = i - homo_idx
            if   delta == 0:  lbl, tag = "HOMO",           "homo"
            elif delta == 1:  lbl, tag = "LUMO",           "lumo"
            elif delta < 0:   lbl, tag = f"HOMO{delta}",   "occ"
            else:             lbl, tag = f"LUMO+{delta-1}", "virt"
            self._mo_tree.insert("", tk.END,
                                 values=(i, lbl, ene_str, f"{occ_val:.2f}"),
                                 iid=str(i), tags=(tag,))

        self._mo_tree.see(str(homo_idx))
        self._mo_tree.selection_set(str(homo_idx))

        nao = mo_data["coeffs"].shape[0]
        n_basis = sum(len(sh["prims"]) for shells in basis.values() for sh in shells)
        self._iso_status.config(
            text=(f"{nmo} MOs loaded  |  {nao} AOs  |  {n_basis} primitives  |  "
                  f"HOMO = MO {homo_idx}   ({energies[homo_idx]:.4f} Eh)  |  "
                  f"Select an MO and click Render."),
            fg="#006600"
        )

        # Also parse Löwdin orbital populations from the same file for decomp tab
        def _parse_pops():
            try:
                pops = parse_loewdin_mo_pops(path)
                self.after(0, lambda: self._on_loewdin_loaded(path, pops))
            except Exception:
                pass
        threading.Thread(target=_parse_pops, daemon=True).start()

    def _show_largeprint_help(self, path: str):
        """Show dialog when .out file does not contain the MO coefficients."""
        self._clear_loaded_mos(
            status_text="No MO section found - see dialog.",
            status_color="red",
        )
        self._iso_status.config(text="No MO section found — see dialog.", fg="red")

        win = tk.Toplevel(self)
        win.title("MO Coefficients Not Found")
        _clamp_geometry(win, 620, 400, 400, 300)

        tk.Label(win, text="MOLECULAR ORBITALS section not found",
                 font=("", 11, "bold"), fg="#cc0000").pack(pady=(14, 2))
        tk.Label(win, text=(
            f"The file  {Path(path).name}  does not contain printed MO coefficients.\n"
            "ORCA only prints them when you enable LargePrint."
        ), font=("", 9), justify=tk.LEFT, wraplength=560).pack(padx=12, pady=6)

        tk.Label(win, text="Add one of these to your ORCA input and re-run:",
                 font=("", 10, "bold")).pack(anchor="w", padx=12, pady=(6, 2))

        cmd = tk.Text(win, wrap=tk.NONE, width=70, height=12,
                      font=("Courier", 10), bg="#1c1c1e", fg="#00FF88",
                      relief=tk.FLAT)
        cmd.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        cmd.insert("1.0",
            "# Option 1 — add the LargePrint keyword:\n"
            "! B3LYP def2-TZVP LargePrint\n\n"
            "# Option 2 — print only MOs (smaller output):\n"
            "%output\n"
            "  Print[ P_MOs ] 1\n"
            "end\n\n"
            "# Then re-run ORCA, transfer the new .out to Windows,\n"
            "# and load it here.  The MO tree will populate."
        )
        cmd.config(state=tk.DISABLED)

        tk.Button(win, text="Close", command=win.destroy,
                  width=10).pack(pady=8)
        win.grab_set()
        win.focus_set()

    def _on_loewdin_loaded(self, path: str, pops: dict):
        """Called (on main thread) when Löwdin populations are parsed."""
        self._loewdin_data = pops
        n = len(pops)
        if hasattr(self, "_decomp_status"):
            if n:
                self._decomp_status.config(
                    text=f"Löwdin data: {n} MOs loaded from {Path(path).name}  — select an MO and click Analyse.",
                    fg="#005500")
            else:
                self._decomp_status.config(
                    text="No Löwdin/Mulliken population section found in file.  "
                         "Run ORCA with  ! LargePrint  to include it.",
                    fg="#994400")
        if hasattr(self, "_decomp_mo_lb") and n:
            self._populate_decomp_mo_list()

    def _populate_decomp_mo_list(self):
        """Fill the decomp-tab MO listbox from self._loewdin_data."""
        lb = self._decomp_mo_lb
        lb.delete(0, tk.END)
        # Also need energies/labels from the loaded MO data
        energies = []
        occs     = []
        homo_idx = 0
        if self._orca_mo_data is not None:
            energies = list(self._orca_mo_data.get("energies", []))
            occs     = list(self._orca_mo_data.get("occs", []))
            occ_mask = [o > 0.5 for o in occs]
            homo_idx = max((i for i, v in enumerate(occ_mask) if v), default=0)

        for mo_idx in sorted(self._loewdin_data.keys()):
            entry = self._loewdin_data[mo_idx]
            raw_lbl = entry.get("label", "")
            if not raw_lbl:
                # derive from occ
                delta = mo_idx - homo_idx
                if   delta == 0:  raw_lbl = "HOMO"
                elif delta == 1:  raw_lbl = "LUMO"
                elif delta < 0:   raw_lbl = f"HOMO{delta}"
                else:             raw_lbl = f"LUMO+{delta-1}"
            ene_s = f"{energies[mo_idx]:.4f} Eh" if mo_idx < len(energies) else ""
            lb.insert(tk.END, f"  {mo_idx:>4}   {raw_lbl:<10}  {ene_s}")

        # Scroll to HOMO
        homo_keys = sorted(self._loewdin_data.keys())
        if homo_idx in homo_keys:
            pos = homo_keys.index(homo_idx)
            lb.see(pos)
            lb.selection_clear(0, tk.END)
            lb.selection_set(pos)

    def _show_mo_load_error(self, detail: str):
        self._iso_status.config(text=f"Error: {detail[:80]}", fg="red")
        messagebox.showerror("MO Loading Error", detail)

    def _on_mo_loaded(self, ccdata):
        self._cclib_data = ccdata
        self._repopulate_mo_tree()
        nspin = len(ccdata.mocoeffs)
        nmo   = len(ccdata.mocoeffs[0])
        self._iso_status.config(
            text=f"{nmo} MOs loaded ({nspin} spin set{'s' if nspin > 1 else ''}).  "
                 "Select one and click Render.", fg="#006600"
        )

    def _repopulate_mo_tree(self):
        if self._cclib_data is None: return
        for item in self._mo_tree.get_children():
            self._mo_tree.delete(item)

        data  = self._cclib_data
        spin  = self._iso_mo_spin.get()
        if spin >= len(data.mocoeffs): return

        energies = data.moenergies[spin] if hasattr(data, "moenergies") else []
        homos    = data.homos if hasattr(data, "homos") else []
        homo_idx = int(homos[min(spin, len(homos)-1)]) if len(homos) > 0 else 0
        self._mo_homo_idx = homo_idx
        nmo = len(data.mocoeffs[spin])

        for i in range(nmo):
            ene_str = f"{energies[i]:.4f}" if i < len(energies) else "N/A"
            occ     = 2.0 if i <= homo_idx else 0.0
            delta   = i - homo_idx
            if   delta == 0:  lbl, tag = "HOMO",          "homo"
            elif delta == 1:  lbl, tag = "LUMO",          "lumo"
            elif delta < 0:   lbl, tag = f"HOMO{delta}",  "occ"
            else:             lbl, tag = f"LUMO+{delta-1}","virt"
            self._mo_tree.insert("", tk.END,
                                 values=(i, lbl, ene_str, f"{occ:.1f}"),
                                 iid=str(i), tags=(tag,))

        self._mo_tree.see(str(homo_idx))
        self._mo_tree.selection_set(str(homo_idx))

    def _on_mo_select(self, _ev=None):
        sel = self._mo_tree.selection()
        if not sel: return
        iorb = int(sel[0])
        vals = self._mo_tree.item(sel[0], "values")

        # Quick composition preview (instant)
        if hasattr(self, "_orca_mo_data") and self._orca_mo_data is not None:
            try:
                comp = compute_mo_composition(self._orca_mo_data, iorb,
                                              reorient_R=self._reorient_R)
                top3 = comp['top_atoms'][:3]
                top_str = ", ".join(f"{k}: {p:.1f}%" for k, p in top3) if top3 else ""
                char_str = ""
                if comp['total_d'] > 5:
                    char_str += f"d={comp['total_d']:.0f}% "
                if comp['total_p'] > 5:
                    char_str += f"p={comp['total_p']:.0f}% "
                if comp['total_s'] > 5:
                    char_str += f"s={comp['total_s']:.0f}% "
                self._iso_status.config(
                    text=f"MO {vals[0]} {vals[1]} {vals[2]} Eh  occ={vals[3]}  |  "
                         f"{char_str} |  {top_str}  |  Click Render",
                    fg="#333"
                )
                # Also update composition panel
                self._show_mo_composition(comp, iorb)
                return
            except Exception:
                pass

        self._iso_status.config(
            text=f"Selected: MO {vals[0]}  {vals[1]}  {vals[2]} eV  occ={vals[3]}.  "
                 "Click Render to compute grid.", fg="#333"
        )

    # ─── Render dispatcher ────────────────────────────────────────────────────

    def _render_iso_tab(self):
        if not _HAS_SKIMAGE:
            messagebox.showerror("Missing", "Run:  pip install scikit-image")
            return
        if self._iso_source.get() == "cube":
            self._render_from_cube()
        else:
            self._render_from_mo()

    def _render_from_cube(self):
        sel = self._cube_lb.curselection()
        if not sel:
            messagebox.showinfo("Isosurface", "Load and select a cube file first.")
            return
        key  = self._cube_lb.get(sel[0])
        cube = self._cubes.get(key)
        if cube:
            self._do_render(cube)

    def _render_from_mo(self):
        """Evaluate the selected MO on a 3D grid using our ORCA parser + GTO evaluator."""
        if not hasattr(self, "_orca_mo_data") or self._orca_mo_data is None:
            messagebox.showinfo("Info", "Click 'Load MOs from .out...' first.")
            return
        sel = self._mo_tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Select an MO from the list.")
            return

        iorb    = int(sel[0])
        spacing = self._iso_spacing.get()
        pad     = self._iso_pad.get()
        out_path = getattr(self, "_orca_mo_path", "")

        vals = self._mo_tree.item(sel[0], "values")
        ro_tag = " [RE-ORIENTED]" if self._reorient_R is not None else ""
        self._iso_status.config(
            text=f"Evaluating MO {vals[0]} ({vals[1]}) on grid  "
                 f"(spacing={spacing:.2f} A, pad={pad:.1f} A){ro_tag} ...",
            fg="gray"
        )
        self.update_idletasks()

        # Compute composition immediately (fast — no grid eval needed)
        try:
            comp = compute_mo_composition(self._orca_mo_data, iorb,
                                          reorient_R=self._reorient_R)
            self.after(0, lambda: self._show_mo_composition(comp, iorb))
        except Exception:
            pass

        # Capture re-orientation state for the background thread
        ro_R = self._reorient_R
        ro_T = self._reorient_T
        ro_R_key = None if ro_R is None else tuple(np.round(ro_R.ravel(), 6))
        ro_T_key = None if ro_T is None else tuple(np.round(ro_T.ravel(), 6))
        grid_key = (out_path, iorb, round(float(spacing), 4), round(float(pad), 4),
                    ro_R_key, ro_T_key)

        if grid_key == self._last_mo_grid_key and self._last_mo_grid_cube is not None:
            self._iso_status.config(
                text=(f"Re-using cached grid for MO {vals[0]} ({vals[1]})  "
                      f"(spacing={spacing:.2f} A, pad={pad:.1f} A){ro_tag}"),
                fg="#006600"
            )
            self._do_render(self._last_mo_grid_cube)
            return

        def _task():
            try:
                cube = build_cube_from_orca_output(out_path, iorb, spacing, pad,
                                                   reorient_R=ro_R, reorient_T=ro_T)
                def _finish():
                    self._last_mo_grid_key = grid_key
                    self._last_mo_grid_cube = cube
                    self._do_render(cube)
                self.after(0, _finish)
            except Exception as exc:
                msg = str(exc)
                self.after(0, lambda: (
                    self._iso_status.config(text=f"Error: {msg[:120]}", fg="red"),
                    messagebox.showerror("MO Evaluation Error", msg)
                ))

        threading.Thread(target=_task, daemon=True).start()

    def _show_mo_composition(self, comp: dict, mo_index: int):
        """Display MO composition breakdown in the composition panel."""
        # Clear previous
        for item in self._comp_tree.get_children():
            self._comp_tree.delete(item)

        if not comp['atom_contribs']:
            self._comp_summary.config(text="No composition data available.", fg="gray")
            self._comp_ao_detail.config(text="")
            return

        # Summary line
        mo_label = ""
        if hasattr(self, "_orca_mo_data") and self._orca_mo_data is not None:
            occs = self._orca_mo_data["occs"]
            occ_mask = occs > 0.5
            homo_idx = int(np.max(np.where(occ_mask))) if np.any(occ_mask) else 0
            delta = mo_index - homo_idx
            if   delta == 0: mo_label = "HOMO"
            elif delta == 1: mo_label = "LUMO"
            elif delta < 0:  mo_label = f"HOMO{delta}"
            else:            mo_label = f"LUMO+{delta-1}"

        self._comp_summary.config(
            text=f"MO {mo_index} ({mo_label})  |  "
                 f"s: {comp['total_s']:.1f}%  p: {comp['total_p']:.1f}%  "
                 f"d: {comp['total_d']:.1f}%  f: {comp['total_f']:.1f}%",
            fg="#003366", font=("", 9, "bold")
        )

        # Populate tree with top contributing atoms
        _METALS = {'Ni', 'Fe', 'Co', 'Cu', 'Pd', 'Pt', 'Ru', 'Rh', 'Ir', 'Os',
                    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Zn'}

        for atom_key, total_pct in comp['top_atoms']:
            if total_pct < 0.5:
                continue  # skip negligible contributors
            info = comp['atom_contribs'][atom_key]
            elem = info['element']

            if elem in _METALS:
                tag = "metal"
            elif total_pct > 5.0:
                tag = "major"
            else:
                tag = "minor"

            values = (
                atom_key,
                f"{total_pct:.1f}",
                f"{info['s']:.1f}" if info['s'] > 0.05 else "-",
                f"{info['p']:.1f}" if info['p'] > 0.05 else "-",
                f"{info['d']:.1f}" if info['d'] > 0.05 else "-",
                f"{info['f']:.1f}" if info['f'] > 0.05 else "-",
            )
            self._comp_tree.insert("", tk.END, values=values, tags=(tag,))

        # Add totals row
        total = sum(v['total'] for v in comp['atom_contribs'].values())
        self._comp_tree.insert("", tk.END, values=(
            "TOTAL", f"{total:.1f}",
            f"{comp['total_s']:.1f}", f"{comp['total_p']:.1f}",
            f"{comp['total_d']:.1f}", f"{comp['total_f']:.1f}",
        ), tags=("total",))

        # Summed AO types (all dx2y2 combined across atoms/shells, etc.)
        summed = comp.get('summed_ao_types', [])
        if summed:
            ao_parts = [f"{ang}: {pct:.1f}%" for ang, pct in summed if pct > 0.3]
            self._comp_ao_detail.config(
                text="Summed AOs: " + "  |  ".join(ao_parts[:15]),
                fg="#333"
            )
        elif comp['top_ao']:
            ao_parts = [f"{ak}({ao}): {pct:.1f}%" for ak, ao, pct in comp['top_ao'][:12]]
            self._comp_ao_detail.config(
                text="Top AOs: " + "  |  ".join(ao_parts),
                fg="#333"
            )
        else:
            self._comp_ao_detail.config(text="")

        # Store last comp for the detail button
        self._last_mo_comp = comp
        self._last_mo_comp_index = mo_index

        # Populate atom checklist (only if not already populated)
        self._populate_comp_atom_list(comp)

    def _show_atom_ao_detail(self):
        """Open a window showing per-atom AO breakdown for the current MO.

        For each atom with significant contribution, shows a table of
        individual orbital types (dx2y2, dz2, px, s, etc.) with their
        percentages summed across shells.
        """
        comp = getattr(self, "_last_mo_comp", None)
        mo_idx = getattr(self, "_last_mo_comp_index", None)
        if comp is None or mo_idx is None:
            messagebox.showinfo("Atom AO Detail", "Render an MO first to see its composition.")
            return

        atom_ao = comp.get('atom_ao_detail', {})
        atom_shells = comp.get('atom_ao_shells', {})
        if not atom_ao:
            messagebox.showinfo("Atom AO Detail", "No AO detail available.")
            return

        # Determine which atoms to show (those checked, or all significant ones)
        checked = [k for k, v in self._comp_atom_vars.items() if v.get()]
        if not checked:
            # Fall back to all atoms > 1%
            checked = [k for k, pct in comp['top_atoms'] if pct > 1.0]

        win = tk.Toplevel(self)
        mo_label = f"MO {mo_idx}"
        if hasattr(self, "_orca_mo_data") and self._orca_mo_data is not None:
            occs = self._orca_mo_data["occs"]
            occ_mask = occs > 0.5
            homo_idx = int(np.max(np.where(occ_mask))) if np.any(occ_mask) else 0
            delta = mo_idx - homo_idx
            if   delta == 0: mo_label += " (HOMO)"
            elif delta == 1: mo_label += " (LUMO)"
            elif delta < 0:  mo_label += f" (HOMO{delta})"
            else:            mo_label += f" (LUMO+{delta-1})"
        reoriented = self._reorient_R is not None
        title_suffix = " [Re-oriented]" if reoriented else ""
        win.title(f"Atom AO Breakdown — {mo_label}{title_suffix}")
        _clamp_geometry(win, 700, 500, 400, 350)

        tk.Label(win, text=f"{mo_label}{title_suffix}  —  Per-Atom Orbital Contributions",
                 font=("", 11, "bold")).pack(pady=6)

        # Notebook with one tab per atom
        nb = ttk.Notebook(win)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        for atom_key in checked:
            ao_dict = atom_ao.get(atom_key, {})
            shells_list = atom_shells.get(atom_key, [])
            if not ao_dict:
                continue

            tab = tk.Frame(nb)
            nb.add(tab, text=f"{atom_key}")

            # Info header
            info = comp['atom_contribs'].get(atom_key, {})
            total_pct = info.get('total', 0.0)
            tk.Label(tab, text=(
                f"{atom_key}  —  Total: {total_pct:.1f}%  |  "
                f"s: {info.get('s', 0):.1f}%  p: {info.get('p', 0):.1f}%  "
                f"d: {info.get('d', 0):.1f}%  f: {info.get('f', 0):.1f}%"
            ), font=("", 9, "bold"), fg="#003366").pack(anchor="w", padx=6, pady=4)

            # ── Summed AO table (across shells) ──────────────────────────
            tk.Label(tab, text="Summed across shells:", font=("", 9, "italic")).pack(anchor="w", padx=6)

            sum_frame = tk.Frame(tab)
            sum_frame.pack(fill=tk.X, padx=6, pady=2)

            sum_cols = ("Orbital", "% Contribution")
            sum_tree = ttk.Treeview(sum_frame, columns=sum_cols, show="headings", height=6)
            for c, w in zip(sum_cols, (150, 150)):
                sum_tree.heading(c, text=c)
                sum_tree.column(c, width=w, anchor="center")
            sum_tree.pack(fill=tk.X)

            # Sort by contribution
            sorted_ao = sorted(ao_dict.items(), key=lambda x: x[1], reverse=True)
            for ang, pct in sorted_ao:
                if pct < 0.05:
                    continue
                tag = ""
                if pct > 10:
                    tag = "major"
                elif pct > 2:
                    tag = "moderate"
                sum_tree.insert("", tk.END, values=(ang, f"{pct:.2f}%"), tags=(tag,))
            sum_tree.tag_configure("major", foreground="#006600", font=("", 9, "bold"))
            sum_tree.tag_configure("moderate", foreground="#336633")

            # ── Per-shell detail ─────────────────────────────────────────
            ttk.Separator(tab).pack(fill=tk.X, padx=6, pady=4)
            tk.Label(tab, text="Per-shell detail:", font=("", 9, "italic")).pack(anchor="w", padx=6)

            shell_frame = tk.Frame(tab)
            shell_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=2)

            sh_cols = ("Shell", "Orbital", "% Contribution")
            sh_tree = ttk.Treeview(shell_frame, columns=sh_cols, show="headings", height=8)
            sh_sb = ttk.Scrollbar(shell_frame, orient=tk.VERTICAL, command=sh_tree.yview)
            sh_tree.config(yscrollcommand=sh_sb.set)
            for c, w in zip(sh_cols, (60, 120, 120)):
                sh_tree.heading(c, text=c)
                sh_tree.column(c, width=w, anchor="center")
            sh_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sh_sb.pack(side=tk.RIGHT, fill=tk.Y)

            # Sort per-shell by contribution
            sorted_shells = sorted(shells_list, key=lambda x: x[2], reverse=True)
            for shell_n, ang, pct in sorted_shells:
                if pct < 0.05:
                    continue
                sh_tree.insert("", tk.END, values=(shell_n, ang, f"{pct:.2f}%"))

    def _show_summed_atom_ao_detail(self):
        """Open a window summing AO breakdown across the currently checked atoms."""
        comp = getattr(self, "_last_mo_comp", None)
        mo_idx = getattr(self, "_last_mo_comp_index", None)
        if comp is None or mo_idx is None:
            messagebox.showinfo("Summed AO Detail", "Render an MO first to see its composition.")
            return

        atom_ao = comp.get('atom_ao_detail', {})
        if not atom_ao:
            messagebox.showinfo("Summed AO Detail", "No AO detail available.")
            return

        checked = [k for k, v in self._comp_atom_vars.items() if v.get()]
        if not checked:
            checked = [k for k, pct in comp['top_atoms'] if pct > 1.0]
        checked = [k for k in checked if k in atom_ao]
        if not checked:
            messagebox.showinfo("Summed AO Detail", "No checked atoms with AO detail were found.")
            return

        mo_label = f"MO {mo_idx}"
        if hasattr(self, "_orca_mo_data") and self._orca_mo_data is not None:
            occs = self._orca_mo_data["occs"]
            occ_mask = occs > 0.5
            homo_idx = int(np.max(np.where(occ_mask))) if np.any(occ_mask) else 0
            delta = mo_idx - homo_idx
            if   delta == 0: mo_label += " (HOMO)"
            elif delta == 1: mo_label += " (LUMO)"
            elif delta < 0:  mo_label += f" (HOMO{delta})"
            else:            mo_label += f" (LUMO+{delta-1})"

        summed_ao: Dict[str, float] = {}
        total_pct = 0.0
        total_s = total_p = total_d = total_f = 0.0
        per_atom_rows = []

        for atom_key in checked:
            info = comp['atom_contribs'].get(atom_key, {})
            total_pct += info.get('total', 0.0)
            total_s += info.get('s', 0.0)
            total_p += info.get('p', 0.0)
            total_d += info.get('d', 0.0)
            total_f += info.get('f', 0.0)
            per_atom_rows.append((
                atom_key,
                info.get('total', 0.0),
                info.get('s', 0.0),
                info.get('p', 0.0),
                info.get('d', 0.0),
                info.get('f', 0.0),
            ))

            for ang, pct in atom_ao.get(atom_key, {}).items():
                summed_ao[ang] = summed_ao.get(ang, 0.0) + pct

        sorted_ao = sorted(summed_ao.items(), key=lambda x: x[1], reverse=True)
        per_atom_rows.sort(key=lambda row: row[1], reverse=True)

        win = tk.Toplevel(self)
        reoriented = self._reorient_R is not None
        title_suffix = " [Re-oriented]" if reoriented else ""
        win.title(f"Summed Atom AO Breakdown — {mo_label}{title_suffix}")
        _clamp_geometry(win, 760, 560, 500, 400)

        tk.Label(
            win,
            text=f"{mo_label}{title_suffix}  —  Summed AO Breakdown for Selected Atoms",
            font=("", 11, "bold")
        ).pack(pady=6)

        atoms_text = ", ".join(checked)
        tk.Label(
            win,
            text=f"Atoms: {atoms_text}",
            fg="#333", justify=tk.LEFT, wraplength=720
        ).pack(fill=tk.X, padx=8)

        tk.Label(
            win,
            text=(f"Total selected: {total_pct:.1f}%  |  "
                  f"s: {total_s:.1f}%  p: {total_p:.1f}%  "
                  f"d: {total_d:.1f}%  f: {total_f:.1f}%"),
            font=("", 9, "bold"), fg="#003366"
        ).pack(fill=tk.X, padx=8, pady=(4, 6))

        top = tk.PanedWindow(win, orient=tk.VERTICAL, sashwidth=5, sashrelief=tk.RAISED)
        top.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        upper = tk.LabelFrame(top, text="Summed AO contributions", padx=4, pady=4)
        lower = tk.LabelFrame(top, text="Per-atom totals in selection", padx=4, pady=4)
        top.add(upper, minsize=180)
        top.add(lower, minsize=140)

        upper_frame = tk.Frame(upper)
        upper_frame.pack(fill=tk.BOTH, expand=True)
        ao_cols = ("Orbital", "% Contribution")
        ao_tree = ttk.Treeview(upper_frame, columns=ao_cols, show="headings", height=12)
        ao_sb = ttk.Scrollbar(upper_frame, orient=tk.VERTICAL, command=ao_tree.yview)
        ao_tree.config(yscrollcommand=ao_sb.set)
        for c, w in zip(ao_cols, (180, 140)):
            ao_tree.heading(c, text=c)
            ao_tree.column(c, width=w, anchor="center")
        ao_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ao_sb.pack(side=tk.RIGHT, fill=tk.Y)

        for ang, pct in sorted_ao:
            if pct < 0.05:
                continue
            tag = ""
            if pct > 10:
                tag = "major"
            elif pct > 2:
                tag = "moderate"
            ao_tree.insert("", tk.END, values=(ang, f"{pct:.2f}%"), tags=(tag,))
        ao_tree.tag_configure("major", foreground="#006600", font=("", 9, "bold"))
        ao_tree.tag_configure("moderate", foreground="#336633")

        lower_frame = tk.Frame(lower)
        lower_frame.pack(fill=tk.BOTH, expand=True)
        atom_cols = ("Atom", "Total", "s", "p", "d", "f")
        atom_tree = ttk.Treeview(lower_frame, columns=atom_cols, show="headings", height=8)
        atom_sb = ttk.Scrollbar(lower_frame, orient=tk.VERTICAL, command=atom_tree.yview)
        atom_tree.config(yscrollcommand=atom_sb.set)
        for c, w in zip(atom_cols, (110, 80, 70, 70, 70, 70)):
            atom_tree.heading(c, text=c)
            atom_tree.column(c, width=w, anchor="center")
        atom_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        atom_sb.pack(side=tk.RIGHT, fill=tk.Y)

        for atom_key, total_v, s_v, p_v, d_v, f_v in per_atom_rows:
            atom_tree.insert("", tk.END, values=(
                atom_key,
                f"{total_v:.2f}%",
                f"{s_v:.2f}%" if s_v > 0.0 else "-",
                f"{p_v:.2f}%" if p_v > 0.0 else "-",
                f"{d_v:.2f}%" if d_v > 0.0 else "-",
                f"{f_v:.2f}%" if f_v > 0.0 else "-",
            ))

    # ─── Atom tracking checklist helpers ─────────────────────────────────────

    def _populate_comp_atom_list(self, comp: dict):
        """Populate the atom checklist from the current MO composition data."""
        # Only rebuild if we have new atoms not yet in the list
        existing = set(self._comp_atom_vars.keys())
        new_atoms = set(comp['atom_contribs'].keys())

        if new_atoms - existing:
            # Rebuild the entire list
            for w in self._comp_atom_inner.winfo_children():
                w.destroy()
            self._comp_atom_vars.clear()

            _METALS = {'Ni', 'Fe', 'Co', 'Cu', 'Pd', 'Pt', 'Ru', 'Rh', 'Ir', 'Os',
                        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Zn'}
            _DONORS = {'P', 'N', 'O', 'S', 'Cl', 'Br', 'I', 'F', 'As', 'Se'}

            # Get all unique atoms from the MO data, sorted by index
            all_atoms = []
            if hasattr(self, '_orca_mo_data') and self._orca_mo_data is not None:
                seen = set()
                for a_idx, a_el, _, _ in self._orca_mo_data['ao_labels']:
                    key = f"{a_el}{a_idx + 1}"
                    if key not in seen:
                        seen.add(key)
                        all_atoms.append((a_idx, a_el, key))
                all_atoms.sort(key=lambda x: x[0])

            for a_idx, a_el, key in all_atoms:
                var = tk.BooleanVar(value=(a_el in _METALS or a_el in _DONORS))
                self._comp_atom_vars[key] = var
                color = _ec(a_el)
                cb = tk.Checkbutton(self._comp_atom_inner, text=key,
                                    variable=var, anchor="w",
                                    font=("", 8), selectcolor=color,
                                    activebackground=color)
                cb.pack(fill=tk.X, padx=2)

    def _comp_select_all(self, state: bool):
        for var in self._comp_atom_vars.values():
            var.set(state)

    def _comp_select_metals(self):
        _METALS = {'Ni', 'Fe', 'Co', 'Cu', 'Pd', 'Pt', 'Ru', 'Rh', 'Ir', 'Os',
                    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Zn'}
        for key, var in self._comp_atom_vars.items():
            elem = ''.join(c for c in key if c.isalpha())
            var.set(elem in _METALS)

    def _comp_select_donors(self):
        _DONORS = {'P', 'N', 'O', 'S', 'Cl', 'Br', 'I', 'F', 'As', 'Se',
                    'Ni', 'Fe', 'Co', 'Cu', 'Pd', 'Pt'}
        for key, var in self._comp_atom_vars.items():
            elem = ''.join(c for c in key if c.isalpha())
            var.set(elem in _DONORS)

    def _show_tracked_atom_table(self):
        """Open a new window showing tracked atoms' contributions across all MOs."""
        if not hasattr(self, '_orca_mo_data') or self._orca_mo_data is None:
            messagebox.showinfo("Info", "Load MOs from an .out file first.")
            return

        # Which atoms are checked?
        tracked = [key for key, var in self._comp_atom_vars.items() if var.get()]
        if not tracked:
            messagebox.showinfo("Info", "Check at least one atom in the 'Track atoms' panel.")
            return

        mo_data = self._orca_mo_data
        occs = mo_data["occs"]
        occ_mask = occs > 0.5
        homo_idx = int(np.max(np.where(occ_mask))) if np.any(occ_mask) else 0
        nmo = mo_data["coeffs"].shape[1]

        # Compute compositions for MOs near the frontier (HOMO-10 to LUMO+10)
        mo_range_start = max(0, homo_idx - 10)
        mo_range_end   = min(nmo, homo_idx + 11)

        self._iso_status.config(text="Computing tracked atom contributions...", fg="gray")
        self.update_idletasks()

        results = []
        for mo_i in range(mo_range_start, mo_range_end):
            comp = compute_mo_composition(mo_data, mo_i,
                                          reorient_R=self._reorient_R)
            delta = mo_i - homo_idx
            if   delta == 0:  lbl = "HOMO"
            elif delta == 1:  lbl = "LUMO"
            elif delta < 0:   lbl = f"H{delta}"
            else:             lbl = f"L+{delta-1}"
            row = {"mo": mo_i, "label": lbl, "energy": mo_data["energies"][mo_i],
                   "occ": mo_data["occs"][mo_i]}
            for ak in tracked:
                ac = comp['atom_contribs'].get(ak, {'s': 0, 'p': 0, 'd': 0, 'f': 0, 'total': 0})
                row[ak] = ac
            results.append(row)

        # Open the table window
        win = tk.Toplevel(self)
        win.title(f"Atom Contributions — {', '.join(tracked[:5])}{'...' if len(tracked) > 5 else ''}")
        _clamp_geometry(win, 1100, 500, 600, 300)

        # Build treeview columns: MO#, Label, Energy, Occ, then per-atom Total(s/p/d)
        base_cols = ("MO#", "Label", "Energy (Eh)", "Occ")
        atom_cols = []
        for ak in tracked:
            atom_cols.append(f"{ak} total")
            atom_cols.append(f"{ak} d%")
            atom_cols.append(f"{ak} p%")
            atom_cols.append(f"{ak} s%")
        all_cols = base_cols + tuple(atom_cols)

        tf = tk.Frame(win)
        tf.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        vsb = ttk.Scrollbar(tf, orient=tk.VERTICAL)
        hsb = ttk.Scrollbar(tf, orient=tk.HORIZONTAL)
        tree = ttk.Treeview(tf, columns=all_cols, show="headings",
                            yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)

        for col in all_cols:
            w = 50 if col not in base_cols else (45, 60, 85, 40)[base_cols.index(col)]
            tree.heading(col, text=col, anchor="center")
            tree.column(col, width=w, anchor="center", stretch=False)

        tree.tag_configure("homo", background="#FFD700", font=("", 9, "bold"))
        tree.tag_configure("lumo", background="#90EE90", font=("", 9, "bold"))
        tree.tag_configure("occ",  background="#F0F5FF")
        tree.tag_configure("virt", background="#FFF5F0")

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)

        # MO range controls at bottom
        ctrl = tk.Frame(win)
        ctrl.pack(fill=tk.X, padx=4, pady=4)
        tk.Label(ctrl, text="MO range:", font=("", 9)).pack(side=tk.LEFT)
        from_var = tk.IntVar(value=mo_range_start)
        to_var   = tk.IntVar(value=mo_range_end - 1)
        min_pct_var = tk.DoubleVar(value=0.5)
        info_var = tk.StringVar(value="")
        tk.Label(ctrl, text="From MO#:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Spinbox(ctrl, textvariable=from_var, from_=0, to=nmo - 1, width=6).pack(side=tk.LEFT)
        tk.Label(ctrl, text="To MO#:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Spinbox(ctrl, textvariable=to_var, from_=0, to=nmo - 1, width=6).pack(side=tk.LEFT)
        tk.Label(ctrl, text="Min % shown:", font=("", 9)).pack(side=tk.LEFT, padx=(10, 2))
        ttk.Spinbox(ctrl, textvariable=min_pct_var, from_=0.0, to=100.0,
                    increment=0.5, width=5, format="%.1f").pack(side=tk.LEFT)

        def _fmt_cell(val: float, threshold: float) -> str:
            return f"{val:.1f}" if val >= threshold else "-"

        def _populate_table(rows):
            for item in tree.get_children():
                tree.delete(item)

            threshold = float(min_pct_var.get())
            shown_rows = 0
            for row in rows:
                meets_threshold = any(row.get(ak, {}).get('total', 0.0) >= threshold for ak in tracked)
                if not meets_threshold:
                    continue

                vals = [
                    row["mo"],
                    row["label"],
                    f"{row['energy']:.5f}",
                    f"{row['occ']:.2f}",
                ]
                for ak in tracked:
                    ac = row.get(ak, {'total': 0, 'd': 0, 'p': 0, 's': 0})
                    vals.append(_fmt_cell(ac['total'], threshold))
                    vals.append(_fmt_cell(ac['d'], threshold))
                    vals.append(_fmt_cell(ac['p'], threshold))
                    vals.append(_fmt_cell(ac['s'], threshold))

                tag = "homo" if row["label"] == "HOMO" else \
                      "lumo" if row["label"] == "LUMO" else \
                      "occ"  if row["occ"] > 0.5 else "virt"
                tree.insert("", tk.END, values=tuple(vals), tags=(tag,))
                shown_rows += 1

            info_var.set(
                f"Tracked: {', '.join(tracked)}  |  {shown_rows} shown  |  "
                f"min = {threshold:.1f}%  |  HOMO = MO {homo_idx}"
            )

        def _refresh_range():
            rng_start = max(0, from_var.get())
            rng_end   = min(nmo, to_var.get() + 1)
            refreshed_rows = []
            for mo_i in range(rng_start, rng_end):
                comp = compute_mo_composition(mo_data, mo_i,
                                              reorient_R=self._reorient_R)
                delta = mo_i - homo_idx
                if   delta == 0:  lbl = "HOMO"
                elif delta == 1:  lbl = "LUMO"
                elif delta < 0:   lbl = f"H{delta}"
                else:             lbl = f"L+{delta-1}"
                row = {"mo": mo_i, "label": lbl, "energy": mo_data["energies"][mo_i],
                       "occ": mo_data["occs"][mo_i]}
                for ak in tracked:
                    c = comp['atom_contribs'].get(ak, {'total': 0, 'd': 0, 'p': 0, 's': 0})
                    row[ak] = c
                refreshed_rows.append(row)
            _populate_table(refreshed_rows)

        tk.Button(ctrl, text="Refresh", bg="#1a3c6e", fg="white",
                  command=_refresh_range).pack(side=tk.LEFT, padx=8)
        tk.Button(ctrl, text="Apply filter", command=lambda: _populate_table(results)
                  ).pack(side=tk.LEFT, padx=(2, 8))
        tk.Label(ctrl, textvariable=info_var, fg="#333", font=("", 8)
                 ).pack(side=tk.LEFT, padx=8)

        _populate_table(results)

        self._iso_status.config(text=f"Atom tracker: {len(tracked)} atoms across {len(results)} MOs", fg="#006600")

    def _show_mo_diagram_picker(self):
        """MO diagram workflow with element/atom pickers and contributor lists."""
        if not hasattr(self, "_orca_mo_data") or self._orca_mo_data is None:
            messagebox.showinfo(
                "MO Diagram",
                "Load an ORCA .out file with MOs first "
                "(use Load Output on the Isosurface tab).",
            )
            return

        mo_data = self._orca_mo_data
        energies = mo_data["energies"]
        occs = mo_data["occs"]
        nmo = mo_data["coeffs"].shape[1]
        homo_idx = int(np.max(np.where(occs > 0.5))) if np.any(occs > 0.5) else 0
        pending_state = self._pending_mo_diagram_state
        self._pending_mo_diagram_state = None

        all_atom_keys: List[str] = []
        all_element_keys: List[str] = []
        for (a_idx, a_el, _, _) in mo_data["ao_labels"]:
            if a_el not in all_element_keys:
                all_element_keys.append(a_el)
            atom_key = f"{a_el}{a_idx + 1}"
            if atom_key not in all_atom_keys:
                all_atom_keys.append(atom_key)

        element_lookup = {el.lower(): el for el in all_element_keys}
        atom_lookup = {ak.lower(): ak for ak in all_atom_keys}
        comp_cache: Dict[int, dict] = {}

        win = tk.Toplevel(self)
        win.title("MO Energy Diagram")
        _clamp_geometry(win, 1480, 920, 700, 500)

        # Wrap in scrollable frame for small screens
        _sf2 = _ScrollableFrame(win)
        _sf2.pack(fill=tk.BOTH, expand=True)
        _win_body2 = _sf2.interior

        top = tk.LabelFrame(_win_body2, text="1. MO selection", padx=6, pady=4)
        top.pack(fill=tk.X, padx=6, pady=(6, 3))

        rng_row = tk.Frame(top)
        rng_row.pack(fill=tk.X)
        tk.Label(rng_row, text="From MO#:").pack(side=tk.LEFT)
        from_var = tk.IntVar(value=max(0, homo_idx - 5))
        ttk.Spinbox(rng_row, textvariable=from_var, from_=0, to=nmo - 1,
                    width=6).pack(side=tk.LEFT, padx=(2, 8))
        tk.Label(rng_row, text="To MO#:").pack(side=tk.LEFT)
        to_var = tk.IntVar(value=min(nmo - 1, homo_idx + 5))
        ttk.Spinbox(rng_row, textvariable=to_var, from_=0, to=nmo - 1,
                    width=6).pack(side=tk.LEFT, padx=(2, 8))

        mo_list_var = tk.StringVar(value="")
        list_row = tk.Frame(top)
        list_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(list_row, text="MO list / ranges:").pack(side=tk.LEFT)
        tk.Entry(list_row, textvariable=mo_list_var, width=42,
                 font=("Courier", 9)).pack(side=tk.LEFT, padx=(4, 4))
        tk.Label(list_row, text="Examples: 118-124, 126, HOMO, LUMO+2",
                 font=("", 8), fg="gray").pack(side=tk.LEFT)

        mo_list_frame = tk.Frame(top)
        mo_list_frame.pack(fill=tk.X, pady=(4, 0))
        mo_canvas = tk.Canvas(mo_list_frame, height=80, highlightthickness=0)
        mo_sb = ttk.Scrollbar(mo_list_frame, orient=tk.HORIZONTAL,
                              command=mo_canvas.xview)
        mo_inner = tk.Frame(mo_canvas)
        mo_canvas.create_window((0, 0), window=mo_inner, anchor="nw")
        mo_inner.bind(
            "<Configure>",
            lambda e: mo_canvas.config(scrollregion=mo_canvas.bbox("all")),
        )
        mo_canvas.config(xscrollcommand=mo_sb.set)
        mo_canvas.pack(fill=tk.X, expand=True)
        mo_sb.pack(fill=tk.X)
        mo_check_vars: Dict[int, tk.BooleanVar] = {}

        mid = tk.Frame(_win_body2)
        mid.pack(fill=tk.X, padx=6, pady=3)

        targets_frame = tk.LabelFrame(mid, text="2. Targets", padx=6, pady=4)
        targets_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        target_help_var = tk.StringVar(
            value="Elements: Ni, Cl, C, P (typed or checked below)."
        )
        tk.Label(targets_frame, textvariable=target_help_var,
                 font=("", 8), fg="#333").pack(anchor="w")
        target_entry_var = tk.StringVar(value="")
        tk.Entry(targets_frame, textvariable=target_entry_var,
                 font=("Courier", 9)).pack(fill=tk.X, pady=(2, 4))
        target_btn_row = tk.Frame(targets_frame)
        target_btn_row.pack(fill=tk.X, pady=(0, 2))
        target_canvas = tk.Canvas(targets_frame, height=112, highlightthickness=0)
        target_sb = ttk.Scrollbar(targets_frame, orient=tk.VERTICAL,
                                  command=target_canvas.yview)
        target_inner = tk.Frame(target_canvas)
        target_canvas.create_window((0, 0), window=target_inner, anchor="nw")
        target_inner.bind(
            "<Configure>",
            lambda e: target_canvas.config(scrollregion=target_canvas.bbox("all")),
        )
        target_canvas.config(yscrollcommand=target_sb.set)
        target_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        target_sb.pack(side=tk.RIGHT, fill=tk.Y)
        target_check_vars: Dict[str, tk.BooleanVar] = {}

        opts_frame = tk.LabelFrame(mid, text="3. Grouping and style", padx=6, pady=4)
        opts_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        group_var = tk.StringVar(value="element")
        mode_var = tk.StringVar(value="ao")
        spacing_var = tk.StringVar(value="even")
        min_pct_var = tk.DoubleVar(value=1.0)
        unit_var = tk.StringVar(value="eV")

        contrib_frame = tk.LabelFrame(mid, text="4. Contributor list", padx=6, pady=4)
        contrib_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(
            contrib_frame,
            text="Refresh the list, then keep only the orbital channels you want to annotate.",
            font=("", 8), fg="#333",
        ).pack(anchor="w")
        contrib_btn_row = tk.Frame(contrib_frame)
        contrib_btn_row.pack(fill=tk.X, pady=(2, 2))
        contrib_list_frame = tk.Frame(contrib_frame)
        contrib_list_frame.pack(fill=tk.BOTH, expand=True)
        contrib_sb = ttk.Scrollbar(contrib_list_frame, orient=tk.VERTICAL)
        contrib_sb.pack(side=tk.RIGHT, fill=tk.Y)
        contrib_lb = tk.Listbox(
            contrib_list_frame,
            selectmode=tk.EXTENDED,
            yscrollcommand=contrib_sb.set,
            height=8,
            font=("Courier", 9),
        )
        contrib_lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        contrib_sb.config(command=contrib_lb.yview)
        contrib_status = tk.Label(
            contrib_frame,
            text="No contributors listed yet.",
            fg="gray",
            font=("", 8),
            anchor="w",
            justify=tk.LEFT,
        )
        contrib_status.pack(fill=tk.X, pady=(2, 0))

        plot_frame = tk.LabelFrame(_win_body2, text="5. Diagram", padx=4, pady=4)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(3, 6))
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()

        status = tk.Label(
            _win_body2,
            text="Pick MOs, targets, and contributors, then build the classic diagram.",
            fg="gray",
            anchor="w",
        )
        status.pack(fill=tk.X, padx=6, pady=(0, 4), before=plot_frame)

        def _get_comp(mo_i: int) -> dict:
            if mo_i not in comp_cache:
                comp_cache[mo_i] = compute_mo_composition(
                    mo_data, mo_i, reorient_R=self._reorient_R
                )
            return comp_cache[mo_i]

        def _get_selected_mos() -> List[int]:
            return sorted([mo_i for mo_i, var in mo_check_vars.items() if var.get()])

        def _populate_mo_checks(selected_mos: Optional[List[int]] = None):
            for w in mo_inner.winfo_children():
                w.destroy()
            mo_check_vars.clear()
            lo = max(0, from_var.get())
            hi = min(nmo - 1, to_var.get())
            if lo > hi:
                lo, hi = hi, lo
            for i in range(lo, hi + 1):
                if selected_mos is None:
                    is_selected = True
                else:
                    is_selected = (i in selected_mos)
                v = tk.BooleanVar(value=is_selected)
                mo_check_vars[i] = v
                frontier = _mo_frontier_label(i, homo_idx)
                if frontier.startswith("HOMO"):
                    tag = frontier if frontier == "HOMO" else frontier.replace("HOMO", "H")
                elif frontier.startswith("LUMO"):
                    tag = frontier if frontier == "LUMO" else frontier.replace("LUMO", "L")
                else:
                    tag = frontier
                tk.Checkbutton(
                    mo_inner,
                    text=f"{i} ({tag})",
                    variable=v,
                    font=("", 8),
                ).pack(side=tk.LEFT)
            mo_inner.update_idletasks()
            mo_canvas.config(scrollregion=mo_canvas.bbox("all"))

        def _apply_mo_list():
            picked, errors = parse_mo_selection_list(mo_list_var.get(), nmo, homo_idx)
            if errors:
                status.config(text="Could not parse: " + ", ".join(errors[:6]), fg="#993300")
                return
            if not picked:
                status.config(text="No MO indices found in the typed list.", fg="#993300")
                return
            from_var.set(min(picked))
            to_var.set(max(picked))
            _populate_mo_checks(selected_mos=picked)
            status.config(text=f"Selected {len(picked)} MOs from typed list.", fg="#006600")

        def _rebuild_target_checks():
            for w in target_inner.winfo_children():
                w.destroy()
            target_check_vars.clear()

            if group_var.get() == "element":
                options = all_element_keys
                target_help_var.set("Elements: Ni, Cl, C, P (typed or checked below).")
            else:
                options = all_atom_keys
                target_help_var.set("Atoms: Ni1, Cl7, P2 (typed or checked below).")

            for key in options:
                var = tk.BooleanVar(value=False)
                target_check_vars[key] = var
                tk.Checkbutton(target_inner, text=key, variable=var, anchor="w",
                               font=("", 8)).pack(fill=tk.X, anchor="w")

            target_inner.update_idletasks()
            target_canvas.config(scrollregion=target_canvas.bbox("all"))

        def _normalize_target_token(token: str) -> Optional[str]:
            compact = token.strip().replace(" ", "")
            if not compact:
                return None

            m = re.match(r"^(\d+)([A-Za-z]+)$", compact)
            if m:
                compact = f"{m.group(2)}{m.group(1)}"

            if group_var.get() == "element":
                m = re.match(r"^([A-Za-z]+)(\d+)$", compact)
                if m:
                    compact = m.group(1)
                elem = compact[:1].upper() + compact[1:].lower()
                return element_lookup.get(elem.lower())

            m = re.match(r"^([A-Za-z]+)(\d+)$", compact)
            if not m:
                return None
            elem = m.group(1)[:1].upper() + m.group(1)[1:].lower()
            atom_key = f"{elem}{m.group(2)}"
            return atom_lookup.get(atom_key.lower())

        def _gather_targets() -> Tuple[List[str], List[str]]:
            chosen: List[str] = []
            invalid: List[str] = []
            seen = set()

            raw = target_entry_var.get().strip()
            if raw:
                for tok in re.split(r"[,\s;]+", raw):
                    if not tok.strip():
                        continue
                    norm = _normalize_target_token(tok)
                    if norm is None:
                        invalid.append(tok)
                        continue
                    if norm not in seen:
                        seen.add(norm)
                        chosen.append(norm)

            for key, var in target_check_vars.items():
                if var.get() and key not in seen:
                    seen.add(key)
                    chosen.append(key)

            return chosen, invalid

        def _target_component_values(comp: dict, target: str) -> Dict[str, float]:
            detail_mode = mode_var.get()

            if group_var.get() == "atom":
                if detail_mode == "ang":
                    info = comp["atom_contribs"].get(target)
                    if not info:
                        return {}
                    return {orb: info.get(orb, 0.0) for orb in ("s", "p", "d", "f")}
                return dict(comp["atom_ao_detail"].get(target, {}))

            if detail_mode == "ang":
                totals = {"s": 0.0, "p": 0.0, "d": 0.0, "f": 0.0}
                for info in comp["atom_contribs"].values():
                    if info.get("element") != target:
                        continue
                    for orb in totals:
                        totals[orb] += info.get(orb, 0.0)
                return totals

            totals: Dict[str, float] = {}
            for atom_key, info in comp["atom_contribs"].items():
                if info.get("element") != target:
                    continue
                for orb, pct in comp["atom_ao_detail"].get(atom_key, {}).items():
                    totals[orb] = totals.get(orb, 0.0) + pct
            return totals

        _METAL_ELEMENTS = {
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
            "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
            "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
            "Ho", "Er", "Tm", "Yb", "Lu",
        }

        def _extract_target_element(target: str) -> str:
            m = re.match(r"^([A-Za-z]+)", str(target).strip())
            if not m:
                return str(target).strip()
            token = m.group(1)
            return token[:1].upper() + token[1:].lower()

        def _is_metal_target(target: str) -> bool:
            return _extract_target_element(target) in _METAL_ELEMENTS

        def _get_selected_contributors(expand_all: bool = True) -> List[str]:
            picked = [contrib_lb.get(i) for i in contrib_lb.curselection()]
            if picked or not expand_all:
                return picked
            return [contrib_lb.get(i) for i in range(contrib_lb.size())]

        def _refresh_contributor_list(select_all: bool = False, keep_selection: bool = True):
            previous = set(_get_selected_contributors(expand_all=False)) if keep_selection else set()
            contrib_lb.delete(0, tk.END)

            picked_mos = _get_selected_mos()
            if not picked_mos:
                contrib_status.config(text="No MOs selected.", fg="#993300")
                return []

            picked_targets, invalid = _gather_targets()
            if not picked_targets:
                contrib_status.config(
                    text=f"Pick at least one {group_var.get()} first.",
                    fg="#993300",
                )
                return []

            min_pct = float(min_pct_var.get())
            contrib_by_target: Dict[str, Dict[str, float]] = {}
            for mo_i in picked_mos:
                comp = _get_comp(mo_i)
                for target in picked_targets:
                    vals = _target_component_values(comp, target)
                    if not vals:
                        continue
                    contrib_by_target.setdefault(target, {})
                    for orb, pct in vals.items():
                        if pct < min_pct:
                            continue
                        contrib_by_target[target][orb] = max(
                            pct, contrib_by_target[target].get(orb, 0.0)
                        )

            labels: List[str] = []
            for target in picked_targets:
                orb_map = contrib_by_target.get(target, {})
                if not orb_map:
                    continue
                if mode_var.get() == "ang":
                    orb_order = [orb for orb in ("s", "p", "d", "f") if orb in orb_map]
                else:
                    orb_order = sorted(orb_map, key=lambda orb: (-orb_map[orb], orb))
                for orb in orb_order:
                    labels.append(f"{target}-{orb}")

            for label in labels:
                contrib_lb.insert(tk.END, label)

            if labels:
                if select_all or not previous:
                    contrib_lb.selection_set(0, tk.END)
                else:
                    for idx, label in enumerate(labels):
                        if label in previous:
                            contrib_lb.selection_set(idx)
                    if not contrib_lb.curselection():
                        contrib_lb.selection_set(0, tk.END)

            extra = ""
            if invalid:
                extra = "  Ignored: " + ", ".join(invalid[:5])
            if labels:
                contrib_status.config(
                    text=(
                        f"{len(labels)} contributors available across "
                        f"{len(picked_mos)} selected MOs.{extra}"
                    ),
                    fg="#006600",
                )
            else:
                contrib_status.config(
                    text=(
                        f"No contributors above {min_pct:.1f}% for the current "
                        f"targets and MO range.{extra}"
                    ),
                    fg="#993300",
                )
            return labels

        popup_views: List[dict] = []
        last_plot_payload: Optional[dict] = None

        def _format_target_entries(entries: List[Tuple[str, float]], max_lines: int = 3) -> str:
            if not entries:
                return ""
            total = sum(pct for _, pct in entries)
            total_txt = f"Total {total:.0f}%" if total >= 10.0 else f"Total {total:.1f}%"
            if max_lines <= 1:
                return total_txt

            lines: List[str] = [total_txt]
            detail_slots = max_lines - 1
            for orb, pct in entries[:detail_slots]:
                pct_txt = f"{pct:.0f}%" if pct >= 10.0 else f"{pct:.1f}%"
                lines.append(f"{orb} {pct_txt}")
            if len(entries) > detail_slots:
                lines[-1] = lines[-1] + "  ..."
            return "\n".join(lines)

        def _render_plot(payload: dict, fig_obj, ax_obj, canvas_obj) -> None:
            row_data = payload["rows"]
            even = payload["even"]
            unit = payload["unit"]
            left_targets = payload["left_targets"]
            right_targets = payload["right_targets"]
            all_targets = left_targets + right_targets
            count_mos = max(1, len(row_data))

            ax_obj.clear()
            ax_obj.set_facecolor("white")

            if even:
                ys = list(range(len(row_data)))
            else:
                ys = [row["energy"] for row in row_data]

            if even:
                arrow_fs = 11 if count_mos <= 12 else 9
                text_fs = 8 if count_mos <= 12 else 7
                max_lines = 3 if count_mos <= 8 else 2
            else:
                arrow_fs = 10
                text_fs = 7
                max_lines = 2

            bar_left = -0.18
            bar_right = 0.18
            x_mid = 0.0
            energy_x = 0.30
            col_spacing = 0.92
            left_anchor = bar_left - 0.78
            right_anchor = bar_right + 0.98

            left_centers: Dict[str, float] = {}
            right_centers: Dict[str, float] = {}

            if left_targets:
                start_left = left_anchor - col_spacing * (len(left_targets) - 1)
                for idx, target in enumerate(left_targets):
                    left_centers[target] = start_left + idx * col_spacing

            if right_targets:
                for idx, target in enumerate(right_targets):
                    right_centers[target] = right_anchor + idx * col_spacing

            if even:
                y_lo = -1
                y_hi = len(row_data)
            else:
                pad = (max(ys) - min(ys)) * 0.05 if len(ys) > 1 else 0.5
                y_lo = min(ys) - pad
                y_hi = max(ys) + pad

            for idx in range(len(right_targets) - 1):
                x_sep = 0.5 * (
                    right_centers[right_targets[idx]] +
                    right_centers[right_targets[idx + 1]]
                )
                ax_obj.plot(
                    [x_sep, x_sep], [y_lo, y_hi],
                    color="#777777", linewidth=1.0, zorder=1
                )

            for row, y in zip(row_data, ys):
                mi = row["mo"]
                occ = row["occ"]
                per_target = row["by_target"]

                ax_obj.plot(
                    [bar_left, bar_right], [y, y],
                    color="black", linewidth=2.2,
                    solid_capstyle="butt", zorder=3,
                )

                if occ > 0.05:
                    n_e = max(0, min(2, int(round(occ))))
                    if n_e == 1:
                        ax_obj.text(
                            x_mid, y, "\u2191",
                            ha="center", va="center",
                            fontsize=arrow_fs, color="black",
                            zorder=5, fontweight="bold",
                        )
                    elif n_e == 2:
                        ax_obj.text(
                            x_mid - 0.06, y, "\u2191",
                            ha="center", va="center",
                            fontsize=arrow_fs, color="black",
                            zorder=5, fontweight="bold",
                        )
                        ax_obj.text(
                            x_mid + 0.06, y, "\u2193",
                            ha="center", va="center",
                            fontsize=arrow_fs, color="black",
                            zorder=5, fontweight="bold",
                        )

                frontier = _mo_frontier_label(mi, homo_idx)
                ax_obj.text(
                    bar_left - 0.08, y, f"MO {mi} ({frontier})",
                    ha="right", va="center", fontsize=text_fs
                )

                energy_txt = f"{row['energy']:.3f} {unit}" if even else f"{row['energy']:.3f}"
                ax_obj.text(
                    energy_x, y, energy_txt,
                    ha="left", va="center", fontsize=text_fs
                )

                for target in left_targets:
                    txt = _format_target_entries(per_target.get(target, []), max_lines=max_lines)
                    if txt:
                        ax_obj.text(
                            left_centers[target], y, txt,
                            ha="center", va="center",
                            fontsize=text_fs, color="#222222",
                            linespacing=1.15,
                        )

                for target in right_targets:
                    txt = _format_target_entries(per_target.get(target, []), max_lines=max_lines)
                    if txt:
                        ax_obj.text(
                            right_centers[target], y, txt,
                            ha="center", va="center",
                            fontsize=text_fs, color="#222222",
                            linespacing=1.15,
                        )

            header_transform = ax_obj.get_xaxis_transform()
            for target in left_targets:
                ax_obj.text(
                    left_centers[target], 1.02, target,
                    transform=header_transform,
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold"
                )
            for target in right_targets:
                ax_obj.text(
                    right_centers[target], 1.02, target,
                    transform=header_transform,
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold"
                )
            if left_targets:
                ax_obj.text(
                    np.mean([left_centers[t] for t in left_targets]), 1.08, "Metal",
                    transform=header_transform,
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold"
                )
            if right_targets:
                ax_obj.text(
                    np.mean([right_centers[t] for t in right_targets]), 1.08, "Ligands",
                    transform=header_transform,
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold"
                )

            x_candidates = [bar_left - 0.45, energy_x + 0.30]
            x_candidates.extend(left_centers.values())
            x_candidates.extend(right_centers.values())
            x_min = min(x_candidates) - 0.55
            x_max = max(x_candidates) + 0.55
            ax_obj.set_xlim(x_min, x_max)

            if even:
                ax_obj.set_ylim(y_lo, y_hi)
                ax_obj.set_yticks([])
                ax_obj.set_xlabel("Classic MO diagram (evenly spaced levels)")
            else:
                ax_obj.set_ylim(y_lo, y_hi)
                ax_obj.set_ylabel(f"Energy ({unit})")
                ax_obj.set_yticks([])

            ax_obj.set_xticks([])
            for spine in ("top", "right", "bottom"):
                ax_obj.spines[spine].set_visible(False)
            if even:
                ax_obj.spines["left"].set_visible(False)
            else:
                ax_obj.spines["left"].set_position(("outward", 8))

            ax_obj.set_title(
                f"MO energy diagram - {payload['mo_count']} MOs, "
                f"{len(all_targets)} targets"
                + (" [Re-oriented]" if self._reorient_R is not None else ""),
                fontsize=11,
            )

            fig_obj.tight_layout(pad=1.4)
            canvas_obj.draw_idle()

        def _save_figure(fig_obj, dialog_title: str = "Save MO diagram") -> None:
            fp = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("All", "*.*")],
                title=dialog_title,
            )
            if not fp:
                return
            try:
                fig_obj.savefig(fp, dpi=220, bbox_inches="tight")
                status.config(text=f"Saved to {fp}", fg="#006600")
            except Exception as exc:
                status.config(text=f"Save failed: {exc}", fg="#993300")

        def _capture_state() -> dict:
            return {
                "version": 1,
                "source_orca_out": getattr(self, "_orca_mo_path", None),
                "from_mo": int(from_var.get()),
                "to_mo": int(to_var.get()),
                "mo_list_text": mo_list_var.get(),
                "selected_mos": _get_selected_mos(),
                "grouping": group_var.get(),
                "target_entry": target_entry_var.get(),
                "checked_targets": [k for k, v in target_check_vars.items() if v.get()],
                "orbital_mode": mode_var.get(),
                "spacing_mode": spacing_var.get(),
                "min_pct": float(min_pct_var.get()),
                "unit": unit_var.get(),
                "selected_contributors": _get_selected_contributors(expand_all=False),
                "auto_build": bool(last_plot_payload),
                "window_geometry": win.geometry(),
            }

        def _apply_state(state: dict) -> None:
            grouping = state.get("grouping", "element")
            if grouping not in ("element", "atom"):
                grouping = "element"
            group_var.set(grouping)
            _rebuild_target_checks()

            mode = state.get("orbital_mode", "ao")
            if mode not in ("ang", "ao"):
                mode = "ao"
            mode_var.set(mode)
            _on_orbital_mode_change()

            spacing = state.get("spacing_mode", "even")
            spacing_var.set("even" if spacing != "true" else "true")

            try:
                min_pct_var.set(float(state.get("min_pct", 1.0)))
            except Exception:
                min_pct_var.set(1.0)

            unit = state.get("unit", "eV")
            unit_var.set(unit if unit in ("eV", "Eh") else "eV")

            lo = max(0, min(nmo - 1, int(state.get("from_mo", max(0, homo_idx - 5)))))
            hi = max(0, min(nmo - 1, int(state.get("to_mo", min(nmo - 1, homo_idx + 5)))))
            from_var.set(lo)
            to_var.set(hi)

            mo_list_var.set(str(state.get("mo_list_text", "")))
            saved_mos = [
                int(mo_i) for mo_i in state.get("selected_mos", [])
                if isinstance(mo_i, int) or (isinstance(mo_i, str) and str(mo_i).isdigit())
            ]
            saved_mos = [mo_i for mo_i in saved_mos if 0 <= mo_i < nmo]
            _populate_mo_checks(selected_mos=saved_mos)

            target_entry_var.set(str(state.get("target_entry", "")))
            checked_targets = set(str(k) for k in state.get("checked_targets", []))
            for key, var in target_check_vars.items():
                var.set(key in checked_targets)

            labels = _refresh_contributor_list(select_all=False, keep_selection=False)
            selected_contributors = set(str(k) for k in state.get("selected_contributors", []))
            contrib_lb.selection_clear(0, tk.END)
            if labels:
                if selected_contributors:
                    for idx, label in enumerate(labels):
                        if label in selected_contributors:
                            contrib_lb.selection_set(idx)
                if not contrib_lb.curselection():
                    contrib_lb.selection_set(0, tk.END)

            geom = state.get("window_geometry")
            if isinstance(geom, str) and "x" in geom:
                try:
                    win.geometry(geom)
                except Exception:
                    pass

            if state.get("auto_build", True):
                _build()
            else:
                status.config(text="Loaded diagram state.", fg="#006600")

        def _save_state():
            fp = filedialog.asksaveasfilename(
                defaultextension=".mo_diagram.json",
                filetypes=[("MO diagram state", "*.mo_diagram.json"), ("JSON", "*.json"), ("All", "*.*")],
                title="Save MO diagram state",
            )
            if not fp:
                return
            try:
                Path(fp).write_text(json.dumps(_capture_state(), indent=2), encoding="utf-8")
                status.config(text=f"Saved diagram state to {fp}", fg="#006600")
            except Exception as exc:
                status.config(text=f"State save failed: {exc}", fg="#993300")

        def _load_state():
            fp = filedialog.askopenfilename(
                title="Load MO diagram state",
                filetypes=[("MO diagram state", "*.mo_diagram.json *.json"), ("All", "*.*")],
            )
            if not fp:
                return
            try:
                state = json.loads(Path(fp).read_text(encoding="utf-8"))
            except Exception as exc:
                messagebox.showerror("Load Diagram State", f"Could not read state file:\n{exc}")
                return

            source_path = state.get("source_orca_out")
            current_path = getattr(self, "_orca_mo_path", None)
            if source_path and current_path != source_path:
                if not Path(source_path).exists():
                    messagebox.showerror(
                        "Load Diagram State",
                        f"Saved source file not found:\n{source_path}"
                    )
                    return

                self._pending_mo_diagram_state = state
                self._load_mo_list_from_path(
                    source_path,
                    show_help_dialog=True,
                    status_prefix="Loading saved diagram",
                )
                win.destroy()

                def _wait_for_saved_source(attempt: int = 0):
                    if getattr(self, "_orca_mo_path", None) == source_path and getattr(self, "_orca_mo_data", None) is not None:
                        self._show_mo_diagram_picker()
                        return
                    if attempt >= 150:
                        self._pending_mo_diagram_state = None
                        messagebox.showerror(
                            "Load Diagram State",
                            f"Timed out loading saved source file:\n{source_path}"
                        )
                        return
                    self.after(200, lambda: _wait_for_saved_source(attempt + 1))

                self.after(250, _wait_for_saved_source)
                return

            _apply_state(state)

        def _open_large_view():
            nonlocal last_plot_payload
            if not last_plot_payload:
                messagebox.showinfo("MO Diagram", "Build the diagram first, then pop it out.")
                return

            pop = tk.Toplevel(win)
            pop.title("MO Energy Diagram - Large View")
            _clamp_geometry(pop, 1700, 1000, 800, 500)

            top_bar = tk.Frame(pop)
            top_bar.pack(fill=tk.X, padx=6, pady=(6, 0))

            pop_fig = Figure(figsize=(16, 9), dpi=110)
            pop_ax = pop_fig.add_subplot(111)
            pop_canvas = FigureCanvasTkAgg(pop_fig, master=pop)
            pop_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=(4, 6))
            pop_toolbar = NavigationToolbar2Tk(pop_canvas, pop)
            pop_toolbar.update()

            tk.Button(
                top_bar, text="Save PNG/SVG",
                command=lambda: _save_figure(pop_fig, dialog_title="Save large MO diagram")
            ).pack(side=tk.LEFT, padx=2)
            tk.Button(top_bar, text="Close", command=pop.destroy).pack(side=tk.RIGHT, padx=2)

            view = {"window": pop, "fig": pop_fig, "ax": pop_ax, "canvas": pop_canvas}
            popup_views.append(view)

            def _remove_popup(_event=None, popup_view=view):
                if _event is not None and _event.widget is not popup_view["window"]:
                    return
                if popup_view in popup_views:
                    popup_views.remove(popup_view)

            pop.bind("<Destroy>", _remove_popup)
            _render_plot(last_plot_payload, pop_fig, pop_ax, pop_canvas)

        def _build():
            nonlocal last_plot_payload
            picked_mos = _get_selected_mos()
            if not picked_mos:
                status.config(text="No MOs ticked.", fg="#993300")
                return

            labels = _refresh_contributor_list(select_all=False, keep_selection=True)
            if not labels:
                status.config(text="No contributors available to plot.", fg="#993300")
                return

            picked_contributors = _get_selected_contributors(expand_all=True)
            if not picked_contributors:
                status.config(text="Select at least one contributor from the list.", fg="#993300")
                return

            unit = unit_var.get()
            even = (spacing_var.get() == "even")

            status.config(
                text=(
                    f"Computing {len(picked_mos)} MOs x "
                    f"{len(picked_contributors)} selected contributors..."
                ),
                fg="gray",
            )
            win.update_idletasks()

            target_labels: List[str] = []
            seen_targets = set()
            for label in picked_contributors:
                target = label.split("-", 1)[0]
                if target not in seen_targets:
                    seen_targets.add(target)
                    target_labels.append(target)

            left_targets = [target for target in target_labels if _is_metal_target(target)]
            right_targets = [target for target in target_labels if target not in left_targets]
            if not left_targets and target_labels:
                right_targets = list(target_labels)

            rows: List[dict] = []
            all_seg_keys = list(picked_contributors)

            for mi in picked_mos:
                comp = _get_comp(mi)
                ene_eh = energies[mi]
                ene = ene_eh * 27.2114 if unit == "eV" else ene_eh
                per_target: Dict[str, List[Tuple[str, float]]] = {}
                target_cache: Dict[str, Dict[str, float]] = {}

                for label in picked_contributors:
                    target, orb = label.split("-", 1)
                    if target not in target_cache:
                        target_cache[target] = _target_component_values(comp, target)
                    pct = float(target_cache[target].get(orb, 0.0))
                    if pct > 0.0:
                        per_target.setdefault(target, []).append((orb, pct))

                for target in per_target:
                    per_target[target].sort(key=lambda item: (-item[1], item[0]))

                rows.append({
                    "mo": mi,
                    "energy": ene,
                    "occ": occs[mi],
                    "by_target": per_target,
                })

            payload = {
                "rows": rows,
                "unit": unit,
                "even": even,
                "left_targets": left_targets,
                "right_targets": right_targets,
                "mo_count": len(picked_mos),
            }
            last_plot_payload = payload

            _render_plot(payload, fig, ax, canvas)
            for popup_view in list(popup_views):
                if popup_view["window"].winfo_exists():
                    _render_plot(payload, popup_view["fig"], popup_view["ax"], popup_view["canvas"])

            status.config(
                text=(
                    f"Built MO diagram: {len(picked_mos)} MOs, "
                    f"{len(target_labels)} targets, "
                    f"{len(all_seg_keys)} contributor channels  "
                    f"|  HOMO = MO {homo_idx}"
                ),
                fg="#006600",
            )

        def _on_group_change():
            _rebuild_target_checks()
            contrib_lb.delete(0, tk.END)
            contrib_status.config(
                text="Target mode changed. Refresh the contributor list.",
                fg="gray",
            )

        def _on_orbital_mode_change():
            contrib_lb.delete(0, tk.END)
            contrib_status.config(
                text="Orbital mode changed. Refresh the contributor list.",
                fg="gray",
            )

        tk.Button(rng_row, text="Apply range", font=("", 8),
                  command=lambda: _populate_mo_checks()).pack(side=tk.LEFT, padx=(8, 4))
        tk.Button(rng_row, text="Apply list", font=("", 8),
                  command=_apply_mo_list).pack(side=tk.LEFT, padx=1)
        tk.Button(rng_row, text="All", font=("", 7), width=4,
                  command=lambda: [v.set(True) for v in mo_check_vars.values()]
                  ).pack(side=tk.LEFT, padx=1)
        tk.Button(rng_row, text="None", font=("", 7), width=4,
                  command=lambda: [v.set(False) for v in mo_check_vars.values()]
                  ).pack(side=tk.LEFT, padx=1)

        tk.Button(target_btn_row, text="All shown", font=("", 8),
                  command=lambda: [v.set(True) for v in target_check_vars.values()]
                  ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(target_btn_row, text="None shown", font=("", 8),
                  command=lambda: [v.set(False) for v in target_check_vars.values()]
                  ).pack(side=tk.LEFT)

        tk.Label(opts_frame, text="Contributor grouping:", font=("", 8, "bold")
                 ).pack(anchor="w")
        tk.Radiobutton(
            opts_frame,
            text="By element (Ni-d, C-px, Cl-p)",
            variable=group_var,
            value="element",
            command=_on_group_change,
            font=("", 8),
        ).pack(anchor="w")
        tk.Radiobutton(
            opts_frame,
            text="By atom (Ni1-d, C12-px, Cl7-p)",
            variable=group_var,
            value="atom",
            command=_on_group_change,
            font=("", 8),
        ).pack(anchor="w")
        ttk.Separator(opts_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        tk.Label(opts_frame, text="Orbital resolution:", font=("", 8, "bold")
                 ).pack(anchor="w")
        tk.Radiobutton(
            opts_frame,
            text="Angular momentum (s / p / d / f)",
            variable=mode_var,
            value="ang",
            command=_on_orbital_mode_change,
            font=("", 8),
        ).pack(anchor="w")
        tk.Radiobutton(
            opts_frame,
            text="Specific orbital labels (px, py, dz2, ...)",
            variable=mode_var,
            value="ao",
            command=_on_orbital_mode_change,
            font=("", 8),
        ).pack(anchor="w")
        ttk.Separator(opts_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        tk.Radiobutton(opts_frame, text="True energy spacing",
                       variable=spacing_var, value="true",
                       font=("", 8)).pack(anchor="w")
        tk.Radiobutton(opts_frame, text="Evenly spaced levels",
                       variable=spacing_var, value="even",
                       font=("", 8)).pack(anchor="w")
        ttk.Separator(opts_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        mp_row = tk.Frame(opts_frame)
        mp_row.pack(anchor="w", fill=tk.X)
        tk.Label(mp_row, text="Min % in list:", font=("", 8)).pack(side=tk.LEFT)
        ttk.Spinbox(mp_row, textvariable=min_pct_var, from_=0.0, to=50.0,
                    increment=0.5, width=5).pack(side=tk.LEFT, padx=2)
        u_row = tk.Frame(opts_frame)
        u_row.pack(anchor="w", fill=tk.X, pady=(4, 0))
        tk.Label(u_row, text="Energy unit:", font=("", 8)).pack(side=tk.LEFT)
        ttk.Combobox(u_row, textvariable=unit_var, values=("eV", "Eh"),
                     width=4, state="readonly").pack(side=tk.LEFT, padx=2)
        tk.Button(opts_frame, text="Refresh contributor list",
                  bg="#1a6e3c", fg="white", font=("", 8, "bold"),
                  command=lambda: _refresh_contributor_list(select_all=True, keep_selection=False)
                  ).pack(anchor="w", fill=tk.X, pady=(8, 0))

        tk.Button(contrib_btn_row, text="All", font=("", 8), width=5,
                  command=lambda: contrib_lb.selection_set(0, tk.END)
                  ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(contrib_btn_row, text="None", font=("", 8), width=5,
                  command=lambda: contrib_lb.selection_clear(0, tk.END)
                  ).pack(side=tk.LEFT)

        btn_row = tk.Frame(_win_body2)
        btn_row.pack(fill=tk.X, padx=6, pady=(0, 6), before=status)
        tk.Button(btn_row, text="Build diagram", bg="#1a3c6e", fg="white",
                  font=("", 9, "bold"), command=_build
                  ).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Pop Out Larger", command=_open_large_view
                  ).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Save State", command=_save_state
                  ).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Load State", command=_load_state
                  ).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Save PNG/SVG", command=lambda: _save_figure(fig)
                  ).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Close", command=win.destroy
                  ).pack(side=tk.RIGHT, padx=2)

        _populate_mo_checks()
        _rebuild_target_checks()
        if pending_state is not None:
            win.after(100, lambda state=pending_state: _apply_state(state))

    def _show_mo_diagram(self):
        """
        Open an MO energy-level diagram window. User picks an MO range,
        unticks unwanted MOs, picks atoms (typed or checked), and a
        contributor mode (atom + ang-momentum, or atom + specific AO).
        Each MO is drawn as a horizontal bar at its energy, segmented by
        contributor and shaded according to electron occupation.
        """
        if not hasattr(self, "_orca_mo_data") or self._orca_mo_data is None:
            messagebox.showinfo("MO Diagram",
                                "Load an ORCA .out file with MOs first "
                                "(use Load Output… on the Isosurface tab).")
            return

        mo_data  = self._orca_mo_data
        energies = mo_data["energies"]
        occs     = mo_data["occs"]
        nmo      = mo_data["coeffs"].shape[1]
        homo_idx = int(np.max(np.where(occs > 0.5))) if np.any(occs > 0.5) else 0

        # Collect all atom keys present in the basis (in atom-index order)
        seen = []
        for (a_idx, a_el, _, _) in mo_data["ao_labels"]:
            key = f"{a_el}{a_idx + 1}"
            if key not in seen:
                seen.append(key)
        all_atom_keys = seen

        win = tk.Toplevel(self)
        win.title("MO Energy Diagram")
        _clamp_geometry(win, 1250, 780, 700, 500)

        # Wrap everything in a scrollable frame so controls survive small screens
        _sf = _ScrollableFrame(win)
        _sf.pack(fill=tk.BOTH, expand=True)
        _win_body = _sf.interior

        # ── Top: MO range + checklist of MOs ────────────────────────────
        top = tk.LabelFrame(_win_body, text="1. MO selection", padx=6, pady=4)
        top.pack(fill=tk.X, padx=6, pady=(6, 3))

        rng_row = tk.Frame(top); rng_row.pack(fill=tk.X)
        tk.Label(rng_row, text="From MO#:").pack(side=tk.LEFT)
        from_var = tk.IntVar(value=max(0, homo_idx - 5))
        ttk.Spinbox(rng_row, textvariable=from_var, from_=0, to=nmo - 1,
                    width=6).pack(side=tk.LEFT, padx=(2, 8))
        tk.Label(rng_row, text="To MO#:").pack(side=tk.LEFT)
        to_var = tk.IntVar(value=min(nmo - 1, homo_idx + 5))
        ttk.Spinbox(rng_row, textvariable=to_var, from_=0, to=nmo - 1,
                    width=6).pack(side=tk.LEFT, padx=(2, 8))

        # Manual MO entry row
        manual_row = tk.Frame(top); manual_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(manual_row, text="Manual MOs:", font=("", 8)
                 ).pack(side=tk.LEFT)
        mo_entry_var = tk.StringVar(value="")
        tk.Entry(manual_row, textvariable=mo_entry_var, width=40,
                 font=("", 8)).pack(side=tk.LEFT, padx=(4, 4), fill=tk.X, expand=True)
        tk.Label(manual_row,
                 text="(comma/space sep, ranges ok: 100-110, 115, 120)",
                 font=("", 7), fg="#666666").pack(side=tk.LEFT)

        # MO checklist (scrollable)
        mo_list_frame = tk.Frame(top)
        mo_list_frame.pack(fill=tk.X, pady=(4, 0))
        mo_canvas = tk.Canvas(mo_list_frame, height=80, highlightthickness=0)
        mo_sb = ttk.Scrollbar(mo_list_frame, orient=tk.HORIZONTAL,
                              command=mo_canvas.xview)
        mo_inner = tk.Frame(mo_canvas)
        mo_canvas.create_window((0, 0), window=mo_inner, anchor="nw")
        mo_inner.bind("<Configure>",
            lambda e: mo_canvas.config(scrollregion=mo_canvas.bbox("all")))
        mo_canvas.config(xscrollcommand=mo_sb.set)
        mo_canvas.pack(fill=tk.X, expand=True)
        mo_sb.pack(fill=tk.X)

        mo_check_vars: Dict[int, tk.BooleanVar] = {}

        def _populate_mo_checks():
            for w in mo_inner.winfo_children():
                w.destroy()
            mo_check_vars.clear()
            lo = max(0, from_var.get())
            hi = min(nmo - 1, to_var.get())
            if lo > hi:
                lo, hi = hi, lo
            for i in range(lo, hi + 1):
                v = tk.BooleanVar(value=True)
                mo_check_vars[i] = v
                delta = i - homo_idx
                if   delta == 0: tag = "HOMO"
                elif delta == 1: tag = "LUMO"
                elif delta < 0:  tag = f"H{delta}"
                else:            tag = f"L+{delta-1}"
                txt = f"{i} ({tag})"
                cb = tk.Checkbutton(mo_inner, text=txt, variable=v,
                                    font=("", 8))
                cb.pack(side=tk.LEFT)
            mo_inner.update_idletasks()
            mo_canvas.config(scrollregion=mo_canvas.bbox("all"))

        tk.Button(rng_row, text="Apply range", font=("", 8),
                  command=_populate_mo_checks).pack(side=tk.LEFT, padx=(8, 4))
        tk.Button(rng_row, text="All", font=("", 7), width=4,
                  command=lambda: [v.set(True) for v in mo_check_vars.values()]
                  ).pack(side=tk.LEFT, padx=1)
        tk.Button(rng_row, text="None", font=("", 7), width=4,
                  command=lambda: [v.set(False) for v in mo_check_vars.values()]
                  ).pack(side=tk.LEFT, padx=1)

        _populate_mo_checks()

        # ── Middle: atoms + contributor mode + style ────────────────────
        mid = tk.Frame(_win_body); mid.pack(fill=tk.X, padx=6, pady=3)

        atoms_frame = tk.LabelFrame(mid, text="2. Atoms (type or check)",
                                    padx=6, pady=4)
        atoms_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        tk.Label(atoms_frame, text="Atom keys (comma sep, e.g. 1Ni, 4P, 7Cl):",
                 font=("", 8)).pack(anchor="w")
        atoms_entry_var = tk.StringVar(value="")
        tk.Entry(atoms_frame, textvariable=atoms_entry_var
                 ).pack(fill=tk.X, pady=(0, 4))

        # Scrollable atom checklist
        ac_frame = tk.Frame(atoms_frame); ac_frame.pack(fill=tk.X)
        ac_canvas = tk.Canvas(ac_frame, height=80, highlightthickness=0)
        ac_sb = ttk.Scrollbar(ac_frame, orient=tk.HORIZONTAL,
                              command=ac_canvas.xview)
        ac_inner = tk.Frame(ac_canvas)
        ac_canvas.create_window((0, 0), window=ac_inner, anchor="nw")
        ac_inner.bind("<Configure>",
            lambda e: ac_canvas.config(scrollregion=ac_canvas.bbox("all")))
        ac_canvas.config(xscrollcommand=ac_sb.set)
        ac_canvas.pack(fill=tk.X)
        ac_sb.pack(fill=tk.X)

        atom_check_vars: Dict[str, tk.BooleanVar] = {}
        for ak in all_atom_keys:
            v = tk.BooleanVar(value=False)
            atom_check_vars[ak] = v
            tk.Checkbutton(ac_inner, text=ak, variable=v,
                           font=("", 8)).pack(side=tk.LEFT)

        opts_frame = tk.LabelFrame(mid, text="3. Contributor mode & style",
                                   padx=6, pady=4)
        opts_frame.pack(side=tk.LEFT, fill=tk.Y)

        mode_var = tk.StringVar(value="ang")  # "ang" or "ao"
        tk.Radiobutton(opts_frame, text="Atom + angular momentum (Ni-d, P-p)",
                       variable=mode_var, value="ang",
                       font=("", 8)).pack(anchor="w")
        tk.Radiobutton(opts_frame, text="Atom + specific AO (Ni-dx²-y², P-pz)",
                       variable=mode_var, value="ao",
                       font=("", 8)).pack(anchor="w")

        ttk.Separator(opts_frame, orient=tk.HORIZONTAL
                      ).pack(fill=tk.X, pady=4)

        spacing_var = tk.StringVar(value="true")
        tk.Radiobutton(opts_frame, text="True energy spacing",
                       variable=spacing_var, value="true",
                       font=("", 8)).pack(anchor="w")
        tk.Radiobutton(opts_frame, text="Evenly spaced (gaps)",
                       variable=spacing_var, value="even",
                       font=("", 8)).pack(anchor="w")

        ttk.Separator(opts_frame, orient=tk.HORIZONTAL
                      ).pack(fill=tk.X, pady=4)

        min_pct_var = tk.DoubleVar(value=2.0)
        mp_row = tk.Frame(opts_frame); mp_row.pack(anchor="w", fill=tk.X)
        tk.Label(mp_row, text="Min %:", font=("", 8)).pack(side=tk.LEFT)
        ttk.Spinbox(mp_row, textvariable=min_pct_var, from_=0.0, to=50.0,
                    increment=0.5, width=5).pack(side=tk.LEFT, padx=2)

        unit_var = tk.StringVar(value="eV")
        u_row = tk.Frame(opts_frame); u_row.pack(anchor="w", fill=tk.X)
        tk.Label(u_row, text="Energy unit:", font=("", 8)).pack(side=tk.LEFT)
        ttk.Combobox(u_row, textvariable=unit_var, values=("eV", "Eh"),
                     width=4, state="readonly").pack(side=tk.LEFT, padx=2)

        # ── Plot canvas ─────────────────────────────────────────────────
        plot_frame = tk.LabelFrame(_win_body, text="4. Diagram", padx=4, pady=4)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(3, 6))

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()

        status = tk.Label(_win_body, text="Pick atoms, click Build diagram.",
                          fg="gray", anchor="w")
        status.pack(fill=tk.X, padx=6, pady=(0, 4))

        def _gather_atoms():
            """Combine typed entry + checked boxes; preserve all_atom_keys order."""
            picked = set()
            raw = atoms_entry_var.get().strip()
            if raw:
                for tok in raw.replace(";", ",").split(","):
                    t = tok.strip()
                    if t and t in atom_check_vars:
                        picked.add(t)
            for ak, v in atom_check_vars.items():
                if v.get():
                    picked.add(ak)
            return [k for k in all_atom_keys if k in picked]

        def _color_for(label, idx, total):
            import matplotlib.cm as cm
            cmap = cm.get_cmap("tab20", max(20, total))
            return cmap(idx % cmap.N)

        def _parse_manual_mos():
            """Parse the manual MO entry: supports comma/space sep, ranges (e.g. 100-110, 115)."""
            raw = mo_entry_var.get().strip()
            if not raw:
                return []
            result = set()
            # Split on commas, semicolons, and whitespace
            for tok in raw.replace(",", " ").replace(";", " ").split():
                tok = tok.strip()
                if not tok:
                    continue
                if "-" in tok and not tok.startswith("-"):
                    # Range like "100-110"
                    parts = tok.split("-", 1)
                    try:
                        lo, hi = int(parts[0]), int(parts[1])
                        for i in range(max(0, lo), min(nmo, hi + 1)):
                            result.add(i)
                    except ValueError:
                        pass
                else:
                    try:
                        v = int(tok)
                        if 0 <= v < nmo:
                            result.add(v)
                    except ValueError:
                        pass
            return sorted(result)

        def _build():
            # Merge: ticked checkboxes + manually typed MOs
            from_checks = sorted([i for i, v in mo_check_vars.items() if v.get()])
            from_manual = _parse_manual_mos()
            picked_mos = sorted(set(from_checks) | set(from_manual))
            if not picked_mos:
                status.config(text="No MOs selected — tick checkboxes or type MO numbers.", fg="#993300")
                return
            picked_atoms = _gather_atoms()
            if not picked_atoms:
                status.config(text="Pick at least one atom (type or check).",
                              fg="#993300")
                return

            mode    = mode_var.get()
            unit    = unit_var.get()
            min_pct = float(min_pct_var.get())
            even    = (spacing_var.get() == "even")

            status.config(text=f"Computing {len(picked_mos)} MOs × "
                               f"{len(picked_atoms)} atoms...", fg="gray")
            win.update_idletasks()

            # Compute compositions
            mo_rows = []  # list of (mo_i, energy_disp, occ, segments_dict)
            all_seg_keys = []  # preserved order across MOs

            for mi in picked_mos:
                comp = compute_mo_composition(mo_data, mi,
                                              reorient_R=self._reorient_R)
                ene_eh = energies[mi]
                ene = ene_eh * 27.2114 if unit == "eV" else ene_eh
                segs: Dict[str, float] = {}
                for ak in picked_atoms:
                    if mode == "ang":
                        ac = comp['atom_contribs'].get(ak)
                        if not ac:
                            continue
                        for ang in ('s', 'p', 'd', 'f'):
                            pct = ac[ang]
                            if pct >= min_pct:
                                key = f"{ak}-{ang}"
                                segs[key] = pct
                                if key not in all_seg_keys:
                                    all_seg_keys.append(key)
                    else:  # specific AO
                        ao_det = comp['atom_ao_detail'].get(ak, {})
                        for ao_lbl, pct in ao_det.items():
                            if pct >= min_pct:
                                key = f"{ak}-{ao_lbl}"
                                segs[key] = pct
                                if key not in all_seg_keys:
                                    all_seg_keys.append(key)
                mo_rows.append((mi, ene, occs[mi], segs))

            if not all_seg_keys:
                status.config(
                    text=f"No contributors above {min_pct:.1f}% — lower the threshold.",
                    fg="#993300")
                ax.clear(); canvas.draw_idle()
                return

            # Color map
            colors = {k: _color_for(k, i, len(all_seg_keys))
                      for i, k in enumerate(all_seg_keys)}

            # ── Draw ───────────────────────────────────────────────
            ax.clear()

            # y positions: true energy or evenly spaced
            if even:
                ys = list(range(len(mo_rows)))
                ene_for_label = [r[1] for r in mo_rows]
            else:
                ys = [r[1] for r in mo_rows]
                ene_for_label = ys

            # Bar geometry
            if even:
                bar_h = 0.5
            else:
                # Choose a bar height ~ a fraction of the energy range
                if len(ys) >= 2:
                    span = max(ys) - min(ys)
                    bar_h = max(span / max(15, len(ys) * 1.5), 0.02)
                else:
                    bar_h = 0.05

            x_left  = 0.0
            x_right = 1.0     # bars span [0, 1]
            x_span  = x_right - x_left

            for (mi, ene, occ, segs), y in zip(mo_rows, ys):
                tot = sum(segs.values())
                # bar background (light grey: portion of MO not in picked atoms)
                ax.add_patch(plt.Rectangle((x_left, y - bar_h / 2),
                                           x_span, bar_h,
                                           facecolor="#eeeeee",
                                           edgecolor="#333333",
                                           linewidth=0.6, zorder=2))
                # Stacked colored segments — proportional to their %
                # Scale so that 100% spans the full bar (so picked sums to <=1.0)
                cursor = x_left
                for k in all_seg_keys:
                    pct = segs.get(k, 0.0)
                    if pct <= 0:
                        continue
                    w = (pct / 100.0) * x_span
                    ax.add_patch(plt.Rectangle((cursor, y - bar_h / 2),
                                               w, bar_h,
                                               facecolor=colors[k],
                                               edgecolor="none", zorder=3))
                    cursor += w

                # Electron filling: draw arrows above the bar
                # occ ~ 2 → ↑↓, occ ~ 1 → ↑, occ ~ 0 → none
                if occ > 0.05:
                    n_e = int(round(occ))
                    n_e = max(0, min(2, n_e))
                    if n_e == 1:
                        ax.text(x_left + x_span * 0.5, y, "↑",
                                ha="center", va="center",
                                fontsize=11, color="black",
                                zorder=5, fontweight="bold")
                    elif n_e == 2:
                        ax.text(x_left + x_span * 0.42, y, "↑",
                                ha="center", va="center",
                                fontsize=11, color="black",
                                zorder=5, fontweight="bold")
                        ax.text(x_left + x_span * 0.58, y, "↓",
                                ha="center", va="center",
                                fontsize=11, color="black",
                                zorder=5, fontweight="bold")

                # MO label on the left
                delta = mi - homo_idx
                if   delta == 0: tag = "HOMO"
                elif delta == 1: tag = "LUMO"
                elif delta < 0:  tag = f"H{delta}"
                else:            tag = f"L+{delta-1}"
                ax.text(x_left - 0.02, y, f"MO {mi} ({tag})",
                        ha="right", va="center", fontsize=8)

                # Energy label on the right
                if even:
                    ax.text(x_right + 0.02, y, f"{ene:.3f} {unit}",
                            ha="left", va="center", fontsize=8)
                else:
                    ax.text(x_right + 0.02, y, f"{ene:.3f}",
                            ha="left", va="center", fontsize=8)

                # Coverage label (sum of selected % at the far right edge)
                ax.text(x_left + x_span + 0.16, y,
                        f"({tot:.0f}%)", ha="left", va="center",
                        fontsize=7, color="#555555")

            # Axes cosmetics
            ax.set_xlim(x_left - 0.25, x_right + 0.32)
            if even:
                ax.set_ylim(-1, len(mo_rows))
                ax.invert_yaxis()
                ax.set_yticks([])
                ax.set_xlabel("(evenly spaced)")
            else:
                # Add some padding above/below
                pad = (max(ys) - min(ys)) * 0.05 if len(ys) > 1 else 0.5
                ax.set_ylim(min(ys) - pad, max(ys) + pad)
                ax.set_ylabel(f"Energy ({unit})")
                ax.set_yticks([])
            ax.set_xticks([])
            for sp in ("top", "right", "bottom"):
                ax.spines[sp].set_visible(False)
            if not even:
                ax.spines["left"].set_position(("outward", 8))
            else:
                ax.spines["left"].set_visible(False)

            ax.set_title(
                f"MO energy diagram — {len(picked_mos)} MOs, "
                f"{len(picked_atoms)} atoms"
                + (" [Re-oriented]" if self._reorient_R is not None else ""),
                fontsize=11)

            # Legend
            from matplotlib.patches import Patch
            handles = [Patch(facecolor=colors[k], edgecolor="none", label=k)
                       for k in all_seg_keys]
            if handles:
                ax.legend(handles=handles, loc="center left",
                          bbox_to_anchor=(1.02, 0.5),
                          fontsize=7, frameon=True,
                          title="Contributors", title_fontsize=8)

            fig.tight_layout()
            canvas.draw_idle()

            status.config(
                text=(f"Built diagram: {len(picked_mos)} MOs, "
                      f"{len(picked_atoms)} atoms, {len(all_seg_keys)} segment types  "
                      f"|  HOMO = MO {homo_idx}"),
                fg="#006600")

        def _save_png():
            from tkinter import filedialog
            fp = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("All", "*.*")],
                title="Save MO diagram")
            if not fp:
                return
            try:
                fig.savefig(fp, dpi=200, bbox_inches="tight")
                status.config(text=f"Saved to {fp}", fg="#006600")
            except Exception as exc:
                status.config(text=f"Save failed: {exc}", fg="#993300")

        # ── Action buttons ──────────────────────────────────────────────
        btn_row = tk.Frame(_win_body); btn_row.pack(fill=tk.X, padx=6, pady=(0, 6))
        tk.Button(btn_row, text="Build diagram", bg="#1a3c6e", fg="white",
                  font=("", 9, "bold"), command=_build
                  ).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Save PNG/SVG", command=_save_png
                  ).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Close", command=win.destroy
                  ).pack(side=tk.RIGHT, padx=2)

    def _rerender_last_iso(self, _event=None):
        """Re-render the current cube quickly without rebuilding the MO grid."""
        cube = getattr(self, "_last_rendered_cube", None)
        if cube is not None:
            self._do_render(cube)
        else:
            self._render_iso_tab()

    def _get_iso_mesh_cache_token(self, cube: dict) -> int:
        token = cube.get("_iso_mesh_cache_token")
        if token is None:
            self._iso_mesh_cache_token_counter += 1
            token = self._iso_mesh_cache_token_counter
            cube["_iso_mesh_cache_token"] = token
        return int(token)

    def _remember_iso_mesh_cache_key(self, key: Tuple[int, float, int]) -> None:
        if key in self._iso_mesh_cache_order:
            self._iso_mesh_cache_order.remove(key)
        self._iso_mesh_cache_order.append(key)
        while len(self._iso_mesh_cache_order) > 24:
            old = self._iso_mesh_cache_order.pop(0)
            self._iso_mesh_cache.pop(old, None)

    def _get_cached_iso_mesh(self, cube: dict, isovalue: float) -> Optional[dict]:
        stride = _ISO_QUALITY_STRIDE.get(self._iso_quality.get(), 1)
        key = (self._get_iso_mesh_cache_token(cube), round(float(isovalue), 6), stride)
        if key in self._iso_mesh_cache:
            self._remember_iso_mesh_cache_key(key)
            return self._iso_mesh_cache[key]

        mesh = _extract_isosurface_mesh(cube, isovalue, stride=stride)
        self._iso_mesh_cache[key] = mesh
        self._remember_iso_mesh_cache_key(key)
        return mesh

    def _do_render(self, cube: dict, ax=None, canvas=None, fig=None):
        """Render the isosurface into the given axes (or the embedded one)."""
        isovalue = self._iso_val_var.get()
        alpha    = self._iso_alpha.get()
        pos_col  = self._iso_pos_color.get()
        neg_col  = self._iso_neg_color.get()
        bg_col   = self._iso_bg_color.get()
        atom_scale = self._iso_atom_scale.get()
        bond_width = self._iso_bond_width.get()
        mol_alpha = self._iso_mol_alpha.get()
        style = self._style_3d.get()
        surface_style = self._iso_surface_style.get()
        show_edges = self._iso_surface_edges.get()

        target_ax     = ax     if ax     is not None else self._ax_iso
        target_canvas = canvas if canvas is not None else self._canvas_iso
        target_fig    = fig    if fig    is not None else self._fig_iso

        target_ax.clear()
        target_fig.set_facecolor(bg_col)
        _clean_3d_ax(target_ax, bg_col)
        self._iso_status.config(text="Rendering...", fg="gray")
        self.update_idletasks()

        # Reset click-to-label state (only for embedded view)
        if ax is None:
            self._iso_atom_annotations = {}
            self._iso_atom_data = []

        rendered = 0

        # Determine bond color: light on dark, dark on light
        bg_lum = sum(int(bg_col.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) / 765.0
        bond_color = "#DDDDDD" if bg_lum < 0.5 else "#333333"
        title_color = "white" if bg_lum < 0.5 else "black"
        on_dark = bg_lum < 0.5

        xs_m = ys_m = zs_m = names_m = []
        extent_sets: List[np.ndarray] = []

        if self._iso_mol_on.get():
            ovr = self._elem_color_overrides
            all_atoms  = cube["atoms"]
            all_names  = [_ATOMIC_NUM_REV.get(a[0], "X") for a in all_atoms]
            # Filter H atoms based on global toggle + force-show set
            show_H = self._show_H.get()
            vis_mask = [show_H or el != "H" or i in self._force_show_H
                        for i, el in enumerate(all_names)]
            xs_m    = [all_atoms[i][2] for i in range(len(all_atoms)) if vis_mask[i]]
            ys_m    = [all_atoms[i][3] for i in range(len(all_atoms)) if vis_mask[i]]
            zs_m    = [all_atoms[i][4] for i in range(len(all_atoms)) if vis_mask[i]]
            names_m = [all_names[i]    for i in range(len(all_atoms)) if vis_mask[i]]
            col_m   = [mcolors.to_rgba(_ec(el, ovr)) for el in names_m]
            if style == "Spacefill":
                atom_scale_base = 280
            elif style == "Stick":
                atom_scale_base = 22
            else:
                atom_scale_base = 88
            siz_m = [ELEM_RADIUS.get(el, _DEFAULT_RADIUS) * atom_scale_base * atom_scale
                     for el in names_m]
            bond_tuples = list(zip(names_m, xs_m, ys_m, zs_m))
            if style != "Spacefill":
                line_w = bond_width * (1.15 if style == "Stick" else 1.0)
                line_alpha = 0.95 if style == "Stick" else 0.82
                for i, j in _detect_bonds(bond_tuples):
                    target_ax.plot([xs_m[i], xs_m[j]], [ys_m[i], ys_m[j]], [zs_m[i], zs_m[j]],
                                   color=bond_color, linewidth=line_w, alpha=line_alpha,
                                   solid_capstyle="round")
            sc = target_ax.scatter(xs_m, ys_m, zs_m, c=col_m, s=siz_m,
                                   depthshade=(style != "Stick"),
                                   alpha=mol_alpha, picker=True)
            sc.set_pickradius(8)
            if xs_m:
                extent_sets.append(np.column_stack((xs_m, ys_m, zs_m)))
            # Store atom data for click-to-label (embedded view only)
            if ax is None:
                self._iso_atom_data = list(zip(xs_m, ys_m, zs_m, names_m))

        if self._iso_pos_on.get():
            mesh = self._get_cached_iso_mesh(cube, +isovalue)
            if mesh is not None:
                edge_col = _surface_edge_color(pos_col, bg_col, on_dark) if show_edges or surface_style == "Mesh" else "none"
                edge_width = 0.4 if surface_style == "Mesh" else (0.18 if show_edges else 0.0)
                if _draw_isosurface_mesh(target_ax, mesh, pos_col, alpha,
                                         style=surface_style,
                                         edge_color=edge_col,
                                         edge_width=edge_width):
                    extent_sets.append(mesh["verts"])
                    rendered += 1
        if self._iso_neg_on.get():
            mesh = self._get_cached_iso_mesh(cube, -isovalue)
            if mesh is not None:
                edge_col = _surface_edge_color(neg_col, bg_col, on_dark) if show_edges or surface_style == "Mesh" else "none"
                edge_width = 0.4 if surface_style == "Mesh" else (0.18 if show_edges else 0.0)
                if _draw_isosurface_mesh(target_ax, mesh, neg_col, alpha,
                                         style=surface_style,
                                         edge_color=edge_col,
                                         edge_width=edge_width):
                    extent_sets.append(mesh["verts"])
                    rendered += 1

        if rendered == 0 and not self._iso_mol_on.get():
            dmin, dmax = cube["data"].min(), cube["data"].max()
            target_ax.text2D(
                0.5, 0.5,
                f"No isosurface at +/-{isovalue:.4f}\n"
                f"Data range: [{dmin:.4f}, {dmax:.4f}]\n"
                f"Try a smaller isovalue.",
                ha="center", va="center",
                transform=target_ax.transAxes, color=title_color, fontsize=9
            )

        if self._show_axes.get():
            if xs_m:
                _draw_axis_indicator(target_ax, np.asarray(xs_m), np.asarray(ys_m), np.asarray(zs_m))
            elif cube.get("atoms"):
                axs = np.array([[a[2], a[3], a[4]] for a in cube["atoms"]], dtype=float)
                if len(axs):
                    _draw_axis_indicator(target_ax, axs[:, 0], axs[:, 1], axs[:, 2])

        if extent_sets:
            pts = np.vstack(extent_sets)
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
            center = 0.5 * (mins + maxs)
            radius = max(float(np.max(maxs - mins)) * 0.62, 1.0)
            target_ax.set_xlim(center[0] - radius, center[0] + radius)
            target_ax.set_ylim(center[1] - radius, center[1] + radius)
            target_ax.set_zlim(center[2] - radius, center[2] + radius)
            try:
                target_ax.set_box_aspect((1, 1, 1))
            except Exception:
                pass

        # Title at top
        comment = cube.get("comment", "")[:70]
        target_ax.set_title(comment, color=title_color, fontsize=9, pad=-8)

        # Connect pick event for atom label toggle (embedded only)
        if ax is None:
            if not hasattr(self, "_iso_pick_cid") or self._iso_pick_cid is None:
                self._iso_pick_cid = self._fig_iso.canvas.mpl_connect(
                    "pick_event", self._on_iso_atom_pick)

        target_canvas.draw()
        self._last_rendered_cube = cube
        lobe_desc = []
        if self._iso_pos_on.get() and rendered > 0:
            lobe_desc.append(f"+ = {pos_col}")
        if self._iso_neg_on.get() and rendered >= (2 if self._iso_pos_on.get() else 1):
            lobe_desc.append(f"- = {neg_col}")
        lobe_str = "  ".join(lobe_desc) if lobe_desc else f"{rendered} lobe(s)"
        self._iso_status.config(
            text=(f"Done  |  isovalue +/-{isovalue:.4f}  |  {lobe_str}  |  "
                  f"{surface_style}, {self._iso_quality.get()}  |  Click atoms to label"),
            fg="#006600"
        )

    # ─── Lobe colour picker ──────────────────────────────────────────────────

    def _pick_lobe_color(self, which: str):
        """Open a colour chooser for the + or - lobe."""
        from tkinter import colorchooser
        if which == "+":
            cur = self._iso_pos_color.get()
        else:
            cur = self._iso_neg_color.get()
        result = colorchooser.askcolor(color=cur, title=f"Choose {which} lobe colour")
        if result and result[1]:
            hex_col = result[1]
            if which == "+":
                self._iso_pos_color.set(hex_col)
                self._pos_swatch.config(bg=hex_col)
            else:
                self._iso_neg_color.set(hex_col)
                self._neg_swatch.config(bg=hex_col)
            self._rerender_last_iso()

    # ─── Background colour ──────────────────────────────────────────────────

    def _pick_bg_color(self):
        """Open a colour chooser for the 3D background."""
        from tkinter import colorchooser
        cur = self._iso_bg_color.get()
        result = colorchooser.askcolor(color=cur, title="Choose background colour")
        if result and result[1]:
            self._set_bg_color(result[1])

    def _set_bg_color(self, hex_col: str):
        """Set the isosurface background colour and re-render if possible."""
        self._iso_bg_color.set(hex_col)
        self._bg_swatch.config(bg=hex_col)
        # Re-render if we have a cube
        cube = getattr(self, "_last_rendered_cube", None)
        if cube is not None:
            self._do_render(cube)

    # ─── Pop-out resizable window ────────────────────────────────────────────

    def _popout_render(self):
        """Open the last rendered isosurface in a separate resizable window
        with editable title, annotation text, font controls, and save."""
        cube = getattr(self, "_last_rendered_cube", None)
        if cube is None:
            messagebox.showinfo("Pop Out", "Render an orbital first, then pop it out.")
            return

        bg_col = self._iso_bg_color.get()

        win = tk.Toplevel(self)
        win.title(f"Orbital Viewer — {cube.get('comment', 'MO')[:60]}")
        _clamp_geometry(win, 1000, 800, 500, 400)
        win.configure(bg=bg_col)

        # ── State for text / fonts ────────────────────────────────────────
        available_fonts = sorted(set([
            "Arial", "Helvetica", "Times New Roman", "Courier New",
            "Calibri", "Cambria", "Georgia", "Verdana", "Consolas",
            "Segoe UI", "Tahoma", "Palatino Linotype", "Garamond",
            "Comic Sans MS", "Impact", "Lucida Console",
        ]))
        title_var   = tk.StringVar(value=cube.get("comment", "MO")[:80])
        annot_var   = tk.StringVar(value="")
        font_var    = tk.StringVar(value="Arial")
        fsize_var   = tk.IntVar(value=14)
        fcolor_var  = tk.StringVar(value="#FFFFFF")
        title_obj   = [None]   # mutable ref to matplotlib Text
        annot_obj   = [None]   # mutable ref to matplotlib Text

        # ── Row 1: Toolbar (actions + BG presets) ─────────────────────────
        toolbar = tk.Frame(win, bg="#2a2a2e")
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(toolbar, text="Re-render", bg="#2a4a00", fg="white",
                  font=("", 9), command=lambda: _rerender()
                  ).pack(side=tk.LEFT, padx=4, pady=2)
        tk.Button(toolbar, text="Save PNG", bg="#1a3c6e", fg="white",
                  font=("", 9), command=lambda: self._save_popout_png(pop_fig)
                  ).pack(side=tk.LEFT, padx=4, pady=2)
        tk.Button(toolbar, text="Save SVG", bg="#1a3c6e", fg="white",
                  font=("", 9), command=lambda: self._save_popout_svg(pop_fig)
                  ).pack(side=tk.LEFT, padx=4, pady=2)

        tk.Label(toolbar, text="  BG:", fg="white", bg="#2a2a2e",
                 font=("", 8)).pack(side=tk.LEFT, padx=(8, 2))
        for label, col in [("Black", "#000000"), ("Dark", "#1c1c1e"),
                           ("White", "#FFFFFF"), ("Grey", "#808080")]:
            tk.Button(toolbar, text=label, width=5, font=("", 7),
                      command=lambda c=col: (
                          self._set_bg_color(c),
                          _rerender())
                      ).pack(side=tk.LEFT, padx=1, pady=2)

        # Composition summary
        comp_lbl = tk.Label(toolbar, text="", fg="#aaaaaa", bg="#2a2a2e",
                            font=("Courier", 8), anchor="w")
        comp_lbl.pack(side=tk.LEFT, padx=(12, 0), fill=tk.X, expand=True)
        if hasattr(self, "_orca_mo_data") and self._orca_mo_data is not None:
            sel = self._mo_tree.selection()
            if sel:
                try:
                    iorb = int(sel[0])
                    comp = compute_mo_composition(self._orca_mo_data, iorb,
                                                  reorient_R=self._reorient_R)
                    top3 = comp['top_atoms'][:4]
                    parts = [f"{k}:{p:.0f}%" for k, p in top3]
                    char = f"s={comp['total_s']:.0f}% p={comp['total_p']:.0f}% d={comp['total_d']:.0f}%"
                    comp_lbl.config(text=f"{char}  |  {', '.join(parts)}")
                except Exception:
                    pass

        # ── Row 2: Title / Annotation / Font controls ─────────────────────
        text_frame = tk.Frame(win, bg="#333338")
        text_frame.pack(side=tk.TOP, fill=tk.X)

        # Title
        tk.Label(text_frame, text="Title:", fg="white", bg="#333338",
                 font=("", 9)).grid(row=0, column=0, padx=(6, 2), pady=2, sticky="e")
        title_ent = tk.Entry(text_frame, textvariable=title_var, width=40,
                             font=("", 9))
        title_ent.grid(row=0, column=1, padx=2, pady=2, sticky="ew")

        # Annotation text
        tk.Label(text_frame, text="Text:", fg="white", bg="#333338",
                 font=("", 9)).grid(row=1, column=0, padx=(6, 2), pady=2, sticky="e")
        annot_ent = tk.Entry(text_frame, textvariable=annot_var, width=40,
                             font=("", 9))
        annot_ent.grid(row=1, column=1, padx=2, pady=2, sticky="ew")

        # Font family
        tk.Label(text_frame, text="Font:", fg="white", bg="#333338",
                 font=("", 9)).grid(row=0, column=2, padx=(12, 2), pady=2, sticky="e")
        font_cb = ttk.Combobox(text_frame, textvariable=font_var,
                               values=available_fonts, width=18, state="readonly")
        font_cb.grid(row=0, column=3, padx=2, pady=2)

        # Font size
        tk.Label(text_frame, text="Size:", fg="white", bg="#333338",
                 font=("", 9)).grid(row=1, column=2, padx=(12, 2), pady=2, sticky="e")
        fsize_spin = tk.Spinbox(text_frame, from_=6, to=72, textvariable=fsize_var,
                                width=4, font=("", 9))
        fsize_spin.grid(row=1, column=3, padx=2, pady=2, sticky="w")

        # Font color
        tk.Label(text_frame, text="Color:", fg="white", bg="#333338",
                 font=("", 9)).grid(row=0, column=4, padx=(12, 2), pady=2, sticky="e")
        fc_btn = tk.Button(text_frame, text="  ", width=3,
                           bg=fcolor_var.get(),
                           command=lambda: _pick_font_color())
        fc_btn.grid(row=0, column=5, padx=2, pady=2)

        # Apply text button
        tk.Button(text_frame, text="Apply Text", bg="#4a4a00", fg="white",
                  font=("", 9, "bold"),
                  command=lambda: _apply_text()
                  ).grid(row=1, column=4, columnspan=2, padx=6, pady=2)

        text_frame.columnconfigure(1, weight=1)

        # ── 3D canvas ────────────────────────────────────────────────────
        pop_fig = Figure(figsize=(9, 7), dpi=100, facecolor=bg_col)
        pop_ax  = pop_fig.add_subplot(111, projection="3d")
        _clean_3d_ax(pop_ax, bg_col)

        pop_canvas = FigureCanvasTkAgg(pop_fig, master=win)
        pop_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(pop_canvas, win)

        # ── Helper functions ──────────────────────────────────────────────
        def _pick_font_color():
            from tkinter import colorchooser
            c = colorchooser.askcolor(fcolor_var.get(), title="Font colour")[1]
            if c:
                fcolor_var.set(c)
                fc_btn.config(bg=c)

        def _apply_text():
            """Update / create title and annotation text on the figure."""
            fname = font_var.get()
            fsz   = fsize_var.get()
            fcol  = fcolor_var.get()

            # Title
            ttxt = title_var.get().strip()
            if title_obj[0] is not None:
                try:
                    title_obj[0].remove()
                except Exception:
                    pass
                title_obj[0] = None
            if ttxt:
                title_obj[0] = pop_ax.set_title(
                    ttxt, fontsize=fsz, fontfamily=fname,
                    color=fcol, pad=10, fontweight="bold")

            # Annotation (bottom-left of figure)
            atxt = annot_var.get().strip()
            if annot_obj[0] is not None:
                try:
                    annot_obj[0].remove()
                except Exception:
                    pass
                annot_obj[0] = None
            if atxt:
                annot_obj[0] = pop_fig.text(
                    0.02, 0.02, atxt, fontsize=max(fsz - 2, 6),
                    fontfamily=fname, color=fcol,
                    transform=pop_fig.transFigure,
                    ha="left", va="bottom")

            pop_canvas.draw_idle()

        def _rerender():
            """Re-render and re-apply text."""
            self._do_render(cube, ax=pop_ax, canvas=pop_canvas, fig=pop_fig)
            _apply_text()

        # Initial render
        self._do_render(cube, ax=pop_ax, canvas=pop_canvas, fig=pop_fig)
        _apply_text()

        # ── Click-to-label atoms ──────────────────────────────────────────
        pop_atom_annotations = {}
        pop_atom_data = []
        if self._iso_mol_on.get():
            atoms = cube["atoms"]
            pop_atom_data = [
                (a[2], a[3], a[4], _ATOMIC_NUM_REV.get(a[0], "X"))
                for a in atoms
            ]

        def _on_pick(event):
            ind = event.ind
            if ind is None or len(ind) == 0:
                return
            idx = int(ind[0])
            if idx >= len(pop_atom_data):
                return
            x, y, z, elem = pop_atom_data[idx]
            fname = font_var.get()
            fsz   = fsize_var.get()
            if idx in pop_atom_annotations:
                ann = pop_atom_annotations.pop(idx)
                ann.remove()
            else:
                ann = pop_ax.text(x, y, z, f"  {elem}{idx+1}",
                                  color="white", fontsize=max(fsz - 2, 8),
                                  fontfamily=fname, fontweight="bold",
                                  ha="left", va="center",
                                  bbox=dict(boxstyle="round,pad=0.15",
                                            facecolor=_ec(elem, self._elem_color_overrides),
                                            alpha=0.85,
                                            edgecolor="white", linewidth=0.5),
                                  zorder=100)
                pop_atom_annotations[idx] = ann
            pop_canvas.draw_idle()

        pop_fig.canvas.mpl_connect("pick_event", _on_pick)

    def _save_popout_png(self, fig):
        """Save the popout figure as a high-resolution PNG."""
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All files", "*.*")],
            title="Save Orbital Image",
        )
        if path:
            fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(),
                        edgecolor="none", bbox_inches="tight")
            messagebox.showinfo("Saved", f"Saved to:\n{path}")

    def _save_popout_svg(self, fig):
        """Save the popout figure as SVG (vector format)."""
        path = filedialog.asksaveasfilename(
            defaultextension=".svg",
            filetypes=[("SVG Image", "*.svg"), ("PDF", "*.pdf"), ("All files", "*.*")],
            title="Save Orbital Image (Vector)",
        )
        if path:
            fig.savefig(path, facecolor=fig.get_facecolor(),
                        edgecolor="none", bbox_inches="tight")
            messagebox.showinfo("Saved", f"Saved to:\n{path}")

    # ─── Molecular Re-orientation Tool ─────────────────────────────────────

    def _open_reorient_tool(self):
        """
        Open a window to re-orient the molecule so the metal center is at the
        origin and ligand atoms define the coordinate axes.

        This is essential for making orbital decomposition meaningful — d-orbital
        labels (dxy, dz2, etc.) only have physical meaning relative to a defined
        coordinate system.

        Method:
        - Pick the metal center → becomes origin
        - Pick atom for +X direction (e.g., a ligand trans to another)
        - Pick atom for XY plane (defines +Y component)
        - Z is computed as X × Y (right-hand rule)
        """
        if not self._current or self._current not in self._cache:
            messagebox.showinfo("Re-orient", "Load a molecule first.")
            return

        atoms = self._cache[self._current]["xyz"]
        if not atoms:
            messagebox.showinfo("Re-orient", "No coordinates available.")
            return

        win = tk.Toplevel(self)
        win.title("Re-orient Molecule")
        _clamp_geometry(win, 520, 560, 400, 350)

        _ro_sf = _ScrollableFrame(win)
        _ro_sf.pack(fill=tk.BOTH, expand=True)
        _ro = _ro_sf.interior

        # ── Instruction text ──────────────────────────────────────────────
        info = tk.Label(_ro, text=(
            "Re-orient the molecule so orbital labels (dxy, dz², etc.) are meaningful.\n"
            "1. Pick the metal center → becomes the origin\n"
            "2. Pick an atom that defines the +X axis direction\n"
            "3. Pick an atom in the XY plane (defines +Y component)\n"
            "Z axis is computed via the right-hand rule (X × Y)."
        ), justify="left", font=("", 9), wraplength=480, padx=10, pady=6)
        info.pack(fill=tk.X)

        ttk.Separator(_ro).pack(fill=tk.X, pady=2)

        # ── Atom list ─────────────────────────────────────────────────────
        list_frame = tk.Frame(_ro)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        cols = ("idx", "elem", "x", "y", "z")
        tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=14,
                            selectmode="browse")
        for c, w in zip(cols, (50, 50, 100, 100, 100)):
            tree.heading(c, text=c.upper())
            tree.column(c, width=w, anchor="center")

        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.config(yscrollcommand=sb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Populate — auto-highlight metals
        metal_indices = []
        for i, (el, x, y, z) in enumerate(atoms):
            tag = "metal" if el in _METALS else ""
            tree.insert("", "end", iid=str(i),
                        values=(i + 1, el, f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"),
                        tags=(tag,))
            if el in _METALS:
                metal_indices.append(i)
        tree.tag_configure("metal", foreground="#FFD700")

        # ── Selection entries ─────────────────────────────────────────────
        sel_frame = tk.Frame(_ro)
        sel_frame.pack(fill=tk.X, padx=6, pady=4)

        center_var = tk.StringVar(value="")
        xaxis_var  = tk.StringVar(value="")
        xyplane_var = tk.StringVar(value="")

        # Auto-fill metal center if exactly one metal
        if len(metal_indices) == 1:
            center_var.set(str(metal_indices[0] + 1))

        tk.Label(sel_frame, text="Metal center (atom #):", font=("", 9)
                 ).grid(row=0, column=0, sticky="e", padx=4, pady=2)
        tk.Entry(sel_frame, textvariable=center_var, width=8, font=("", 9)
                 ).grid(row=0, column=1, padx=4, pady=2, sticky="w")
        tk.Button(sel_frame, text="← Use selected", font=("", 8),
                  command=lambda: _fill_from_tree(center_var)
                  ).grid(row=0, column=2, padx=4)

        tk.Label(sel_frame, text="+X axis atom (atom #):", font=("", 9)
                 ).grid(row=1, column=0, sticky="e", padx=4, pady=2)
        tk.Entry(sel_frame, textvariable=xaxis_var, width=8, font=("", 9)
                 ).grid(row=1, column=1, padx=4, pady=2, sticky="w")
        tk.Button(sel_frame, text="← Use selected", font=("", 8),
                  command=lambda: _fill_from_tree(xaxis_var)
                  ).grid(row=1, column=2, padx=4)

        tk.Label(sel_frame, text="XY plane atom (atom #):", font=("", 9)
                 ).grid(row=2, column=0, sticky="e", padx=4, pady=2)
        tk.Entry(sel_frame, textvariable=xyplane_var, width=8, font=("", 9)
                 ).grid(row=2, column=1, padx=4, pady=2, sticky="w")
        tk.Button(sel_frame, text="← Use selected", font=("", 8),
                  command=lambda: _fill_from_tree(xyplane_var)
                  ).grid(row=2, column=2, padx=4)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_frame = tk.Frame(_ro)
        btn_frame.pack(fill=tk.X, padx=6, pady=6)

        tk.Button(btn_frame, text="Apply Re-orientation", font=("", 10, "bold"),
                  bg="#2a4a00", fg="white",
                  command=lambda: _apply()
                  ).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Reset to Original", font=("", 10),
                  bg="#4a2a00", fg="white",
                  command=lambda: _reset()
                  ).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Export re-oriented XYZ…", font=("", 9),
                  command=lambda: _export_xyz()
                  ).pack(side=tk.RIGHT, padx=4)

        status_lbl = tk.Label(_ro, text="", font=("", 9), fg="green")
        status_lbl.pack(fill=tk.X, padx=6)

        # ── Helper closures ───────────────────────────────────────────────
        def _fill_from_tree(var):
            sel = tree.selection()
            if sel:
                var.set(str(int(sel[0]) + 1))

        def _apply():
            try:
                ci = int(center_var.get()) - 1
                xi = int(xaxis_var.get()) - 1
                yi = int(xyplane_var.get()) - 1
            except ValueError:
                messagebox.showerror("Re-orient", "Enter valid atom numbers (1-based).")
                return
            n = len(atoms)
            if not (0 <= ci < n and 0 <= xi < n and 0 <= yi < n):
                messagebox.showerror("Re-orient", f"Atom numbers must be 1–{n}.")
                return
            if len({ci, xi, yi}) < 3:
                messagebox.showerror("Re-orient", "Pick three different atoms.")
                return

            # Build rotation matrix
            origin = np.array([atoms[ci][1], atoms[ci][2], atoms[ci][3]])
            p_x    = np.array([atoms[xi][1], atoms[xi][2], atoms[xi][3]])
            p_y    = np.array([atoms[yi][1], atoms[yi][2], atoms[yi][3]])

            vx = p_x - origin
            vx_norm = np.linalg.norm(vx)
            if vx_norm < 1e-10:
                messagebox.showerror("Re-orient", "X-axis atom is at the same position as center.")
                return
            ex = vx / vx_norm

            vy_raw = p_y - origin
            # Gram-Schmidt: remove component along ex
            vy = vy_raw - np.dot(vy_raw, ex) * ex
            vy_norm = np.linalg.norm(vy)
            if vy_norm < 1e-10:
                messagebox.showerror("Re-orient",
                                     "XY-plane atom is collinear with center and X-axis atom.")
                return
            ey = vy / vy_norm

            ez = np.cross(ex, ey)
            ez = ez / np.linalg.norm(ez)  # should already be unit

            # Rotation matrix: rows are the new basis vectors
            R = np.array([ex, ey, ez])

            self._reorient_R = R
            self._reorient_T = origin

            status_lbl.config(
                text=(f"Applied: center={atoms[ci][0]}{ci+1}, "
                      f"X→{atoms[xi][0]}{xi+1}, "
                      f"XY-plane→{atoms[yi][0]}{yi+1}"))
            self._redraw_3d()

        def _reset():
            self._reorient_R = None
            self._reorient_T = None
            status_lbl.config(text="Reset to original orientation.")
            self._redraw_3d()

        def _export_xyz():
            if self._reorient_R is None:
                messagebox.showinfo("Export", "Apply a re-orientation first.")
                return
            transformed = _apply_reorient(atoms, self._reorient_R, self._reorient_T)
            path = filedialog.asksaveasfilename(
                defaultextension=".xyz",
                filetypes=[("XYZ file", "*.xyz"), ("All files", "*.*")],
                title="Export re-oriented coordinates",
            )
            if not path:
                return
            with open(path, "w") as f:
                f.write(f"{len(transformed)}\n")
                f.write(f"Re-oriented coordinates from {self._current}\n")
                for el, x, y, z in transformed:
                    f.write(f"{el:4s} {x:14.8f} {y:14.8f} {z:14.8f}\n")
            messagebox.showinfo("Saved", f"Exported to:\n{path}")

        def _compare_composition():
            """Show before/after comparison of orbital decomposition for selected MO."""
            if not hasattr(self, "_orca_mo_data") or self._orca_mo_data is None:
                messagebox.showinfo("Compare", "Load a file with MO data first.")
                return
            sel = self._mo_tree.selection()
            if not sel:
                messagebox.showinfo("Compare", "Select an MO in the Isosurface tab first.")
                return
            iorb = int(sel[0])

            comp_orig = compute_mo_composition(self._orca_mo_data, iorb, reorient_R=None)
            comp_rot  = compute_mo_composition(self._orca_mo_data, iorb,
                                               reorient_R=self._reorient_R)

            cwin = tk.Toplevel(win)
            cwin.title(f"Composition Comparison — MO {iorb}")
            _clamp_geometry(cwin, 750, 500, 450, 350)

            tk.Label(cwin, text=f"MO {iorb} — Orbital Decomposition Before vs After Re-orientation",
                     font=("", 11, "bold")).pack(pady=6)

            cols = ("AO", "Original %", "Re-oriented %", "Change")
            tree_c = ttk.Treeview(cwin, columns=cols, show="headings", height=20)
            for c, w in zip(cols, (200, 120, 120, 120)):
                tree_c.heading(c, text=c)
                tree_c.column(c, width=w, anchor="center")
            tree_c.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

            # Summary row
            for label, key in [("Total s", "total_s"), ("Total p", "total_p"),
                               ("Total d", "total_d"), ("Total f", "total_f")]:
                v1, v2 = comp_orig[key], comp_rot[key]
                delta = v2 - v1
                tree_c.insert("", "end", values=(
                    label, f"{v1:.1f}%", f"{v2:.1f}%",
                    f"{delta:+.1f}%"))

            tree_c.insert("", "end", values=("───────", "───────", "───────", "───────"))

            # Per-AO detail
            all_ao_keys = set()
            orig_ao = {(a, l): p for a, l, p in comp_orig['top_ao']}
            rot_ao  = {(a, l): p for a, l, p in comp_rot['top_ao']}
            all_ao_keys = set(orig_ao.keys()) | set(rot_ao.keys())
            sorted_keys = sorted(all_ao_keys, key=lambda k: max(orig_ao.get(k, 0), rot_ao.get(k, 0)),
                                 reverse=True)

            for ao_key in sorted_keys[:30]:
                atom_lbl, ang_lbl = ao_key
                v1 = orig_ao.get(ao_key, 0.0)
                v2 = rot_ao.get(ao_key, 0.0)
                delta = v2 - v1
                tag = ""
                if abs(delta) > 5:
                    tag = "big_change"
                tree_c.insert("", "end", values=(
                    f"{atom_lbl} {ang_lbl}", f"{v1:.1f}%", f"{v2:.1f}%",
                    f"{delta:+.1f}%"), tags=(tag,))

            tree_c.tag_configure("big_change", foreground="#FF6600")

        # Add compare button to the button frame
        tk.Button(btn_frame, text="Compare Before/After", font=("", 9),
                  command=_compare_composition).pack(side=tk.LEFT, padx=8)

    # ─── Click-to-label atoms on isosurface view ─────────────────────────────

    def _on_iso_atom_pick(self, event):
        """Toggle element label when an atom is clicked in the isosurface view."""
        if not hasattr(self, "_iso_atom_data") or not self._iso_atom_data:
            return
        ind = event.ind
        if ind is None or len(ind) == 0:
            return

        # Pick the first atom in the selection
        idx = int(ind[0])
        if idx >= len(self._iso_atom_data):
            return
        x, y, z, elem = self._iso_atom_data[idx]

        if not hasattr(self, "_iso_atom_annotations"):
            self._iso_atom_annotations = {}

        if idx in self._iso_atom_annotations:
            # Already labelled — remove (toggle off)
            ann = self._iso_atom_annotations.pop(idx)
            ann.remove()
        else:
            # Add label
            ann = self._ax_iso.text(x, y, z, f"  {elem}{idx+1}",
                                     color="white", fontsize=9, fontweight="bold",
                                     ha="left", va="center",
                                     bbox=dict(boxstyle="round,pad=0.15",
                                               facecolor=_ec(elem, self._elem_color_overrides),
                                               alpha=0.85,
                                               edgecolor="white", linewidth=0.5),
                                     zorder=100)
            self._iso_atom_annotations[idx] = ann

        self._canvas_iso.draw_idle()

    # ─── Orbital Decomp tab ───────────────────────────────────────────────────

    _DECOMP_LCOLOURS = [
        "#4e9a4e", "#4d79c7", "#cc5555", "#888800",
        "#8855aa", "#cc7722", "#227766", "#aa3388",
    ]
    _LORB_COLOURS = {   # angular-momentum → bar segment colour
        "s":     "#98c379",   # green
        "p":     "#61afef",   # blue
        "d":     "#e06c75",   # red/salmon
        "f_orb": "#c678dd",   # purple
    }

    def _build_tab_decomp(self, p):
        # ── Status bar ────────────────────────────────────────────────────────
        self._decomp_status = tk.Label(
            p, text="Load an ORCA .out file from the Isosurface tab first.",
            fg="gray", font=("", 8), anchor="w", padx=6)
        self._decomp_status.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))

        top_btn = tk.Frame(p)
        top_btn.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        tk.Button(top_btn, text="  Analyse  ", bg="#002255", fg="white",
                  font=("", 10, "bold"),
                  command=self._decomp_analyse).pack(side=tk.LEFT)
        tk.Button(top_btn, text="Add Group", font=("", 9),
                  command=self._decomp_add_row).pack(side=tk.LEFT, padx=(8, 0))
        tk.Label(top_btn, text="  Double-click an MO to analyse immediately",
                 font=("", 8), fg="gray").pack(side=tk.LEFT, padx=8)

        # ── Main content area ─────────────────────────────────────────────────
        content = tk.Frame(p)
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: MO selector
        mo_frame = tk.LabelFrame(content, text="Select MO", padx=4, pady=4)
        mo_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 2), pady=4)

        sb = tk.Scrollbar(mo_frame, orient=tk.VERTICAL)
        self._decomp_mo_lb = tk.Listbox(
            mo_frame, font=("Courier", 9), width=30, height=26,
            selectmode=tk.SINGLE, yscrollcommand=sb.set,
            activestyle="dotbox")
        sb.config(command=self._decomp_mo_lb.yview)
        self._decomp_mo_lb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        self._decomp_mo_lb.bind("<Double-Button-1>", lambda _e: self._decomp_analyse())

        # Right: groups + plot
        right = tk.Frame(content)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 4), pady=4)

        # Atom group editor
        grp_frame = tk.LabelFrame(right, text="Atom Groups  (type atom labels exactly as in ORCA, e.g. Ni1  P2  Cl85)", padx=4, pady=4)
        grp_frame.pack(side=tk.TOP, fill=tk.X)

        # Header row
        hdr = tk.Frame(grp_frame)
        hdr.pack(fill=tk.X, pady=(0, 2))
        tk.Label(hdr, text="Group name", font=("", 8, "bold"), width=18, anchor="w").pack(side=tk.LEFT)
        tk.Label(hdr, text="Atoms (comma-separated)", font=("", 8, "bold"), anchor="w").pack(side=tk.LEFT, padx=(4, 0))

        # Scrollable rows container
        rows_canvas = tk.Canvas(grp_frame, height=130, highlightthickness=0)
        rows_canvas.pack(fill=tk.X, expand=False)
        self._decomp_rows_frame = tk.Frame(rows_canvas)
        rows_canvas.create_window((0, 0), window=self._decomp_rows_frame, anchor="nw")
        self._decomp_rows_frame.bind(
            "<Configure>",
            lambda e: rows_canvas.config(scrollregion=rows_canvas.bbox("all")))
        rows_sb = tk.Scrollbar(grp_frame, orient=tk.VERTICAL,
                                command=rows_canvas.yview)
        rows_canvas.config(yscrollcommand=rows_sb.set)

        # Default rows
        self._decomp_rows = []
        for name_hint in ["Metal", "Donor 1", "Donor 2", "Ligand"]:
            self._decomp_add_row(name_hint=name_hint)

        # matplotlib figure for results
        plot_frame = tk.LabelFrame(right, text="Orbital character", padx=2, pady=2)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(4, 0))

        self._decomp_fig = Figure(figsize=(7, 3.5), dpi=96)
        self._decomp_ax  = self._decomp_fig.add_subplot(111)
        self._decomp_ax.set_visible(False)
        self._decomp_canvas = FigureCanvasTkAgg(self._decomp_fig, master=plot_frame)
        self._decomp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _decomp_add_row(self, name_hint: str = ""):
        idx  = len(self._decomp_rows)
        col  = self._DECOMP_LCOLOURS[idx % len(self._DECOMP_LCOLOURS)]
        frame = tk.Frame(self._decomp_rows_frame)
        frame.pack(fill=tk.X, pady=1)

        tk.Label(frame, bg=col, width=2, relief=tk.RAISED).pack(side=tk.LEFT, padx=(0, 3))

        name_var = tk.StringVar(value=name_hint)
        tk.Entry(frame, textvariable=name_var, width=16,
                 font=("", 9)).pack(side=tk.LEFT)

        atoms_var = tk.StringVar(value="")
        tk.Entry(frame, textvariable=atoms_var, width=38,
                 font=("Courier", 9)).pack(side=tk.LEFT, padx=(4, 0))

        def _remove(i=idx):
            self._decomp_remove_row(i)

        tk.Button(frame, text="\u2715", width=2, font=("", 8), relief=tk.FLAT,
                  command=_remove).pack(side=tk.LEFT, padx=(4, 0))

        self._decomp_rows.append((name_var, atoms_var, frame))

    def _decomp_remove_row(self, idx: int):
        if idx < len(self._decomp_rows):
            _, _, frame = self._decomp_rows[idx]
            frame.destroy()
            self._decomp_rows.pop(idx)

    def _decomp_analyse(self):
        """Compute s/p/d/f contributions for each group and draw stacked bar chart."""
        if not self._loewdin_data:
            self._decomp_status.config(
                text="No Löwdin data loaded.  Go to the Isosurface tab and load an .out file first.",
                fg="red")
            return

        # Get selected MO
        sel = self._decomp_mo_lb.curselection()
        if not sel:
            self._decomp_status.config(
                text="Select an MO from the list first.", fg="#994400")
            return

        mo_keys = sorted(self._loewdin_data.keys())
        mo_idx  = mo_keys[sel[0]]
        entry   = self._loewdin_data[mo_idx]
        atoms   = entry["atoms"]   # {atom_key: {s,p,d,f_orb,total}}
        label   = entry.get("label", "") or f"MO {mo_idx}"

        # Collect all atoms represented in this MO
        all_keys = set(atoms.keys())
        assigned = set()

        # Build per-group data
        groups = []
        for name_var, atoms_var, _ in self._decomp_rows:
            name = name_var.get().strip() or "(unnamed)"
            raw  = atoms_var.get().strip()
            if not raw:
                continue
            atom_list = [a.strip() for a in raw.replace(";", ",").split(",") if a.strip()]

            g_s = g_p = g_d = g_f = 0.0
            found_any = False
            for ak in atom_list:
                if ak in atoms:
                    d = atoms[ak]
                    g_s += d["s"]
                    g_p += d["p"]
                    g_d += d["d"]
                    g_f += d["f_orb"]
                    assigned.add(ak)
                    found_any = True
            if found_any:
                groups.append({"name": name, "s": g_s, "p": g_p, "d": g_d, "f": g_f})

        if not groups:
            self._decomp_status.config(
                text="No matching atoms found.  Check atom labels (case-sensitive, e.g. Ni1, P2).",
                fg="red")
            return

        # Draw the chart
        ax = self._decomp_ax
        ax.cla()
        ax.set_visible(True)

        y_pos   = list(range(len(groups)))
        y_labels = [g["name"] for g in groups]
        lc = self._LORB_COLOURS

        lefts = [0.0] * len(groups)
        for comp, colour, label_lc in [
            ("s", lc["s"],     "s"),
            ("p", lc["p"],     "p"),
            ("d", lc["d"],     "d"),
            ("f", lc["f_orb"], "f"),
        ]:
            vals = [g[comp] for g in groups]
            bars = ax.barh(y_pos, vals, left=lefts, color=colour,
                           label=label_lc, height=0.55)
            # Annotate each segment if large enough
            for bar, v in zip(bars, vals):
                if v > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_y() + bar.get_height() / 2,
                            f"{v*100:.1f}%", ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
            lefts = [l + v for l, v in zip(lefts, vals)]

        # Total % label at end of each bar
        for i, g in enumerate(groups):
            tot = g["s"] + g["p"] + g["d"] + g["f"]
            ax.text(tot + 0.004, i, f"{tot*100:.1f}%",
                    va="center", fontsize=8, color="black")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_xlabel("Population contribution", fontsize=9)
        ax.set_xlim(0, max(sum(g[k] for k in ("s","p","d","f")) for g in groups) * 1.20 + 0.02)
        ax.invert_yaxis()
        ax.set_title(f"Orbital character — {label}  (MO {mo_idx})", fontsize=10)
        ax.legend(loc="lower right", fontsize=8, title="\u2113", title_fontsize=8,
                  framealpha=0.7)
        ax.grid(axis="x", alpha=0.2, linestyle=":")

        self._decomp_fig.tight_layout()
        self._decomp_canvas.draw_idle()

        self._decomp_status.config(
            text=(f"MO {mo_idx} ({label})  |  "
                  f"{len(groups)} plotted group(s)  |  "
                  f"{len(assigned)}/{len(all_keys)} atoms matched"),
            fg="#004400")

    def _save_cube(self):
        cube = getattr(self, "_last_rendered_cube", None)
        if cube is None:
            messagebox.showinfo("Save", "Render an orbital first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".cube",
            filetypes=[("Cube files", "*.cube"), ("All files", "*.*")],
            title="Save Cube File",
        )
        if not path: return
        try:
            _write_cube_file(path, cube)
            messagebox.showinfo("Saved", f"Saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))


# ─────────────────────────────────────────────────────────────────────────────
#  Cube file writer
# ─────────────────────────────────────────────────────────────────────────────

def _write_cube_file(path: str, cube: dict) -> None:
    """Write a cube-format dict back to a standard .cube file (Bohr units)."""
    ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
    lines: List[str] = []
    comment = cube.get("comment", "NBO Viewer cube").split("|")[0].strip()
    lines.append(comment[:78])
    lines.append("Generated by NBO Analysis Viewer")

    atoms  = cube["atoms"]
    origin = np.asarray(cube["origin"]) * ANG_TO_BOHR
    axes   = np.asarray(cube["axes"])   * ANG_TO_BOHR
    n      = cube["n"]

    lines.append(f"{len(atoms):5d}  {origin[0]:12.6f}  {origin[1]:12.6f}  {origin[2]:12.6f}")
    for i in range(3):
        lines.append(f"{n[i]:5d}  {axes[i,0]:12.6f}  {axes[i,1]:12.6f}  {axes[i,2]:12.6f}")
    for z, q, x, y, zc in atoms:
        xb = x * ANG_TO_BOHR; yb = y * ANG_TO_BOHR; zb = zc * ANG_TO_BOHR
        lines.append(f"{int(z):5d}  {float(q):12.6f}  {xb:12.6f}  {yb:12.6f}  {zb:12.6f}")

    vals = cube["data"].ravel()
    for i in range(0, len(vals), 6):
        lines.append("  ".join(f"{v:13.5e}" for v in vals[i:i+6]))

    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    try:
        import numpy, matplotlib  # noqa: F401
    except ImportError:
        import tkinter.messagebox as mb
        mb.showerror("Missing dependency",
                     "numpy and matplotlib are required.\n\nRun:  pip install numpy matplotlib")
        return

    app = NBOViewerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
