"""
fdmnes_input.py — FDMNES input-file builder and output parser for Binah.

FDMNES uses a card-based text input (one card per "section"):

    Filout
       <output_basename>

    Range
       <emin> <estep> <emax>     ! eV relative to threshold

    Edge
       K

    Radius
       6.0                       ! cluster radius in Å

    Quadrupole                   ! enables E2 (1s→3d for transition metals)

    SCF
    Convergence
       0.0001

    Arc                          ! enable Lorentzian broadening output

    Atom
       28 3                      ! Z, l_max — Ni → 3
       6  2                      ! C  → 2
       ...

    Molecule
       <Z> <x> <y> <z>           ! one row per atom, absorber FIRST
       ...

    End

Output format (`<basename>.txt`):
    ! Energy   Sigma  ...
    -10.000   0.000345   ...
    ...

Convolution adds a `<basename>_conv.txt` that's the Lorentzian-broadened
version (matches experiment better).

This module reuses ``crop_structure_around_absorber`` and
``_ATOMIC_NUMBERS`` from ``structure_converter.py`` so the cropping /
absorber-by-element behaviour matches the existing FEFF flow.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

from structure_converter import (
    XYZStructure, parse_xyz_file, crop_structure_around_absorber,
    _ATOMIC_NUMBERS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Edge-energy lookup. FDMNES emits energies relative to the absorption
# threshold; converting back to absolute eV (so the FDMNES spectrum lines up
# with experimental spectra in the plot widget) needs an E0.
#
# Values in eV from standard X-ray data tables (Bearden 1967 / NIST).
# ─────────────────────────────────────────────────────────────────────────────
_K_EDGE_E0_EV = {
    19: 3608,    # K
    20: 4038,    # Ca
    21: 4492,    # Sc
    22: 4966,    # Ti
    23: 5465,    # V
    24: 5989,    # Cr
    25: 6539,    # Mn
    26: 7112,    # Fe
    27: 7709,    # Co
    28: 8333,    # Ni
    29: 8979,    # Cu
    30: 9659,    # Zn
    31: 10367,   # Ga
    32: 11103,   # Ge
    33: 11867,   # As
    34: 12658,   # Se
    35: 13474,   # Br
    42: 20000,   # Mo
    47: 25514,   # Ag
}


def edge_energy_eV(z: int, edge: str = "K") -> float:
    """Return the absolute E0 (eV) for absorber Z and edge name.

    Returns 0.0 if unknown — caller can still operate on relative energies.
    """
    edge = (edge or "K").strip().upper()
    if edge != "K":
        return 0.0
    return float(_K_EDGE_E0_EV.get(int(z), 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# l_max defaults per Z. Wrong l_max silently produces bad spectra (FDMNES
# truncates the basis), so we hard-code sensible values keyed by row /
# block. Override via the `l_max_overrides` dict.
# ─────────────────────────────────────────────────────────────────────────────
def _default_l_max(z: int) -> int:
    if z <= 2:           # H, He
        return 0
    if z <= 10:          # Li-Ne (s/p block)
        return 1
    if z <= 18:          # Na-Ar
        return 2
    if 19 <= z <= 36:    # K-Kr (3d transition metals + 4p block)
        return 3
    if z <= 54:          # Rb-Xe (4d + 5p)
        return 3
    return 3


# ─────────────────────────────────────────────────────────────────────────────
# Input builder
# ─────────────────────────────────────────────────────────────────────────────
def build_fdmnes_input(
    structure: XYZStructure,
    absorber_index: int,
    *,
    output_basename: str = "fdmnes_out",
    edge: str = "K",
    cluster_radius: float = 6.0,
    e_min: float = -10.0,
    e_step: float = 0.1,
    e_max: float = 50.0,
    use_quadrupole: bool = True,
    use_nondipole: bool = False,
    use_nonquadrupole: bool = False,
    use_noninterf: bool = False,
    use_scf: bool = True,
    scf_convergence: float = 1e-4,
    convolution: bool = True,
    use_green: bool = False,
    use_energpho: bool = False,
    use_spinorbit: bool = False,
    use_magnetism: bool = False,
    use_relativism: bool = False,
    use_nonrelat: bool = False,
    use_allsite: bool = False,
    use_cartesian: bool = False,
    use_spherical: bool = False,
    polarization: tuple[float, float, float] | None = None,
    l_max_overrides: dict[int, int] | None = None,
    extra_cards: list[str] | None = None,
) -> str:
    """Build the FDMNES input text (the contents of ``fdmnes_in.txt``).

    Parameters mirror the most common FDMNES knobs. The atom list is built
    from `structure.symbols` (one entry per unique element); the molecule
    block writes the absorber first (always ipot 1 in FDMNES convention)
    then other atoms sorted by distance from the absorber.

    Returns the full input text ready to be written to disk. The caller is
    expected to write a thin wrapper file (``fdmfile.txt``) pointing at this
    file — that's how FDMNES discovers input.
    """
    abs_idx = int(absorber_index) - 1
    if abs_idx < 0 or abs_idx >= structure.atom_count:
        raise ValueError(
            f"Absorber index must be between 1 and {structure.atom_count}."
        )

    abs_sym = structure.symbols[abs_idx]
    abs_z = _ATOMIC_NUMBERS.get(abs_sym)
    if abs_z is None:
        raise ValueError(f"Unknown element symbol for absorber: {abs_sym}")
    abs_xyz = np.asarray(structure.coords[abs_idx], dtype=float)

    overrides = dict(l_max_overrides or {})

    lines: list[str] = []
    lines.append(f"! FDMNES input generated by Binah for {abs_sym} {edge}-edge")
    lines.append(f"! Source structure: {os.path.basename(structure.source_file)}"
                 f" (atom #{abs_idx + 1})")
    lines.append("")

    lines.append("Filout")
    lines.append(f"   {output_basename}")
    lines.append("")

    lines.append("Range")
    lines.append(f"   {float(e_min):.3f} {float(e_step):.4f} {float(e_max):.3f}")
    lines.append("")

    lines.append("Edge")
    lines.append(f"   {edge}")
    lines.append("")

    lines.append("Radius")
    lines.append(f"   {float(cluster_radius):.3f}")
    lines.append("")

    # In molecule mode FDMNES defaults to the chemical species of the first
    # listed atom. `Absorbeur 1` makes the selected, reordered XYZ atom the
    # unique absorbing site even when there are several atoms with the same Z.
    lines.append("Absorbeur")
    lines.append("   1")
    lines.append("")

    if use_green:
        lines.append("Green")
        lines.append("")

    if use_energpho:
        lines.append("Energpho")
        lines.append("")

    if use_quadrupole:
        lines.append("Quadrupole")
        lines.append("")

    if use_nondipole:
        lines.append("Nondipole")
        lines.append("")

    if use_nonquadrupole:
        lines.append("Nonquadrupole")
        lines.append("")

    if use_noninterf:
        lines.append("Noninterf")
        lines.append("")

    if use_spinorbit:
        lines.append("Spinorbite")
        lines.append("")

    if use_magnetism:
        lines.append("Magnetism")
        lines.append("")

    if use_relativism:
        lines.append("Relativism")
        lines.append("")

    if use_nonrelat:
        lines.append("Nonrelat")
        lines.append("")

    if use_allsite:
        lines.append("Allsite")
        lines.append("")

    if use_cartesian:
        lines.append("Cartesian")
        lines.append("")

    if use_spherical:
        lines.append("Spherical")
        lines.append("")

    if use_scf:
        # FDMNES turns on self-consistency with the bare ``SCF`` keyword.
        # The convergence threshold has a sensible default — advanced users
        # can override with ``Delta_E_conv``/``N_self`` via extra_cards.
        # (Earlier versions of this file emitted a ``Convergence`` block,
        #  which FDMNES does NOT understand and rejects with an indata
        #  error.)
        lines.append("SCF")
        if float(scf_convergence) > 0 and abs(float(scf_convergence) - 1e-4) > 1e-9:
            # Only emit Delta_E_conv if the user moved off the default.
            lines.append("Delta_E_conv")
            lines.append(f"   {float(scf_convergence):.6f}")
        lines.append("")

    if convolution:
        # Manual section C: adding convolution keywords to the main input runs
        # the broadening step immediately. Bare `Arc` uses FDMNES defaults and
        # produces the `<filout>_conv.txt` file.
        lines.append("Arc")
        lines.append("")

    if polarization is not None:
        px, py, pz = polarization
        lines.append("Polarize")
        lines.append(f"   {float(px):.4f} {float(py):.4f} {float(pz):.4f}")
        lines.append("")

    # FDMNES uses default neutral electronic configurations when no `Atom`
    # block is present. The `Atom` keyword's true syntax is `Z N (n l occ)*N`
    # — a full electronic-configuration override — and is only useful for
    # ionic states (Ni²⁺ etc.). For neutral molecular calculations we skip
    # it entirely and let FDMNES use its built-in atomic configs.
    #
    # The l_max table built from `_default_l_max` is therefore unused in the
    # default path. It's still consulted via `l_max_overrides` if a power
    # user wants to pass explicit per-element values via extra_cards.
    _ = overrides  # silence "unused variable" linters

    # Molecule card. Per the FDMNES manual (II-2 "Cluster or crystal
    # structure"), the line right under "molecule" gives the cell parameters
    # (a, b, c, α, β, γ), and the atom positions that follow are FRACTIONAL
    # coordinates of those parameters.
    #
    # For a non-periodic molecular cluster we use scale = (1 Å, 1 Å, 1 Å),
    # so positions in mesh-parameter units are numerically equal to cartesian Å
    # and FDMNES's `Radius` cluster cutoff (also in Å) lines up directly.
    # In molecule mode no periodic images are generated, so fractional
    # coords outside [0, 1] are fine (the manual's FeO_6 example uses ±1.0).
    lines.append("Molecule")
    lines.append("   1.0  1.0  1.0   90.0  90.0  90.0    "
                 "! a, b, c (Angstrom), alpha, beta, gamma (deg); scale=1")
    lines.append("! Z    x[Ang]      y[Ang]      z[Ang]")
    lines.append(f"   {abs_z:3d}   {0.0:11.6f} {0.0:11.6f} {0.0:11.6f}    "
                 f"! {abs_sym}{abs_idx + 1} (absorber)")
    others = []
    for i, sym in enumerate(structure.symbols):
        if i == abs_idx:
            continue
        rel = np.asarray(structure.coords[i], dtype=float) - abs_xyz
        d = float(np.linalg.norm(rel))
        others.append((d, rel, sym, i + 1))
    others.sort(key=lambda t: t[0])
    for d, rel, sym, idx_1based in others:
        z = _ATOMIC_NUMBERS.get(sym)
        if z is None:
            raise ValueError(f"Unknown element symbol in molecule block: {sym}")
        lines.append(
            f"   {z:3d}   {rel[0]:11.6f} {rel[1]:11.6f} {rel[2]:11.6f}    "
            f"! {sym}{idx_1based} ({d:.3f} A)"
        )
    lines.append("")

    if extra_cards:
        lines.append("! User-supplied extra cards")
        lines.extend(extra_cards)
        lines.append("")

    lines.append("End")
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Bundle exporter (XYZ → cropped + fdmnes_in.txt + fdmfile.txt)
# ─────────────────────────────────────────────────────────────────────────────
def export_xyz_as_fdmnes_bundle(
    xyz_path: str,
    work_dir: str,
    *,
    absorber_index: int,
    cluster_radius: float | None = None,
    basename: str = "",
    **build_kwargs,
) -> dict:
    """Write an FDMNES input bundle from an XYZ file.

    Returns a dict mirroring ``export_xyz_as_feff_bundle``:
        {
            "structure":         XYZStructure (possibly cropped),
            "input_path":        path to fdmnes_in.txt,
            "fdmfile_path":      path to fdmfile.txt (FDMNES discovery file),
            "xyz_copy_path":     path to <basename>(_<R>A).xyz,
            "output_basename":   what FDMNES will name its outputs,
            "cluster_radius":    float or None,
            "atoms_dropped":     int,
            "atoms_total_input": int,
            "atoms_used":        int,
        }

    The caller runs FDMNES with ``cwd=work_dir`` and ``fdmfile.txt`` will
    point it at ``fdmnes_in.txt``. FDMNES then writes
    ``<output_basename>.txt`` and (if Convolution is on) ``..._conv.txt``
    next to the input.
    """
    full_structure = parse_xyz_file(xyz_path)
    target = int(absorber_index)
    if target < 1 or target > full_structure.atom_count:
        raise ValueError(
            f"Absorber index must be 1..{full_structure.atom_count} for this XYZ."
        )

    dropped_count = 0
    if cluster_radius is not None and float(cluster_radius) > 0:
        structure, target, dropped_count = crop_structure_around_absorber(
            full_structure, target, float(cluster_radius)
        )
    else:
        structure = full_structure

    work_dir_path = Path(work_dir)
    work_dir_path.mkdir(parents=True, exist_ok=True)

    base = str(basename).strip() or full_structure.basename
    safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("._") or "structure"

    if cluster_radius is not None and float(cluster_radius) > 0:
        radius_tag = f"{float(cluster_radius):.1f}A"
        output_basename = f"{safe_base}_{radius_tag}"
        xyz_copy_path = work_dir_path / f"{output_basename}.xyz"
    else:
        output_basename = safe_base
        xyz_copy_path = work_dir_path / f"{safe_base}.xyz"

    input_path = work_dir_path / "fdmnes_in.txt"
    fdmfile_path = work_dir_path / "fdmfile.txt"

    # Write a (possibly cropped) XYZ copy alongside the input.
    if cluster_radius is not None and float(cluster_radius) > 0:
        xyz_lines = [str(structure.atom_count), structure.title or ""]
        for sym, xyz in zip(structure.symbols, structure.coords):
            xyz_lines.append(f"{sym}  {xyz[0]:.6f}  {xyz[1]:.6f}  {xyz[2]:.6f}")
        xyz_copy_path.write_text("\n".join(xyz_lines) + "\n", encoding="utf-8")
    else:
        xyz_copy_path.write_text(
            Path(xyz_path).read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )

    # Build and write the input file.
    build_options = dict(build_kwargs)
    if cluster_radius is not None and float(cluster_radius) > 0:
        build_options.setdefault("cluster_radius", float(cluster_radius))
    input_text = build_fdmnes_input(
        structure,
        absorber_index=target,
        output_basename=output_basename,
        **build_options,
    )
    input_path.write_text(input_text, encoding="utf-8")

    # fdmfile.txt — FDMNES's discovery wrapper. Format: number of input files,
    # then one path per line (relative to cwd).
    fdmfile_path.write_text("1\n" + input_path.name + "\n", encoding="utf-8")

    return {
        "structure":         structure,
        "input_path":        str(input_path),
        "fdmfile_path":      str(fdmfile_path),
        "xyz_copy_path":     str(xyz_copy_path),
        "output_basename":   output_basename,
        "cluster_radius":    (float(cluster_radius)
                              if cluster_radius is not None else None),
        "atoms_dropped":     int(dropped_count),
        "atoms_total_input": int(full_structure.atom_count),
        "atoms_used":        int(structure.atom_count),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_fdmnes_output(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Parse an FDMNES output file (``<basename>.txt`` or ``..._conv.txt``).

    FDMNES output layout:
        ! Possibly several header lines starting with '!'
        ! Energy   <signal_name(s)>
           <e1> <s1> [...]
           <e2> <s2> [...]

    Returns ``(energy_eV_relative, signal, meta)``. ``signal`` is the first
    non-energy column (FDMNES adds extra columns when convolution / multiple
    polarizations are enabled — we just take the first one and report the
    column name in ``meta``).

    The energy axis stays *relative* to the absorption edge — the caller
    converts to absolute eV using ``edge_energy_eV(z, edge)`` if needed.
    """
    path_str = str(path)
    header_cols: list[str] = []
    data_rows: list[list[float]] = []
    with open(path_str, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("!") or stripped.startswith("#"):
                # Look for a column-name line. FDMNES typically writes
                # "  Energy    Sigma  ..." sometimes with units in parens.
                tokens = re.findall(r"[A-Za-z][A-Za-z0-9_()/\-]*", stripped)
                if tokens and tokens[0].lower().startswith(("energy", "e ")):
                    header_cols = tokens
                continue
            # Data row.
            parts = stripped.split()
            try:
                row = [float(p) for p in parts]
            except ValueError:
                # First non-comment line might still be a column header
                # without a leading "!"; pull it as headers if all alpha.
                if all(re.match(r"^[A-Za-z][A-Za-z0-9_()/\-]*$", p) for p in parts):
                    header_cols = parts
                    continue
                # Otherwise skip silently.
                continue
            data_rows.append(row)

    if not data_rows:
        raise ValueError(f"No numeric data found in {path_str}")

    arr = np.asarray(data_rows, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2:
        raise ValueError(
            f"FDMNES output {path_str} has only {arr.shape[1]} column(s), "
            f"expected at least 2 (energy + signal)."
        )

    energy = arr[:, 0]
    signal = arr[:, 1]
    signal_name = (header_cols[1] if len(header_cols) >= 2 else "signal")

    meta = {
        "source": "fdmnes",
        "filename": os.path.basename(path_str),
        "n_points": int(arr.shape[0]),
        "n_columns": int(arr.shape[1]),
        "signal_column": signal_name,
        "energy_range_relative_eV": [float(energy[0]), float(energy[-1])],
        "all_columns": list(header_cols),
    }
    return energy, signal, meta
