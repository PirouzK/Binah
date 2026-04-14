"""
Structure conversion helpers for FEFF-oriented workflows.

This module focuses on turning simple XYZ structures into FEFF-friendly
artifacts:
  - a valid P1 CIF with cell metrics, symmetry metadata, and fractional sites
  - a matching feff.inp template using FEFF's CIF + RECIPROCAL workflow

The generated bundle is most appropriate for periodic / boxed-P1 workflows.
For isolated molecular clusters, FEFF's traditional ATOMS-based real-space
input can still be the more physically direct representation.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class XYZStructure:
    source_file: str
    title: str
    symbols: list[str]
    coords: np.ndarray

    @property
    def atom_count(self) -> int:
        return len(self.symbols)

    @property
    def basename(self) -> str:
        stem = Path(self.source_file).stem or "structure"
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
        return cleaned or "structure"

    @property
    def formula(self) -> str:
        counts = Counter(self.symbols)
        pieces = []
        for sym in sorted(counts.keys()):
            count = counts[sym]
            pieces.append(sym if count == 1 else f"{sym}{count}")
        return "".join(pieces)


def _canonicalize_symbol(raw: str) -> str:
    token = str(raw).strip()
    if not token:
        raise ValueError("Empty element symbol in XYZ file.")
    match = re.match(r"[A-Za-z]+", token)
    if not match:
        raise ValueError(f"Could not interpret element symbol '{raw}'.")
    alpha = match.group(0)
    if len(alpha) == 1:
        return alpha.upper()
    return alpha[0].upper() + alpha[1:].lower()


def parse_xyz_file(path: str) -> XYZStructure:
    file_path = Path(path)
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ file is too short.")

    try:
        natoms = int(lines[0].strip())
    except Exception as exc:
        raise ValueError("First line of XYZ file must be the atom count.") from exc

    title = lines[1].strip()
    atom_lines = [line for line in lines[2:] if line.strip()]
    if len(atom_lines) < natoms:
        raise ValueError(
            f"XYZ file declares {natoms} atoms but only {len(atom_lines)} atom lines were found."
        )

    symbols: list[str] = []
    coords: list[list[float]] = []
    for i, line in enumerate(atom_lines[:natoms], start=1):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Atom line {i} does not contain symbol + x y z coordinates.")
        symbol = _canonicalize_symbol(parts[0])
        try:
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        except Exception as exc:
            raise ValueError(f"Could not parse coordinates on atom line {i}.") from exc
        symbols.append(symbol)
        coords.append(xyz)

    return XYZStructure(
        source_file=str(file_path),
        title=title,
        symbols=symbols,
        coords=np.asarray(coords, dtype=float),
    )


def _infer_box(coords: np.ndarray, padding: float = 6.0,
               cubic: bool = False, min_length: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0:
        raise ValueError("Coordinates must be an N x 3 array.")

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = maxs - mins

    lengths = np.maximum(span + 2.0 * float(padding), float(min_length))
    if cubic:
        cube = float(np.max(lengths))
        lengths = np.array([cube, cube, cube], dtype=float)

    offset = 0.5 * (lengths - span)
    shifted = coords - mins + offset
    return lengths, shifted


def build_p1_cif_text(structure: XYZStructure, padding: float = 6.0,
                      cubic: bool = False) -> str:
    lengths, shifted = _infer_box(structure.coords, padding=padding, cubic=cubic)
    frac = shifted / lengths

    data_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", structure.basename).strip("._") or "structure"
    title = structure.title or structure.basename
    safe_title = title.replace("'", "")

    lines = [
        f"data_{data_name}",
        "_audit_creation_method 'Binah XYZ-to-CIF converter'",
        f"_chemical_name_common '{safe_title}'",
        f"_chemical_formula_sum '{structure.formula}'",
        f"_cell_length_a {lengths[0]:.6f}",
        f"_cell_length_b {lengths[1]:.6f}",
        f"_cell_length_c {lengths[2]:.6f}",
        "_cell_angle_alpha 90.000000",
        "_cell_angle_beta 90.000000",
        "_cell_angle_gamma 90.000000",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_symmetry_Int_Tables_number 1",
        "",
        "loop_",
        "_space_group_symop_operation_xyz",
        "x,y,z",
        "",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]

    for i, (sym, fxyz) in enumerate(zip(structure.symbols, frac), start=1):
        lines.append(
            f"{sym}{i} {sym} {fxyz[0]:.8f} {fxyz[1]:.8f} {fxyz[2]:.8f}"
        )

    return "\n".join(lines) + "\n"


def write_p1_cif(structure: XYZStructure, output_path: str, padding: float = 6.0,
                 cubic: bool = False) -> dict:
    out_path = Path(output_path)
    out_path.write_text(
        build_p1_cif_text(structure, padding=padding, cubic=cubic),
        encoding="utf-8",
    )
    lengths, shifted = _infer_box(structure.coords, padding=padding, cubic=cubic)
    return {
        "path": str(out_path),
        "cell_lengths": lengths,
        "shifted_coords": shifted,
    }


def build_feff_cif_input(cif_filename: str, absorber_index: int,
                         edge: str = "K", spectrum: str = "EXAFS",
                         kmesh: int = 200, equivalence: int = 2,
                         corehole: str = "RPA", s02: float = 1.0,
                         scf_radius: float = 4.0, fms_radius: float = 6.0,
                         rpath: float = 8.0, title: str = "Generated by Binah") -> str:
    edge = str(edge).strip() or "K"
    spectrum = str(spectrum).strip().upper() or "EXAFS"
    corehole = str(corehole).strip() or "RPA"
    kmesh = max(1, int(kmesh))
    equivalence = min(4, max(1, int(equivalence)))
    absorber_index = max(1, int(absorber_index))

    lines = [
        f"TITLE {title}",
        f"EDGE {edge}",
        f"S02 {float(s02):.3f}",
        f"COREHOLE {corehole}",
        "CONTROL 1 1 1 1 1 1",
        "PRINT 5 1 1 1 1 3",
        "EXCHANGE 2 0.0 0.0 2",
        f"SCF {float(scf_radius):.1f}",
        f"FMS {float(fms_radius):.1f}",
    ]

    if spectrum == "XANES":
        lines.append("XANES 20.0 0.07 0.0")
    else:
        lines.append("EXAFS")
        lines.append(f"RPATH {float(rpath):.1f}")

    lines.extend([
        "RECIPROCAL",
        f"KMESH {kmesh} 0 0 1 0",
        f"TARGET {absorber_index}",
        f"CIF {cif_filename}",
        f"EQUIVALENCE {equivalence}",
        "END",
    ])
    return "\n".join(lines) + "\n"


def write_feff_cif_input(output_path: str, cif_filename: str, absorber_index: int,
                         edge: str = "K", spectrum: str = "EXAFS",
                         kmesh: int = 200, equivalence: int = 2,
                         corehole: str = "RPA", s02: float = 1.0,
                         scf_radius: float = 4.0, fms_radius: float = 6.0,
                         rpath: float = 8.0, title: str = "Generated by Binah") -> str:
    out_path = Path(output_path)
    out_path.write_text(
        build_feff_cif_input(
            cif_filename=cif_filename,
            absorber_index=absorber_index,
            edge=edge,
            spectrum=spectrum,
            kmesh=kmesh,
            equivalence=equivalence,
            corehole=corehole,
            s02=s02,
            scf_radius=scf_radius,
            fms_radius=fms_radius,
            rpath=rpath,
            title=title,
        ),
        encoding="utf-8",
    )
    return str(out_path)


def export_xyz_as_feff_bundle(xyz_path: str, workdir: str, *,
                              basename: str = "",
                              padding: float = 6.0,
                              cubic: bool = False,
                              absorber_index: int = 1,
                              edge: str = "K",
                              spectrum: str = "EXAFS",
                              kmesh: int = 200,
                              equivalence: int = 2,
                              corehole: str = "RPA",
                              s02: float = 1.0,
                              scf_radius: float = 4.0,
                              fms_radius: float = 6.0,
                              rpath: float = 8.0) -> dict:
    structure = parse_xyz_file(xyz_path)
    target = int(absorber_index)
    if target < 1 or target > structure.atom_count:
        raise ValueError(
            f"Absorber index must be between 1 and {structure.atom_count} for this XYZ file."
        )

    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)

    base = str(basename).strip() or structure.basename
    safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("._") or "structure"

    cif_path = workdir_path / f"{safe_base}.cif"
    feff_path = workdir_path / "feff.inp"
    xyz_copy_path = workdir_path / f"{safe_base}.xyz"

    xyz_copy_path.write_text(Path(xyz_path).read_text(encoding="utf-8", errors="replace"),
                             encoding="utf-8")
    cif_meta = write_p1_cif(structure, str(cif_path), padding=padding, cubic=cubic)
    write_feff_cif_input(
        str(feff_path),
        cif_filename=cif_path.name,
        absorber_index=target,
        edge=edge,
        spectrum=spectrum,
        kmesh=kmesh,
        equivalence=equivalence,
        corehole=corehole,
        s02=s02,
        scf_radius=scf_radius,
        fms_radius=fms_radius,
        rpath=rpath,
        title=structure.title or f"{structure.formula} from XYZ",
    )

    return {
        "structure": structure,
        "cif_path": str(cif_path),
        "feff_inp_path": str(feff_path),
        "xyz_copy_path": str(xyz_copy_path),
        "cell_lengths": cif_meta["cell_lengths"],
        "padding": float(padding),
        "cubic": bool(cubic),
    }
