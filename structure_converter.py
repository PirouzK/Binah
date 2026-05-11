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
import shlex
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class XYZStructure:
    source_file: str
    title: str
    symbols: list[str]
    coords: np.ndarray
    metadata: dict = field(default_factory=dict)

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


@dataclass
class CIFSite:
    label: str
    symbol: str
    coord: np.ndarray
    occupancy: float
    row_index: int
    frac: np.ndarray | None = None


def crop_structure_around_absorber(structure: XYZStructure,
                                   absorber_index: int,
                                   radius: float) -> tuple[XYZStructure, int, int]:
    """Return a new XYZStructure containing only atoms within `radius` Å of
    the absorber atom (1-based `absorber_index`).

    The absorber itself is always kept and placed first in the cropped list,
    so the new absorber index is always 1.

    Returns (cropped_structure, new_absorber_index, dropped_count).
    """
    abs_idx = int(absorber_index) - 1
    if abs_idx < 0 or abs_idx >= structure.atom_count:
        raise ValueError(
            f"Absorber index must be between 1 and {structure.atom_count}."
        )
    r = float(radius)
    if not r > 0:
        return structure, int(absorber_index), 0

    abs_xyz = np.asarray(structure.coords[abs_idx], dtype=float)
    abs_sym = structure.symbols[abs_idx]

    kept_symbols: list[str] = [abs_sym]
    kept_coords: list[list[float]] = [list(abs_xyz)]
    dropped = 0
    for i, sym in enumerate(structure.symbols):
        if i == abs_idx:
            continue
        d = float(np.linalg.norm(np.asarray(structure.coords[i]) - abs_xyz))
        if d <= r:
            kept_symbols.append(sym)
            kept_coords.append(list(structure.coords[i]))
        else:
            dropped += 1

    cropped = XYZStructure(
        source_file=structure.source_file,
        title=(structure.title or "") + f" [cropped to {r:.2f} A around atom {absorber_index}]",
        symbols=kept_symbols,
        coords=np.asarray(kept_coords, dtype=float),
    )
    return cropped, 1, dropped


_COVALENT_RADII = {
    "H": 0.31, "B": 0.85, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Ni": 1.24, "Cu": 1.32,
    "Zn": 1.22, "Br": 1.20, "I": 1.39,
}


def keep_connected_fragment_around_absorber(
    structure: XYZStructure,
    absorber_index: int,
    *,
    bond_scale: float = 1.25,
) -> tuple[XYZStructure, int, int]:
    """Keep only the covalent graph component containing the absorber."""
    abs_idx = int(absorber_index) - 1
    if abs_idx < 0 or abs_idx >= structure.atom_count:
        raise ValueError(
            f"Absorber index must be between 1 and {structure.atom_count}."
        )
    n = structure.atom_count
    coords = np.asarray(structure.coords, dtype=float)
    adj = [set() for _ in range(n)]
    for i in range(n):
        ri = float(_COVALENT_RADII.get(structure.symbols[i], 0.75))
        for j in range(i + 1, n):
            rj = float(_COVALENT_RADII.get(structure.symbols[j], 0.75))
            cutoff = float(bond_scale) * (ri + rj)
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if 0.35 < d <= cutoff:
                adj[i].add(j)
                adj[j].add(i)

    stack = [abs_idx]
    seen = {abs_idx}
    while stack:
        i = stack.pop()
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                stack.append(j)

    kept_indices = [i for i in range(n) if i in seen]
    removed = n - len(kept_indices)
    if removed <= 0:
        return structure, int(absorber_index), 0

    new_absorber = kept_indices.index(abs_idx) + 1
    metadata = {
        k: v for k, v in dict(structure.metadata or {}).items()
        if k != "cleaned_cif_text"
    }
    metadata["solvent_removed_atoms"] = int(removed)
    metadata["solvent_kept_atoms"] = int(len(kept_indices))
    filtered = XYZStructure(
        source_file=structure.source_file,
        title=(structure.title or "") + " [absorber-connected fragment]",
        symbols=[structure.symbols[i] for i in kept_indices],
        coords=np.asarray([structure.coords[i] for i in kept_indices], dtype=float),
        metadata=metadata,
    )
    return filtered, new_absorber, removed


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


def _parse_cif_number(value: str) -> float:
    token = str(value).strip().strip("'\"")
    if token in ("?", "."):
        raise ValueError(f"Missing CIF numeric value '{value}'.")
    token = re.sub(r"\([0-9]+\)$", "", token)
    try:
        return float(token)
    except Exception as exc:
        raise ValueError(f"Could not parse CIF numeric value '{value}'.") from exc


def _tokenize_cif_line(line: str) -> list[str]:
    # This covers ordinary CIF quoted/unquoted values. Multi-line semicolon
    # text fields are ignored by the parser below because structure geometry
    # should not live there.
    body = line.split("#", 1)[0].strip()
    if not body:
        return []
    return shlex.split(body, comments=False, posix=True)


def _symbol_from_cif_atom(label: str, type_symbol: str = "") -> str:
    candidate = str(type_symbol or "").strip()
    if not candidate or candidate in ("?", "."):
        candidate = str(label or "").strip()
    match = re.match(r"[A-Za-z]{1,3}", candidate)
    if not match:
        raise ValueError(f"Could not infer element symbol from CIF atom '{label}'.")
    return _canonicalize_symbol(match.group(0))


def _cell_matrix_from_lengths_angles(a: float, b: float, c: float,
                                     alpha: float, beta: float,
                                     gamma: float) -> np.ndarray:
    ar, br, gr = np.deg2rad([alpha, beta, gamma])
    va = np.array([a, 0.0, 0.0], dtype=float)
    vb = np.array([b * np.cos(gr), b * np.sin(gr), 0.0], dtype=float)
    cx = c * np.cos(br)
    cy = c * (np.cos(ar) - np.cos(br) * np.cos(gr)) / max(np.sin(gr), 1e-12)
    cz_sq = c * c - cx * cx - cy * cy
    vc = np.array([cx, cy, np.sqrt(max(cz_sq, 0.0))], dtype=float)
    return np.vstack([va, vb, vc])


def _parse_cif_occupancy(row: dict[str, str]) -> float:
    raw = row.get("_atom_site_occupancy", "1")
    try:
        return _parse_cif_number(raw)
    except Exception:
        return 1.0


def _clean_cif_sites(sites: list[CIFSite], *,
                     min_occupancy: float = 0.01,
                     min_distance: float = 0.85) -> tuple[list[CIFSite], dict]:
    """Drop impossible/alternate CIF sites for simulation input.

    Olex2/SHELXL CIFs often contain zero-occupancy atoms and alternate
    disorder positions. FDMNES/FEFF molecule-style inputs do not understand
    that refinement bookkeeping, so nearby alternate sites must be reduced to
    one concrete geometry.
    """
    active = [site for site in sites if float(site.occupancy) > float(min_occupancy)]
    removed_zero = len(sites) - len(active)
    removed_overlap = 0

    changed = True
    while changed:
        changed = False
        remove_index: int | None = None
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                d = float(np.linalg.norm(active[i].coord - active[j].coord))
                if d >= float(min_distance):
                    continue
                a, b = active[i], active[j]
                if abs(a.occupancy - b.occupancy) > 1e-6:
                    remove_index = i if a.occupancy < b.occupancy else j
                else:
                    remove_index = i if a.row_index > b.row_index else j
                changed = True
                break
            if changed:
                break
        if remove_index is not None:
            active.pop(remove_index)
            removed_overlap += 1

    return active, {
        "cif_atoms_input": len(sites),
        "cif_atoms_cleaned": len(active),
        "cif_removed_zero_occupancy": removed_zero,
        "cif_removed_overlap": removed_overlap,
        "cif_cleaning_min_occupancy": float(min_occupancy),
        "cif_cleaning_min_distance": float(min_distance),
    }


def _cif_tag_value(tags: dict[str, str], key: str, default: str) -> str:
    return str(tags.get(key.lower(), default)).strip("'\"")


def build_cleaned_cif_text(structure: XYZStructure) -> str:
    text = structure.metadata.get("cleaned_cif_text") if structure.metadata else None
    if text:
        return str(text)
    return build_p1_cif_text(structure)


def _build_cleaned_cif_text(data_name: str, tags: dict[str, str],
                            sites: list[CIFSite], cleaning: dict) -> str:
    counts = Counter(site.symbol for site in sites)
    formula = " ".join(
        sym if count == 1 else f"{sym}{count}"
        for sym, count in sorted(counts.items())
    )
    lines = [
        f"data_{data_name}_cleaned",
        "_audit_creation_method 'Binah CIF disorder/overlap cleaner'",
        f"_chemical_formula_sum '{formula}'",
        f"_cell_length_a {_cif_tag_value(tags, '_cell_length_a', '1')}",
        f"_cell_length_b {_cif_tag_value(tags, '_cell_length_b', '1')}",
        f"_cell_length_c {_cif_tag_value(tags, '_cell_length_c', '1')}",
        f"_cell_angle_alpha {_cif_tag_value(tags, '_cell_angle_alpha', '90')}",
        f"_cell_angle_beta {_cif_tag_value(tags, '_cell_angle_beta', '90')}",
        f"_cell_angle_gamma {_cif_tag_value(tags, '_cell_angle_gamma', '90')}",
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
        "_atom_site_occupancy",
    ]
    for index, site in enumerate(sites, start=1):
        frac = site.frac
        if frac is None:
            frac = np.asarray(site.coord, dtype=float)
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", site.label).strip("._")
        label = label or f"{site.symbol}{index}"
        lines.append(
            f"{label} {site.symbol} {frac[0]:.8f} {frac[1]:.8f} {frac[2]:.8f} "
            f"{float(site.occupancy):.5f}"
        )
    lines.extend([
        "",
        f"# Binah cleaning: input_sites={cleaning.get('cif_atoms_input', '?')}",
        f"# Binah cleaning: kept_sites={cleaning.get('cif_atoms_cleaned', '?')}",
        f"# Binah cleaning: removed_zero_occupancy={cleaning.get('cif_removed_zero_occupancy', '?')}",
        f"# Binah cleaning: removed_close_overlap={cleaning.get('cif_removed_overlap', '?')}",
        "",
    ])
    return "\n".join(lines)


def parse_cif_file(path: str) -> XYZStructure:
    """Parse a common crystallographic CIF into cartesian coordinates.

    The parser intentionally targets the geometry subset Binah needs:
    cell lengths/angles plus an ``_atom_site`` loop with fractional or
    cartesian coordinates. It does not expand symmetry operations; it uses
    the explicit sites present in the CIF.
    """
    file_path = Path(path)
    raw_lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()

    tags: dict[str, str] = {}
    atom_rows: list[dict[str, str]] = []
    title = file_path.stem
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        if line.lower().startswith("data_"):
            title = line[5:].strip() or title
            i += 1
            continue
        if line.lower() == "loop_":
            i += 1
            headers: list[str] = []
            while i < len(raw_lines):
                tokens = _tokenize_cif_line(raw_lines[i])
                if len(tokens) == 1 and tokens[0].startswith("_"):
                    headers.append(tokens[0].lower())
                    i += 1
                    continue
                break
            values: list[str] = []
            while i < len(raw_lines):
                stripped = raw_lines[i].strip()
                low = stripped.lower()
                if low == "loop_" or stripped.startswith("_") or low.startswith("data_"):
                    break
                values.extend(_tokenize_cif_line(raw_lines[i]))
                i += 1
            coord_headers = {
                "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z",
                "_atom_site_cartn_x", "_atom_site_cartn_y", "_atom_site_cartn_z",
            }
            if headers and any(h in headers for h in coord_headers):
                ncols = len(headers)
                for start in range(0, len(values), ncols):
                    row_values = values[start:start + ncols]
                    if len(row_values) == ncols:
                        atom_rows.append(dict(zip(headers, row_values)))
            continue
        tokens = _tokenize_cif_line(line)
        if len(tokens) >= 2 and tokens[0].startswith("_"):
            tags[tokens[0].lower()] = tokens[1]
        i += 1

    if not atom_rows:
        raise ValueError("CIF does not contain a readable _atom_site loop.")

    sites: list[CIFSite] = []

    fract_keys = ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")
    cart_keys = ("_atom_site_cartn_x", "_atom_site_cartn_y", "_atom_site_cartn_z")
    has_fract = all(all(key in row for key in fract_keys) for row in atom_rows)
    has_cart = all(all(key in row for key in cart_keys) for row in atom_rows)

    if has_fract:
        try:
            cell = _cell_matrix_from_lengths_angles(
                _parse_cif_number(tags["_cell_length_a"]),
                _parse_cif_number(tags["_cell_length_b"]),
                _parse_cif_number(tags["_cell_length_c"]),
                _parse_cif_number(tags.get("_cell_angle_alpha", "90")),
                _parse_cif_number(tags.get("_cell_angle_beta", "90")),
                _parse_cif_number(tags.get("_cell_angle_gamma", "90")),
            )
        except KeyError as exc:
            raise ValueError("CIF fractional coordinates require cell lengths a, b, and c.") from exc
        for row_index, row in enumerate(atom_rows):
            label = row.get("_atom_site_label", "")
            sym = _symbol_from_cif_atom(label, row.get("_atom_site_type_symbol", ""))
            frac = np.array([
                _parse_cif_number(row["_atom_site_fract_x"]),
                _parse_cif_number(row["_atom_site_fract_y"]),
                _parse_cif_number(row["_atom_site_fract_z"]),
            ], dtype=float)
            xyz = frac @ cell
            sites.append(CIFSite(
                label=label or f"{sym}{row_index + 1}",
                symbol=sym,
                coord=np.asarray(xyz, dtype=float),
                occupancy=_parse_cif_occupancy(row),
                row_index=row_index,
                frac=frac,
            ))
    elif has_cart:
        for row_index, row in enumerate(atom_rows):
            label = row.get("_atom_site_label", "")
            sym = _symbol_from_cif_atom(label, row.get("_atom_site_type_symbol", ""))
            xyz = [
                _parse_cif_number(row["_atom_site_cartn_x"]),
                _parse_cif_number(row["_atom_site_cartn_y"]),
                _parse_cif_number(row["_atom_site_cartn_z"]),
            ]
            sites.append(CIFSite(
                label=label or f"{sym}{row_index + 1}",
                symbol=sym,
                coord=np.asarray(xyz, dtype=float),
                occupancy=_parse_cif_occupancy(row),
                row_index=row_index,
                frac=None,
            ))
    else:
        raise ValueError("CIF atom loop must include fractional or cartesian coordinates.")

    cleaned_sites, cleaning = _clean_cif_sites(sites)
    symbols = [site.symbol for site in cleaned_sites]
    coords = [site.coord.tolist() for site in cleaned_sites]
    data_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", title).strip("._") or file_path.stem
    metadata = dict(cleaning)
    metadata["cleaned_cif_text"] = _build_cleaned_cif_text(data_name, tags, cleaned_sites, cleaning)

    return XYZStructure(
        source_file=str(file_path),
        title=title,
        symbols=symbols,
        coords=np.asarray(coords, dtype=float),
        metadata=metadata,
    )


def parse_structure_file(path: str) -> XYZStructure:
    suffix = Path(path).suffix.lower()
    if suffix == ".cif":
        return parse_cif_file(path)
    return parse_xyz_file(path)


def structure_to_xyz_text(structure: XYZStructure) -> str:
    lines = [str(structure.atom_count), structure.title or ""]
    for sym, xyz in zip(structure.symbols, structure.coords):
        lines.append(f"{sym}  {xyz[0]:.6f}  {xyz[1]:.6f}  {xyz[2]:.6f}")
    return "\n".join(lines) + "\n"


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
                         rpath: float = 8.0, title: str = "Generated by Binah",
                         xanes_emin: float = -30.0, xanes_emax: float = 250.0,
                         xanes_estep: float = 0.25,
                         exchange: int = 0, exchange_vr: float = 0.0,
                         exchange_vi: float = 0.0,
                         scf_nscf: int = 30, scf_ca: float = 0.2,
                         exafs_kmax: float = 20.0,
                         nleg: int = 0,
                         multipole_lmax: int = 0,
                         multipole_iorder: int = 2,
                         polarization: tuple | None = None,
                         ellipticity: tuple | None = None,
                         debye: tuple | None = None) -> str:
    edge = str(edge).strip() or "K"
    spectrum = str(spectrum).strip().upper() or "EXAFS"
    corehole = str(corehole).strip() or "RPA"
    kmesh = max(1, int(kmesh))
    equivalence = min(4, max(1, int(equivalence)))
    absorber_index = max(1, int(absorber_index))
    xanes_emin = float(xanes_emin)
    xanes_emax = float(xanes_emax)
    xanes_estep = max(0.01, float(xanes_estep))

    lines = [
        f"TITLE {title}",
        f"EDGE {edge}",
        f"S02 {float(s02):.3f}",
        f"COREHOLE {corehole}",
        "CONTROL 1 1 1 1 1 1",
        "PRINT 1 0 0 0 0 0",
        f"EXCHANGE {int(exchange)} {float(exchange_vr):.1f} {float(exchange_vi):.1f} 2",
        f"SCF {float(scf_radius):.1f} 0 {int(scf_nscf)} {float(scf_ca):.2f} 3",
        f"FMS {float(fms_radius):.1f} 0",
    ]

    if int(nleg) > 0:
        lines.append(f"NLEG {int(nleg)}")

    if int(multipole_lmax) >= 2:
        lines.append(f"MULTIPOLE {int(multipole_lmax)} {int(multipole_iorder)}")

    if polarization is not None:
        px, py, pz = polarization
        lines.append(f"POLARIZATION {float(px):.4f} {float(py):.4f} {float(pz):.4f}")

    if ellipticity is not None:
        ellip, ex, ey, ez = ellipticity
        lines.append(
            f"ELLIPTICITY {float(ellip):.4f} {float(ex):.4f} {float(ey):.4f} {float(ez):.4f}"
        )

    if debye is not None:
        temp, dtemp = debye
        lines.append(f"DEBYE {float(temp):.1f} {float(dtemp):.1f}")

    if spectrum == "XANES":
        lines.append(f"XANES {xanes_emax:.1f} {xanes_estep:.4f} 0.0")
        npts = max(1, int(round((xanes_emax - xanes_emin) / xanes_estep)))
        lines.append("EGRID")
        lines.append(f"  {xanes_emin:.2f}  {xanes_estep:.4f}  {npts}")
    else:
        lines.append(f"EXAFS {float(exafs_kmax):.1f}")
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


_ATOMIC_NUMBERS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
}


def build_feff_atoms_input(structure: XYZStructure, absorber_index: int,
                           edge: str = "K", spectrum: str = "EXAFS",
                           corehole: str = "RPA", s02: float = 1.0,
                           scf_radius: float = 4.0, fms_radius: float = 6.0,
                           rpath: float = 8.0, title: str = "Generated by Binah",
                           xanes_emin: float = -30.0, xanes_emax: float = 250.0,
                           xanes_estep: float = 0.25,
                           exchange: int = 0, exchange_vr: float = 0.0,
                           exchange_vi: float = 0.0,
                           scf_nscf: int = 30, scf_ca: float = 0.2,
                           exafs_kmax: float = 20.0,
                           nleg: int = 0,
                           multipole_lmax: int = 0,
                           multipole_iorder: int = 2,
                           polarization: tuple | None = None,
                           ellipticity: tuple | None = None,
                           debye: tuple | None = None) -> str:
    """Build a real-space FEFF input for an isolated molecular cluster.

    Uses the ATOMS / POTENTIALS cards directly with cartesian coordinates
    relative to the absorber. No CIF, no RECIPROCAL, no KMESH — drastically
    faster than the periodic workflow for isolated molecules.
    """
    abs_idx = int(absorber_index) - 1
    if abs_idx < 0 or abs_idx >= structure.atom_count:
        raise ValueError(
            f"Absorber index must be between 1 and {structure.atom_count}."
        )

    abs_sym = structure.symbols[abs_idx]
    abs_xyz = np.asarray(structure.coords[abs_idx], dtype=float)

    # Assign potential indices: 0 = absorber, then one per unique non-absorber
    # element in order of first appearance.
    pot_for_element: dict[str, int] = {}
    next_pot = 1
    for i, sym in enumerate(structure.symbols):
        if i == abs_idx:
            continue
        if sym not in pot_for_element:
            pot_for_element[sym] = next_pot
            next_pot += 1

    edge = str(edge).strip() or "K"
    spectrum = str(spectrum).strip().upper() or "EXAFS"
    corehole = str(corehole).strip() or "RPA"

    def _z(sym: str) -> int:
        z = _ATOMIC_NUMBERS.get(sym)
        if z is None:
            raise ValueError(f"Unknown element symbol: {sym}")
        return z

    lines = [
        f"TITLE {title}",
        f"EDGE {edge}",
        f"S02 {float(s02):.3f}",
        f"COREHOLE {corehole}",
        "CONTROL 1 1 1 1 1 1",
        "PRINT 1 0 0 0 0 0",
        f"EXCHANGE {int(exchange)} {float(exchange_vr):.1f} {float(exchange_vi):.1f} 2",
        f"SCF {float(scf_radius):.1f} 0 {int(scf_nscf)} {float(scf_ca):.2f} 3",
        f"FMS {float(fms_radius):.1f} 0",
    ]

    if int(nleg) > 0:
        lines.append(f"NLEG {int(nleg)}")

    if int(multipole_lmax) >= 2:
        lines.append(f"MULTIPOLE {int(multipole_lmax)} {int(multipole_iorder)}")

    if polarization is not None:
        px, py, pz = polarization
        lines.append(f"POLARIZATION {float(px):.4f} {float(py):.4f} {float(pz):.4f}")

    if ellipticity is not None:
        ellip, ex, ey, ez = ellipticity
        lines.append(
            f"ELLIPTICITY {float(ellip):.4f} {float(ex):.4f} {float(ey):.4f} {float(ez):.4f}"
        )

    if debye is not None:
        temp, dtemp = debye
        lines.append(f"DEBYE {float(temp):.1f} {float(dtemp):.1f}")

    if spectrum == "XANES":
        lines.append(f"XANES {float(xanes_emax):.1f} {float(xanes_estep):.4f} 0.0")
        npts = max(1, int(round((float(xanes_emax) - float(xanes_emin)) / float(xanes_estep))))
        lines.append("EGRID")
        lines.append(f"  {float(xanes_emin):.2f}  {float(xanes_estep):.4f}  {npts}")
    else:
        lines.append(f"EXAFS {float(exafs_kmax):.1f}")
        lines.append(f"RPATH {float(rpath):.1f}")

    # POTENTIALS card.
    lines.append("")
    lines.append("POTENTIALS")
    lines.append("*  ipot   Z   tag   l_scmt  l_fms  stoichiometry")
    lines.append(f"   0   {_z(abs_sym):3d}   {abs_sym:<4s}   -1     -1     0.001")
    elem_counts = Counter(s for j, s in enumerate(structure.symbols) if j != abs_idx)
    # Emit in pot-index order so the file is easy to read.
    for sym, ipot in sorted(pot_for_element.items(), key=lambda kv: kv[1]):
        lines.append(
            f"   {ipot}   {_z(sym):3d}   {sym:<4s}   -1     -1     {elem_counts[sym]}"
        )

    # ATOMS card with absorber at origin, others sorted by distance.
    lines.append("")
    lines.append("ATOMS")
    lines.append("*  x[A]        y[A]        z[A]        ipot  tag        distance[A]")
    lines.append(
        f"  {0.0:11.6f} {0.0:11.6f} {0.0:11.6f}    0   {abs_sym}{abs_idx+1:<5d}   {0.0:9.5f}"
    )
    others = []
    for i, sym in enumerate(structure.symbols):
        if i == abs_idx:
            continue
        rel = np.asarray(structure.coords[i], dtype=float) - abs_xyz
        d = float(np.linalg.norm(rel))
        others.append((d, rel, sym, i + 1))
    others.sort(key=lambda t: t[0])
    for d, rel, sym, idx_1based in others:
        ipot = pot_for_element[sym]
        lines.append(
            f"  {rel[0]:11.6f} {rel[1]:11.6f} {rel[2]:11.6f}    {ipot}   "
            f"{sym}{idx_1based:<5d}   {d:9.5f}"
        )

    lines.append("")
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_feff_cif_input(output_path: str, cif_filename: str, absorber_index: int,
                         **kwargs) -> str:
    out_path = Path(output_path)
    out_path.write_text(
        build_feff_cif_input(cif_filename=cif_filename,
                             absorber_index=absorber_index, **kwargs),
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
                              rpath: float = 8.0,
                              xanes_emin: float = -30.0,
                              xanes_emax: float = 250.0,
                              xanes_estep: float = 0.25,
                              exchange: int = 0,
                              exchange_vr: float = 0.0,
                              exchange_vi: float = 0.0,
                              scf_nscf: int = 30,
                              scf_ca: float = 0.2,
                              exafs_kmax: float = 20.0,
                              nleg: int = 0,
                              multipole_lmax: int = 0,
                              multipole_iorder: int = 2,
                              polarization: tuple | None = None,
                              ellipticity: tuple | None = None,
                              debye: tuple | None = None,
                              molecular_mode: bool = False,
                              cluster_radius: float | None = None,
                              remove_disconnected: bool = False) -> dict:
    full_structure = parse_structure_file(xyz_path)
    target = int(absorber_index)
    if target < 1 or target > full_structure.atom_count:
        raise ValueError(
            f"Absorber index must be between 1 and {full_structure.atom_count} for this XYZ file."
        )

    solvent_removed = 0
    if remove_disconnected:
        full_structure, target, solvent_removed = keep_connected_fragment_around_absorber(
            full_structure, target
        )

    # Optionally crop the structure to a sphere around the absorber. This is
    # the biggest speedup lever for FEFF on a small molecule where the user
    # only cares about the local coordination environment — atoms beyond the
    # FMS radius contribute nothing to the calculation.
    dropped_count = 0
    if cluster_radius is not None and float(cluster_radius) > 0:
        structure, target, dropped_count = crop_structure_around_absorber(
            full_structure, target, float(cluster_radius)
        )
    else:
        structure = full_structure

    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)

    base = str(basename).strip() or full_structure.basename
    safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("._") or "structure"

    if cluster_radius is not None and float(cluster_radius) > 0:
        # Tag the cropped CIF/XYZ filenames with the cutoff so a user can keep
        # multiple variants side-by-side (PK-26a_6.0A.cif, PK-26a_8.0A.cif, …).
        radius_tag = f"{float(cluster_radius):.1f}A"
        cif_path = workdir_path / f"{safe_base}_{radius_tag}.cif"
        xyz_copy_path = workdir_path / f"{safe_base}_{radius_tag}.xyz"
        cleaned_cif_path = workdir_path / f"{safe_base}_{radius_tag}_cleaned.cif"
    else:
        cif_path = workdir_path / f"{safe_base}.cif"
        xyz_copy_path = workdir_path / f"{safe_base}.xyz"
        cleaned_cif_path = workdir_path / f"{safe_base}_cleaned.cif"
    feff_path = workdir_path / "feff.inp"

    # Write a (possibly cropped) XYZ copy alongside the CIF, so the user has
    # a faithful record of the atoms that were actually used.
    if cluster_radius is not None and float(cluster_radius) > 0:
        xyz_copy_path.write_text(structure_to_xyz_text(structure), encoding="utf-8")
    elif Path(xyz_path).suffix.lower() == ".cif":
        xyz_copy_path.write_text(structure_to_xyz_text(structure), encoding="utf-8")
        cleaned_cif_path.write_text(build_cleaned_cif_text(full_structure), encoding="utf-8")
    else:
        xyz_copy_path.write_text(
            Path(xyz_path).read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )

    # Always write the CIF (cheap, useful for visualisation), but only reference
    # it from feff.inp when running in CIF/RECIPROCAL mode.
    cif_meta = write_p1_cif(structure, str(cif_path), padding=padding, cubic=cubic)

    if molecular_mode:
        # Real-space ATOMS-card input — much faster for isolated molecules.
        feff_text = build_feff_atoms_input(
            structure,
            absorber_index=target,
            edge=edge,
            spectrum=spectrum,
            corehole=corehole,
            s02=s02,
            scf_radius=scf_radius,
            fms_radius=fms_radius,
            rpath=rpath,
            title=structure.title or f"{structure.formula} from XYZ (molecular mode)",
            xanes_emin=xanes_emin,
            xanes_emax=xanes_emax,
            xanes_estep=xanes_estep,
            exchange=exchange,
            exchange_vr=exchange_vr,
            exchange_vi=exchange_vi,
            scf_nscf=scf_nscf,
            scf_ca=scf_ca,
            exafs_kmax=exafs_kmax,
            nleg=nleg,
            multipole_lmax=multipole_lmax,
            multipole_iorder=multipole_iorder,
            polarization=polarization,
            ellipticity=ellipticity,
            debye=debye,
        )
        Path(feff_path).write_text(feff_text, encoding="utf-8")
        return {
            "structure": structure,
            "cif_path": str(cif_path),
            "feff_inp_path": str(feff_path),
            "xyz_copy_path": str(xyz_copy_path),
            "cleaned_cif_path": (str(cleaned_cif_path)
                                 if Path(xyz_path).suffix.lower() == ".cif" else ""),
            "cell_lengths": cif_meta["cell_lengths"],
            "padding": float(padding),
            "cubic": bool(cubic),
            "molecular_mode": True,
            "cluster_radius": (float(cluster_radius)
                               if cluster_radius is not None else None),
            "atoms_dropped": int(dropped_count),
            "atoms_total_input": int(full_structure.atom_count),
            "atoms_used": int(structure.atom_count),
            "solvent_removed_atoms": int(solvent_removed),
            "cif_cleaning": {
                k: v for k, v in dict(full_structure.metadata or {}).items()
                if k != "cleaned_cif_text"
            },
        }

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
        xanes_emin=xanes_emin,
        xanes_emax=xanes_emax,
        xanes_estep=xanes_estep,
        exchange=exchange,
        exchange_vr=exchange_vr,
        exchange_vi=exchange_vi,
        scf_nscf=scf_nscf,
        scf_ca=scf_ca,
        exafs_kmax=exafs_kmax,
        nleg=nleg,
        multipole_lmax=multipole_lmax,
        multipole_iorder=multipole_iorder,
        polarization=polarization,
        ellipticity=ellipticity,
        debye=debye,
    )

    return {
        "structure": structure,
        "cif_path": str(cif_path),
        "feff_inp_path": str(feff_path),
        "xyz_copy_path": str(xyz_copy_path),
        "cleaned_cif_path": (str(cleaned_cif_path)
                             if Path(xyz_path).suffix.lower() == ".cif" else ""),
        "cell_lengths": cif_meta["cell_lengths"],
        "padding": float(padding),
        "cubic": bool(cubic),
        "molecular_mode": False,
        "cluster_radius": (float(cluster_radius)
                           if cluster_radius is not None else None),
        "atoms_dropped": int(dropped_count),
        "atoms_total_input": int(full_structure.atom_count),
        "atoms_used": int(structure.atom_count),
        "solvent_removed_atoms": int(solvent_removed),
        "cif_cleaning": {
            k: v for k, v in dict(full_structure.metadata or {}).items()
            if k != "cleaned_cif_text"
        },
    }
