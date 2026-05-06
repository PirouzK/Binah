"""
Generate XYZ files for rigid-fragment bond-angle, bond-distance, or dihedral scans.

For angle scans, the central atom B is kept fixed. The whole fragment attached
through A and the whole fragment attached through C are rotated as rigid bodies
so each output structure has the requested A-B-C angle.

For A-B bond-distance scans, B is kept fixed and the whole fragment attached
through A is translated along the B-to-A direction.

For A-B-C-D dihedral scans, B-C is the rotation axis. By default, the whole
fragment attached through D is rotated while A, B, and C stay fixed.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


COVALENT_RADII = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39,
    "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20,
    "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Tc": 1.47,
    "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39,
    "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Sm": 1.98,
    "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90,
    "Yb": 1.87, "Lu": 1.87, "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44,
    "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48,
}


@dataclass
class XYZ:
    title: str
    symbols: list[str]
    coords: np.ndarray


def canonical_symbol(raw: str) -> str:
    match = re.match(r"[A-Za-z]+", raw.strip())
    if not match:
        raise ValueError(f"Could not read element symbol from {raw!r}.")
    text = match.group(0)
    return text[:1].upper() + text[1:].lower()


def read_xyz(path: Path) -> XYZ:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ file must contain at least atom-count and title lines.")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("First XYZ line must be the atom count.") from exc

    symbols: list[str] = []
    coords: list[list[float]] = []
    for line_no, line in enumerate(lines[2:2 + atom_count], start=3):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Line {line_no} does not look like an XYZ atom line.")
        symbols.append(canonical_symbol(parts[0]))
        try:
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError as exc:
            raise ValueError(f"Line {line_no} has non-numeric coordinates.") from exc

    if len(symbols) != atom_count:
        raise ValueError(f"Expected {atom_count} atom lines, found {len(symbols)}.")
    return XYZ(title=lines[1].strip(), symbols=symbols, coords=np.asarray(coords, dtype=float))


def write_xyz(path: Path, xyz: XYZ, title: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{len(xyz.symbols)}\n")
        handle.write(f"{title}\n")
        for symbol, (x, y, z) in zip(xyz.symbols, xyz.coords):
            handle.write(f"{symbol:<2s} {x:16.8f} {y:16.8f} {z:16.8f}\n")


def parse_index_list(text: str | None, atom_count: int, zero_based: bool) -> set[int] | None:
    if not text:
        return None
    items: set[int] = set()
    for chunk in re.split(r"[,\s]+", text.strip()):
        if not chunk:
            continue
        value = int(chunk)
        idx = value if zero_based else value - 1
        if idx < 0 or idx >= atom_count:
            raise ValueError(f"Atom index {value} is outside the structure.")
        items.add(idx)
    return items


def distance_graph(symbols: list[str], coords: np.ndarray, scale: float, tolerance: float) -> list[set[int]]:
    graph = [set() for _ in symbols]
    for i in range(len(symbols)):
        ri = COVALENT_RADII.get(symbols[i], 0.77)
        for j in range(i + 1, len(symbols)):
            rj = COVALENT_RADII.get(symbols[j], 0.77)
            cutoff = scale * (ri + rj) + tolerance
            if float(np.linalg.norm(coords[i] - coords[j])) <= cutoff:
                graph[i].add(j)
                graph[j].add(i)
    return graph


def component_without_center(graph: list[set[int]], start: int, center: int) -> set[int]:
    seen = {center}
    queue: deque[int] = deque([start])
    component: set[int] = set()
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        component.add(node)
        for neighbor in graph[node]:
            if neighbor not in seen:
                queue.append(neighbor)
    return component


def component_without_centers(graph: list[set[int]], start: int, centers: set[int]) -> set[int]:
    seen = set(centers)
    queue: deque[int] = deque([start])
    component: set[int] = set()
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        component.add(node)
        for neighbor in graph[node]:
            if neighbor not in seen:
                queue.append(neighbor)
    return component


def infer_angle_fragments(
    symbols: list[str],
    coords: np.ndarray,
    a_idx: int,
    b_idx: int,
    c_idx: int,
    scale: float,
    tolerance: float,
) -> tuple[set[int], set[int]]:
    graph = distance_graph(symbols, coords, scale, tolerance)
    if a_idx not in graph[b_idx]:
        print(f"Warning: atom A is not connected to B by the distance cutoff.")
    if c_idx not in graph[b_idx]:
        print(f"Warning: atom C is not connected to B by the distance cutoff.")

    a_fragment = component_without_center(graph, a_idx, b_idx)
    c_fragment = component_without_center(graph, c_idx, b_idx)
    overlap = a_fragment & c_fragment
    if overlap:
        one_based = ", ".join(str(i + 1) for i in sorted(overlap))
        raise ValueError(
            "Automatic fragment inference found atoms shared by both moving fragments "
            f"({one_based}). This usually means a ring/bridge; pass --a-fragment and "
            "--c-fragment explicitly."
        )
    return a_fragment, c_fragment


def infer_bond_fragment(
    symbols: list[str],
    coords: np.ndarray,
    a_idx: int,
    b_idx: int,
    scale: float,
    tolerance: float,
) -> set[int]:
    graph = distance_graph(symbols, coords, scale, tolerance)
    if a_idx not in graph[b_idx]:
        print(f"Warning: atom A is not connected to B by the distance cutoff.")
    return component_without_center(graph, a_idx, b_idx)


def infer_dihedral_fragment(
    symbols: list[str],
    coords: np.ndarray,
    start_idx: int,
    axis_start_idx: int,
    axis_end_idx: int,
    scale: float,
    tolerance: float,
) -> set[int]:
    graph = distance_graph(symbols, coords, scale, tolerance)
    return component_without_centers(graph, start_idx, {axis_start_idx, axis_end_idx})


def unit(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError(f"{label} has zero length.")
    return vector / norm


def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = unit(axis, "rotation axis")
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_c = 1.0 - c
    return np.array([
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
    ])


def choose_plane_normal(u_a: np.ndarray, u_c: np.ndarray, requested: Iterable[float] | None) -> np.ndarray:
    if requested is not None:
        normal = np.asarray(list(requested), dtype=float)
        normal = normal - np.dot(normal, u_a) * u_a
        return unit(normal, "--plane-normal projected perpendicular to B-A")

    normal = np.cross(u_a, u_c)
    if np.linalg.norm(normal) > 1e-10:
        return unit(normal, "A-B-C plane normal")

    trial = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(trial, u_a))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    normal = trial - np.dot(trial, u_a) * u_a
    return unit(normal, "automatic plane normal")


def current_angle_deg(coords: np.ndarray, a_idx: int, b_idx: int, c_idx: int) -> float:
    u_a = unit(coords[a_idx] - coords[b_idx], "B-A vector")
    u_c = unit(coords[c_idx] - coords[b_idx], "B-C vector")
    dot = max(-1.0, min(1.0, float(np.dot(u_a, u_c))))
    return math.degrees(math.acos(dot))


def current_distance(coords: np.ndarray, first_idx: int, second_idx: int) -> float:
    return float(np.linalg.norm(coords[first_idx] - coords[second_idx]))


def current_dihedral_deg(coords: np.ndarray, a_idx: int, b_idx: int, c_idx: int, d_idx: int) -> float:
    p0 = coords[a_idx]
    p1 = coords[b_idx]
    p2 = coords[c_idx]
    p3 = coords[d_idx]

    b0 = -(p1 - p0)
    b1 = unit(p2 - p1, "B-C vector")
    b2 = p3 - p2

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    v = unit(v, "A-B-C dihedral plane projection")
    w = unit(w, "B-C-D dihedral plane projection")

    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    return math.degrees(math.atan2(y, x))


def wrap_degrees(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def bend_structure(
    xyz: XYZ,
    a_idx: int,
    b_idx: int,
    c_idx: int,
    a_fragment: set[int],
    c_fragment: set[int],
    target_angle_deg: float,
    plane_normal: Iterable[float] | None,
) -> XYZ:
    coords = np.array(xyz.coords, dtype=float, copy=True)
    b_coord = coords[b_idx].copy()
    u_a = unit(coords[a_idx] - b_coord, "B-A vector")
    u_c = unit(coords[c_idx] - b_coord, "B-C vector")
    theta = math.radians(current_angle_deg(coords, a_idx, b_idx, c_idx))
    target = math.radians(float(target_angle_deg))
    if not (0.0 < target <= math.pi):
        raise ValueError("Target angle must be greater than 0 and no more than 180 degrees.")

    normal = choose_plane_normal(u_a, u_c, plane_normal)
    half_delta = 0.5 * (theta - target)
    rot_a = rotation_matrix(normal, half_delta)
    rot_c = rotation_matrix(normal, -half_delta)

    for idx in sorted(a_fragment):
        coords[idx] = b_coord + rot_a @ (coords[idx] - b_coord)
    for idx in sorted(c_fragment):
        coords[idx] = b_coord + rot_c @ (coords[idx] - b_coord)

    return XYZ(title=xyz.title, symbols=list(xyz.symbols), coords=coords)


def stretch_bond_structure(
    xyz: XYZ,
    a_idx: int,
    b_idx: int,
    a_fragment: set[int],
    target_distance: float,
) -> XYZ:
    if target_distance <= 0:
        raise ValueError("Target bond distance must be positive.")
    coords = np.array(xyz.coords, dtype=float, copy=True)
    b_coord = coords[b_idx].copy()
    u_ab = unit(coords[a_idx] - b_coord, "B-A vector")
    shift = (float(target_distance) - current_distance(coords, a_idx, b_idx)) * u_ab
    for idx in sorted(a_fragment):
        coords[idx] = coords[idx] + shift
    return XYZ(title=xyz.title, symbols=list(xyz.symbols), coords=coords)


def rotate_dihedral_structure(
    xyz: XYZ,
    a_idx: int,
    b_idx: int,
    c_idx: int,
    d_idx: int,
    moving_fragment: set[int],
    target_dihedral_deg: float,
    side: str,
) -> XYZ:
    coords = np.array(xyz.coords, dtype=float, copy=True)
    current = current_dihedral_deg(coords, a_idx, b_idx, c_idx, d_idx)
    delta = math.radians(wrap_degrees(float(target_dihedral_deg) - current))
    axis = unit(coords[c_idx] - coords[b_idx], "B-C rotation axis")
    if side == "a":
        delta *= -1.0
    rot = rotation_matrix(axis, delta)
    origin = coords[b_idx].copy()
    for idx in sorted(moving_fragment):
        coords[idx] = origin + rot @ (coords[idx] - origin)
    return XYZ(title=xyz.title, symbols=list(xyz.symbols), coords=coords)


def make_series(start: float, end: float, step_value: float | None, count: int | None, step_name: str) -> list[float]:
    if count is not None:
        if count < 2:
            return [float(start)]
        return [float(x) for x in np.linspace(float(start), float(end), int(count))]
    if step_value is None:
        raise ValueError(f"Provide either {step_name} or --count.")
    step = abs(float(step_value))
    if step <= 0:
        raise ValueError(f"{step_name} must be positive.")
    direction = 1.0 if end >= start else -1.0
    values: list[float] = []
    value = float(start)
    while (direction > 0 and value <= end + 1e-9) or (direction < 0 and value >= end - 1e-9):
        values.append(round(value, 10))
        value += direction * step
    if abs(values[-1] - end) > 1e-8:
        values.append(float(end))
    return values


def make_angle_series(start: float, end: float, step_deg: float | None, count: int | None) -> list[float]:
    return make_series(start, end, step_deg, count, "--step")


def make_distance_series(start: float, end: float, step_dist: float | None, count: int | None) -> list[float]:
    return make_series(start, end, step_dist, count, "--bond-step")


def make_dihedral_series(start: float, end: float, step_deg: float | None, count: int | None) -> list[float]:
    return make_series(start, end, step_deg, count, "--dihedral-step")


def parse_angle_range(text: str) -> tuple[float, float]:
    cleaned = text.strip().replace(" ", "")
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)-([+-]?\d+(?:\.\d+)?)", cleaned)
    if not match:
        raise ValueError("Angle range must look like 180-145 or 145-180.")
    return float(match.group(1)), float(match.group(2))


def parse_distance_range(text: str) -> tuple[float, float]:
    cleaned = text.strip().replace(" ", "")
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)-([+-]?\d+(?:\.\d+)?)", cleaned)
    if not match:
        raise ValueError("Bond range must look like 1.80-2.20.")
    return float(match.group(1)), float(match.group(2))


def parse_dihedral_range(text: str) -> tuple[float, float]:
    cleaned = text.strip()
    match = re.fullmatch(
        r"\s*([+-]?\d+(?:\.\d+)?)\s*(?:,|:|\s+to\s+)\s*([+-]?\d+(?:\.\d+)?)\s*",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError("Dihedral range must look like 0,180 or -180 to 180.")
    return float(match.group(1)), float(match.group(2))


def safe_angle_label(angle: float) -> str:
    if abs(angle - round(angle)) < 1e-8:
        return f"{int(round(angle)):03d}"
    return f"{angle:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def safe_distance_label(distance: float) -> str:
    return f"{distance:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def safe_signed_angle_label(angle: float) -> str:
    prefix = "m" if angle < 0 else ""
    text = f"{abs(angle):.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return prefix + text


def generate(args: argparse.Namespace) -> list[Path]:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    xyz = read_xyz(input_path)
    offset = 0 if args.zero_based else 1
    a_idx = int(args.a) - offset
    b_idx = int(args.b) - offset
    c_idx = int(args.c) - offset if args.c is not None else None
    d_idx = int(args.d) - offset if getattr(args, "d", None) is not None else None
    if args.dihedral_range or args.dihedral_start is not None or args.dihedral_end is not None:
        scan_mode = "dihedral"
    elif args.bond_range or args.bond_start is not None or args.bond_end is not None:
        scan_mode = "bond"
    else:
        scan_mode = "angle"
    for label, idx in [("A", a_idx), ("B", b_idx)]:
        if idx < 0 or idx >= len(xyz.symbols):
            raise ValueError(f"{label} atom index is outside the structure.")
    if a_idx == b_idx:
        raise ValueError("A and B must be two different atoms.")
    if scan_mode == "angle":
        if c_idx is None:
            raise ValueError("Angle scans require --c.")
        if c_idx < 0 or c_idx >= len(xyz.symbols):
            raise ValueError("C atom index is outside the structure.")
        if len({a_idx, b_idx, c_idx}) != 3:
            raise ValueError("A, B, and C must be three different atoms.")
    if scan_mode == "dihedral":
        if c_idx is None or d_idx is None:
            raise ValueError("Dihedral scans require --c and --d.")
        for label, idx in [("C", c_idx), ("D", d_idx)]:
            if idx < 0 or idx >= len(xyz.symbols):
                raise ValueError(f"{label} atom index is outside the structure.")
        if len({a_idx, b_idx, c_idx, d_idx}) != 4:
            raise ValueError("A, B, C, and D must be four different atoms.")

    explicit_a = parse_index_list(args.a_fragment, len(xyz.symbols), args.zero_based)
    explicit_c = parse_index_list(args.c_fragment, len(xyz.symbols), args.zero_based)
    explicit_d = parse_index_list(args.d_fragment, len(xyz.symbols), args.zero_based)
    if scan_mode == "angle":
        if explicit_a is None or explicit_c is None:
            a_fragment, c_fragment = infer_angle_fragments(
                xyz.symbols, xyz.coords, a_idx, b_idx, c_idx, args.bond_scale, args.bond_tolerance
            )
        else:
            a_fragment, c_fragment = explicit_a, explicit_c
    elif scan_mode == "bond":
        a_fragment = explicit_a or infer_bond_fragment(
            xyz.symbols, xyz.coords, a_idx, b_idx, args.bond_scale, args.bond_tolerance
        )
        c_fragment = set()
    if scan_mode == "dihedral":
        c_fragment = set()
        if args.dihedral_side == "a":
            a_fragment = explicit_a or infer_dihedral_fragment(
                xyz.symbols, xyz.coords, a_idx, b_idx, c_idx, args.bond_scale, args.bond_tolerance
            )
            moving_fragment = a_fragment
        else:
            a_fragment = set()
            moving_fragment = explicit_d or infer_dihedral_fragment(
                xyz.symbols, xyz.coords, d_idx, b_idx, c_idx, args.bond_scale, args.bond_tolerance
            )

    if b_idx in a_fragment or b_idx in c_fragment:
        raise ValueError("Atom B must not be included in a moving fragment.")
    if scan_mode in {"angle", "bond"} and a_idx not in a_fragment:
        raise ValueError("The A-side fragment must include atom A.")
    if scan_mode == "angle" and c_idx not in c_fragment:
        raise ValueError("The C-side fragment must include atom C.")
    if a_fragment & c_fragment:
        raise ValueError("A-side and C-side fragments must not overlap.")
    if scan_mode == "dihedral":
        if b_idx in moving_fragment or c_idx in moving_fragment:
            raise ValueError("The dihedral moving fragment must not include B or C.")
        expected_idx = a_idx if args.dihedral_side == "a" else d_idx
        if expected_idx not in moving_fragment:
            raise ValueError(f"The dihedral moving fragment must include atom {args.dihedral_side.upper()}.")

    stem = input_path.stem
    written: list[Path] = []
    if scan_mode == "angle":
        plane_normal = args.plane_normal
        start = args.start
        end = args.end
        if getattr(args, "angle_range", None):
            start, end = parse_angle_range(args.angle_range)
        angles = make_angle_series(start, end, args.step_deg, args.count)
        for angle in angles:
            bent = bend_structure(xyz, a_idx, b_idx, c_idx, a_fragment, c_fragment, angle, plane_normal)
            measured = current_angle_deg(bent.coords, a_idx, b_idx, c_idx)
            title = (
                f"{xyz.title} | target A-B-C={angle:.6g} deg | "
                f"measured={measured:.6f} deg | atoms {args.a}-{args.b}-{args.c}"
            ).strip()
            out_path = output_dir / f"{stem}_angle_{safe_angle_label(angle)}.xyz"
            write_xyz(out_path, bent, title)
            written.append(out_path)
    elif scan_mode == "bond":
        start = args.bond_start
        end = args.bond_end
        if args.bond_range:
            start, end = parse_distance_range(args.bond_range)
        distances = make_distance_series(start, end, args.bond_step, args.count)
        for distance in distances:
            stretched = stretch_bond_structure(xyz, a_idx, b_idx, a_fragment, distance)
            measured = current_distance(stretched.coords, a_idx, b_idx)
            title = (
                f"{xyz.title} | target A-B={distance:.6g} A | "
                f"measured={measured:.6f} A | atoms {args.a}-{args.b}"
            ).strip()
            out_path = output_dir / f"{stem}_bond_{safe_distance_label(distance)}A.xyz"
            write_xyz(out_path, stretched, title)
            written.append(out_path)
    if scan_mode == "dihedral":
        start = args.dihedral_start
        end = args.dihedral_end
        if args.dihedral_range:
            start, end = parse_dihedral_range(args.dihedral_range)
        dihedrals = make_dihedral_series(start, end, args.dihedral_step, args.count)
        for dihedral in dihedrals:
            rotated = rotate_dihedral_structure(
                xyz, a_idx, b_idx, c_idx, d_idx, moving_fragment, dihedral, args.dihedral_side
            )
            measured = current_dihedral_deg(rotated.coords, a_idx, b_idx, c_idx, d_idx)
            title = (
                f"{xyz.title} | target A-B-C-D={dihedral:.6g} deg | "
                f"measured={measured:.6f} deg | atoms {args.a}-{args.b}-{args.c}-{args.d}"
            ).strip()
            out_path = output_dir / f"{stem}_dihedral_{safe_signed_angle_label(dihedral)}.xyz"
            write_xyz(out_path, rotated, title)
            written.append(out_path)

    print(f"Wrote {len(written)} XYZ files to {output_dir}")
    if scan_mode != "dihedral" or args.dihedral_side == "a":
        print(f"A-side moving atoms: {', '.join(str(i + 1) for i in sorted(a_fragment))}")
    if scan_mode == "angle":
        print(f"C-side moving atoms: {', '.join(str(i + 1) for i in sorted(c_fragment))}")
    if scan_mode == "dihedral":
        print(f"{args.dihedral_side.upper()}-side moving atoms: {', '.join(str(i + 1) for i in sorted(moving_fragment))}")
    return written


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("XYZ Geometry Scan")
    root.resizable(False, False)

    fields: dict[str, tk.StringVar] = {
        "input": tk.StringVar(),
        "output_dir": tk.StringVar(),
        "a": tk.StringVar(),
        "b": tk.StringVar(),
        "c": tk.StringVar(),
        "d": tk.StringVar(),
        "start": tk.StringVar(value="180"),
        "end": tk.StringVar(value="160"),
        "step_deg": tk.StringVar(value="5"),
        "bond_start": tk.StringVar(),
        "bond_end": tk.StringVar(),
        "bond_step": tk.StringVar(value="0.05"),
        "dihedral_start": tk.StringVar(),
        "dihedral_end": tk.StringVar(),
        "dihedral_step": tk.StringVar(value="10"),
        "dihedral_side": tk.StringVar(value="d"),
        "plane_normal": tk.StringVar(),
        "a_fragment": tk.StringVar(),
        "c_fragment": tk.StringVar(),
        "d_fragment": tk.StringVar(),
    }

    def browse_input() -> None:
        value = filedialog.askopenfilename(filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")])
        if value:
            fields["input"].set(value)

    def browse_output() -> None:
        value = filedialog.askdirectory()
        if value:
            fields["output_dir"].set(value)

    def add_row(row: int, label: str, key: str, button_text: str | None = None, command=None) -> None:
        tk.Label(root, text=label, anchor="w").grid(row=row, column=0, sticky="w", padx=8, pady=4)
        tk.Entry(root, textvariable=fields[key], width=54).grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        if button_text:
            tk.Button(root, text=button_text, command=command, width=10).grid(row=row, column=2, padx=8, pady=4)

    add_row(0, "Input XYZ", "input", "Browse", browse_input)
    add_row(1, "Output folder", "output_dir", "Browse", browse_output)
    add_row(2, "A atom index", "a")
    add_row(3, "B atom index", "b")
    add_row(4, "C atom index", "c")
    add_row(5, "D atom index", "d")
    add_row(6, "Start angle", "start")
    add_row(7, "End angle", "end")
    add_row(8, "Step degrees", "step_deg")
    add_row(9, "Bond start A", "bond_start")
    add_row(10, "Bond end A", "bond_end")
    add_row(11, "Bond step A", "bond_step")
    add_row(12, "Dihedral start", "dihedral_start")
    add_row(13, "Dihedral end", "dihedral_end")
    add_row(14, "Dihedral step", "dihedral_step")
    add_row(15, "Dihedral side a/d", "dihedral_side")
    add_row(16, "Plane normal x y z", "plane_normal")
    add_row(17, "A fragment atoms", "a_fragment")
    add_row(18, "C fragment atoms", "c_fragment")
    add_row(19, "D fragment atoms", "d_fragment")

    def run() -> None:
        try:
            normal = fields["plane_normal"].get().strip()
            bond_start = fields["bond_start"].get().strip()
            bond_end = fields["bond_end"].get().strip()
            bond_step = fields["bond_step"].get().strip()
            dihedral_start = fields["dihedral_start"].get().strip()
            dihedral_end = fields["dihedral_end"].get().strip()
            dihedral_step = fields["dihedral_step"].get().strip()
            namespace = argparse.Namespace(
                input=fields["input"].get().strip(),
                output_dir=fields["output_dir"].get().strip(),
                a=int(fields["a"].get()),
                b=int(fields["b"].get()),
                c=int(fields["c"].get()) if fields["c"].get().strip() else None,
                d=int(fields["d"].get()) if fields["d"].get().strip() else None,
                start=float(fields["start"].get()) if fields["start"].get().strip() else None,
                end=float(fields["end"].get()) if fields["end"].get().strip() else None,
                angle_range=None,
                step_deg=float(fields["step_deg"].get()),
                count=None,
                bond_range=None,
                bond_start=float(bond_start) if bond_start else None,
                bond_end=float(bond_end) if bond_end else None,
                bond_step=float(bond_step) if bond_step else None,
                dihedral_range=None,
                dihedral_start=float(dihedral_start) if dihedral_start else None,
                dihedral_end=float(dihedral_end) if dihedral_end else None,
                dihedral_step=float(dihedral_step) if dihedral_step else None,
                dihedral_side=fields["dihedral_side"].get().strip().lower() or "d",
                plane_normal=[float(x) for x in normal.split()] if normal else None,
                a_fragment=fields["a_fragment"].get().strip() or None,
                c_fragment=fields["c_fragment"].get().strip() or None,
                d_fragment=fields["d_fragment"].get().strip() or None,
                zero_based=False,
                bond_scale=1.25,
                bond_tolerance=0.20,
            )
            written = generate(namespace)
            messagebox.showinfo("XYZ Geometry Scan", f"Wrote {len(written)} files.")
        except Exception as exc:
            messagebox.showerror("XYZ Geometry Scan", str(exc))

    tk.Button(root, text="Generate XYZ Files", command=run).grid(row=20, column=0, columnspan=3, pady=10)
    root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate rigid-fragment XYZ files for angle, bond-distance, or dihedral scans.")
    parser.add_argument("--input", "-i", help="Input .xyz file.")
    parser.add_argument("--output-dir", "-o", help="Directory for generated .xyz files.")
    parser.add_argument("--a", type=int, help="A atom index in A-B-C.")
    parser.add_argument("--b", type=int, help="Central B atom index in A-B-C.")
    parser.add_argument("--c", type=int, help="C atom index in A-B-C.")
    parser.add_argument("--d", type=int, help="D atom index in A-B-C-D for dihedral scans.")
    parser.add_argument("--start", type=float, help="Starting angle in degrees.")
    parser.add_argument("--end", type=float, help="Ending angle in degrees.")
    parser.add_argument("--range", dest="angle_range", help="Angle range as START-END, for example 180-145.")
    parser.add_argument("--step-deg", "--step", dest="step_deg", type=float,
                        help="Angle spacing in degrees, inclusive of endpoints.")
    parser.add_argument("--bond-range", help="A-B bond-distance range in Angstrom as START-END, for example 1.80-2.20.")
    parser.add_argument("--bond-start", type=float, help="Starting A-B bond distance in Angstrom.")
    parser.add_argument("--bond-end", type=float, help="Ending A-B bond distance in Angstrom.")
    parser.add_argument("--bond-step", type=float, help="A-B bond-distance spacing in Angstrom.")
    parser.add_argument("--dihedral-range", help="A-B-C-D dihedral range in degrees, for example '0,180' or '-180 to 180'.")
    parser.add_argument("--dihedral-start", type=float, help="Starting A-B-C-D dihedral in degrees.")
    parser.add_argument("--dihedral-end", type=float, help="Ending A-B-C-D dihedral in degrees.")
    parser.add_argument("--dihedral-step", type=float, help="A-B-C-D dihedral spacing in degrees.")
    parser.add_argument("--dihedral-side", choices=["a", "d"], default="d",
                        help="Which side of the B-C axis to rotate for dihedral scans. Default: d.")
    parser.add_argument("--count", type=int, help="Number of output structures including both endpoints.")
    parser.add_argument("--a-fragment", help="Explicit A-side atom indices, comma/space separated. Must include A.")
    parser.add_argument("--c-fragment", help="Explicit C-side atom indices, comma/space separated. Must include C.")
    parser.add_argument("--d-fragment", help="Explicit D-side atom indices, comma/space separated. Must include D.")
    parser.add_argument("--plane-normal", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Bend-plane normal. Useful for exactly linear starting geometries.")
    parser.add_argument("--zero-based", action="store_true", help="Treat atom indices as 0-based instead of 1-based.")
    parser.add_argument("--bond-scale", type=float, default=1.25, help="Covalent-radii multiplier for inferred bonds.")
    parser.add_argument("--bond-tolerance", type=float, default=0.20, help="Extra Angstrom tolerance for inferred bonds.")
    parser.add_argument("--gui", action="store_true", help="Open a small Tkinter GUI.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.gui or len(sys.argv) == 1:
        launch_gui()
        return

    dihedral_mode = bool(args.dihedral_range or args.dihedral_start is not None or args.dihedral_end is not None)
    bond_mode = bool(args.bond_range or args.bond_start is not None or args.bond_end is not None)
    required = ["input", "output_dir", "a", "b"]
    missing = [name.replace("_", "-") for name in required if getattr(args, name) in (None, "")]
    if dihedral_mode:
        if args.c is None:
            missing.append("c")
        if args.d is None:
            missing.append("d")
        if not args.dihedral_range and (args.dihedral_start is None or args.dihedral_end is None):
            missing.extend(["dihedral-start/dihedral-end or dihedral-range"])
        if args.dihedral_step is None and args.count is None:
            missing.append("dihedral-step or count")
    elif bond_mode:
        if not args.bond_range and (args.bond_start is None or args.bond_end is None):
            missing.extend(["bond-start/bond-end or bond-range"])
        if args.bond_step is None and args.count is None:
            missing.append("bond-step or count")
    else:
        if args.c is None:
            missing.append("c")
        if not args.angle_range and (args.start is None or args.end is None):
            missing.extend(["start/end or range"])
        if args.step_deg is None and args.count is None:
            missing.append("step or count")
    if missing:
        parser.error("Missing required arguments: " + ", ".join(f"--{name}" for name in missing))
    generate(args)


if __name__ == "__main__":
    main()
