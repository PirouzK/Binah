"""
Clean Olex2/SHELXL-style CIF files for FEFF/FDMNES simulation inputs.

Examples:
  python clean_cif.py path/to/structure.cif
  python clean_cif.py path/to/cif_folder
  python clean_cif.py path/to/cif_folder --recursive
  python clean_cif.py path/to/cif_folder --output-dir path/to/cleaned
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from structure_converter import (
    build_cleaned_cif_text, keep_connected_fragment_around_absorber,
    parse_structure_file,
)


def _iter_cif_paths(target: Path, recursive: bool) -> list[Path]:
    if target.is_file():
        if target.suffix.lower() != ".cif":
            raise ValueError(f"Input file is not a .cif file: {target}")
        return [target]
    if target.is_dir():
        pattern = "**/*.cif" if recursive else "*.cif"
        return sorted(
            p for p in target.glob(pattern)
            if p.is_file() and not p.name.lower().endswith("_cleaned.cif")
        )
    raise ValueError(f"Input path does not exist: {target}")


def _output_path_for(src: Path, *, output_dir: Path | None,
                     input_root: Path | None, suffix: str) -> Path:
    out_name = f"{src.stem}{suffix}.cif"
    if output_dir is None:
        return src.with_name(out_name)
    if input_root is not None and input_root.is_dir():
        rel_parent = src.parent.relative_to(input_root)
        out_dir = output_dir / rel_parent
    else:
        out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / out_name


def _resolve_absorber_index(text: str, structure) -> tuple[int, str]:
    raw = (text or "").strip()
    if not raw:
        return 1, "using atom #1"
    try:
        idx = int(raw)
    except ValueError:
        idx = None
    if idx is not None:
        if idx < 1 or idx > structure.atom_count:
            raise ValueError(f"Absorber index {idx} is out of range (1..{structure.atom_count}).")
        return idx, f"using atom #{idx} ({structure.symbols[idx - 1]})"
    matches = [
        i + 1 for i, sym in enumerate(structure.symbols)
        if sym.lower() == raw.lower()
    ]
    if not matches:
        raise ValueError(f"No absorber atom matching '{raw}' was found.")
    return matches[0], f"using {raw} at atom #{matches[0]}"


def clean_one(src: Path, *, output_dir: Path | None = None,
              input_root: Path | None = None, suffix: str = "_cleaned",
              overwrite: bool = False, remove_solvent: bool = False,
              absorber: str = "Ni") -> tuple[Path, dict]:
    structure = parse_structure_file(str(src))
    solvent_removed = 0
    if remove_solvent:
        absorber_index, _note = _resolve_absorber_index(absorber, structure)
        structure, _absorber_index, solvent_removed = keep_connected_fragment_around_absorber(
            structure, absorber_index
        )
    out_path = _output_path_for(
        src, output_dir=output_dir, input_root=input_root, suffix=suffix
    )
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists, use --overwrite to replace it: {out_path}"
        )
    out_path.write_text(build_cleaned_cif_text(structure), encoding="utf-8")
    cleaning = {
        k: v for k, v in (structure.metadata or {}).items()
        if k != "cleaned_cif_text"
    }
    cleaning["solvent_removed_atoms"] = int(solvent_removed)
    return out_path, cleaning


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write *_cleaned.cif files with overlapping/disordered CIF sites removed."
    )
    parser.add_argument(
        "input",
        help="A .cif file or a directory containing .cif files.",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="When input is a directory, search subdirectories too.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Optional directory for cleaned CIFs. Defaults to next to each input file.",
    )
    parser.add_argument(
        "--suffix",
        default="_cleaned",
        help="Suffix before .cif for output files. Default: _cleaned",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cleaned CIF files.",
    )
    parser.add_argument(
        "--remove-solvent",
        action="store_true",
        help="Keep only the covalent fragment connected to the absorber.",
    )
    parser.add_argument(
        "--absorber",
        default="Ni",
        help="Absorber atom for --remove-solvent. Accepts element symbol or 1-based index. Default: Ni",
    )
    args = parser.parse_args(argv)

    target = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    try:
        paths = _iter_cif_paths(target, args.recursive)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not paths:
        print(f"No .cif files found in {target}")
        return 0

    failures = 0
    input_root = target if target.is_dir() else None
    for src in paths:
        try:
            out_path, cleaning = clean_one(
                src,
                output_dir=output_dir,
                input_root=input_root,
                suffix=args.suffix,
                overwrite=args.overwrite,
                remove_solvent=args.remove_solvent,
                absorber=args.absorber,
            )
            print(
                f"OK  {src} -> {out_path} | "
                f"kept {cleaning.get('cif_atoms_cleaned', '?')}/"
                f"{cleaning.get('cif_atoms_input', '?')} sites, "
                f"removed overlap={cleaning.get('cif_removed_overlap', '?')}, "
                f"zero_occ={cleaning.get('cif_removed_zero_occupancy', '?')}, "
                f"solvent={cleaning.get('solvent_removed_atoms', 0)}"
            )
        except Exception as exc:
            failures += 1
            print(f"FAIL {src} | {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
