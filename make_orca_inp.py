#!/usr/bin/env python
"""Interactive ORCA .inp + SLURM .sh generator from XYZ files.

Run with no flags to use the Tk GUI:

    python make_orca_inp.py

Or pass one folder, one XYZ file, or many XYZ files directly:

    python make_orca_inp.py C:\\path\\to\\xyz_files
    python make_orca_inp.py a.xyz b.xyz c.xyz
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_SH_TEMPLATE = """#!/bin/bash

#SBATCH --account={account}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={nprocs}
#SBATCH --mem=0
#SBATCH --time={walltime}
#SBATCH --output={base_name}.out

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 orca/{orca_version}
{gaussian_module}
{nbo_exports}

$EBROOTORCA/orca {base_name}.inp

echo "Program finished with exit code $? at: `date`"
"""


FUNCTIONAL_SUGGESTIONS = [
    "wB97X",
    "wB97X-D3BJ",
    "B3LYP",
    "PBE0",
    "PBE",
    "BP86",
    "TPSSh",
    "M06",
    "M06-2X",
    "CAM-B3LYP",
    "r2SCAN-3c",
    "B97-3c",
    "HF",
]

BASIS_SUGGESTIONS = [
    "def2-SVP",
    "def2-TZVP",
    "def2-TZVPP",
    "def2-QZVP",
    "x2c-SVPall",
    "x2c-TZVPall",
]

POPULATION_SUGGESTIONS = [
    "None",
    "Hirshfeld",
    "NBO",
    "Hirshfeld+NBO",
    "Mulliken",
]

CALC_TYPE_SUGGESTIONS = [
    "Ground/Opt/Freq",
    "Single point",
    "TDDFT UV-vis",
    "TDDFT K-edge XAS",
    "ROCIS L-edge XAS",
    "QTAIM/Molden prep",
]

ORCA_VERSION_SUGGESTIONS = [
    "6.0.1",
    "5.0.4",
]

SCF_SUGGESTIONS = [
    "TightSCF",
    "VeryTightSCF",
    "NormalSCF",
    "SlowConv",
    "VerySlowConv",
]

GRID_SUGGESTIONS = [
    "DEFGRID2",
    "DEFGRID3",
    "DEFGRID1",
    "None",
]

RI_SUGGESTIONS = [
    "RIJCOSX",
    "AutoAux",
    "RIJCOSX AutoAux",
    "RI-JK",
    "NoRI",
    "None",
]

SOLVENT_MODEL_SUGGESTIONS = [
    "None",
    "CPCM",
    "SMD",
    "ALPB",
]

SOLVENT_SUGGESTIONS = [
    "water",
    "h2o",
    "acetonitrile",
    "mecn",
    "ch3cn",
    "acetone",
    "acetic acid",
    "aceticacid",
    "acetophenone",
    "ammonia",
    "aniline",
    "anisole",
    "benzaldehyde",
    "benzene",
    "benzonitrile",
    "benzyl alcohol",
    "benzylalcohol",
    "bromobenzene",
    "bromoethane",
    "bromoform",
    "butanal",
    "butanoic acid",
    "butanone",
    "butanonitrile",
    "butyl acetate",
    "butyl ethanoate",
    "butylacetate",
    "butylamine",
    "butylbenzene",
    "carbon disulfide",
    "carbondisulfide",
    "cs2",
    "carbon tetrachloride",
    "ccl4",
    "chlorobenzene",
    "chloroform",
    "chcl3",
    "cyclohexane",
    "cyclohexanone",
    "cyclopentane",
    "cyclopentanol",
    "cyclopentanone",
    "decalin",
    "cis-decalin",
    "trans-decalin",
    "decane",
    "dibromomethane",
    "dibutylether",
    "dichloromethane",
    "ch2cl2",
    "dcm",
    "diethyl ether",
    "diethylether",
    "diethyl sulfide",
    "diethylamine",
    "diiodomethane",
    "diisopropyl ether",
    "diisopropylether",
    "dimethyl disulfide",
    "dimethylacetamide",
    "n,n-dimethylacetamide",
    "dimethylformamide",
    "n,n-dimethylformamide",
    "dmf",
    "dimethylsulfoxide",
    "dmso",
    "diphenylether",
    "dipropylamine",
    "dodecane",
    "ethanethiol",
    "ethanol",
    "ethyl acetate",
    "ethylacetate",
    "ethyl ethanoate",
    "ethyl methanoate",
    "ethyl phenyl ether",
    "ethoxybenzene",
    "ethylbenzene",
    "fluorobenzene",
    "formamide",
    "formic acid",
    "furan",
    "furane",
    "heptane",
    "hexadecane",
    "hexane",
    "hexanoic acid",
    "iodobenzene",
    "iodoethane",
    "iodomethane",
    "isopropylbenzene",
    "isopropyltoluene",
    "mesitylene",
    "methanol",
    "methyl benzoate",
    "methyl butanoate",
    "methyl ethanoate",
    "methyl methanoate",
    "methyl propanoate",
    "methylcyclohexane",
    "methylformamide",
    "n-methylformamide",
    "nitrobenzene",
    "phno2",
    "nitroethane",
    "nitromethane",
    "meno2",
    "nonane",
    "octane",
    "pentadecane",
    "octanol",
    "octanol(wet)",
    "wetoctanol",
    "woctanol",
    "pentanal",
    "pentane",
    "pentanoic acid",
    "pentyl ethanoate",
    "pentylamine",
    "perfluorobenzene",
    "hexafluorobenzene",
    "phenol",
    "propanal",
    "propanoic acid",
    "propanonitrile",
    "propyl ethanoate",
    "propylamine",
    "pyridine",
    "tetrachloroethene",
    "c2cl4",
    "tetrahydrofuran",
    "thf",
    "sulfolane",
    "tetrahydrothiophenedioxide",
    "tetralin",
    "thiophene",
    "thiophenol",
    "toluene",
    "tributylphosphate",
    "trichloroethene",
    "triethylamine",
    "undecane",
    "xylene",
    "m-xylene",
    "o-xylene",
    "p-xylene",
    "1,1,1-trichloroethane",
    "1,1,2-trichloroethane",
    "1,2,4-trimethylbenzene",
    "1,2-dibromoethane",
    "1,2-dichloroethane",
    "1,2-ethanediol",
    "1,4-dioxane",
    "dioxane",
    "1-bromo-2-methylpropane",
    "1-bromooctane",
    "bromooctane",
    "1-bromopentane",
    "1-bromopropane",
    "1-butanol",
    "butanol",
    "1-chlorohexane",
    "chlorohexane",
    "1-chloropentane",
    "1-chloropropane",
    "1-decanol",
    "decanol",
    "1-fluorooctane",
    "1-heptanol",
    "heptanol",
    "1-hexanol",
    "hexanol",
    "1-hexene",
    "1-hexyne",
    "1-iodobutane",
    "1-iodohexadecane",
    "hexadecyliodide",
    "1-iodopentane",
    "1-iodopropane",
    "1-nitropropane",
    "1-nonanol",
    "nonanol",
    "1-octanol",
    "1-pentanol",
    "pentanol",
    "1-pentene",
    "1-propanol",
    "propanol",
    "2,2,2-trifluoroethanol",
    "2,2,4-trimethylpentane",
    "isooctane",
    "2,4-dimethylpentane",
    "2,4-dimethylpyridine",
    "2,6-dimethylpyridine",
    "2-bromopropane",
    "2-butanol",
    "secbutanol",
    "2-chlorobutane",
    "2-heptanone",
    "2-hexanone",
    "2-methoxyethanol",
    "methoxyethanol",
    "2-methyl-1-propanol",
    "isobutanol",
    "2-methyl-2-propanol",
    "2-methylpentane",
    "2-methylpyridine",
    "2methylpyridine",
    "2-nitropropane",
    "2-octanone",
    "2-pentanone",
    "2-propanol",
    "isopropanol",
    "2-propen-1-ol",
    "e-2-pentene",
    "3-methylpyridine",
    "3-pentanone",
    "4-heptanone",
    "4-methyl-2-pentanone",
    "4methyl2pentanone",
    "4-methylpyridine",
    "5-nonanone",
    "a-chlorotoluene",
    "o-chlorotoluene",
    "m-cresol",
    "mcresol",
    "o-cresol",
    "o-dichlorobenzene",
    "odichlorobenzene",
    "e-1,2-dichloroethene",
    "z-1,2-dichloroethene",
    "cis-1,2-dimethylcyclohexane",
    "n-butylbenzene",
    "sec-butylbenzene",
    "secbutylbenzene",
    "tert-butylbenzene",
    "tbutylbenzene",
    "n-decane",
    "n-dodecane",
    "n-heptane",
    "n-hexadecane",
    "n-hexane",
    "n-methylaniline",
    "n-nonane",
    "n-octane",
    "n-pentadecane",
    "n-pentane",
    "n-undecane",
    "o-nitrotoluene",
    "onitrotoluene",
    "p-isopropyltoluene",
]


@dataclass
class ScanSettings:
    kind: str = "none"
    atoms: list[int] = field(default_factory=list)
    start: float | None = None
    end: float | None = None
    steps: int | None = None


@dataclass
class OrcaSettings:
    root: Path
    xyz_files: list[Path] = field(default_factory=list)
    output_folder: str = "ORCA"
    calculation_type: str = "Ground/Opt/Freq"
    charge: int = 0
    multiplicity: int = 1
    level_of_theory: str = "wB97X"
    basis_set: str = "def2-TZVP"
    aux_basis: str = "def2/J"
    ri_keywords: str = "RIJCOSX"
    scf_convergence: str = "TightSCF"
    grid: str = "DEFGRID2"
    extra_keywords: str = ""
    optimization: bool = True
    freq: bool = True
    dispersion: str = "D3BJ"
    large_print: bool = False
    population: str = "Hirshfeld"
    nprocs: int = 32
    maxcore: int = 12000
    account: str = "def-pierre-ab"
    walltime: str = "01-21:00"
    orca_version: str = "6.0.1"
    solvent_model: str = "None"
    solvent: str = "water"
    tddft: bool = False
    tddft_nroots: int = 50
    tddft_tda: bool = True
    tddft_triplets: bool = True
    tddft_orbwin: str = ""
    tddft_xasloc: str = ""
    rocis_nroots: int = 40
    rocis_maxdim: int = 360
    rocis_orbwin: str = ""
    rocis_soc: bool = True
    rocis_dftcis: bool = True
    rocis_higher_mult: bool = True
    rocis_lower_mult: bool = True
    rocis_pno: bool = False
    rocis_xas_elems: str = ""
    rocis_tcutpno: str = "1e-11"
    scan: ScanSettings = field(default_factory=ScanSettings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ORCA .inp and SLURM .sh files from .xyz files."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Directory to scan, or one/more .xyz files to convert.",
    )
    parser.add_argument("--console", action="store_true", help="Use console prompts instead of the GUI.")
    return parser.parse_args()


def ask_console(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default != "" else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else default


def ask_console_yes_no(prompt: str, default: bool = True) -> bool:
    default_text = "Y/n" if default else "y/N"
    value = input(f"{prompt} [{default_text}]: ").strip().lower()
    if not value:
        return default
    return value.startswith("y")


class TkPrompter:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, simpledialog

        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.simpledialog = simpledialog
        self.root = tk.Tk()
        self.root.withdraw()

    def ask_dir(self, title: str, initialdir: str) -> str:
        return self.filedialog.askdirectory(title=title, initialdir=initialdir)

    def ask_xyz_files(self, title: str, initialdir: str) -> tuple[str, ...]:
        return self.filedialog.askopenfilenames(
            title=title,
            initialdir=initialdir,
            filetypes=(("XYZ files", "*.xyz"), ("All files", "*.*")),
            parent=self.root,
        )

    def ask_string(self, title: str, prompt: str, default: str = "") -> str:
        value = self.simpledialog.askstring(title, prompt, initialvalue=default, parent=self.root)
        return default if value is None else value.strip()

    def ask_integer(self, title: str, prompt: str, default: int) -> int:
        value = self.simpledialog.askinteger(title, prompt, initialvalue=default, parent=self.root)
        return default if value is None else int(value)

    def ask_yes_no(self, title: str, prompt: str, default: bool = True) -> bool:
        if default:
            return bool(self.messagebox.askyesno(title, prompt, parent=self.root))
        return bool(self.messagebox.askyesno(title, prompt, parent=self.root))

    def info(self, title: str, message: str) -> None:
        self.messagebox.showinfo(title, message, parent=self.root)


def parse_bool_text(value: str, default: bool) -> bool:
    text = value.strip().lower()
    if not text:
        return default
    return text in {"y", "yes", "true", "t", "1", "on"}


def parse_atom_pair_list(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for chunk in text.replace(";", ",").split(","):
        pieces = chunk.split()
        if not pieces:
            continue
        if len(pieces) != 2:
            raise ValueError("Bond distance requests must look like '1 2, 3 4'.")
        a, b = int(pieces[0]), int(pieces[1])
        if a < 1 or b < 1:
            raise ValueError("Atom numbers are 1-based and must be positive.")
        pairs.append((a, b))
    return pairs


def parse_atom_triple_list(text: str) -> list[tuple[int, int, int]]:
    triples: list[tuple[int, int, int]] = []
    for chunk in text.replace(";", ",").split(","):
        pieces = chunk.split()
        if not pieces:
            continue
        if len(pieces) != 3:
            raise ValueError("Angle requests must look like '1 2 3, 4 5 6'.")
        a, b, c = int(pieces[0]), int(pieces[1]), int(pieces[2])
        if min(a, b, c) < 1:
            raise ValueError("Atom numbers are 1-based and must be positive.")
        triples.append((a, b, c))
    return triples


def parse_scan(text: str) -> ScanSettings:
    text = text.strip()
    if not text or text.lower() in {"none", "no", "n"}:
        return ScanSettings()

    pieces = text.split()
    kind = pieces[0].lower()
    if kind == "bond":
        if len(pieces) != 6:
            raise ValueError("Bond scan format: bond A B start end steps")
        atoms = [int(pieces[1]), int(pieces[2])]
        start, end, steps = float(pieces[3]), float(pieces[4]), int(pieces[5])
    elif kind == "angle":
        if len(pieces) != 7:
            raise ValueError("Angle scan format: angle A B C start end steps")
        atoms = [int(pieces[1]), int(pieces[2]), int(pieces[3])]
        start, end, steps = float(pieces[4]), float(pieces[5]), int(pieces[6])
    else:
        raise ValueError("Scan must start with 'bond', 'angle', or 'none'.")

    if min(atoms) < 1:
        raise ValueError("Scan atom numbers are 1-based and must be positive.")
    if steps < 2:
        raise ValueError("Scan steps must be at least 2.")
    return ScanSettings(kind=kind, atoms=atoms, start=start, end=end, steps=steps)


def parse_bond_scan(text: str) -> ScanSettings:
    text = text.strip()
    if not text:
        return ScanSettings()
    pieces = text.split()
    if len(pieces) != 5:
        raise ValueError("Bond distance scan format: A B start end steps")
    atoms = [int(pieces[0]), int(pieces[1])]
    if min(atoms) < 1:
        raise ValueError("Bond scan atom numbers are 1-based and must be positive.")
    steps = int(pieces[4])
    if steps < 2:
        raise ValueError("Bond scan steps must be at least 2.")
    return ScanSettings(
        kind="bond",
        atoms=atoms,
        start=float(pieces[2]),
        end=float(pieces[3]),
        steps=steps,
    )


def parse_angle_scan(text: str) -> ScanSettings:
    text = text.strip()
    if not text:
        return ScanSettings()
    pieces = text.split()
    if len(pieces) != 6:
        raise ValueError("Angle scan format: A B C start end steps")
    atoms = [int(pieces[0]), int(pieces[1]), int(pieces[2])]
    if min(atoms) < 1:
        raise ValueError("Angle scan atom numbers are 1-based and must be positive.")
    steps = int(pieces[5])
    if steps < 2:
        raise ValueError("Angle scan steps must be at least 2.")
    return ScanSettings(
        kind="angle",
        atoms=atoms,
        start=float(pieces[3]),
        end=float(pieces[4]),
        steps=steps,
    )


def choose_scan(bond_text: str, angle_text: str) -> ScanSettings:
    scans = [
        scan for scan in (
            parse_bond_scan(bond_text),
            parse_angle_scan(angle_text),
        )
        if scan.kind != "none"
    ]
    if len(scans) > 1:
        raise ValueError("Choose only one scan per generated input set.")
    return scans[0] if scans else ScanSettings()


def common_parent(paths: list[Path]) -> Path:
    if not paths:
        return Path.cwd()
    resolved = [path.resolve() for path in paths]
    if len(resolved) == 1:
        return resolved[0].parent
    return Path(os.path.commonpath([str(path.parent) for path in resolved]))


def resolve_inputs(input_texts: list[str]) -> tuple[Path, list[Path]]:
    if not input_texts:
        return Path.cwd(), []

    paths = [Path(text) for text in input_texts]
    directories = [path for path in paths if path.is_dir()]
    files = [path for path in paths if path.is_file()]
    missing = [path for path in paths if not path.exists()]

    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise ValueError(f"Input path does not exist: {joined}")
    if directories and files:
        raise ValueError("Pass either a directory or selected .xyz files, not both.")
    if len(directories) > 1:
        raise ValueError("Pass only one directory, or pass selected .xyz files.")
    if directories:
        return directories[0], []

    xyz_files = [path for path in files if is_xyz_file(path)]
    skipped = [path for path in files if path.suffix.lower() != ".xyz"]
    if skipped:
        joined = ", ".join(str(path) for path in skipped)
        raise ValueError(f"Selected file is not an .xyz file: {joined}")
    return common_parent(xyz_files), xyz_files


def prompt_settings(args: argparse.Namespace) -> OrcaSettings:
    use_tk = not args.console
    prompter = None
    if use_tk:
        try:
            prompter = TkPrompter()
        except Exception:
            prompter = None

    if args.inputs:
        root, xyz_files = resolve_inputs(args.inputs)
    elif prompter:
        chosen_files = [Path(path) for path in prompter.ask_xyz_files("Choose one or more .xyz files", os.getcwd())]
        if chosen_files:
            root, xyz_files = common_parent(chosen_files), chosen_files
        else:
            chosen = prompter.ask_dir("Or choose a folder with .xyz files", os.getcwd())
            if not chosen:
                raise SystemExit("No .xyz files or folder selected.")
            root, xyz_files = Path(chosen), []
    else:
        selected = ask_console(
            "Folder to scan, or .xyz files separated by semicolons",
            ".",
        )
        input_texts = [chunk.strip().strip('"') for chunk in selected.split(";") if chunk.strip()]
        root, xyz_files = resolve_inputs(input_texts)

    settings = OrcaSettings(root=root, xyz_files=xyz_files)

    if prompter:
        settings.output_folder = prompter.ask_string("Output", "Output folder name", settings.output_folder)
        settings.calculation_type = prompter.ask_string(
            "Calculation Type",
            "Calculation type: Ground/Opt/Freq, Single point, TDDFT UV-vis, TDDFT K-edge XAS, ROCIS L-edge XAS, QTAIM/Molden prep",
            settings.calculation_type,
        )
        settings.charge = prompter.ask_integer("Charge", "Molecular charge", settings.charge)
        settings.multiplicity = prompter.ask_integer("Multiplicity", "Spin multiplicity", settings.multiplicity)
        settings.level_of_theory = prompter.ask_string("Level of Theory", "Functional / method", settings.level_of_theory)
        settings.basis_set = prompter.ask_string("Basis Set", "Basis set", settings.basis_set)
        settings.aux_basis = prompter.ask_string("Aux Basis", "Auxiliary basis; leave blank for none", settings.aux_basis)
        settings.ri_keywords = prompter.ask_string("RI / Aux", "RI/COSX keywords", settings.ri_keywords)
        settings.scf_convergence = prompter.ask_string("SCF", "SCF convergence keyword", settings.scf_convergence)
        settings.grid = prompter.ask_string("Grid", "Grid keyword", settings.grid)
        settings.extra_keywords = prompter.ask_string("Keywords", "Extra ORCA keywords", settings.extra_keywords)
        settings.optimization = prompter.ask_yes_no("Optimization", "Include geometry optimization?")
        settings.freq = prompter.ask_yes_no("Frequency", "Include frequency calculation?")
        settings.large_print = prompter.ask_yes_no("LargePrint", "Include LargePrint?")
        settings.population = prompter.ask_string(
            "Population Analysis",
            "Population analysis: None, Hirshfeld, NBO, Hirshfeld+NBO, or custom keyword(s)",
            settings.population,
        )
        bond_text = prompter.ask_string(
            "Bond Distance Scan",
            "Bond scan: A B start end steps, e.g. '1 2 1.8 2.4 7', or leave blank",
            "",
        )
        angle_text = prompter.ask_string(
            "Angle Scan",
            "Angle scan: A B C start end steps, e.g. '1 2 3 90 140 11', or leave blank",
            "",
        )
        settings.nprocs = prompter.ask_integer("Cores", "Number of ORCA cores", settings.nprocs)
        settings.maxcore = prompter.ask_integer("MaxCore", "MaxCore per core in MB", settings.maxcore)
        settings.walltime = prompter.ask_string("Walltime", "SLURM walltime", settings.walltime)
        settings.orca_version = prompter.ask_string("ORCA Version", "ORCA version", settings.orca_version)
        settings.solvent_model = prompter.ask_string("Solvent Model", "Solvent model: None, CPCM, SMD, ALPB", settings.solvent_model)
        if settings.solvent_model.lower() != "none":
            settings.solvent = prompter.ask_string("Solvent", "Solvent name", settings.solvent)
        if settings.solvent_model.strip().upper() == "SMD":
            settings.freq = False
        settings.tddft = is_tddft_job(settings) or prompter.ask_yes_no("TDDFT", "Include TDDFT block?", settings.tddft)
        if settings.tddft or is_tddft_job(settings):
            settings.tddft_nroots = prompter.ask_integer("TDDFT Roots", "TDDFT roots", settings.tddft_nroots)
            settings.tddft_tda = prompter.ask_yes_no("TDDFT TDA", "TDDFT TDA?", settings.tddft_tda)
            settings.tddft_triplets = prompter.ask_yes_no("TDDFT Triplets", "TDDFT triplets?", settings.tddft_triplets)
            settings.tddft_orbwin = prompter.ask_string("TDDFT OrbWin", "TDDFT/XAS OrbWin[0], e.g. 0,0,-1,-1; blank for none", settings.tddft_orbwin)
            settings.tddft_xasloc = prompter.ask_string("TDDFT XASLoc", "TDDFT/XAS XASLoc[0], e.g. 1,4; blank for none", settings.tddft_xasloc)
        if is_rocis_job(settings):
            settings.rocis_nroots = prompter.ask_integer("ROCIS Roots", "ROCIS roots", settings.rocis_nroots)
            settings.rocis_maxdim = prompter.ask_integer("ROCIS MaxDim", "ROCIS MaxDim", settings.rocis_maxdim)
            settings.rocis_orbwin = prompter.ask_string("ROCIS OrbWin", "ROCIS OrbWin: donor_start,donor_end,acceptor_start,acceptor_end", settings.rocis_orbwin)
    else:
        settings.output_folder = ask_console("Output folder name", settings.output_folder)
        settings.calculation_type = ask_console(
            "Calculation type: Ground/Opt/Freq, Single point, TDDFT UV-vis, TDDFT K-edge XAS, ROCIS L-edge XAS, QTAIM/Molden prep",
            settings.calculation_type,
        )
        settings.charge = int(ask_console("Molecular charge", str(settings.charge)))
        settings.multiplicity = int(ask_console("Spin multiplicity", str(settings.multiplicity)))
        settings.level_of_theory = ask_console("Functional / method", settings.level_of_theory)
        settings.basis_set = ask_console("Basis set", settings.basis_set)
        settings.aux_basis = ask_console("Auxiliary basis; blank for none", settings.aux_basis)
        settings.ri_keywords = ask_console("RI/COSX keywords", settings.ri_keywords)
        settings.scf_convergence = ask_console("SCF convergence keyword", settings.scf_convergence)
        settings.grid = ask_console("Grid keyword", settings.grid)
        settings.extra_keywords = ask_console("Extra ORCA keywords", settings.extra_keywords)
        settings.optimization = ask_console_yes_no("Include geometry optimization?", settings.optimization)
        settings.freq = ask_console_yes_no("Include frequency calculation?", settings.freq)
        settings.large_print = ask_console_yes_no("Include LargePrint?", settings.large_print)
        settings.population = ask_console("Population analysis", settings.population)
        bond_text = ask_console("Bond distance scan, e.g. '1 2 1.8 2.4 7'", "")
        angle_text = ask_console("Angle scan, e.g. '1 2 3 90 140 11'", "")
        settings.nprocs = int(ask_console("Number of ORCA cores", str(settings.nprocs)))
        settings.maxcore = int(ask_console("MaxCore per core in MB", str(settings.maxcore)))
        settings.walltime = ask_console("SLURM walltime", settings.walltime)
        settings.orca_version = ask_console("ORCA version", settings.orca_version)
        settings.solvent_model = ask_console("Solvent model: None, CPCM, SMD, ALPB", settings.solvent_model)
        if settings.solvent_model.lower() != "none":
            settings.solvent = ask_console("Solvent name", settings.solvent)
        if settings.solvent_model.strip().upper() == "SMD":
            settings.freq = False
        settings.tddft = ask_console_yes_no("Include TDDFT block?", settings.tddft)
        if is_tddft_job(settings):
            settings.tddft = True
        if settings.tddft:
            settings.tddft_nroots = int(ask_console("TDDFT roots", str(settings.tddft_nroots)))
            settings.tddft_tda = ask_console_yes_no("TDDFT TDA?", settings.tddft_tda)
            settings.tddft_triplets = ask_console_yes_no("TDDFT triplets?", settings.tddft_triplets)
            settings.tddft_orbwin = ask_console("TDDFT/XAS OrbWin[0], e.g. 0,0,-1,-1; blank for none", settings.tddft_orbwin)
            settings.tddft_xasloc = ask_console("TDDFT/XAS XASLoc[0], e.g. 1,4; blank for none", settings.tddft_xasloc)
        if is_rocis_job(settings):
            settings.rocis_nroots = int(ask_console("ROCIS roots", str(settings.rocis_nroots)))
            settings.rocis_maxdim = int(ask_console("ROCIS MaxDim", str(settings.rocis_maxdim)))
            settings.rocis_orbwin = ask_console(
                "ROCIS OrbWin: donor_start,donor_end,acceptor_start,acceptor_end",
                settings.rocis_orbwin,
            )
            settings.rocis_soc = ask_console_yes_no("ROCIS SOC?", settings.rocis_soc)
            settings.rocis_dftcis = ask_console_yes_no("Use DFT/ROCIS?", settings.rocis_dftcis)
            settings.rocis_higher_mult = ask_console_yes_no("Include higher multiplicity roots?", settings.rocis_higher_mult)
            settings.rocis_lower_mult = ask_console_yes_no("Include lower multiplicity roots?", settings.rocis_lower_mult)
            settings.rocis_pno = ask_console_yes_no("Use PNO-ROCIS speedup?", settings.rocis_pno)
            if settings.rocis_pno:
                settings.rocis_xas_elems = ask_console("XASElems for PNO-ROCIS, e.g. 28 for Ni", settings.rocis_xas_elems)
                settings.rocis_tcutpno = ask_console("TCutPNO", settings.rocis_tcutpno)
    settings.scan = choose_scan(bond_text, angle_text)
    return settings


def is_xyz_file(path: Path) -> bool:
    lower_name = path.name.lower()
    return (
        path.is_file()
        and path.suffix.lower() == ".xyz"
        and not lower_name.endswith("_trj.xyz")
        and not lower_name.endswith("_xas.xyz")
    )


def find_xyz_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for child in sorted(root.iterdir()):
        if is_xyz_file(child):
            files.append(child)
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.lower() in {"orca", "tddft"}:
            continue
        try:
            for grandchild in sorted(child.iterdir()):
                if is_xyz_file(grandchild):
                    files.append(grandchild)
        except PermissionError:
            continue
    return files


def unique_output_dir(parent: Path, preferred_name: str) -> Path:
    candidate = parent / preferred_name
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        candidate = parent / f"{preferred_name}-{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def normalize_population(text: str) -> list[str]:
    value = text.strip()
    lower = value.lower()
    if not value or lower in {"none", "no", "n"}:
        return []
    if lower == "hirshfeld+nbo":
        return ["Hirshfeld", "NBO"]
    if lower in {"hirshfeld", "nbo", "mulliken"}:
        if lower == "mulliken":
            return []
        return [value]
    return value.split()


def wants_nbo(settings: OrcaSettings) -> bool:
    return "nbo" in settings.population.lower()


def calc_type_key(settings: OrcaSettings) -> str:
    return settings.calculation_type.strip().lower()


def is_tddft_job(settings: OrcaSettings) -> bool:
    key = calc_type_key(settings)
    return settings.tddft or "tddft" in key


def is_xas_tddft_job(settings: OrcaSettings) -> bool:
    key = calc_type_key(settings)
    return "k-edge" in key or ("xas" in key and "tddft" in key)


def is_rocis_job(settings: OrcaSettings) -> bool:
    key = calc_type_key(settings)
    return "rocis" in key or "l-edge" in key


def is_spectroscopy_job(settings: OrcaSettings) -> bool:
    return is_tddft_job(settings) or is_rocis_job(settings)


def allows_opt_freq(settings: OrcaSettings) -> bool:
    key = calc_type_key(settings)
    return not is_spectroscopy_job(settings) and ("ground" in key or "opt" in key or not key)


def keyword_tokens(text: str) -> list[str]:
    value = text.strip()
    if not value or value.lower() == "none":
        return []
    return value.split()


def dispersion_needed(settings: OrcaSettings) -> bool:
    method_upper = settings.level_of_theory.upper()
    dispersion_upper = settings.dispersion.upper()
    return bool(settings.dispersion.strip()) and dispersion_upper not in method_upper


def solvent_keywords(settings: OrcaSettings) -> list[str]:
    model = settings.solvent_model.strip().upper()
    solvent = settings.solvent.strip()
    if not solvent or model in {"", "NONE"}:
        return []
    if model in {"CPCM", "ALPB"}:
        return [f"{model}({solvent})"]
    if model == "SMD":
        return [f"CPCM({solvent})"]
    return []


def build_cpcm_block(settings: OrcaSettings) -> str:
    if settings.solvent_model.strip().upper() != "SMD":
        return ""
    solvent = settings.solvent.strip()
    if not solvent:
        return ""
    return "\n".join([
        "%cpcm",
        "  smd true",
        f'  SMDsolvent "{solvent}"',
        "end",
        "",
    ])


def build_tddft_block(settings: OrcaSettings) -> str:
    if not is_tddft_job(settings):
        return ""
    tda = "true" if settings.tddft_tda else "false"
    triplets = "true" if settings.tddft_triplets else "false"
    lines = [
        "%tddft",
        f"  nroots {settings.tddft_nroots}",
        f"  tda {tda}",
        f"  triplets {triplets}",
    ]
    if settings.tddft_xasloc.strip():
        lines.append(f"  XASLoc[0] = {settings.tddft_xasloc.strip()}")
    if settings.tddft_orbwin.strip():
        lines.append(f"  OrbWin[0] = {settings.tddft_orbwin.strip()}")
    if is_xas_tddft_job(settings):
        lines.append("  DoHigherMoments true")
        lines.append("  DoFullSemiclassical true")
    lines.extend(["end", ""])
    return "\n".join(lines)


def build_rocis_block(settings: OrcaSettings) -> str:
    if not is_rocis_job(settings):
        return ""
    soc = "true" if settings.rocis_soc else "false"
    dftcis = "true" if settings.rocis_dftcis else "false"
    higher = "true" if settings.rocis_higher_mult else "false"
    lower = "true" if settings.rocis_lower_mult else "false"
    pno = "true" if settings.rocis_pno else "false"
    lines = [
        "%rocis",
        f"  NRoots {settings.rocis_nroots}",
        f"  MaxDim {settings.rocis_maxdim}",
        f"  MaxCore {settings.maxcore}",
        f"  SOC {soc}",
        f"  DoHigherMult {higher}",
        f"  DoLowerMult {lower}",
        "  DoRI true",
        f"  DoDFTCIS {dftcis}",
        "  DFTCIS_c = 0.18, 0.20, 0.40",
        "  PrintLevel 3",
    ]
    if settings.rocis_orbwin.strip():
        lines.append(f"  OrbWin = {settings.rocis_orbwin.strip()}")
    else:
        lines.append("  # TODO: Set OrbWin to the metal 2p donor MOs and valence/virtual acceptor window.")
        lines.append("  # Example only: OrbWin = 5,7,50,120")
    if settings.rocis_pno:
        lines.append(f"  DoPNO {pno}")
        lines.append(f"  TCutPNO {settings.rocis_tcutpno.strip() or '1e-11'}")
        if settings.rocis_xas_elems.strip():
            lines.append(f"  XASElems {settings.rocis_xas_elems.strip()}")
    lines.extend(["end", ""])
    return "\n".join(lines)


def build_geom_block(settings: OrcaSettings) -> str:
    lines: list[str] = []
    if settings.scan.kind != "none":
        lines.append("%geom")
        lines.append("  # Scan atom numbers were entered as 1-based indices and converted to ORCA indices.")
    if settings.scan.kind == "bond":
        a, b = [idx - 1 for idx in settings.scan.atoms]
        lines.append("  Scan")
        lines.append(f"    B {a} {b} = {settings.scan.start:.6g}, {settings.scan.end:.6g}, {settings.scan.steps}")
        lines.append("  end")
    elif settings.scan.kind == "angle":
        a, b, c = [idx - 1 for idx in settings.scan.atoms]
        lines.append("  Scan")
        lines.append(f"    A {a} {b} {c} = {settings.scan.start:.6g}, {settings.scan.end:.6g}, {settings.scan.steps}")
        lines.append("  end")
    if lines:
        lines.append("end")
        return "\n".join(lines) + "\n\n"
    return ""


def build_inp_text(settings: OrcaSettings, xyz_name: str) -> str:
    keywords = [
        settings.level_of_theory,
        settings.basis_set,
    ]
    if settings.aux_basis:
        keywords.append(settings.aux_basis)
    keywords.extend(keyword_tokens(settings.ri_keywords))
    keywords.extend(keyword_tokens(settings.scf_convergence))
    keywords.extend(keyword_tokens(settings.grid))
    keywords.extend(solvent_keywords(settings))
    keywords.extend(keyword_tokens(settings.extra_keywords))
    if settings.optimization and allows_opt_freq(settings):
        keywords.append("Opt")
    if settings.freq and allows_opt_freq(settings) and settings.solvent_model.strip().upper() != "SMD":
        keywords.append("Freq")
    if dispersion_needed(settings):
        keywords.append(settings.dispersion)
    if settings.large_print:
        keywords.append("LargePrint")
    keywords.extend(normalize_population(settings.population))

    lines = [
        f"%pal nprocs {settings.nprocs} end",
        f"%maxcore {settings.maxcore}",
        "",
        "! " + " ".join(keywords),
        "",
    ]

    if wants_nbo(settings):
        lines.append("# NBO requested. The generated .sh loads Gaussian and exports NBO executables.")
        lines.append("")

    cpcm_block = build_cpcm_block(settings)
    if cpcm_block:
        lines.append(cpcm_block.rstrip())
        lines.append("")

    tddft_block = build_tddft_block(settings)
    if tddft_block:
        lines.append(tddft_block.rstrip())
        lines.append("")

    rocis_block = build_rocis_block(settings)
    if rocis_block:
        lines.append(rocis_block.rstrip())
        lines.append("")

    geom_block = build_geom_block(settings)
    if geom_block and allows_opt_freq(settings):
        lines.append(geom_block.rstrip())
        lines.append("")

    lines.append(f"*xyzfile {settings.charge} {settings.multiplicity} {xyz_name}")
    return "\n".join(lines) + "\n"


def build_sh_text(settings: OrcaSettings, base_name: str) -> str:
    use_nbo = wants_nbo(settings)
    gaussian_module = "module load gaussian/g16.c01\n" if use_nbo else ""
    nbo_exports = (
        "export GENEXE=`which gennbo.i4.exe`\n"
        "export NBOEXE=`which nbo7.i4.exe`"
        if use_nbo else ""
    )
    return DEFAULT_SH_TEMPLATE.format(
        account=settings.account,
        nprocs=settings.nprocs,
        walltime=settings.walltime,
        orca_version=settings.orca_version,
        gaussian_module=gaussian_module,
        nbo_exports=nbo_exports,
        base_name=base_name,
    )


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="ascii", newline="\n")


def create_jobs(settings: OrcaSettings) -> list[Path]:
    root = settings.root.resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    xyz_files = settings.xyz_files or find_xyz_files(root)
    if not xyz_files:
        return []

    output_root = root / settings.output_folder
    output_root.mkdir(exist_ok=True)
    job_dirs: list[Path] = []

    for xyz_file in xyz_files:
        base_name = xyz_file.stem
        job_dir = unique_output_dir(output_root, base_name)
        job_dir.mkdir(parents=True)
        shutil.copy2(xyz_file, job_dir / xyz_file.name)

        inp_path = job_dir / f"{base_name}.inp"
        sh_path = job_dir / f"{base_name}.sh"
        write_text(inp_path, build_inp_text(settings, xyz_file.name))
        write_text(sh_path, build_sh_text(settings, base_name))
        job_dirs.append(job_dir)

    return job_dirs


def run_gui(args: argparse.Namespace) -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    selected_files: list[Path] = []
    selected_folder: Path | None = None

    def initial_inputs() -> None:
        nonlocal selected_folder
        if not args.inputs:
            return
        root_path, xyz_files = resolve_inputs(args.inputs)
        if xyz_files:
            selected_files.extend(xyz_files)
        else:
            selected_folder = root_path

    initial_inputs()

    app = tk.Tk()
    app.title("ORCA Input Builder")
    app.geometry("1040x900")
    app.minsize(900, 780)

    style = ttk.Style(app)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    main_frame = ttk.Frame(app, padding=12)
    main_frame.grid(row=0, column=0, sticky="nsew")
    app.columnconfigure(0, weight=1)
    app.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(1, weight=1)

    file_frame = ttk.LabelFrame(main_frame, text="XYZ structures", padding=10)
    file_frame.grid(row=0, column=0, sticky="ew")
    file_frame.columnconfigure(0, weight=1)

    file_list = tk.Listbox(file_frame, height=5, selectmode=tk.EXTENDED)
    file_list.grid(row=0, column=0, rowspan=3, sticky="ew")
    file_scroll = ttk.Scrollbar(file_frame, orient="vertical", command=file_list.yview)
    file_scroll.grid(row=0, column=1, rowspan=3, sticky="ns")
    file_list.configure(yscrollcommand=file_scroll.set)

    def refresh_files() -> None:
        file_list.delete(0, tk.END)
        if selected_folder is not None:
            file_list.insert(tk.END, f"Scan folder: {selected_folder}")
        for path in selected_files:
            file_list.insert(tk.END, str(path))

    def add_files() -> None:
        nonlocal selected_folder
        paths = filedialog.askopenfilenames(
            title="Choose one or more .xyz files",
            filetypes=(("XYZ files", "*.xyz"), ("All files", "*.*")),
            parent=app,
        )
        if not paths:
            return
        selected_folder = None
        existing = {path.resolve() for path in selected_files}
        for text in paths:
            path = Path(text)
            if not is_xyz_file(path):
                continue
            if path.resolve() not in existing:
                selected_files.append(path)
                existing.add(path.resolve())
        refresh_files()

    def choose_folder() -> None:
        nonlocal selected_folder
        path = filedialog.askdirectory(title="Choose folder to scan for .xyz files", parent=app)
        if not path:
            return
        selected_files.clear()
        selected_folder = Path(path)
        refresh_files()

    def clear_files() -> None:
        nonlocal selected_folder
        selected_files.clear()
        selected_folder = None
        refresh_files()

    ttk.Button(file_frame, text="Add XYZ files", command=add_files).grid(row=0, column=2, padx=(10, 0), sticky="ew")
    ttk.Button(file_frame, text="Scan folder", command=choose_folder).grid(row=1, column=2, padx=(10, 0), pady=4, sticky="ew")
    ttk.Button(file_frame, text="Clear", command=clear_files).grid(row=2, column=2, padx=(10, 0), sticky="ew")

    form_frame = ttk.Frame(main_frame)
    form_frame.grid(row=1, column=0, pady=(12, 0), sticky="nsew")
    form_frame.columnconfigure(0, weight=1)
    form_frame.columnconfigure(1, weight=1)

    calc_frame = ttk.LabelFrame(form_frame, text="Calculation", padding=10)
    calc_frame.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
    job_frame = ttk.LabelFrame(form_frame, text="Job", padding=10)
    job_frame.grid(row=0, column=1, padx=(6, 0), sticky="nsew")
    scan_frame = ttk.LabelFrame(form_frame, text="Scan", padding=10)
    scan_frame.grid(row=1, column=0, columnspan=2, pady=(12, 0), sticky="ew")
    mode_notebook = ttk.Notebook(form_frame)
    mode_notebook.grid(row=2, column=0, columnspan=2, pady=(12, 0), sticky="ew")
    opt_tab = ttk.Frame(mode_notebook, padding=10)
    sp_tab = ttk.Frame(mode_notebook, padding=10)
    uvvis_tab = ttk.Frame(mode_notebook, padding=10)
    kedge_tab = ttk.Frame(mode_notebook, padding=10)
    ledge_tab = ttk.Frame(mode_notebook, padding=10)
    qtaim_tab = ttk.Frame(mode_notebook, padding=10)
    mode_notebook.add(opt_tab, text="Optimization")
    mode_notebook.add(sp_tab, text="Single Point")
    mode_notebook.add(uvvis_tab, text="UV-Vis")
    mode_notebook.add(kedge_tab, text="K-edge XAS")
    mode_notebook.add(ledge_tab, text="L-edge XAS")
    mode_notebook.add(qtaim_tab, text="QTAIM/Molden")

    def entry(parent: ttk.Frame, row: int, label: str, default: str, width: int = 24) -> tk.StringVar:
        var = tk.StringVar(value=default)
        ttk.Label(parent, text=label).grid(row=row, column=0, padx=(0, 8), pady=4, sticky="w")
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=1, pady=4, sticky="ew")
        parent.columnconfigure(1, weight=1)
        return var

    def combo(
        parent: ttk.Frame,
        row: int,
        label: str,
        default: str,
        values: list[str],
        width: int = 24,
    ) -> tk.StringVar:
        var = tk.StringVar(value=default)
        ttk.Label(parent, text=label).grid(row=row, column=0, padx=(0, 8), pady=4, sticky="w")
        ttk.Combobox(parent, textvariable=var, values=values, width=width).grid(row=row, column=1, pady=4, sticky="ew")
        parent.columnconfigure(1, weight=1)
        return var

    output_var = entry(calc_frame, 0, "Output folder", "ORCA")
    charge_var = entry(calc_frame, 1, "Charge", "0")
    mult_var = entry(calc_frame, 2, "Multiplicity", "1")
    method_var = combo(calc_frame, 3, "Level of theory", "wB97X", FUNCTIONAL_SUGGESTIONS)
    basis_var = combo(calc_frame, 4, "Basis set", "def2-TZVP", BASIS_SUGGESTIONS)
    aux_var = entry(calc_frame, 5, "Aux basis", "def2/J")
    ri_var = combo(calc_frame, 6, "RI / COSX", "RIJCOSX", RI_SUGGESTIONS)
    scf_var = combo(calc_frame, 7, "SCF", "TightSCF", SCF_SUGGESTIONS)
    grid_var = combo(calc_frame, 8, "Grid", "DEFGRID2", GRID_SUGGESTIONS)
    keywords_var = entry(calc_frame, 9, "Extra keywords", "")
    population_var = combo(calc_frame, 10, "Population", "Hirshfeld", POPULATION_SUGGESTIONS)

    opt_var = tk.BooleanVar(value=True)
    freq_var = tk.BooleanVar(value=True)
    large_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(calc_frame, text="LargePrint", variable=large_var).grid(row=11, column=0, pady=(8, 2), sticky="w")

    cores_var = entry(job_frame, 0, "Cores", "32")
    maxcore_var = entry(job_frame, 1, "MaxCore MB", "12000")
    account_var = entry(job_frame, 2, "SLURM account", "def-pierre-ab")
    walltime_var = entry(job_frame, 3, "Walltime", "01-21:00")
    orca_version_var = combo(job_frame, 4, "ORCA version", "6.0.1", ORCA_VERSION_SUGGESTIONS)

    solvent_model_var = combo(job_frame, 5, "Solvation", "None", SOLVENT_MODEL_SUGGESTIONS)
    solvent_var = combo(job_frame, 6, "Solvent", "water", SOLVENT_SUGGESTIONS)
    ttk.Checkbutton(opt_tab, text="Optimization", variable=opt_var).grid(row=0, column=0, padx=(0, 18), sticky="w")
    ttk.Checkbutton(opt_tab, text="Freq", variable=freq_var).grid(row=0, column=1, padx=(0, 18), sticky="w")
    ttk.Label(opt_tab, text="SMD automatically disables Freq. Use this tab for geometry, frequency, NBO, or QTAIM prep.").grid(
        row=1, column=0, columnspan=4, pady=(8, 0), sticky="w"
    )
    ttk.Label(sp_tab, text="Single-point job: no Opt/Freq, but keeps your method, basis, solvation, and population settings.").grid(
        row=0, column=0, columnspan=4, sticky="w"
    )
    ttk.Label(qtaim_tab, text="QTAIM/Molden prep: generate a clean single-point wavefunction, then convert the .gbw with orca_2mkl.").grid(
        row=0, column=0, columnspan=4, sticky="w"
    )

    tddft_var = tk.BooleanVar(value=False)
    tddft_tda_var = tk.BooleanVar(value=True)
    tddft_triplets_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(uvvis_tab, text="TDDFT block", variable=tddft_var).grid(row=0, column=0, padx=(0, 12), sticky="w")
    tddft_roots_var = tk.StringVar(value="50")
    ttk.Label(uvvis_tab, text="TDDFT roots").grid(row=0, column=1, padx=(12, 6), sticky="w")
    ttk.Entry(uvvis_tab, textvariable=tddft_roots_var, width=8).grid(row=0, column=2, sticky="w")
    ttk.Checkbutton(uvvis_tab, text="TDA", variable=tddft_tda_var).grid(row=0, column=3, padx=(18, 6), sticky="w")
    ttk.Checkbutton(uvvis_tab, text="Triplets", variable=tddft_triplets_var).grid(row=0, column=4, padx=(6, 0), sticky="w")
    ttk.Label(uvvis_tab, text="Use this tab for ordinary valence excited states / UV-Vis.").grid(
        row=1, column=0, columnspan=5, pady=(6, 0), sticky="w"
    )
    tddft_orbwin_var = tk.StringVar(value="")
    tddft_xasloc_var = tk.StringVar(value="")
    ttk.Label(kedge_tab, text="TDDFT roots").grid(row=0, column=0, padx=(0, 6), sticky="w")
    ttk.Entry(kedge_tab, textvariable=tddft_roots_var, width=8).grid(row=0, column=1, sticky="w")
    ttk.Checkbutton(kedge_tab, text="TDA", variable=tddft_tda_var).grid(row=0, column=2, padx=(18, 6), sticky="w")
    ttk.Checkbutton(kedge_tab, text="Triplets", variable=tddft_triplets_var).grid(row=0, column=3, padx=(6, 0), sticky="w")
    ttk.Label(kedge_tab, text="TDDFT OrbWin[0]").grid(row=1, column=0, padx=(0, 6), pady=(8, 0), sticky="w")
    ttk.Entry(kedge_tab, textvariable=tddft_orbwin_var, width=22).grid(row=1, column=1, columnspan=2, pady=(8, 0), sticky="ew")
    ttk.Label(kedge_tab, text="XASLoc[0]").grid(row=1, column=3, padx=(18, 6), pady=(8, 0), sticky="w")
    ttk.Entry(kedge_tab, textvariable=tddft_xasloc_var, width=16).grid(row=1, column=4, pady=(8, 0), sticky="ew")
    ttk.Label(kedge_tab, text="Use K-edge for 1s core excitations. For Ni L-edge, use the L-edge tab.").grid(
        row=2, column=0, columnspan=5, pady=(6, 0), sticky="w"
    )

    rocis_roots_var = tk.StringVar(value="40")
    rocis_maxdim_var = tk.StringVar(value="360")
    rocis_orbwin_var = tk.StringVar(value="")
    ttk.Label(ledge_tab, text="ROCIS roots").grid(row=0, column=0, padx=(0, 6), sticky="w")
    ttk.Entry(ledge_tab, textvariable=rocis_roots_var, width=8).grid(row=0, column=1, sticky="w")
    ttk.Label(ledge_tab, text="MaxDim").grid(row=0, column=2, padx=(12, 6), sticky="w")
    ttk.Entry(ledge_tab, textvariable=rocis_maxdim_var, width=8).grid(row=0, column=3, sticky="w")
    ttk.Label(ledge_tab, text="ROCIS OrbWin").grid(row=1, column=0, padx=(0, 6), pady=(8, 0), sticky="w")
    ttk.Entry(ledge_tab, textvariable=rocis_orbwin_var, width=36).grid(row=1, column=1, columnspan=4, pady=(8, 0), sticky="ew")
    rocis_soc_var = tk.BooleanVar(value=True)
    rocis_dftcis_var = tk.BooleanVar(value=True)
    rocis_higher_var = tk.BooleanVar(value=True)
    rocis_lower_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(ledge_tab, text="SOC", variable=rocis_soc_var).grid(row=2, column=0, pady=(8, 0), sticky="w")
    ttk.Checkbutton(ledge_tab, text="DFT/ROCIS", variable=rocis_dftcis_var).grid(row=2, column=1, pady=(8, 0), sticky="w")
    ttk.Checkbutton(ledge_tab, text="Higher mult", variable=rocis_higher_var).grid(row=2, column=2, pady=(8, 0), sticky="w")
    ttk.Checkbutton(ledge_tab, text="Lower mult", variable=rocis_lower_var).grid(row=2, column=3, pady=(8, 0), sticky="w")
    ttk.Label(
        ledge_tab,
        text="For Ni L-edge: set ROCIS OrbWin after finding the Ni 2p MO numbers in a ground-state output.",
    ).grid(row=3, column=0, columnspan=5, pady=(6, 0), sticky="w")

    scan_kind_var = tk.StringVar(value="none")
    ttk.Radiobutton(scan_frame, text="No scan", value="none", variable=scan_kind_var).grid(row=0, column=0, sticky="w")
    ttk.Radiobutton(scan_frame, text="Bond distance", value="bond", variable=scan_kind_var).grid(row=0, column=1, sticky="w")
    ttk.Radiobutton(scan_frame, text="Angle", value="angle", variable=scan_kind_var).grid(row=0, column=2, sticky="w")

    atoms_var = entry(scan_frame, 1, "Atoms", "")
    start_var = entry(scan_frame, 2, "Start", "")
    end_var = entry(scan_frame, 3, "End", "")
    steps_var = entry(scan_frame, 4, "Steps", "")
    ttk.Label(
        scan_frame,
        text="Use 1-based atom numbers. Bond atoms: '1 2'. Angle atoms: '1 2 3'.",
    ).grid(row=5, column=0, columnspan=3, pady=(6, 0), sticky="w")

    status_var = tk.StringVar(value="")
    ttk.Label(main_frame, textvariable=status_var).grid(row=2, column=0, pady=(10, 0), sticky="w")

    def scan_from_gui() -> ScanSettings:
        kind = scan_kind_var.get()
        if kind == "none":
            return ScanSettings()
        atoms = atoms_var.get().split()
        if kind == "bond":
            if len(atoms) != 2:
                raise ValueError("Bond scan needs exactly 2 atoms, for example: 1 2")
            text = f"{atoms[0]} {atoms[1]} {start_var.get()} {end_var.get()} {steps_var.get()}"
            return parse_bond_scan(text)
        if len(atoms) != 3:
            raise ValueError("Angle scan needs exactly 3 atoms, for example: 1 2 3")
        text = f"{atoms[0]} {atoms[1]} {atoms[2]} {start_var.get()} {end_var.get()} {steps_var.get()}"
        return parse_angle_scan(text)

    def calculation_type_from_tab() -> str:
        tab_text = mode_notebook.tab(mode_notebook.select(), "text")
        return {
            "Optimization": "Ground/Opt/Freq",
            "Single Point": "Single point",
            "UV-Vis": "TDDFT UV-vis",
            "K-edge XAS": "TDDFT K-edge XAS",
            "L-edge XAS": "ROCIS L-edge XAS",
            "QTAIM/Molden": "QTAIM/Molden prep",
        }.get(tab_text, "Ground/Opt/Freq")

    def settings_from_gui() -> OrcaSettings:
        if selected_folder is None and not selected_files:
            raise ValueError("Choose one or more .xyz files, or choose a folder to scan.")
        root_path = selected_folder if selected_folder is not None else common_parent(selected_files)
        settings = OrcaSettings(root=root_path, xyz_files=list(selected_files))
        settings.output_folder = output_var.get().strip() or "ORCA"
        settings.calculation_type = calculation_type_from_tab()
        settings.charge = int(charge_var.get())
        settings.multiplicity = int(mult_var.get())
        settings.level_of_theory = method_var.get().strip() or "wB97X"
        settings.basis_set = basis_var.get().strip() or "def2-TZVP"
        settings.aux_basis = aux_var.get().strip()
        settings.ri_keywords = ri_var.get().strip()
        settings.scf_convergence = scf_var.get().strip()
        settings.grid = grid_var.get().strip()
        settings.extra_keywords = keywords_var.get().strip()
        settings.optimization = bool(opt_var.get())
        settings.freq = bool(freq_var.get())
        settings.large_print = bool(large_var.get())
        settings.population = population_var.get().strip()
        settings.nprocs = int(cores_var.get())
        settings.maxcore = int(maxcore_var.get())
        settings.account = account_var.get().strip() or "def-pierre-ab"
        settings.walltime = walltime_var.get().strip() or "01-21:00"
        settings.orca_version = orca_version_var.get().strip() or "6.0.1"
        settings.solvent_model = solvent_model_var.get().strip() or "None"
        settings.solvent = solvent_var.get().strip() or "water"
        if settings.solvent_model.strip().upper() == "SMD":
            settings.freq = False
        settings.tddft = bool(tddft_var.get())
        if is_tddft_job(settings):
            settings.tddft = True
        settings.tddft_nroots = int(tddft_roots_var.get())
        settings.tddft_tda = bool(tddft_tda_var.get())
        settings.tddft_triplets = bool(tddft_triplets_var.get())
        settings.tddft_orbwin = tddft_orbwin_var.get().strip()
        settings.tddft_xasloc = tddft_xasloc_var.get().strip()
        settings.rocis_nroots = int(rocis_roots_var.get())
        settings.rocis_maxdim = int(rocis_maxdim_var.get())
        settings.rocis_orbwin = rocis_orbwin_var.get().strip()
        settings.rocis_soc = bool(rocis_soc_var.get())
        settings.rocis_dftcis = bool(rocis_dftcis_var.get())
        settings.rocis_higher_mult = bool(rocis_higher_var.get())
        settings.rocis_lower_mult = bool(rocis_lower_var.get())
        settings.scan = scan_from_gui()
        return settings

    def generate() -> None:
        try:
            settings = settings_from_gui()
            job_dirs = create_jobs(settings)
        except Exception as exc:
            messagebox.showerror("ORCA Input Builder", str(exc), parent=app)
            return
        if not job_dirs:
            messagebox.showwarning("ORCA Input Builder", "No .xyz files found.", parent=app)
            return
        output_root = settings.root / settings.output_folder
        status_var.set(f"Created {len(job_dirs)} ORCA job folder(s) in {output_root}")
        messagebox.showinfo(
            "ORCA Input Builder",
            f"Created {len(job_dirs)} ORCA job folder(s).\n\nOutput:\n{output_root}",
            parent=app,
        )

    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=3, column=0, pady=(12, 0), sticky="e")
    ttk.Button(button_frame, text="Generate", command=generate).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(button_frame, text="Close", command=app.destroy).grid(row=0, column=1)

    refresh_files()
    app.mainloop()
    return 0


def main() -> int:
    args = parse_args()
    if not args.console:
        return run_gui(args)
    settings = prompt_settings(args)
    job_dirs = create_jobs(settings)
    if not job_dirs:
        print("[WARNING] No .xyz files found.")
        return 0
    print(f"[DONE] Created {len(job_dirs)} ORCA job folder(s) in {settings.root / settings.output_folder}")
    for job_dir in job_dirs:
        print(f"  - {job_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
