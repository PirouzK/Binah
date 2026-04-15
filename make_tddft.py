#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import shutil
import sys


TDDFT_INP_TEMPLATE = """%pal nprocs 32 end

! wB97X-D3BJ def2-TZVP def2/J RIJCOSX LargePrint Hirshfeld
!NBO
%maxcore 12000

%tddft
  nroots 100         # plenty to catch MLCT band(s)
  tda true
  triplet true
  maxdim 5
end

*xyzfile 0 1 {xyz_name}
"""


XAS_INP_TEMPLATE = """%pal nprocs 32 end
! BP86 x2c x2c-SVPall x2c/J TightSCF
!LargePrint NBO

%basis
  newgto Ni "CP(PPP)"
end
end

%method
  SpecialGridAtoms 28
  SpecialGridIntACC 7
end

%xes
CoreOrb 1,1
NRoots 100
OrbOp 1,1
CoreOrbSOC 1,1
DoXAS true
DoSOC true
DoDipoleLength True
DoFullSemiclassical true
end


*xyzfile 0 1 {xyz_name}
"""


SH_TEMPLATE = """#!/bin/bash

#SBATCH --account=def-pierre-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=0
#SBATCH --time=01-21:00
#SBATCH --output={base_name}.out

module load StdEnv/2020  gcc/10.3.0  openmpi/4.1.1 orca/5.0.4
module load gaussian/g16.c01

export GENEXE=`which gennbo.i4.exe`
export NBOEXE=`which nbo7.i4.exe`

$EBROOTORCA/orca {base_name}.inp

echo "Program finished with exit code $? at: `date`"
"""


TRJ_SUFFIX = "_trj.xyz"
XAS_SUFFIX = "_xas"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a TDDFT folder in the chosen directory, scan that directory "
            "plus one subdirectory layer for .xyz files, and generate matching "
            ".inp and .sh files."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Directory to scan. Defaults to the current working directory.",
    )
    return parser.parse_args()


def is_xyz_file(path):
    return os.path.isfile(path) and path.lower().endswith(".xyz")


def find_xyz_files(root):
    xyz_files = []

    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if is_xyz_file(path):
            xyz_files.append(path)

    for name in sorted(os.listdir(root)):
        child_dir = os.path.join(root, name)
        if not os.path.isdir(child_dir):
            continue
        if name == "TDDFT":
            continue

        for child_name in sorted(os.listdir(child_dir)):
            child_path = os.path.join(child_dir, child_name)
            if is_xyz_file(child_path):
                xyz_files.append(child_path)

    return xyz_files


def split_xyz_groups(xyz_files):
    primary_files = []
    associated_by_key = {}
    matched_assoc_keys = {}
    unmatched_assoc_files = []

    for xyz_file in xyz_files:
        xyz_dir = os.path.dirname(xyz_file)
        xyz_name = os.path.basename(xyz_file)
        xyz_stem, _ = os.path.splitext(xyz_name)

        if xyz_name.lower().endswith(TRJ_SUFFIX):
            primary_stem = xyz_stem[:-4]
            key = (xyz_dir.lower(), primary_stem.lower())
            if key not in associated_by_key:
                associated_by_key[key] = []
            associated_by_key[key].append(xyz_file)
            continue

        primary_files.append(xyz_file)

    jobs = []
    for primary_file in primary_files:
        primary_dir = os.path.dirname(primary_file)
        primary_name = os.path.basename(primary_file)
        primary_stem, _ = os.path.splitext(primary_name)
        key = (primary_dir.lower(), primary_stem.lower())
        associated_files = associated_by_key.get(key, [])
        matched_assoc_keys[key] = True
        jobs.append({
            "primary": primary_file,
            "associated": associated_files,
        })

    for key in associated_by_key:
        if key not in matched_assoc_keys:
            unmatched_assoc_files.extend(associated_by_key[key])

    return jobs, unmatched_assoc_files


def strip_variant_suffixes(base_name):
    lower_name = base_name.lower()
    if lower_name.endswith(XAS_SUFFIX):
        return base_name[:-len(XAS_SUFFIX)]
    return base_name


def choose_inp_template(xyz_name):
    base_name, _ = os.path.splitext(xyz_name)
    variant_name = strip_variant_suffixes(base_name)
    lower_variant_name = variant_name.lower()

    if lower_variant_name.endswith("b"):
        return XAS_INP_TEMPLATE, "XAS-B"

    return TDDFT_INP_TEMPLATE, "TDDFT-A"


def unique_output_dir(parent, preferred_name):
    candidate = os.path.join(parent, preferred_name)
    if not os.path.exists(candidate):
        return candidate

    suffix = 2
    while True:
        candidate = os.path.join(parent, "{0}-{1}".format(preferred_name, suffix))
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


def write_text_file(path, content):
    handle = open(path, "wb")
    try:
        handle.write(content.encode("ascii"))
    finally:
        handle.close()


def ensure_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def main():
    args = parse_args()
    root = os.path.abspath(args.root)

    if not os.path.exists(root):
        print("[ERROR] Directory does not exist: {0}".format(root))
        return 1

    if not os.path.isdir(root):
        print("[ERROR] Path is not a directory: {0}".format(root))
        return 1

    print("[INFO] Working directory: {0}".format(root))
    print("[INFO] Scanning for .xyz files in the current directory and one subdirectory layer.")

    xyz_files = find_xyz_files(root)
    if not xyz_files:
        print("[WARNING] No .xyz files found.")
        return 0

    print("[INFO] Found {0} total .xyz file(s):".format(len(xyz_files)))
    for xyz_file in xyz_files:
        print("  - {0}".format(xyz_file))

    jobs, unmatched_assoc_files = split_xyz_groups(xyz_files)
    if not jobs:
        print("[WARNING] No primary .xyz files found for TDDFT job creation.")
        if unmatched_assoc_files:
            print("[WARNING] The following *_trj.xyz files were ignored because no matching primary .xyz file was found:")
            for assoc_file in unmatched_assoc_files:
                print("  - {0}".format(assoc_file))
        return 0

    print("[INFO] Creating {0} TDDFT job(s) from primary .xyz files.".format(len(jobs)))
    for job in jobs:
        primary_file = job["primary"]
        associated_count = len(job["associated"])
        primary_name = os.path.basename(primary_file)
        _, mode_label = choose_inp_template(primary_name)
        print("  [JOB] {0}".format(primary_file))
        print("        mode: {0}".format(mode_label))
        if associated_count:
            print("        associated *_trj.xyz file(s): {0}".format(associated_count))

    if unmatched_assoc_files:
        print("[WARNING] The following *_trj.xyz files were ignored because no matching primary .xyz file was found:")
        for assoc_file in unmatched_assoc_files:
            print("  - {0}".format(assoc_file))

    tddft_dir = os.path.join(root, "TDDFT")
    ensure_directory(tddft_dir)
    print("[INFO] Output folder ready: {0}".format(tddft_dir))

    for job in jobs:
        xyz_file = job["primary"]
        associated_files = job["associated"]
        xyz_name = os.path.basename(xyz_file)
        base_name, _ = os.path.splitext(xyz_name)
        molecule_dir = unique_output_dir(tddft_dir, base_name)
        ensure_directory(molecule_dir)
        print("[INFO] Creating folder: {0}".format(molecule_dir))

        copied_xyz = os.path.join(molecule_dir, xyz_name)
        shutil.copy2(xyz_file, copied_xyz)
        print("  [COPY] {0} -> {1}".format(xyz_name, copied_xyz))

        for associated_file in associated_files:
            associated_name = os.path.basename(associated_file)
            copied_assoc = os.path.join(molecule_dir, associated_name)
            shutil.copy2(associated_file, copied_assoc)
            print("  [COPY-ASSOCIATED] {0} -> {1}".format(associated_name, copied_assoc))

        inp_path = os.path.join(molecule_dir, "{0}.inp".format(base_name))
        sh_path = os.path.join(molecule_dir, "{0}.sh".format(base_name))

        inp_template, mode_label = choose_inp_template(xyz_name)
        write_text_file(inp_path, inp_template.format(xyz_name=xyz_name))
        print("  [WRITE] {0}".format(inp_path))
        print("         template: {0}".format(mode_label))

        write_text_file(sh_path, SH_TEMPLATE.format(base_name=base_name))
        print("  [WRITE] {0}".format(sh_path))

    print("[DONE] TDDFT folder generation finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
