"""
simulation_studio_tab.py — XANES/EXAFS simulation workspace for Binah.

Hosts two engines behind an engine selector:
  * **FEFF** — multiple-scattering, great for above-edge XANES + EXAFS.
              The UI is reused verbatim from EXAFSAnalysisTab (with its
              new ``show_analysis_panel=False`` flag), so all existing FEFF
              behaviour — Run, Batch Run, FEFF Options dialog, parallel
              build wrapper, output archiving, auto-load — keeps working.

  * **FDMNES** — finite-difference DFT XANES, much better at the bound-state
                 pre-edge region (1s→3d for transition metals). Configured
                 manually via the Help → FDMNES Setup dialog.

Layout (top → bottom):
    1. Engine bar: combobox + status line for the active engine.
    2. Stacked panels: only the active engine's panel is visible.
       — FEFF panel = inner EXAFSAnalysisTab(show_analysis_panel=False)
       — FDMNES panel = native widgets in this file.

The class exposes the same surface area binah.py expects from
EXAFSAnalysisTab — ``get_params`` / ``set_params`` / ``refresh_scan_list``
— so it slots into the existing project save/load and notebook plumbing.
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

import numpy as np

import fdmnes_manager
from fdmnes_input import (
    edge_energy_eV, export_xyz_as_fdmnes_bundle, parse_fdmnes_output,
)
from exafs_analysis_tab import (
    EXAFSAnalysisTab, FeffPathData, _coerce_float, _parse_optional_radius,
    _resolve_absorber_index,
)
from experimental_parser import ExperimentalScan
from structure_converter import parse_xyz_file


class SimulationStudioTab(tk.Frame):
    """Composite tab: engine selector + FEFF (inner EXAFSAnalysisTab) + FDMNES."""

    def __init__(self, parent, *,
                 get_scans_fn: Callable,
                 add_exp_scan_fn: Optional[Callable] = None,
                 replot_fn: Optional[Callable] = None,
                 cfg_path: str = ""):
        super().__init__(parent)
        self._get_scans = get_scans_fn
        self._add_exp_scan_fn = add_exp_scan_fn
        self._replot_fn = replot_fn
        self._cfg_path = cfg_path

        self._engine_var = tk.StringVar(value="FEFF")

        # FDMNES Tk vars (shared structure + FDMNES-specific knobs).
        self._fd_xyz_path_var = tk.StringVar(value="")
        self._fd_workdir_var = tk.StringVar(value="")
        self._fd_basename_var = tk.StringVar(value="")
        self._fd_absorber_var = tk.StringVar(value="Ni")
        self._fd_edge_var = tk.StringVar(value="K")
        self._fd_cluster_radius_var = tk.StringVar(value="6.0")
        self._fd_emin_var = tk.DoubleVar(value=-10.0)
        self._fd_estep_var = tk.DoubleVar(value=0.1)
        self._fd_emax_var = tk.DoubleVar(value=50.0)
        self._fd_quadrupole_var = tk.BooleanVar(value=True)
        self._fd_nondipole_var = tk.BooleanVar(value=False)
        self._fd_nonquadrupole_var = tk.BooleanVar(value=False)
        self._fd_noninterf_var = tk.BooleanVar(value=False)
        self._fd_scf_var = tk.BooleanVar(value=True)
        self._fd_convergence_var = tk.DoubleVar(value=1e-4)
        self._fd_convolution_var = tk.BooleanVar(value=True)
        self._fd_green_var = tk.BooleanVar(value=False)
        self._fd_energpho_var = tk.BooleanVar(value=False)
        self._fd_spinorbit_var = tk.BooleanVar(value=False)
        self._fd_magnetism_var = tk.BooleanVar(value=False)
        self._fd_relativism_var = tk.BooleanVar(value=False)
        self._fd_nonrelat_var = tk.BooleanVar(value=False)
        self._fd_allsite_var = tk.BooleanVar(value=False)
        self._fd_cartesian_var = tk.BooleanVar(value=False)
        self._fd_spherical_var = tk.BooleanVar(value=False)
        # Parallel (WSL) toggle + N-procs. Disabled by default; the Setup
        # dialog is what arms the launcher path.
        self._fd_parallel_var = tk.BooleanVar(value=False)
        self._fd_n_procs_var = tk.IntVar(value=4)
        self._fd_batch_mode_var = tk.StringVar(value="Linear")
        self._fd_batch_jobs_var = tk.IntVar(value=2)
        self._fd_batch_recursive_var = tk.BooleanVar(value=False)
        self._fd_status_var = tk.StringVar(
            value="FDMNES not configured — Help → FDMNES Setup to point at fdmnes_win64.exe."
        )

        self._fdmnes_run_btn: Optional[tk.Button] = None
        self._fdmnes_batch_btn: Optional[tk.Button] = None

        self._build_ui()
        self._refresh_fdmnes_status()

    # ------------------------------------------------------------------ #
    #  Layout                                                             #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        # ── Engine selector bar ─────────────────────────────────────────
        bar = tk.Frame(self, bd=1, relief=tk.GROOVE, padx=6, pady=4)
        bar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(bar, text="Engine:", font=("", 9, "bold")).pack(side=tk.LEFT)
        cb = ttk.Combobox(bar, textvariable=self._engine_var,
                          values=("FEFF", "FDMNES"),
                          state="readonly", width=10)
        cb.pack(side=tk.LEFT, padx=(4, 8))
        cb.bind("<<ComboboxSelected>>", self._on_engine_change)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        tk.Label(bar, textvariable=self._fd_status_var,
                 fg="#444444", font=("", 8)).pack(side=tk.LEFT, fill=tk.X,
                                                    expand=True)

        # ── Stacked engine panels ───────────────────────────────────────
        body = tk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # FEFF panel: an EXAFSAnalysisTab with the analysis half hidden.
        # The whole existing FEFF UI (browse, options dialog, run, batch,
        # path table, log, etc.) lives inside.
        self._feff_frame = tk.Frame(body)
        self._feff_panel = EXAFSAnalysisTab(
            self._feff_frame,
            get_scans_fn=self._get_scans,
            replot_fn=self._replot_fn,
            add_exp_scan_fn=self._add_exp_scan_fn,
            show_analysis_panel=False,
            show_feff_panel=True,
        )
        self._feff_panel.pack(fill=tk.BOTH, expand=True)

        # FDMNES panel — native to this module.
        self._fdmnes_frame = tk.Frame(body)
        self._build_fdmnes_panel(self._fdmnes_frame)

        self._show_engine_panel("FEFF")

    def _show_engine_panel(self, engine: str):
        for f in (self._feff_frame, self._fdmnes_frame):
            f.pack_forget()
        if engine == "FDMNES":
            self._fdmnes_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        else:
            self._feff_frame.pack(fill=tk.BOTH, expand=True)

    def _on_engine_change(self, *_):
        self._show_engine_panel(self._engine_var.get())

    # ------------------------------------------------------------------ #
    #  FDMNES panel                                                       #
    # ------------------------------------------------------------------ #
    def _build_fdmnes_panel(self, parent):
        # Workdir + executable status row
        row = tk.Frame(parent, bd=1, relief=tk.GROOVE, padx=5, pady=4)
        row.pack(side=tk.TOP, fill=tk.X)

        tk.Label(row, text="Workdir:", font=("", 8, "bold"), width=10,
                 anchor="w").grid(row=0, column=0, sticky="w")
        ttk.Entry(row, textvariable=self._fd_workdir_var, width=60).grid(
            row=0, column=1, sticky="ew", padx=4)
        tk.Button(row, text="Browse", font=("", 8),
                  command=self._browse_fdmnes_workdir).grid(row=0, column=2)
        row.columnconfigure(1, weight=1)

        # XYZ + structure params
        xyz_box = tk.LabelFrame(parent, text="Structure", padx=6, pady=4)
        xyz_box.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))

        tk.Label(xyz_box, text="XYZ file:", font=("", 8, "bold")).grid(
            row=0, column=0, sticky="w")
        ttk.Entry(xyz_box, textvariable=self._fd_xyz_path_var, width=58).grid(
            row=0, column=1, columnspan=4, sticky="ew", padx=4)
        tk.Button(xyz_box, text="Browse", font=("", 8),
                  command=self._browse_fdmnes_xyz).grid(row=0, column=5)
        xyz_box.columnconfigure(1, weight=1)

        tk.Label(xyz_box, text="Base:", font=("", 8, "bold")).grid(
            row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(xyz_box, textvariable=self._fd_basename_var, width=22).grid(
            row=1, column=1, sticky="w", padx=4, pady=(4, 0))
        tk.Label(xyz_box, text="Absorber:", font=("", 8, "bold")).grid(
            row=1, column=2, sticky="e", pady=(4, 0))
        ttk.Entry(xyz_box, textvariable=self._fd_absorber_var, width=8).grid(
            row=1, column=3, sticky="w", padx=4, pady=(4, 0))
        tk.Label(xyz_box, text="Edge:", font=("", 8, "bold")).grid(
            row=1, column=4, sticky="e", pady=(4, 0))
        ttk.Combobox(xyz_box, textvariable=self._fd_edge_var,
                     values=("K", "L1", "L2", "L3"),
                     state="readonly", width=6).grid(
            row=1, column=5, sticky="w", padx=4, pady=(4, 0))

        tk.Label(xyz_box, text="Cluster (Å):", font=("", 8, "bold")).grid(
            row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(xyz_box, textvariable=self._fd_cluster_radius_var, width=8).grid(
            row=2, column=1, sticky="w", padx=4, pady=(4, 0))

        # Energy range
        e_box = tk.LabelFrame(parent, text="Energy range (relative to edge, eV)",
                              padx=6, pady=4)
        e_box.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))

        tk.Label(e_box, text="E min:", font=("", 8)).grid(row=0, column=0)
        ttk.Entry(e_box, textvariable=self._fd_emin_var, width=8).grid(
            row=0, column=1, padx=4)
        tk.Label(e_box, text="E step:", font=("", 8)).grid(row=0, column=2)
        ttk.Entry(e_box, textvariable=self._fd_estep_var, width=8).grid(
            row=0, column=3, padx=4)
        tk.Label(e_box, text="E max:", font=("", 8)).grid(row=0, column=4)
        ttk.Entry(e_box, textvariable=self._fd_emax_var, width=8).grid(
            row=0, column=5, padx=4)

        # Method toggles
        m_box = tk.LabelFrame(parent, text="Method", padx=6, pady=4)
        m_box.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))

        ttk.Checkbutton(m_box, text="Quadrupole (E2 — needed for 1s→3d pre-edge)",
                        variable=self._fd_quadrupole_var).grid(
            row=0, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(m_box, text="SCF (self-consistent field)",
                        variable=self._fd_scf_var).grid(
            row=1, column=0, sticky="w")
        tk.Label(m_box, text="Convergence:", font=("", 8)).grid(
            row=1, column=1, sticky="e")
        ttk.Entry(m_box, textvariable=self._fd_convergence_var, width=10).grid(
            row=1, column=2, sticky="w", padx=4)
        ttk.Checkbutton(m_box, text="Convolution (Lorentzian broadening output)",
                        variable=self._fd_convolution_var).grid(
            row=2, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(m_box, text="Green (multiple scattering, faster muffin-tin)",
                        variable=self._fd_green_var).grid(
            row=3, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(m_box, text="Energpho (output photon-energy scale)",
                        variable=self._fd_energpho_var).grid(
            row=4, column=0, columnspan=3, sticky="w")

        adv_box = tk.LabelFrame(parent, text="FDMNES keyword toggles",
                                padx=6, pady=4)
        adv_box.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))
        toggles = [
            ("Nondipole", self._fd_nondipole_var),
            ("Nonquadrupole", self._fd_nonquadrupole_var),
            ("Noninterf", self._fd_noninterf_var),
            ("Spinorbite", self._fd_spinorbit_var),
            ("Magnetism", self._fd_magnetism_var),
            ("Relativism", self._fd_relativism_var),
            ("Nonrelat", self._fd_nonrelat_var),
            ("Allsite", self._fd_allsite_var),
            ("Cartesian tensors", self._fd_cartesian_var),
            ("Spherical tensors", self._fd_spherical_var),
        ]
        for i, (label, var) in enumerate(toggles):
            ttk.Checkbutton(adv_box, text=label, variable=var).grid(
                row=i // 3, column=i % 3, sticky="w", padx=(0, 14), pady=1)

        # Parallelism (via WSL — Linux MPI build of FDMNES)
        p_box = tk.LabelFrame(parent, text="Parallel execution", padx=6, pady=4)
        p_box.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))

        ttk.Checkbutton(
            p_box,
            text="Run via WSL (parallel Linux build of FDMNES)",
            variable=self._fd_parallel_var,
            command=self._refresh_fdmnes_status,
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        tk.Label(p_box, text="Processes (N):", font=("", 8)).grid(
            row=1, column=0, sticky="w", padx=(20, 4))
        ttk.Entry(p_box, textvariable=self._fd_n_procs_var, width=6).grid(
            row=1, column=1, sticky="w")
        tk.Label(
            p_box,
            text=("Requires WSL + the Linux parallel bundle. Configure the "
                  "launcher path in Help → FDMNES Setup."),
            fg="gray", font=("", 8), wraplength=520, justify="left",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 0))

        b_box = tk.LabelFrame(parent, text="Batch scheduling", padx=6, pady=4)
        b_box.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))
        tk.Label(b_box, text="Run selected XYZ files:", font=("", 8)).grid(
            row=0, column=0, sticky="w")
        ttk.Combobox(
            b_box,
            textvariable=self._fd_batch_mode_var,
            values=("Linear", "Parallel jobs"),
            state="readonly",
            width=14,
        ).grid(row=0, column=1, sticky="w", padx=(4, 10))
        tk.Label(b_box, text="Concurrent jobs:", font=("", 8)).grid(
            row=0, column=2, sticky="e")
        ttk.Entry(b_box, textvariable=self._fd_batch_jobs_var, width=6).grid(
            row=0, column=3, sticky="w", padx=4)
        ttk.Checkbutton(
            b_box,
            text="Recursive folder scan",
            variable=self._fd_batch_recursive_var,
        ).grid(row=0, column=4, sticky="w", padx=(8, 0))
        tk.Label(
            b_box,
            text=("Linear runs one input folder at a time. Parallel jobs launches "
                  "several independent FDMNES folders at once; if WSL parallel is "
                  "also on, total processes = concurrent jobs x Processes (N)."),
            fg="gray", font=("", 8), wraplength=720, justify="left",
        ).grid(row=1, column=0, columnspan=5, sticky="w", pady=(2, 0))

        # Action row
        a_row = tk.Frame(parent, padx=4, pady=4)
        a_row.pack(side=tk.TOP, fill=tk.X)

        self._fdmnes_run_btn = tk.Button(
            a_row, text="Run FDMNES", font=("", 9, "bold"),
            bg="#2A4D14", fg="black", activebackground="#3E6E1E",
            command=self._run_fdmnes,
        )
        self._fdmnes_run_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._fdmnes_batch_btn = tk.Button(
            a_row, text="Batch Run...", font=("", 8, "bold"),
            bg="#5C2A0E", fg="black", activebackground="#7A3D17",
            command=self._on_fdmnes_batch_clicked,
        )
        self._fdmnes_batch_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._fdmnes_batch_folder_btn = tk.Button(
            a_row, text="Batch Folder...", font=("", 8, "bold"),
            bg="#5C2A0E", fg="black", activebackground="#7A3D17",
            command=self._on_fdmnes_batch_folder_clicked,
        )
        self._fdmnes_batch_folder_btn.pack(side=tk.LEFT)

        # Log
        log_frame = tk.Frame(parent)
        log_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(4, 0))
        self._fd_log = tk.Text(log_frame, height=14, font=("Consolas", 9),
                               wrap=tk.WORD, state=tk.DISABLED)
        scroll = tk.Scrollbar(log_frame, orient=tk.VERTICAL,
                              command=self._fd_log.yview)
        self._fd_log.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._fd_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                          padx=(4, 0), pady=(2, 0))

    # ------------------------------------------------------------------ #
    #  FDMNES helpers                                                     #
    # ------------------------------------------------------------------ #
    def _append_fd_log(self, text: str):
        self._fd_log.config(state=tk.NORMAL)
        self._fd_log.insert(tk.END, text.rstrip() + "\n")
        self._fd_log.see(tk.END)
        self._fd_log.config(state=tk.DISABLED)

    def _refresh_fdmnes_status(self):
        if self._fd_parallel_var.get():
            launcher = fdmnes_manager.discover_parallel_fdmnes_launcher(
                cfg_path=self._cfg_path,
            )
            wsl = fdmnes_manager.discover_wsl_executable()
            if not wsl:
                self._fd_status_var.set(
                    "Parallel mode: wsl.exe not found. Install WSL "
                    "(`wsl --install` from an admin shell, then reboot)."
                )
            elif launcher:
                self._fd_status_var.set(
                    f"FDMNES (parallel via WSL): {launcher}"
                )
            else:
                self._fd_status_var.set(
                    "Parallel mode: launcher not found. Help → FDMNES Setup "
                    "to point at C:\\FDMNES\\parallel_fdmnes\\mpirun_fdmnes."
                )
            return
        exe = fdmnes_manager.discover_fdmnes_executable(cfg_path=self._cfg_path)
        if exe:
            self._fd_status_var.set(f"FDMNES: {exe}")
        else:
            self._fd_status_var.set(
                "FDMNES not configured — Help → FDMNES Setup to point at fdmnes_win64.exe."
            )

    def _browse_fdmnes_workdir(self):
        path = filedialog.askdirectory(parent=self,
                                       title="Pick FDMNES working directory")
        if path:
            self._fd_workdir_var.set(path)

    def _browse_fdmnes_xyz(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Pick XYZ file",
            filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")],
        )
        if path:
            self._fd_xyz_path_var.set(path)
            if not self._fd_basename_var.get().strip():
                self._fd_basename_var.set(Path(path).stem)

    # ------------------------------------------------------------------ #
    #  Runner spec: chooses between Windows-serial and WSL-parallel       #
    # ------------------------------------------------------------------ #
    def _build_runner_spec(self) -> tuple[Optional[dict], str]:
        """Return ``(spec, error_message)``.

        ``spec`` is a dict describing how to invoke FDMNES for the current
        engine settings (parallel-WSL or serial-Windows). ``error_message``
        is non-empty when the requested mode isn't usable (and ``spec`` is
        ``None``).
        """
        if self._fd_parallel_var.get():
            wsl = fdmnes_manager.discover_wsl_executable()
            if not wsl:
                return None, (
                    "wsl.exe not found on PATH. Install WSL first "
                    "(`wsl --install` from an admin shell, then reboot)."
                )
            launcher = fdmnes_manager.discover_parallel_fdmnes_launcher(
                cfg_path=self._cfg_path
            )
            if not launcher:
                return None, (
                    "Parallel FDMNES launcher not configured. Use "
                    "Help → FDMNES Setup → Parallel launcher."
                )
            try:
                n = max(1, int(self._fd_n_procs_var.get()))
            except Exception:
                n = 4
            return {"mode": "parallel", "launcher": launcher,
                    "n_procs": n, "wsl": wsl}, ""
        # Serial Windows path.
        exe = fdmnes_manager.discover_fdmnes_executable(cfg_path=self._cfg_path)
        if not exe or not os.path.isfile(exe):
            return None, (
                "FDMNES executable not configured. Use Help → FDMNES Setup."
            )
        return {"mode": "serial", "exe": exe}, ""

    @staticmethod
    def _spec_to_command(spec: dict, workdir: str) -> tuple[list, Optional[str]]:
        """Translate a runner spec to a ``(argv, cwd)`` pair for ``subprocess``.

        For parallel runs the bash payload `cd`s into the WSL-translated
        workdir itself, so we set ``cwd=None`` and let WSL handle pathing.
        """
        if spec["mode"] == "parallel":
            argv = fdmnes_manager.build_parallel_fdmnes_command(
                spec["launcher"], workdir, spec["n_procs"]
            )
            return argv, None
        return [spec["exe"]], workdir

    # ------------------------------------------------------------------ #
    #  Single FDMNES run                                                  #
    # ------------------------------------------------------------------ #
    def _run_fdmnes(self):
        spec, err = self._build_runner_spec()
        if err:
            messagebox.showwarning("FDMNES", err, parent=self)
            return

        xyz_path = self._fd_xyz_path_var.get().strip()
        workdir = self._fd_workdir_var.get().strip()
        if not xyz_path or not os.path.isfile(xyz_path):
            messagebox.showwarning("FDMNES", "Pick a valid XYZ file first.",
                                   parent=self)
            return
        if not workdir:
            messagebox.showwarning("FDMNES", "Pick a workdir first.",
                                   parent=self)
            return

        try:
            structure = parse_xyz_file(xyz_path)
            absorber_idx, abs_note = _resolve_absorber_index(
                self._fd_absorber_var.get(), structure
            )
        except Exception as exc:
            messagebox.showerror("FDMNES", str(exc), parent=self)
            return
        self._append_fd_log(f"Absorber: {abs_note}")

        radius = _parse_optional_radius(self._fd_cluster_radius_var.get())
        base = (self._fd_basename_var.get().strip()
                or Path(xyz_path).stem)
        safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("._") or "structure"
        sub = (Path(workdir) / f"{safe_base}_{radius:.1f}A"
               if radius is not None else Path(workdir) / safe_base)

        try:
            settings = self._snapshot_fdmnes_settings()
        except Exception as exc:
            messagebox.showerror("FDMNES",
                                 f"Could not read FDMNES settings:\n{exc}",
                                 parent=self)
            return

        self._fdmnes_run_btn.config(state=tk.DISABLED)
        self._fdmnes_batch_btn.config(state=tk.DISABLED)
        self._fdmnes_batch_folder_btn.config(state=tk.DISABLED)
        self._append_fd_log("")
        self._append_fd_log(f"=== FDMNES: {Path(xyz_path).name} ===")
        if spec["mode"] == "parallel":
            self._append_fd_log(f"  Mode: parallel (WSL, np={spec['n_procs']})")
        else:
            self._append_fd_log(f"  Mode: serial ({Path(spec['exe']).name})")
        threading.Thread(
            target=self._run_fdmnes_worker,
            args=(spec, xyz_path, str(sub), absorber_idx, radius,
                  safe_base, settings),
            daemon=True,
        ).start()

    def _snapshot_fdmnes_settings(self) -> dict:
        return {
            "edge":            self._fd_edge_var.get().strip() or "K",
            "e_min":           float(self._fd_emin_var.get()),
            "e_step":          float(self._fd_estep_var.get()),
            "e_max":           float(self._fd_emax_var.get()),
            "use_quadrupole":  bool(self._fd_quadrupole_var.get()),
            "use_nondipole":   bool(self._fd_nondipole_var.get()),
            "use_nonquadrupole": bool(self._fd_nonquadrupole_var.get()),
            "use_noninterf":   bool(self._fd_noninterf_var.get()),
            "use_scf":         bool(self._fd_scf_var.get()),
            "scf_convergence": float(self._fd_convergence_var.get()),
            "convolution":     bool(self._fd_convolution_var.get()),
            "use_green":       bool(self._fd_green_var.get()),
            "use_energpho":    bool(self._fd_energpho_var.get()),
            "use_spinorbit":   bool(self._fd_spinorbit_var.get()),
            "use_magnetism":   bool(self._fd_magnetism_var.get()),
            "use_relativism":  bool(self._fd_relativism_var.get()),
            "use_nonrelat":    bool(self._fd_nonrelat_var.get()),
            "use_allsite":     bool(self._fd_allsite_var.get()),
            "use_cartesian":   bool(self._fd_cartesian_var.get()),
            "use_spherical":   bool(self._fd_spherical_var.get()),
        }

    def _run_fdmnes_worker(self, spec: dict, xyz_path: str, sub: str,
                           absorber_idx: int, cluster_radius,
                           safe_base: str, settings: dict):
        def log(msg: str):
            self.after(0, lambda m=msg: self._append_fd_log(m))

        def push_scan(scan):
            if scan is not None and self._add_exp_scan_fn is not None:
                self.after(0, lambda s=scan: self._add_exp_scan_fn(s))

        try:
            sub_path = Path(sub)
            sub_path.mkdir(parents=True, exist_ok=True)
            log(f"  Workdir: {sub_path}")

            bundle = export_xyz_as_fdmnes_bundle(
                xyz_path,
                str(sub_path),
                absorber_index=absorber_idx,
                cluster_radius=cluster_radius,
                basename=safe_base,
                edge=settings["edge"],
                e_min=settings["e_min"],
                e_step=settings["e_step"],
                e_max=settings["e_max"],
                use_quadrupole=settings["use_quadrupole"],
                use_nondipole=settings["use_nondipole"],
                use_nonquadrupole=settings["use_nonquadrupole"],
                use_noninterf=settings["use_noninterf"],
                use_scf=settings["use_scf"],
                scf_convergence=settings["scf_convergence"],
                convolution=settings["convolution"],
                use_green=settings["use_green"],
                use_energpho=settings["use_energpho"],
                use_spinorbit=settings["use_spinorbit"],
                use_magnetism=settings["use_magnetism"],
                use_relativism=settings["use_relativism"],
                use_nonrelat=settings["use_nonrelat"],
                use_allsite=settings["use_allsite"],
                use_cartesian=settings["use_cartesian"],
                use_spherical=settings["use_spherical"],
            )
            log(f"  Wrote: {Path(bundle['input_path']).name}, "
                f"{Path(bundle['fdmfile_path']).name}, "
                f"{Path(bundle['xyz_copy_path']).name}")
            log(f"  Atoms: {bundle['atoms_used']} / {bundle['atoms_total_input']}"
                + (f" (dropped {bundle['atoms_dropped']})"
                   if bundle['atoms_dropped'] else ""))

            argv, cwd = self._spec_to_command(spec, str(sub_path))
            if spec["mode"] == "parallel":
                log(f"  Running FDMNES via WSL with -np {spec['n_procs']} ...")
            else:
                log(f"  Running FDMNES ...")
            run_started = time.time()
            proc = subprocess.run(
                argv, cwd=cwd,
                capture_output=True, text=True, check=False,
            )
            output_basename = bundle["output_basename"]
            output_file = self._fdmnes_output_path(sub_path, output_basename)
            if output_file and output_file.stat().st_mtime < run_started - 1.0:
                output_file = None
            if proc.returncode != 0:
                details = [f"  FDMNES returned {proc.returncode}."]
                if output_file:
                    details.append("  Checking output written before exit.")
                else:
                    details[-1] += " No usable output was written."
                    self._append_fdmnes_diagnostics(
                        details, sub_path, proc.stdout, proc.stderr,
                    )
                for ln in details:
                    log(ln)
                if not output_file:
                    return
            else:
                log(f"  FDMNES finished (rc=0)")

            # Locate output files. Convolution path is preferred for plotting.
            if output_file is None or not output_file.exists():
                log(f"  Could not find FDMNES output ({output_basename}.txt).")
                # FDMNES sometimes exits cleanly (rc=0) even when its input
                # parsing fails — surface fdmnes_error.txt if present so the
                # user knows what went wrong.
                details = []
                self._append_fdmnes_diagnostics(
                    details, sub_path, proc.stdout, proc.stderr,
                    max_lines=25,
                )
                for ln in details:
                    log(ln)
                return
            log(f"  Output: {output_file.name}")

            # Convert relative energies to absolute eV using a tabulated edge.
            try:
                z_abs = self._absorber_z_from_xyz(xyz_path, absorber_idx)
            except Exception:
                z_abs = 0
            e0 = edge_energy_eV(z_abs, settings["edge"]) if z_abs else 0.0

            scan = self._scan_from_fdmnes_output(
                output_file, label=output_basename, e0=e0,
                edge=settings["edge"],
            )
            if not self._fdmnes_output_covers_range(scan, settings):
                log("  Output is incomplete; not loading partial spectrum.")
                return
            push_scan(scan)
            log(f"  Pushed to plot as: {scan.label}")
        except Exception as exc:
            log(f"  FAILED: {exc}")
        finally:
            self.after(0, self._fdmnes_finish)

    def _fdmnes_finish(self):
        try:
            self._fdmnes_run_btn.config(state=tk.NORMAL)
            self._fdmnes_batch_btn.config(state=tk.NORMAL)
            self._fdmnes_batch_folder_btn.config(state=tk.NORMAL)
        except Exception:
            pass

    @staticmethod
    def _absorber_z_from_xyz(xyz_path: str, absorber_index: int) -> int:
        from structure_converter import _ATOMIC_NUMBERS
        s = parse_xyz_file(xyz_path)
        sym = s.symbols[absorber_index - 1]
        return int(_ATOMIC_NUMBERS.get(sym, 0))

    @staticmethod
    def _scan_from_fdmnes_output(path: Path, label: str, e0: float,
                                 edge: str) -> ExperimentalScan:
        e_rel, signal, meta = parse_fdmnes_output(str(path))
        # FDMNES writes energies relative to threshold; shift to absolute eV
        # so the spectrum lines up with experimental data on the same axis.
        absolute_energy = e_rel + float(e0) if e0 else e_rel
        return ExperimentalScan(
            label=f"FDMNES: {label}",
            source_file=str(path),
            energy_ev=np.asarray(absolute_energy, dtype=float),
            mu=np.asarray(signal, dtype=float),
            e0=float(e0),
            is_normalized=True,
            scan_type="FDMNES calculation",
            metadata={
                "source": "fdmnes",
                "filename": Path(path).name,
                "edge": edge,
                "e0_eV": float(e0),
                **meta,
            },
        )

    @staticmethod
    def _fdmnes_output_covers_range(scan: ExperimentalScan, settings: dict,
                                    tolerance_eV: float = 0.25) -> bool:
        if scan is None or scan.energy_ev is None or len(scan.energy_ev) == 0:
            return False
        rel_energy = np.asarray(scan.energy_ev, dtype=float) - float(scan.e0 or 0.0)
        requested_min = float(settings["e_min"])
        requested_max = float(settings["e_max"])
        return (
            float(np.nanmin(rel_energy)) <= requested_min + tolerance_eV
            and float(np.nanmax(rel_energy)) >= requested_max - tolerance_eV
        )

    @staticmethod
    def _fdmnes_output_path(folder: Path, output_basename: str) -> Optional[Path]:
        conv_out = folder / f"{output_basename}_conv.txt"
        raw_out = folder / f"{output_basename}.txt"
        if conv_out.exists():
            return conv_out
        if raw_out.exists():
            return raw_out
        return None

    @staticmethod
    def _append_fdmnes_diagnostics(logs: list[str], folder: Path,
                                   stdout: str = "", stderr: str = "",
                                   max_lines: int = 10) -> None:
        if stderr:
            for ln in stderr.splitlines()[-max_lines:]:
                if ln.strip():
                    logs.append(f"    {ln}")
            return
        if stdout:
            for ln in stdout.splitlines()[-max_lines:]:
                if ln.strip():
                    logs.append(f"    {ln}")

        err_file = folder / "fdmnes_error.txt"
        if err_file.exists():
            logs.append(f"  {err_file.name}:")
            try:
                for ln in err_file.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()[-max_lines:]:
                    if ln.strip():
                        logs.append(f"    {ln}")
            except Exception as exc:
                logs.append(f"    (could not read: {exc})")

        bav_files = sorted(folder.glob("*_bav.txt"),
                           key=lambda p: p.stat().st_mtime)
        if bav_files:
            bav_file = bav_files[-1]
            logs.append(f"  {bav_file.name} tail:")
            try:
                for ln in bav_file.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()[-max_lines:]:
                    if ln.strip():
                        logs.append(f"    {ln}")
            except Exception as exc:
                logs.append(f"    (could not read: {exc})")

    # ------------------------------------------------------------------ #
    #  Batch FDMNES                                                       #
    # ------------------------------------------------------------------ #
    def _on_fdmnes_batch_clicked(self):
        spec, err = self._build_runner_spec()
        if err:
            messagebox.showwarning("Batch FDMNES", err, parent=self)
            return
        workdir = self._fd_workdir_var.get().strip()
        if not workdir:
            messagebox.showwarning("Batch FDMNES",
                                   "Pick a workdir first.", parent=self)
            return

        paths = filedialog.askopenfilenames(
            title="Pick .xyz files for batch FDMNES",
            filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")],
            parent=self,
        )
        if not paths:
            return

        self._start_fdmnes_batch(list(paths), workdir, spec,
                                 source_label="selected files")

    def _on_fdmnes_batch_folder_clicked(self):
        spec, err = self._build_runner_spec()
        if err:
            messagebox.showwarning("Batch FDMNES Folder", err, parent=self)
            return
        workdir = self._fd_workdir_var.get().strip()
        if not workdir:
            messagebox.showwarning("Batch FDMNES Folder",
                                   "Pick a workdir first.", parent=self)
            return

        folder = filedialog.askdirectory(
            parent=self,
            title="Pick folder containing XYZ files",
        )
        if not folder:
            return

        root = Path(folder)
        pattern = "**/*.xyz" if self._fd_batch_recursive_var.get() else "*.xyz"
        paths = sorted(
            str(p) for p in root.glob(pattern)
            if p.is_file() and p.suffix.lower() == ".xyz"
        )
        if not paths:
            mode = " recursively" if self._fd_batch_recursive_var.get() else ""
            messagebox.showinfo(
                "Batch FDMNES Folder",
                f"No .xyz files found{mode} in:\n{folder}",
                parent=self,
            )
            return

        self._start_fdmnes_batch(paths, workdir, spec,
                                 source_label=f"folder: {folder}")

    def _start_fdmnes_batch(self, paths: list[str], workdir: str, spec: dict,
                            *, source_label: str):

        try:
            settings = self._snapshot_fdmnes_settings()
        except Exception as exc:
            messagebox.showerror("Batch FDMNES",
                                 f"Could not read FDMNES settings:\n{exc}",
                                 parent=self)
            return
        radius = _parse_optional_radius(self._fd_cluster_radius_var.get())
        absorber_spec = self._fd_absorber_var.get().strip() or "1"
        batch_mode = self._fd_batch_mode_var.get().strip() or "Linear"
        try:
            batch_jobs = max(1, int(self._fd_batch_jobs_var.get()))
        except Exception:
            batch_jobs = 1
        if batch_mode != "Parallel jobs":
            batch_jobs = 1

        mode_line = (f"  Mode:       parallel (WSL, np={spec['n_procs']})"
                     if spec["mode"] == "parallel"
                     else f"  Mode:       serial ({Path(spec['exe']).name})")
        sched_line = (f"  Schedule:   {batch_jobs} concurrent FDMNES job(s)"
                      if batch_jobs > 1 else
                      "  Schedule:   linear (one FDMNES job at a time)")
        if not messagebox.askyesno(
            "Batch FDMNES",
            f"Run FDMNES on {len(paths)} XYZ file(s)?\n\n"
            f"  Workdir:   {workdir}\n"
            f"  Absorber:   {absorber_spec}\n"
            f"  Cluster:    {f'{radius:.1f} Å' if radius else '(no crop)'}\n"
            f"  Edge:       {settings['edge']}\n"
            f"  E range:    {settings['e_min']} → {settings['e_max']} eV "
            f"(step {settings['e_step']})\n"
            f"{mode_line}\n"
            f"{sched_line}\n\n"
            "Each XYZ gets its own input folder under the workdir before runs start.\n"
            "Failed runs are logged and skipped — the batch keeps going.",
            parent=self,
        ):
            return

        self._fdmnes_run_btn.config(state=tk.DISABLED)
        self._fdmnes_batch_btn.config(state=tk.DISABLED)
        self._fdmnes_batch_folder_btn.config(state=tk.DISABLED)
        self._append_fd_log("")
        self._append_fd_log(f"=== Batch FDMNES: {len(paths)} XYZ file(s) queued ===")
        self._append_fd_log(f"Source: {source_label}")
        self._append_fd_log(mode_line.lstrip())
        self._append_fd_log(sched_line.lstrip())
        threading.Thread(
            target=self._batch_fdmnes_worker,
            args=(list(paths), workdir, spec, absorber_spec, radius, settings,
                  batch_jobs),
            daemon=True,
        ).start()

    def _batch_fdmnes_worker(self, xyz_paths, base_workdir, spec,
                             absorber_spec, cluster_radius, settings,
                             batch_jobs: int = 1):
        def log(msg: str):
            self.after(0, lambda m=msg: self._append_fd_log(m))

        def push_scan(scan):
            if scan is not None and self._add_exp_scan_fn is not None:
                self.after(0, lambda s=scan: self._add_exp_scan_fn(s))

        n = len(xyz_paths)
        succeeded = 0
        failed = []

        for idx, xyz_path in enumerate(xyz_paths, start=1):
            xyz_name = os.path.basename(xyz_path)
            log("")
            log(f"--- [{idx}/{n}] {xyz_name} ---")
            try:
                stem = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                              Path(xyz_path).stem).strip("._") or f"xyz_{idx}"
                folder = (f"{stem}_{cluster_radius:.1f}A"
                          if cluster_radius is not None else stem)
                sub = Path(base_workdir) / folder
                sub.mkdir(parents=True, exist_ok=True)

                structure = parse_xyz_file(xyz_path)
                absorber_idx, abs_note = _resolve_absorber_index(
                    absorber_spec, structure
                )
                log(f"  Absorber: {abs_note}")

                bundle = export_xyz_as_fdmnes_bundle(
                    xyz_path, str(sub),
                    absorber_index=absorber_idx,
                    cluster_radius=cluster_radius,
                    basename=stem,
                    edge=settings["edge"],
                    e_min=settings["e_min"],
                    e_step=settings["e_step"],
                    e_max=settings["e_max"],
                    use_quadrupole=settings["use_quadrupole"],
                    use_nondipole=settings["use_nondipole"],
                    use_nonquadrupole=settings["use_nonquadrupole"],
                    use_noninterf=settings["use_noninterf"],
                    use_scf=settings["use_scf"],
                    scf_convergence=settings["scf_convergence"],
                    convolution=settings["convolution"],
                    use_green=settings["use_green"],
                    use_energpho=settings["use_energpho"],
                    use_spinorbit=settings["use_spinorbit"],
                    use_magnetism=settings["use_magnetism"],
                    use_relativism=settings["use_relativism"],
                    use_nonrelat=settings["use_nonrelat"],
                    use_allsite=settings["use_allsite"],
                    use_cartesian=settings["use_cartesian"],
                    use_spherical=settings["use_spherical"],
                )
                argv, cwd = self._spec_to_command(spec, str(sub))
                if spec["mode"] == "parallel":
                    log(f"  Wrote {Path(bundle['input_path']).name}; "
                        f"running FDMNES via WSL with -np {spec['n_procs']} ...")
                else:
                    log(f"  Wrote {Path(bundle['input_path']).name}; "
                        f"running FDMNES ...")
                proc = subprocess.run(
                    argv, cwd=cwd,
                    capture_output=True, text=True, check=False,
                )
                if proc.returncode != 0:
                    log(f"  FDMNES rc={proc.returncode}; skipping load.")
                    if proc.stderr:
                        for ln in proc.stderr.splitlines()[-4:]:
                            log(f"    {ln}")
                    elif proc.stdout:
                        for ln in proc.stdout.splitlines()[-4:]:
                            log(f"    {ln}")
                    failed.append(xyz_name)
                    continue
                log(f"  FDMNES finished (rc=0)")

                out_basename = bundle["output_basename"]
                conv_out = sub / f"{out_basename}_conv.txt"
                raw_out = sub / f"{out_basename}.txt"
                output_file = conv_out if conv_out.exists() else raw_out
                if not output_file.exists():
                    log(f"  Output not found ({out_basename}.txt).")
                    # Surface fdmnes_error.txt — FDMNES often returns rc=0
                    # even when input parsing fails.
                    err_file = sub / "fdmnes_error.txt"
                    if err_file.exists():
                        try:
                            for ln in err_file.read_text(
                                encoding="utf-8", errors="replace"
                            ).splitlines()[:8]:
                                if ln.strip():
                                    log(f"    {ln}")
                        except Exception:
                            pass
                    failed.append(xyz_name)
                    continue

                z_abs = self._absorber_z_from_xyz(xyz_path, absorber_idx)
                e0 = edge_energy_eV(z_abs, settings["edge"]) if z_abs else 0.0
                scan = self._scan_from_fdmnes_output(
                    output_file, label=out_basename, e0=e0,
                    edge=settings["edge"],
                )
                push_scan(scan)
                log(f"  Pushed to plot: {scan.label}")
                succeeded += 1
            except Exception as exc:
                log(f"  FAILED: {exc}")
                failed.append(xyz_name)

        log("")
        log(f"=== Batch complete: {succeeded}/{n} succeeded"
            + (f", failed: {', '.join(failed)}" if failed else "")
            + " ===")
        self.after(0, self._fdmnes_finish)

    @staticmethod
    def _fdmnes_export_kwargs(settings: dict) -> dict:
        return {
            "edge": settings["edge"],
            "e_min": settings["e_min"],
            "e_step": settings["e_step"],
            "e_max": settings["e_max"],
            "use_quadrupole": settings["use_quadrupole"],
            "use_nondipole": settings["use_nondipole"],
            "use_nonquadrupole": settings["use_nonquadrupole"],
            "use_noninterf": settings["use_noninterf"],
            "use_scf": settings["use_scf"],
            "scf_convergence": settings["scf_convergence"],
            "convolution": settings["convolution"],
            "use_green": settings["use_green"],
            "use_energpho": settings["use_energpho"],
            "use_spinorbit": settings["use_spinorbit"],
            "use_magnetism": settings["use_magnetism"],
            "use_relativism": settings["use_relativism"],
            "use_nonrelat": settings["use_nonrelat"],
            "use_allsite": settings["use_allsite"],
            "use_cartesian": settings["use_cartesian"],
            "use_spherical": settings["use_spherical"],
        }

    def _execute_prepared_fdmnes_task(self, task: dict, spec: dict,
                                      settings: dict) -> dict:
        logs: list[str] = []
        sub = Path(task["sub"])
        xyz_name = task["xyz_name"]
        bundle = task["bundle"]
        try:
            argv, cwd = self._spec_to_command(spec, str(sub))
            if spec["mode"] == "parallel":
                logs.append(
                    f"  Running FDMNES via WSL with -np {spec['n_procs']} ..."
                )
            else:
                logs.append("  Running FDMNES ...")
            run_started = time.time()
            proc = subprocess.run(
                argv, cwd=cwd, capture_output=True, text=True, check=False,
            )
            out_basename = bundle["output_basename"]
            output_file = self._fdmnes_output_path(sub, out_basename)
            if output_file and output_file.stat().st_mtime < run_started - 1.0:
                output_file = None
            if proc.returncode != 0:
                logs.append(f"  FDMNES rc={proc.returncode}.")
                if output_file:
                    logs.append("  Checking output written before exit.")
                else:
                    logs[-1] += " No usable output was written."
                    self._append_fdmnes_diagnostics(
                        logs, sub, proc.stdout, proc.stderr,
                    )
                    return {"ok": False, "xyz_name": xyz_name, "logs": logs}
            else:
                logs.append("  FDMNES finished (rc=0)")

            if output_file is None or not output_file.exists():
                logs.append(f"  Output not found ({out_basename}.txt).")
                self._append_fdmnes_diagnostics(
                    logs, sub, proc.stdout, proc.stderr,
                )
                return {"ok": False, "xyz_name": xyz_name, "logs": logs}

            z_abs = self._absorber_z_from_xyz(
                task["xyz_path"], task["absorber_idx"]
            )
            e0 = edge_energy_eV(z_abs, settings["edge"]) if z_abs else 0.0
            scan = self._scan_from_fdmnes_output(
                output_file, label=out_basename, e0=e0, edge=settings["edge"],
            )
            if not self._fdmnes_output_covers_range(scan, settings):
                logs.append("  Output is incomplete; not loading partial spectrum.")
                return {"ok": False, "xyz_name": xyz_name, "logs": logs}
            logs.append(f"  Pushed to plot: {scan.label}")
            return {
                "ok": True,
                "xyz_name": xyz_name,
                "logs": logs,
                "scan": scan,
            }
        except Exception as exc:
            logs.append(f"  FAILED: {exc}")
            return {"ok": False, "xyz_name": xyz_name, "logs": logs}

    def _batch_fdmnes_worker(self, xyz_paths, base_workdir, spec,
                             absorber_spec, cluster_radius, settings,
                             batch_jobs: int = 1):
        def log(msg: str):
            self.after(0, lambda m=msg: self._append_fd_log(m))

        def push_scan(scan):
            if scan is not None and self._add_exp_scan_fn is not None:
                self.after(0, lambda s=scan: self._add_exp_scan_fn(s))

        n = len(xyz_paths)
        succeeded = 0
        failed = []
        tasks = []
        used_folders: set[str] = set()
        export_kwargs = self._fdmnes_export_kwargs(settings)

        log("")
        log("Preparing FDMNES input folders ...")
        for idx, xyz_path in enumerate(xyz_paths, start=1):
            xyz_name = os.path.basename(xyz_path)
            log("")
            log(f"--- prepare [{idx}/{n}] {xyz_name} ---")
            try:
                stem = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                              Path(xyz_path).stem).strip("._") or f"xyz_{idx}"
                base_folder = (f"{stem}_{cluster_radius:.1f}A"
                               if cluster_radius is not None else stem)
                folder = base_folder
                suffix = 2
                while folder.lower() in used_folders:
                    folder = f"{base_folder}_{suffix}"
                    suffix += 1
                used_folders.add(folder.lower())
                sub = Path(base_workdir) / folder
                sub.mkdir(parents=True, exist_ok=True)

                structure = parse_xyz_file(xyz_path)
                absorber_idx, abs_note = _resolve_absorber_index(
                    absorber_spec, structure
                )
                log(f"  Absorber: {abs_note}")

                bundle = export_xyz_as_fdmnes_bundle(
                    xyz_path, str(sub),
                    absorber_index=absorber_idx,
                    cluster_radius=cluster_radius,
                    basename=stem,
                    **export_kwargs,
                )
                log(f"  Wrote: {Path(bundle['input_path']).name}, "
                    f"{Path(bundle['fdmfile_path']).name}, "
                    f"{Path(bundle['xyz_copy_path']).name}")
                tasks.append({
                    "idx": idx,
                    "n": n,
                    "xyz_path": xyz_path,
                    "xyz_name": xyz_name,
                    "sub": str(sub),
                    "bundle": bundle,
                    "absorber_idx": absorber_idx,
                })
            except Exception as exc:
                log(f"  FAILED: {exc}")
                failed.append(xyz_name)

        if not tasks:
            log("")
            log(f"=== Batch complete: 0/{n} succeeded"
                + (f", failed: {', '.join(failed)}" if failed else "")
                + " ===")
            self.after(0, self._fdmnes_finish)
            return

        batch_jobs = max(1, min(int(batch_jobs or 1), len(tasks)))
        log("")
        if batch_jobs == 1:
            log("Running prepared inputs linearly ...")
            for task in tasks:
                log("")
                log(f"--- run [{task['idx']}/{n}] {task['xyz_name']} ---")
                result = self._execute_prepared_fdmnes_task(task, spec, settings)
                for ln in result.get("logs", []):
                    log(ln)
                if result.get("ok"):
                    push_scan(result.get("scan"))
                    succeeded += 1
                else:
                    failed.append(result.get("xyz_name", task["xyz_name"]))
        else:
            log(f"Running prepared inputs with {batch_jobs} concurrent job(s) ...")
            with ThreadPoolExecutor(max_workers=batch_jobs) as pool:
                future_to_task = {
                    pool.submit(
                        self._execute_prepared_fdmnes_task, task, spec, settings
                    ): task
                    for task in tasks
                }
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    log("")
                    log(f"--- done [{task['idx']}/{n}] {task['xyz_name']} ---")
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "ok": False,
                            "xyz_name": task["xyz_name"],
                            "logs": [f"  FAILED: {exc}"],
                        }
                    for ln in result.get("logs", []):
                        log(ln)
                    if result.get("ok"):
                        push_scan(result.get("scan"))
                        succeeded += 1
                    else:
                        failed.append(result.get("xyz_name", task["xyz_name"]))

        log("")
        log(f"=== Batch complete: {succeeded}/{n} succeeded"
            + (f", failed: {', '.join(failed)}" if failed else "")
            + " ===")
        self.after(0, self._fdmnes_finish)

    # ------------------------------------------------------------------ #
    #  Forwarders for binah.py / project_manager / EXAFS Studio           #
    # ------------------------------------------------------------------ #
    def get_feff_paths(self) -> list:
        """Return the inner FEFF panel's loaded path list (for EXAFS R-space
        marker overlay)."""
        return self._feff_panel.get_feff_paths()

    def refresh_scan_list(self):
        """Forward to the inner FEFF panel so its scan combobox stays in sync
        with the global experimental-scan list."""
        try:
            self._feff_panel.refresh_scan_list()
        except Exception:
            pass

    def get_params(self) -> dict:
        """Return tab state for project save."""
        feff_params = self._feff_panel.get_params() if self._feff_panel else {}
        return {
            "engine":         self._engine_var.get(),
            "feff":           feff_params,
            "fdmnes": {
                "xyz_path":        self._fd_xyz_path_var.get(),
                "workdir":         self._fd_workdir_var.get(),
                "basename":        self._fd_basename_var.get(),
                "absorber":        self._fd_absorber_var.get(),
                "edge":            self._fd_edge_var.get(),
                "cluster_radius":  self._fd_cluster_radius_var.get(),
                "e_min":           float(self._fd_emin_var.get()),
                "e_step":          float(self._fd_estep_var.get()),
                "e_max":           float(self._fd_emax_var.get()),
                "use_quadrupole":  bool(self._fd_quadrupole_var.get()),
                "use_nondipole":   bool(self._fd_nondipole_var.get()),
                "use_nonquadrupole": bool(self._fd_nonquadrupole_var.get()),
                "use_noninterf":   bool(self._fd_noninterf_var.get()),
                "use_scf":         bool(self._fd_scf_var.get()),
                "scf_convergence": float(self._fd_convergence_var.get()),
                "convolution":     bool(self._fd_convolution_var.get()),
                "use_green":       bool(self._fd_green_var.get()),
                "use_energpho":    bool(self._fd_energpho_var.get()),
                "use_spinorbit":   bool(self._fd_spinorbit_var.get()),
                "use_magnetism":   bool(self._fd_magnetism_var.get()),
                "use_relativism":  bool(self._fd_relativism_var.get()),
                "use_nonrelat":    bool(self._fd_nonrelat_var.get()),
                "use_allsite":     bool(self._fd_allsite_var.get()),
                "use_cartesian":   bool(self._fd_cartesian_var.get()),
                "use_spherical":   bool(self._fd_spherical_var.get()),
                "parallel":        bool(self._fd_parallel_var.get()),
                "n_procs":         int(self._fd_n_procs_var.get()),
                "batch_mode":      self._fd_batch_mode_var.get(),
                "batch_jobs":      int(self._fd_batch_jobs_var.get()),
                "batch_recursive": bool(self._fd_batch_recursive_var.get()),
            },
        }

    def set_params(self, data: dict, *,
                   legacy_exafs_params: Optional[dict] = None) -> None:
        """Restore tab state from a project save.

        Backward compatibility: if ``data`` lacks the new "feff"/"fdmnes"
        sub-dicts (i.e. an old project where FEFF settings lived inside
        ``exafs_params`` directly), pull from ``legacy_exafs_params`` for
        the FEFF half.
        """
        data = data or {}
        if "engine" in data:
            try:
                self._engine_var.set(str(data["engine"]) or "FEFF")
                self._show_engine_panel(self._engine_var.get())
            except Exception:
                pass

        # FEFF half — prefer the new "feff" sub-dict, else fall back to the
        # full legacy exafs_params (which contained both halves pre-refactor).
        feff_data = data.get("feff")
        if not feff_data and legacy_exafs_params is not None:
            feff_data = legacy_exafs_params
        if feff_data:
            try:
                self._feff_panel.set_params(feff_data)
            except Exception:
                pass

        # FDMNES half.
        fd = data.get("fdmnes") or {}
        if "xyz_path" in fd:
            self._fd_xyz_path_var.set(str(fd["xyz_path"]))
        if "workdir" in fd:
            self._fd_workdir_var.set(str(fd["workdir"]))
        if "basename" in fd:
            self._fd_basename_var.set(str(fd["basename"]))
        if "absorber" in fd:
            self._fd_absorber_var.set(str(fd["absorber"]))
        if "edge" in fd:
            self._fd_edge_var.set(str(fd["edge"]))
        if "cluster_radius" in fd:
            self._fd_cluster_radius_var.set(str(fd["cluster_radius"]))
        for var, key in [(self._fd_emin_var, "e_min"),
                         (self._fd_estep_var, "e_step"),
                         (self._fd_emax_var, "e_max"),
                         (self._fd_convergence_var, "scf_convergence")]:
            if key in fd:
                try:
                    var.set(float(fd[key]))
                except Exception:
                    pass
        for var, key in [(self._fd_quadrupole_var, "use_quadrupole"),
                         (self._fd_nondipole_var, "use_nondipole"),
                         (self._fd_nonquadrupole_var, "use_nonquadrupole"),
                         (self._fd_noninterf_var, "use_noninterf"),
                         (self._fd_scf_var, "use_scf"),
                         (self._fd_convolution_var, "convolution"),
                         (self._fd_green_var, "use_green"),
                         (self._fd_energpho_var, "use_energpho"),
                         (self._fd_spinorbit_var, "use_spinorbit"),
                         (self._fd_magnetism_var, "use_magnetism"),
                         (self._fd_relativism_var, "use_relativism"),
                         (self._fd_nonrelat_var, "use_nonrelat"),
                         (self._fd_allsite_var, "use_allsite"),
                         (self._fd_cartesian_var, "use_cartesian"),
                         (self._fd_spherical_var, "use_spherical"),
                         (self._fd_parallel_var, "parallel")]:
            if key in fd:
                try:
                    var.set(bool(fd[key]))
                except Exception:
                    pass
        if "n_procs" in fd:
            try:
                self._fd_n_procs_var.set(int(fd["n_procs"]))
            except Exception:
                pass
        if "batch_mode" in fd:
            mode = str(fd["batch_mode"])
            self._fd_batch_mode_var.set(
                mode if mode in ("Linear", "Parallel jobs") else "Linear"
            )
        if "batch_jobs" in fd:
            try:
                self._fd_batch_jobs_var.set(max(1, int(fd["batch_jobs"])))
            except Exception:
                pass
        if "batch_recursive" in fd:
            try:
                self._fd_batch_recursive_var.set(bool(fd["batch_recursive"]))
            except Exception:
                pass

        self._refresh_fdmnes_status()
