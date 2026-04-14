"""
Binah - ORCA TDDFT XAS Viewer
Run with: python binah.py
Requires: matplotlib, numpy, scipy, xraylarch  (see requirements.txt)
"""

import os
import sys

try:
    import tkinter as tk
except ImportError:
    print(
        "\n"
        "ERROR: tkinter is not installed.\n"
        "tkinter ships with Python but must be enabled at install time.\n"
        "\n"
        "  Windows : Reinstall Python → Custom → check 'tcl/tk and IDLE'\n"
        "  macOS   : brew install python-tk@3.11\n"
        "  Linux   : sudo apt install python3-tk\n"
        "\n"
        "Test it with:  python -c \"import tkinter; print('ok')\"\n",
        file=sys.stderr,
    )
    sys.exit(1)

from tkinter import ttk, filedialog, messagebox

try:
    from sgm_xas_loader import SGMLoaderApp as _SGMLoaderApp
    _HAS_SGM = True
except Exception:
    _HAS_SGM = False

from orca_parser import OrcaParser, TDDFTSpectrum, ParseResult, ParseDiagnosis
from experimental_parser import ExperimentalParser, ExperimentalScan
from plot_widget import PlotWidget
from xas_analysis_tab import XASAnalysisTab
import project_manager as pm


class OrcaTDDFTApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Binah")
        self.geometry("1100x720")
        self.minsize(800, 550)

        self._parser     = OrcaParser()
        self._exp_parser = ExperimentalParser()

        self._spectra: list[TDDFTSpectrum] = []
        self._current_file: str = ""
        self._file_section_idx: dict = {}   # remembers last selected section per file path
        self._project_path: str = ""        # path of currently open .otproj (or "")
        self._recent_projects: list = []    # up to 10 recently opened/saved projects
        self._cfg_path = os.path.join(
            os.path.expanduser("~"), ".binah_config.json")
        self._load_recent_projects()

        self._build_menu()
        self._build_top_bar()
        self._build_main_area()
        self._build_status_bar()

    # ------------------------------------------------------------------ #
    #  Menu bar                                                             #
    # ------------------------------------------------------------------ #
    def _build_menu(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)

        # ── Project operations ───────────────────────────────────────────────
        file_menu.add_command(label="New Project",             accelerator="Ctrl+N",
                              command=self._new_project)
        file_menu.add_command(label="Open Project…",           accelerator="Ctrl+Shift+O",
                              command=self._open_project)
        file_menu.add_command(label="Save Project",            accelerator="Ctrl+S",
                              command=self._save_project)
        file_menu.add_command(label="Save Project As…",        accelerator="Ctrl+Shift+S",
                              command=self._save_project_as)
        file_menu.add_separator()

        # ── Recent projects submenu ───────────────────────────────────────────
        self._recent_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent Projects", menu=self._recent_menu)
        file_menu.add_separator()

        # ── Individual file operations ────────────────────────────────────────
        file_menu.add_command(label="Open .out File…",         accelerator="Ctrl+O",
                              command=self._open_file)
        file_menu.add_command(label="Open Multiple Files…",
                              command=self._open_multiple)
        file_menu.add_separator()
        file_menu.add_command(label="Load Experimental Data…", accelerator="Ctrl+E",
                              command=self._load_experimental)
        file_menu.add_command(label="Load SGM Stack…",
                              command=self._load_sgm_stack)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)
        self.bind_all("<Control-n>",       lambda _: self._new_project())
        self.bind_all("<Control-o>",       lambda _: self._open_file())
        self.bind_all("<Control-O>",       lambda _: self._open_project())
        self.bind_all("<Control-s>",       lambda _: self._save_project())
        self.bind_all("<Control-S>",       lambda _: self._save_project_as())
        self.bind_all("<Control-e>",       lambda _: self._load_experimental())
        # Populate recent-projects menu (needs self._recent_menu to exist first)
        self._rebuild_recent_menu()

    # ------------------------------------------------------------------ #
    #  Top toolbar                                                          #
    # ------------------------------------------------------------------ #
    def _build_top_bar(self):
        bar = tk.Frame(self, bd=1, relief=tk.RAISED, padx=6, pady=4)
        bar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(bar, text="Open File",  width=10, command=self._open_file).pack(side=tk.LEFT, padx=2)
        tk.Button(bar, text="Reload",     width=8,  command=self._reload_file).pack(side=tk.LEFT, padx=2)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        tk.Label(bar, text="TDDFT Section:").pack(side=tk.LEFT)
        self._section_var = tk.StringVar()
        self._section_cb = ttk.Combobox(
            bar, textvariable=self._section_var,
            state="readonly", width=45
        )
        self._section_cb.pack(side=tk.LEFT, padx=4)
        self._section_cb.bind("<<ComboboxSelected>>", self._on_section_change)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self._file_label = tk.Label(bar, text="No file loaded", fg="gray", anchor="w")
        self._file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    # ------------------------------------------------------------------ #
    #  Main area: notebook with Spectra + XAS Analysis tabs                #
    # ------------------------------------------------------------------ #
    def _build_main_area(self):
        nb = ttk.Notebook(self)
        nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ── Tab 1: Spectra (existing layout) ──────────────────────────────────
        spectra_frame = tk.Frame(nb)
        nb.add(spectra_frame, text="\U0001f4c8 Spectra")

        pane = tk.PanedWindow(spectra_frame, orient=tk.HORIZONTAL,
                              sashwidth=5, sashrelief=tk.RAISED)
        pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Left sidebar ---
        sidebar = tk.Frame(pane, width=230, bd=1, relief=tk.SUNKEN)
        pane.add(sidebar, minsize=180)

        tk.Label(sidebar, text="Loaded Files", font=("", 9, "bold")).pack(anchor="w", padx=4, pady=2)

        self._file_listbox = tk.Listbox(sidebar, height=8, selectmode=tk.SINGLE,
                                         exportselection=False)
        self._file_listbox.pack(fill=tk.X, padx=4)
        self._file_listbox.bind("<<ListboxSelect>>", self._on_file_select)

        sb_scroll = ttk.Scrollbar(sidebar, orient=tk.VERTICAL,
                                   command=self._file_listbox.yview)
        self._file_listbox.config(yscrollcommand=sb_scroll.set)

        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        tk.Button(
            sidebar, text="+ Add to Overlay", bg="#003d7a", fg="white",
            activebackground="#0055aa", font=("", 9, "bold"),
            command=self._add_current_to_overlay
        ).pack(fill=tk.X, padx=4, pady=(0, 2))

        tk.Button(
            sidebar, text="Load Exp. Data\u2026", bg="#6B0000", fg="white",
            activebackground="#8B0000", font=("", 9, "bold"),
            command=self._load_experimental
        ).pack(fill=tk.X, padx=4, pady=(0, 2))

        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        tk.Label(sidebar, text="Spectrum Info", font=("", 9, "bold")).pack(anchor="w", padx=4)
        self._info_text = tk.Text(sidebar, height=14, width=28, state=tk.DISABLED,
                                  font=("Courier", 8), wrap=tk.WORD, bd=0,
                                  bg=self.cget("bg"))
        self._info_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        # --- Right: plot widget ---
        plot_frame = tk.Frame(pane)
        pane.add(plot_frame, minsize=500)

        self._plot = PlotWidget(plot_frame)
        self._plot.pack(fill=tk.BOTH, expand=True)

        # ── Tab 2: XAS Analysis ───────────────────────────────────────────────
        xas_frame = tk.Frame(nb)
        nb.add(xas_frame, text="\U0001f52c XAS Analysis")

        self._xas_tab = XASAnalysisTab(
            xas_frame,
            get_scans_fn=lambda: self._plot._exp_scans,
            replot_fn=lambda: self._plot._replot(),
            add_scan_fn=self._add_exp_scan_to_plot,
        )
        self._xas_tab.pack(fill=tk.BOTH, expand=True)

        # Auto-run all scans when XAS Analysis tab is selected
        def _on_tab_changed(event):
            try:
                selected = nb.tab(nb.select(), "text")
                if "XAS" in selected:
                    self._xas_tab.refresh_scan_list()
                    self._xas_tab.auto_run_all()
            except Exception:
                pass
        nb.bind("<<NotebookTabChanged>>", _on_tab_changed)

    # ------------------------------------------------------------------ #
    #  Status bar                                                           #
    # ------------------------------------------------------------------ #
    def _build_status_bar(self):
        self._status = tk.StringVar(value="Ready. Open an ORCA .out file to begin.")
        bar = tk.Label(self, textvariable=self._status, bd=1, relief=tk.SUNKEN,
                       anchor="w", padx=6, font=("", 8))
        bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ------------------------------------------------------------------ #
    #  ORCA file operations                                                 #
    # ------------------------------------------------------------------ #
    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open ORCA Output File",
            filetypes=[("ORCA Output", "*.out"), ("All files", "*.*")]
        )
        if path:
            self._load_file(path)

    def _open_multiple(self):
        paths = filedialog.askopenfilenames(
            title="Open ORCA Output Files",
            filetypes=[("ORCA Output", "*.out"), ("All files", "*.*")]
        )
        for path in paths:
            self._load_file(path, switch=False)
        if paths:
            self._file_listbox.selection_clear(0, tk.END)
            self._file_listbox.selection_set(tk.END)
            self._on_file_select()

    def _reload_file(self):
        if self._current_file:
            self._load_file(self._current_file)

    def _load_file(self, path: str, switch: bool = True):
        self._status.set(f"Parsing: {os.path.basename(path)}\u2026")
        self.update_idletasks()
        try:
            result: ParseResult = self._parser.parse(path)
        except Exception as e:
            messagebox.showerror("Parse Error", f"Failed to parse file:\n{e}")
            self._status.set("Error during parsing.")
            return

        diag = result.diagnosis
        spectra = result.spectra

        if not spectra:
            self._show_no_data_dialog(path, diag)
            self._status.set(f"No spectrum data found — {diag.termination_reason or 'unknown reason'}.")
            return

        if not hasattr(self, "_file_data"):
            self._file_data: dict = {}
        self._file_data[path] = spectra

        names = [self._file_listbox.get(i) for i in range(self._file_listbox.size())]
        short = os.path.basename(path)
        if short not in names:
            self._file_listbox.insert(tk.END, short)
            self._file_listbox._paths = getattr(self._file_listbox, "_paths", [])
            self._file_listbox._paths.append(path)

        if switch:
            self._current_file = path
            idx = self._file_listbox._paths.index(path)
            self._file_listbox.selection_clear(0, tk.END)
            self._file_listbox.selection_set(idx)
            self._switch_to_file(path)

        n = len(spectra)
        self._status.set(
            f"Loaded: {short}  —  {n} TDDFT section{'s' if n != 1 else ''} found."
        )

    def _on_file_select(self, event=None):
        sel = self._file_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        paths = getattr(self._file_listbox, "_paths", [])
        if idx < len(paths):
            path = paths[idx]
            self._current_file = path
            self._switch_to_file(path)

    def _switch_to_file(self, path: str):
        spectra = getattr(self, "_file_data", {}).get(path, [])
        self._spectra = spectra
        self._file_label.config(text=os.path.basename(path), fg="black")

        labels = [s.display_name() for s in spectra]
        self._section_cb["values"] = labels
        if labels:
            saved_idx = self._file_section_idx.get(path, 0)
            restore = saved_idx if saved_idx < len(labels) else 0
            self._section_cb.current(restore)
            self._on_section_change()

    # ------------------------------------------------------------------ #
    #  Section selection                                                    #
    # ------------------------------------------------------------------ #
    def _on_section_change(self, event=None):
        idx = self._section_cb.current()
        if idx < 0 or idx >= len(self._spectra):
            return
        if self._current_file:
            self._file_section_idx[self._current_file] = idx
        spectrum = self._spectra[idx]
        self._plot.load_spectrum(spectrum)
        self._update_info(spectrum)

    def _add_current_to_overlay(self):
        idx = self._section_cb.current()
        if idx < 0 or idx >= len(self._spectra):
            messagebox.showinfo("No Spectrum", "Select a spectrum section first.")
            return
        sp = self._spectra[idx]
        short = os.path.basename(self._current_file)
        label = f"{short} — {sp.display_name()}"
        self._plot.add_overlay(label, sp)
        self._status.set(f"Added to overlay: {label}")

    # ------------------------------------------------------------------ #
    #  Experimental data loading                                            #
    # ------------------------------------------------------------------ #
    def _load_experimental(self):
        """Open a file dialog and load experimental XAS scan(s)."""
        path = filedialog.askopenfilename(
            title="Load Experimental XAS Scan",
            filetypes=[
                ("All supported", "*.dat *.prj *.nor *.csv *.txt"),
                ("SXRMB / BioXAS (.dat)", "*.dat"),
                ("Athena project (.prj)", "*.prj"),
                ("Athena normalized (.nor)", "*.nor"),
                ("CSV / text", "*.csv *.txt"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()

        try:
            if ext == ".dat" and self._exp_parser.is_sxrmb(path):
                self._load_sxrmb_with_dialog(path)
            elif ext == ".dat":
                self._load_dat_with_dialog(path)
            elif ext == ".prj":
                self._load_prj_with_dialog(path)
            elif ext == ".nor":
                scans = self._exp_parser.parse_nor(path)
                for scan in scans:
                    self._add_exp_scan_to_plot(scan)
                self._status.set(
                    f"Loaded {len(scans)} scan(s) from {os.path.basename(path)}")
            else:
                # Generic CSV / two-column text
                scan = self._exp_parser.parse_csv(path)
                self._add_exp_scan_to_plot(scan)
                self._status.set(f"Loaded experimental scan: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load experimental file:\n{e}")
            self._status.set("Error loading experimental file.")

    def _load_sxrmb_with_dialog(self, path: str):
        """Show signal-selection dialog for SXRMB .dat files, then load."""
        win = tk.Toplevel(self)
        win.title("SXRMB Import — Select Signal")
        win.resizable(False, False)
        win.grab_set()

        hdr = tk.Frame(win, bg="#003366", pady=6)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="CLS SXRMB Beamline Import",
                 font=("", 11, "bold"), bg="#003366", fg="white").pack(padx=12)
        tk.Label(hdr, text=os.path.basename(path),
                 font=("", 8), bg="#003366", fg="#AACCFF").pack(padx=12)

        body = tk.Frame(win, padx=16, pady=10)
        body.pack(fill=tk.BOTH)

        tk.Label(body, text="Which signal(s) to load?",
                 font=("", 9, "bold")).pack(anchor="w", pady=(0, 6))

        _signal_var = tk.StringVar(value="both")
        for _val, _txt, _desc in [
            ("tey",  "TEY only",
             "Total Electron Yield (TEYDetector / I0)"),
            ("fluor","Fluorescence only",
             "norm_*Ka1 fluorescence channel"),
            ("both", "Both TEY and Fluorescence",
             "Load as two separate scans"),
        ]:
            f = tk.Frame(body)
            f.pack(anchor="w", pady=2)
            tk.Radiobutton(f, text=_txt, variable=_signal_var, value=_val,
                           font=("", 9)).pack(side=tk.LEFT)
            tk.Label(f, text=f"  — {_desc}", font=("", 8),
                     fg="gray").pack(side=tk.LEFT)

        btn_row = tk.Frame(win, pady=8)
        btn_row.pack()

        def do_load():
            sig = _signal_var.get()
            win.destroy()
            try:
                scans = self._exp_parser.parse_sxrmb(path, signal=sig)
                for scan in scans:
                    self._add_exp_scan_to_plot(scan)
                self._status.set(
                    f"Loaded {len(scans)} SXRMB scan(s) from {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("SXRMB Load Error",
                                     f"Failed to load SXRMB file:\n{e}")
                self._status.set("Error loading SXRMB file.")

        tk.Button(btn_row, text="Load", width=12, bg="#003366", fg="white",
                  activebackground="#0055aa", command=do_load).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="Cancel", width=10,
                  command=win.destroy).pack(side=tk.LEFT, padx=4)

    def _load_dat_with_dialog(self, path: str):
        """Show options dialog for BioXAS .dat files, then load."""
        win = tk.Toplevel(self)
        win.title("Load .dat — Options")
        win.resizable(False, False)
        win.grab_set()

        # Header
        hdr = tk.Frame(win, bg="#6B0000", padx=12, pady=8)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="BioXAS XDI Import Options",
                 bg="#6B0000", fg="white", font=("", 11, "bold")).pack(anchor="w")
        tk.Label(hdr, text=os.path.basename(path),
                 bg="#6B0000", fg="#ffaaaa", font=("", 9)).pack(anchor="w")

        body = tk.Frame(win, padx=16, pady=12)
        body.pack(fill=tk.BOTH)

        # Mode
        mode_var = tk.StringVar(value="fluorescence")
        tk.Label(body, text="Measurement mode:", font=("", 9, "bold")).pack(anchor="w", pady=(0, 4))
        tk.Radiobutton(body, text="Fluorescence  (NiKa1_InB + NiKa1_OutB) / I0",
                       variable=mode_var, value="fluorescence").pack(anchor="w")
        tk.Radiobutton(body, text="Transmission  ln(I0 / I1)",
                       variable=mode_var, value="transmission").pack(anchor="w")

        ttk.Separator(body, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # Normalization
        norm_var = tk.BooleanVar(value=True)
        tk.Checkbutton(body, text="Apply Athena-style normalization\n"
                       "   (pre-edge linear fit + edge-step normalization)",
                       variable=norm_var, justify=tk.LEFT).pack(anchor="w")

        ttk.Separator(body, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        status_lbl = tk.Label(body, text="", fg="red", font=("", 9))
        status_lbl.pack(anchor="w")

        def do_load():
            try:
                scan = self._exp_parser.parse_dat(
                    path,
                    mode=mode_var.get(),
                    normalize=norm_var.get(),
                )
                win.destroy()
                self._add_exp_scan_to_plot(scan)
                self._status.set(
                    f"Loaded experimental: {scan.label} [{scan.scan_type}]"
                )
            except Exception as e:
                status_lbl.config(text=f"Error: {e}")

        btn_row = tk.Frame(win)
        btn_row.pack(pady=(0, 10))
        tk.Button(btn_row, text="Load", width=12, bg="#6B0000", fg="white",
                  activebackground="#8B0000", command=do_load).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="Cancel", width=10,
                  command=win.destroy).pack(side=tk.LEFT, padx=4)

        win.update_idletasks()
        x = self.winfo_x() + (self.winfo_width()  - win.winfo_width())  // 2
        y = self.winfo_y() + (self.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{x}+{y}")

    def _load_prj_with_dialog(self, path: str):
        """Parse .prj file, then show a scan selection dialog."""
        self._status.set(f"Parsing Athena project: {os.path.basename(path)}\u2026")
        self.update_idletasks()

        try:
            scans = self._exp_parser.parse_prj(path)
        except Exception as e:
            messagebox.showerror("Parse Error", f"Failed to read .prj file:\n{e}")
            self._status.set("Error reading .prj file.")
            return

        if not scans:
            messagebox.showwarning("No Scans", "No valid scan groups found in this .prj file.")
            self._status.set("No scans found in .prj file.")
            return

        if len(scans) == 1:
            # Only one scan — load it directly
            self._add_exp_scan_to_plot(scans[0])
            self._status.set(f"Loaded 1 scan from {os.path.basename(path)}")
            return

        # Multiple scans → show selection dialog
        win = tk.Toplevel(self)
        win.title(f"Select Scans — {os.path.basename(path)}")
        win.resizable(True, True)
        win.grab_set()

        hdr = tk.Frame(win, bg="#6B0000", padx=12, pady=8)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Athena Project — Select Scans to Load",
                 bg="#6B0000", fg="white", font=("", 11, "bold")).pack(anchor="w")
        tk.Label(hdr, text=f"{len(scans)} scan groups found  |  {os.path.basename(path)}",
                 bg="#6B0000", fg="#ffaaaa", font=("", 9)).pack(anchor="w")

        body = tk.Frame(win, padx=10, pady=8)
        body.pack(fill=tk.BOTH, expand=True)

        tk.Label(body, text="Select which scans to load (Ctrl+click for multiple):",
                 font=("", 9)).pack(anchor="w", pady=(0, 4))

        list_frame = tk.Frame(body)
        list_frame.pack(fill=tk.BOTH, expand=True)

        lb = tk.Listbox(list_frame, selectmode=tk.EXTENDED, height=min(len(scans), 14),
                        font=("Courier", 9), exportselection=False)
        lb_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=lb.yview)
        lb.config(yscrollcommand=lb_scroll.set)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lb_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        for sc in scans:
            e0_str = f"E0={sc.e0:.1f} eV  " if sc.e0 else ""
            n_pts  = len(sc.energy_ev)
            lb.insert(tk.END, f"{sc.label:<30}  {e0_str}({n_pts} pts)")

        # Select all by default
        lb.selection_set(0, tk.END)

        btn_row = tk.Frame(win)
        btn_row.pack(pady=8)

        def do_load():
            sel = lb.curselection()
            if not sel:
                messagebox.showwarning("Nothing Selected",
                                       "Select at least one scan to load.",
                                       parent=win)
                return
            win.destroy()
            loaded = 0
            for idx in sel:
                self._add_exp_scan_to_plot(scans[idx])
                loaded += 1
            self._status.set(
                f"Loaded {loaded} scan{'s' if loaded != 1 else ''} "
                f"from {os.path.basename(path)}"
            )

        tk.Button(btn_row, text="Load Selected", width=14,
                  bg="#6B0000", fg="white", activebackground="#8B0000",
                  command=do_load).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="Select All",  width=10,
                  command=lambda: lb.selection_set(0, tk.END)).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="Cancel", width=10,
                  command=win.destroy).pack(side=tk.LEFT, padx=4)

        win.update_idletasks()
        x = self.winfo_x() + (self.winfo_width()  - win.winfo_width())  // 2
        y = self.winfo_y() + (self.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{x}+{y}")
        win.minsize(400, 200)

    def _load_sgm_stack(self):
        """Open the SGM Stack Loader as a child Toplevel window."""
        if not _HAS_SGM:
            messagebox.showerror(
                "SGM Loader",
                "SGM loader not available.\n\n"
                "The bundled SGM loader could not be imported.\n"
                "Check that the repository files are present and the dependencies\n"
                "from requirements.txt are installed.")
            return
        try:
            # SGMLoaderApp is now a tk.Toplevel — pass self as master so it
            # shares Binah's event loop.  wait_window() blocks until the user
            # closes the SGM window, keeping Binah responsive throughout.
            app = _SGMLoaderApp(master=self, on_load_cb=self._add_exp_scan_to_plot)
            self.wait_window(app)
        except Exception as e:
            messagebox.showerror("SGM Error", f"Could not open SGM loader:\n{e}")

    def _add_exp_scan_to_plot(self, scan: ExperimentalScan):
        """Forward a loaded experimental scan to the plot widget."""
        short_src = os.path.basename(scan.source_file)
        label = f"{scan.label}  [{short_src}]"
        self._plot.add_exp_scan(label, scan)
        if hasattr(self, "_xas_tab"):
            self._xas_tab.refresh_scan_list()

    # ------------------------------------------------------------------ #
    #  Diagnostic dialog for missing spectrum data                          #
    # ------------------------------------------------------------------ #
    def _show_no_data_dialog(self, path: str, diag: ParseDiagnosis):
        win = tk.Toplevel(self)
        win.title("No Spectrum Data Found")
        win.resizable(False, False)
        win.grab_set()

        hdr = tk.Frame(win, bg="#8B0000", padx=12, pady=8)
        hdr.pack(fill=tk.X)
        tk.Label(
            hdr, text="No TDDFT Spectrum Data Found",
            bg="#8B0000", fg="white", font=("", 11, "bold")
        ).pack(anchor="w")
        tk.Label(
            hdr, text=os.path.basename(path),
            bg="#8B0000", fg="#ffaaaa", font=("", 9)
        ).pack(anchor="w")

        body = tk.Frame(win, padx=14, pady=10)
        body.pack(fill=tk.BOTH)

        txt = tk.Text(body, width=64, height=14, wrap=tk.WORD,
                      font=("Courier", 9), relief=tk.FLAT, bg="#f8f8f8")
        txt.pack(fill=tk.BOTH)

        def ins(text, tag=None):
            txt.insert(tk.END, text, tag or ())

        txt.tag_config("warn",   foreground="#8B0000", font=("Courier", 9, "bold"))
        txt.tag_config("ok",     foreground="#006400", font=("Courier", 9, "bold"))
        txt.tag_config("head",   font=("Courier", 9, "bold"))
        txt.tag_config("indent", lmargin1=20, lmargin2=20)

        if diag.is_complete:
            ins("Status: ", "head"); ins("ORCA terminated normally\n", "ok")
        else:
            ins("Status: ", "head"); ins("Calculation INCOMPLETE\n", "warn")

        if diag.termination_reason:
            ins(f"Reason: {diag.termination_reason}\n\n", "warn" if not diag.is_complete else ())
        else:
            ins("\n")

        if diag.tddft_started:
            ins("TD-DFT block:   ", "head"); ins("Initialised\n")
            if diag.xas_mode:
                ins("Mode:           ", "head"); ins("XAS / core-excitation\n")
            if diag.n_roots_requested:
                ins("Roots requested:", "head"); ins(f" {diag.n_roots_requested}\n")
            ins("Davidson iters: ", "head")
            if diag.tddft_converged:
                ins("Converged\n", "ok")
            else:
                ins(f"{diag.davidson_iterations} (NOT converged)\n", "warn")
        else:
            ins("TD-DFT block:   ", "head"); ins("Not detected\n", "warn")

        if diag.partial_states:
            ins(f"\nPartial eigenvalues from last Davidson iteration:\n", "head")
            for s in diag.partial_states[:10]:
                ins(f"  Root {s.index:>2}: {s.energy_ev:.4f} eV  "
                    f"({1e7/s.energy_cm:.1f} nm  |  {s.energy_cm:.0f} cm\u207b\u00b9)\n", "indent")
            if len(diag.partial_states) > 10:
                ins(f"  ... and {len(diag.partial_states)-10} more\n", "indent")

        ins("\nWhat to do:\n", "head")
        if not diag.is_complete:
            ins("  \u2022 Resubmit the job with a longer wall time\n", "indent")
            ins("  \u2022 Or increase %maxcore / reduce nroots\n", "indent")
        if diag.xas_mode:
            ins("  \u2022 Check the donor orbital window setting\n", "indent")
        if diag.tddft_started and not diag.tddft_converged:
            ins("  \u2022 Consider increasing maxdim or switching to TDA\n", "indent")
        ins("  \u2022 Once finished successfully, reload the .out file\n", "indent")

        txt.config(state=tk.DISABLED)

        tk.Button(win, text="OK", width=10, command=win.destroy).pack(pady=(0, 10))
        win.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - win.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{x}+{y}")

    # ------------------------------------------------------------------ #
    #  Spectrum info panel                                                  #
    # ------------------------------------------------------------------ #
    def _update_info(self, spectrum: TDDFTSpectrum):
        import numpy as np
        self._info_text.config(state=tk.NORMAL)
        self._info_text.delete("1.0", tk.END)

        n = len(spectrum.states)
        is_cd = spectrum.is_cd()
        use_ev = spectrum.is_xas

        lines = [
            f"Section: {spectrum.label}",
            f"Type:    {'XAS' if spectrum.is_xas else 'UV/Vis'}",
            f"States:  {n}",
            "",
        ]

        if not is_cd and spectrum.fosc:
            fosc = spectrum.fosc
            lines += [
                f"Max f:   {max(fosc):.6f}",
                f"Sum f:   {sum(fosc):.4f}",
                "",
                "Top 5 (by f):",
            ]
            sorted_idx = sorted(range(n), key=lambda i: fosc[i], reverse=True)[:5]
            for i in sorted_idx:
                if use_ev:
                    ev = spectrum.energies_ev[i] if i < len(spectrum.energies_ev) else 0
                    lines.append(f"  S{spectrum.states[i]:>3}: {ev:>8.3f} eV  f={fosc[i]:.5f}")
                else:
                    nm = spectrum.wavelengths_nm[i] if i < len(spectrum.wavelengths_nm) else 0
                    lines.append(f"  S{spectrum.states[i]:>3}: {nm:>7.1f} nm   f={fosc[i]:.5f}")

        elif is_cd and spectrum.rotatory_strength:
            r = spectrum.rotatory_strength
            lines += [
                f"Max |R|: {max(abs(x) for x in r):.4f}",
                "",
                "Top 5 (by |R|):",
            ]
            sorted_idx = sorted(range(n), key=lambda i: abs(r[i]), reverse=True)[:5]
            for i in sorted_idx:
                nm = spectrum.wavelengths_nm[i] if i < len(spectrum.wavelengths_nm) else 0
                lines.append(f"  S{spectrum.states[i]:>3}: {nm:>7.1f} nm  R={r[i]:.4f}")

        if spectrum.is_combined() and spectrum.fosc_m2:
            lines += ["", "Includes M2/Q2: yes"]

        if spectrum.excited_states:
            lines += ["", f"MO transitions: {len(spectrum.excited_states)} states"]

        self._info_text.insert(tk.END, "\n".join(lines))
        self._info_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------ #
    #  Project save / open                                                  #
    # ------------------------------------------------------------------ #
    _PROJ_FILETYPES = [
        ("ORCA TDDFT Project", "*.otproj"),
        ("All files",          "*.*"),
    ]

    # ------------------------------------------------------------------ #
    #  Recent projects                                                      #
    # ------------------------------------------------------------------ #
    def _load_recent_projects(self):
        """Load the recent-projects list from the shared config file."""
        try:
            import json
            if os.path.exists(self._cfg_path):
                with open(self._cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                self._recent_projects = [
                    p for p in cfg.get("recent_projects", [])
                    if isinstance(p, str)
                ][:10]
        except Exception:
            self._recent_projects = []

    def _save_recent_projects(self):
        """Persist the recent-projects list to the shared config file."""
        try:
            import json
            cfg = {}
            if os.path.exists(self._cfg_path):
                with open(self._cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            cfg["recent_projects"] = self._recent_projects[:10]
            with open(self._cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

    def _add_recent(self, path: str):
        """Add path to the front of recent projects, deduplicate, cap at 10."""
        path = os.path.normpath(os.path.abspath(path))
        self._recent_projects = (
            [path] + [p for p in self._recent_projects if p != path]
        )[:10]
        self._save_recent_projects()
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self):
        """Refresh the Recent Projects submenu entries."""
        self._recent_menu.delete(0, tk.END)
        if not self._recent_projects:
            self._recent_menu.add_command(label="(no recent projects)",
                                          state=tk.DISABLED)
            return
        for path in self._recent_projects:
            label = os.path.basename(path)
            self._recent_menu.add_command(
                label=label,
                command=lambda p=path: self._open_recent_project(p),
            )
        self._recent_menu.add_separator()
        self._recent_menu.add_command(label="Clear Recent",
                                      command=self._clear_recent)

    def _open_recent_project(self, path: str):
        """Open a project from the recent list."""
        if not os.path.exists(path):
            messagebox.showerror(
                "File Not Found",
                f"Cannot find:\n{path}\n\nIt will be removed from recent projects.",
                parent=self,
            )
            self._recent_projects = [p for p in self._recent_projects if p != path]
            self._save_recent_projects()
            self._rebuild_recent_menu()
            return
        self._status.set("Loading project…")
        self.update_idletasks()
        try:
            doc = pm.load_project(path)
        except Exception as exc:
            messagebox.showerror("Open Error",
                                 f"Could not read project file:\n{exc}", parent=self)
            self._status.set("Open failed.")
            return
        warnings = pm.restore_project(doc, self)
        self._project_path = path
        self.title(f"Binah — {os.path.basename(path)}")
        self._add_recent(path)
        n_exp  = len(self._plot._exp_scans)
        n_orca = self._file_listbox.size()
        n_ov   = len(self._plot._overlay_spectra)
        self._status.set(
            f"Project loaded: {os.path.basename(path)}  |  "
            f"{n_orca} ORCA file(s)  |  {n_exp} exp. scan(s)  |  "
            f"{n_ov} TDDFT overlay(s)")
        if warnings:
            messagebox.showwarning(
                "Project Loaded with Warnings",
                "Some items could not be restored:\n\n" + "\n".join(f"• {w}" for w in warnings),
                parent=self)

    def _clear_recent(self):
        self._recent_projects = []
        self._save_recent_projects()
        self._rebuild_recent_menu()

    def _new_project(self):
        """Clear all state and start fresh."""
        if not messagebox.askyesno(
            "New Project",
            "Start a new project?\nAll unsaved work will be lost.",
            default="no", parent=self,
        ):
            return
        # Clear experimental scans
        self._plot._exp_scans.clear()
        self._plot._overlay_spectra.clear()
        self._plot._refresh_panel_content()
        # Clear ORCA files
        if hasattr(self, "_file_data"):
            self._file_data.clear()
        self._file_section_idx.clear()
        self._file_listbox.delete(0, tk.END)
        self._file_listbox._paths = []
        self._spectra = []
        self._current_file = ""
        self._project_path = ""
        self._file_label.config(text="No file loaded", fg="gray")
        self._section_cb["values"] = []
        self._section_cb.set("")
        self._plot._replot()
        self._xas_tab.refresh_scan_list()
        self.title("Binah")
        self._status.set("New project started.")

    def _save_project(self):
        """Save to current project path, or prompt for one if unsaved."""
        if not self._project_path:
            self._save_project_as()
            return
        self._do_save(self._project_path)

    def _save_project_as(self):
        path = filedialog.asksaveasfilename(
            title="Save Project As…",
            defaultextension=".otproj",
            filetypes=self._PROJ_FILETYPES,
        )
        if not path:
            return
        self._do_save(path)

    def _do_save(self, path: str):
        self._status.set(f"Saving project…")
        self.update_idletasks()
        try:
            pm.save_project(path, self)
            self._project_path = path
            self.title(f"Binah — {os.path.basename(path)}")
            self._status.set(f"Project saved: {os.path.basename(path)}")
            self._add_recent(path)
        except Exception as exc:
            messagebox.showerror("Save Error",
                                 f"Could not save project:\n{exc}", parent=self)
            self._status.set("Save failed.")

    def _open_project(self):
        path = filedialog.askopenfilename(
            title="Open Project…",
            filetypes=self._PROJ_FILETYPES,
        )
        if not path:
            return
        self._status.set(f"Loading project…")
        self.update_idletasks()
        try:
            doc = pm.load_project(path)
        except Exception as exc:
            messagebox.showerror("Open Error",
                                 f"Could not read project file:\n{exc}", parent=self)
            self._status.set("Open failed.")
            return

        warnings = pm.restore_project(doc, self)
        self._project_path = path
        self.title(f"Binah — {os.path.basename(path)}")
        self._add_recent(path)

        n_exp   = len(self._plot._exp_scans)
        n_orca  = self._file_listbox.size()
        n_ov    = len(self._plot._overlay_spectra)
        msg = (f"Project loaded: {os.path.basename(path)}  |  "
               f"{n_orca} ORCA file(s)  |  {n_exp} exp. scan(s)  |  "
               f"{n_ov} TDDFT overlay(s)")
        self._status.set(msg)

        if warnings:
            messagebox.showwarning(
                "Project Loaded with Warnings",
                "Some items could not be restored:\n\n" + "\n".join(f"• {w}" for w in warnings),
                parent=self,
            )

    # ------------------------------------------------------------------ #
    #  About dialog                                                         #
    # ------------------------------------------------------------------ #
    def _show_about(self):
        messagebox.showinfo(
            "About Binah",
            "Binah\n"
            "Parses and interactively plots TDDFT spectra\n"
            "from ORCA quantum chemistry output files.\n\n"
            "Supported TDDFT sections:\n"
            "  \u2022 Electric Dipole, Velocity Dipole\n"
            "  \u2022 CD Spectrum (all variants)\n"
            "  \u2022 Combined D2+m2+Q2 (all variants)\n"
            "  \u2022 Origin-independent and semi-classical\n\n"
            "Experimental XAS overlay:\n"
            "  \u2022 BioXAS XDI .dat files (fluorescence / transmission)\n"
            "  \u2022 Athena/Demeter .prj files (gzip Perl format)\n"
            "  \u2022 Athena normalized .nor files (XDI export)\n"
            "  \u2022 Generic CSV / two-column text\n\n"
            "Features: Gaussian/Lorentzian broadening,\n"
            "unit switching (nm/eV/cm\u207b\u00b9), \u0394E shift alignment,\n"
            "hover tooltips, twin y-axis for experiment,\n"
            "figure export (PNG/PDF/SVG), CSV export.\n\n"
            "Built for ORCA \u2265 4.x output format."
        )


def main():
    try:
        import numpy
        import matplotlib
    except ImportError:
        print("Missing dependencies. Run: pip install numpy matplotlib")
        sys.exit(1)

    app = OrcaTDDFTApp()
    app.mainloop()


if __name__ == "__main__":
    main()
