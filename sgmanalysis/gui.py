import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .plotting import plot_xeol
from .scans import MapScan, StackScan


TOM_NOTES = (
    "Tom's guidance:\n"
    "- TEY is Total Electron Yield and comes from the MCC signal (typically ch2).\n"
    "- PFY is an SDD integration over a selected emission-channel ROI.\n"
    "- TFY integrates over nearly all emission channels and is often less useful when other elements contribute.\n"
    "- Use the emission/excitation matrix to choose the best emission ROI before plotting PFY."
)

HDF5_EXTENSIONS = (".h5", ".hdf5", ".nxs")


class SGMAnalysisGUI:
    """Desktop GUI for the core SGM analysis workflows."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SGM Analysis Desktop GUI")
        self.root.minsize(1024, 760)

        self.map_scan = None
        self.stack_scan = None
        self.source_records = {}

        self.status_var = tk.StringVar(
            value="Choose a map or stack HDF5 file to begin. Plots open in separate Matplotlib windows."
        )
        self.map_info_var = tk.StringVar(value="No map scan loaded.")
        self.stack_info_var = tk.StringVar(value="No stack scan loaded.")
        self.source_root_var = tk.StringVar()
        self.source_summary_var = tk.StringVar(
            value="No directory indexed yet. Point this at a parent folder to discover nested stacks and scans."
        )

        self._build_ui()

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(3, weight=1)

        title = ttk.Label(container, text="SGM Analysis Desktop GUI")
        title.grid(row=0, column=0, sticky="w")

        notes = ttk.Label(container, text=TOM_NOTES, justify="left", wraplength=980)
        notes.grid(row=1, column=0, sticky="ew", pady=(6, 12))

        self._build_source_browser(container)

        notebook = ttk.Notebook(container)
        notebook.grid(row=3, column=0, sticky="nsew")

        self._build_map_tab(notebook)
        self._build_stack_tab(notebook)
        self._build_heatmap_tab(notebook)
        self._build_pca_tab(notebook)
        self._build_xeol_tab(notebook)

        status = ttk.Label(container, textvariable=self.status_var, justify="left", wraplength=980)
        status.grid(row=4, column=0, sticky="ew", pady=(12, 0))

    def _build_source_browser(self, parent):
        frame = ttk.LabelFrame(parent, text="Source Explorer", padding=12)
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(2, weight=1)

        ttk.Label(frame, text="Root directory").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(frame, textvariable=self.source_root_var).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(frame, text="Browse Folder", command=self._browse_source_directory).grid(
            row=0, column=2, sticky="ew", padx=8, pady=4
        )
        ttk.Button(frame, text="Scan Recursively", command=self._scan_source_directory).grid(
            row=0, column=3, sticky="ew", pady=4
        )

        button_row = ttk.Frame(frame)
        button_row.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 8))
        ttk.Button(button_row, text="Use Selected As Stack", command=self._use_selected_source_for_stack).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(button_row, text="Use Selected As Map", command=self._use_selected_source_for_map).pack(side="left")

        tree_frame = ttk.Frame(frame)
        tree_frame.grid(row=2, column=0, columnspan=4, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        self.source_tree = ttk.Treeview(
            tree_frame,
            columns=("kind", "details"),
            show="tree headings",
            height=8,
        )
        self.source_tree.heading("#0", text="Discovered Source")
        self.source_tree.heading("kind", text="Kind")
        self.source_tree.heading("details", text="Details")
        self.source_tree.column("#0", width=420, stretch=True)
        self.source_tree.column("kind", width=120, stretch=False)
        self.source_tree.column("details", width=420, stretch=True)
        self.source_tree.grid(row=0, column=0, sticky="nsew")
        self.source_tree.bind("<<TreeviewSelect>>", self._on_source_tree_select)
        self.source_tree.bind("<Double-1>", self._on_source_tree_double_click)

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.source_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.source_tree.configure(yscrollcommand=scrollbar.set)

        ttk.Label(frame, textvariable=self.source_summary_var, justify="left", wraplength=980).grid(
            row=3, column=0, columnspan=4, sticky="ew", pady=(8, 0)
        )

    def _build_map_tab(self, notebook: ttk.Notebook):
        frame = ttk.Frame(notebook, padding=12)
        frame.columnconfigure(1, weight=1)
        notebook.add(frame, text="Map Scan")

        self.map_path_var = tk.StringVar()
        self.map_channel_roi_var = tk.StringVar(value="80,101")
        self.map_roll_shift_var = tk.StringVar(value="0")
        self.map_map_roi_var = tk.StringVar()
        self.map_contrast_var = tk.StringVar()
        self.map_mcc_channels_var = tk.StringVar()
        self.map_scatter_var = tk.BooleanVar(value=False)

        self._build_file_picker_row(
            frame,
            row=0,
            label="Map HDF5 file",
            variable=self.map_path_var,
            browse_command=lambda: self._browse_open_file(self.map_path_var, [("HDF5 files", "*.h5"), ("All files", "*.*")]),
            load_command=self._load_map_scan,
        )

        ttk.Label(frame, textvariable=self.map_info_var, wraplength=900, justify="left").grid(
            row=1, column=0, columnspan=4, sticky="ew", pady=(6, 12)
        )

        options = ttk.LabelFrame(frame, text="Overview Options", padding=12)
        options.grid(row=2, column=0, columnspan=4, sticky="ew")
        options.columnconfigure(1, weight=1)

        self._build_entry_row(options, 0, "Channel ROI", self.map_channel_roi_var, "start,end")
        self._build_entry_row(options, 1, "Roll shift", self.map_roll_shift_var, "0")
        self._build_entry_row(options, 2, "Map ROI", self.map_map_roi_var, "x1,x2,y1,y2 (optional)")
        self._build_entry_row(options, 3, "Contrast", self.map_contrast_var, "vmin,vmax (optional)")
        self._build_entry_row(options, 4, "MCC channels", self.map_mcc_channels_var, "3,4,5 (optional)")
        ttk.Checkbutton(
            options,
            text="Scatter plot instead of interpolated heatmap",
            variable=self.map_scatter_var,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))

        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=4, sticky="w", pady=(12, 0))
        ttk.Button(buttons, text="Plot Map Overview", command=self._plot_map_overview).pack(side="left", padx=(0, 8))

    def _build_stack_tab(self, notebook: ttk.Notebook):
        frame = ttk.Frame(notebook, padding=12)
        frame.columnconfigure(1, weight=1)
        notebook.add(frame, text="Stack Summary")

        self.stack_path_var = tk.StringVar()
        self.stack_channel_roi_var = tk.StringVar(value="80,101")
        self.stack_roll_shift_var = tk.StringVar(value="0")
        self.stack_map_roi_var = tk.StringVar()
        self.stack_contrast_var = tk.StringVar()
        self.stack_mcc_channels_var = tk.StringVar()
        self.stack_detectors_var = tk.StringVar()
        self.stack_xeol_roi_var = tk.StringVar()
        self.stack_scatter_var = tk.BooleanVar(value=False)
        self.stack_export_var = tk.StringVar()

        self._build_file_picker_row(
            frame,
            row=0,
            label="Stack HDF5 file",
            variable=self.stack_path_var,
            browse_command=lambda: self._browse_open_file(self.stack_path_var, [("HDF5 files", "*.h5"), ("All files", "*.*")]),
            load_command=self._load_stack_scan,
        )

        ttk.Label(frame, textvariable=self.stack_info_var, wraplength=900, justify="left").grid(
            row=1, column=0, columnspan=4, sticky="ew", pady=(6, 12)
        )

        options = ttk.LabelFrame(frame, text="Summary and Export Options", padding=12)
        options.grid(row=2, column=0, columnspan=4, sticky="ew")
        options.columnconfigure(1, weight=1)

        self._build_entry_row(options, 0, "PFY channel ROI", self.stack_channel_roi_var, "start,end")
        self._build_entry_row(options, 1, "Map ROI", self.stack_map_roi_var, "x1,x2,y1,y2 (optional)")
        self._build_entry_row(options, 2, "Roll shift", self.stack_roll_shift_var, "0")
        self._build_entry_row(options, 3, "Contrast", self.stack_contrast_var, "vmin,vmax (optional)")
        self._build_entry_row(options, 4, "MCC channels", self.stack_mcc_channels_var, "3,4,5 (optional)")
        self._build_entry_row(options, 5, "SDD detectors", self.stack_detectors_var, "sdd1,sdd2 (blank = all)")
        self._build_entry_row(options, 6, "XEOL ROI", self.stack_xeol_roi_var, "start,end (optional)")
        ttk.Checkbutton(
            options,
            text="Scatter plot instead of interpolated heatmap",
            variable=self.stack_scatter_var,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(6, 0))

        export_frame = ttk.LabelFrame(frame, text="CSV Export", padding=12)
        export_frame.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(12, 0))
        export_frame.columnconfigure(1, weight=1)
        self._build_file_picker_row(
            export_frame,
            row=0,
            label="Output CSV",
            variable=self.stack_export_var,
            browse_command=lambda: self._browse_save_file(self.stack_export_var, [("CSV files", "*.csv"), ("All files", "*.*")]),
            load_command=self._export_stack_csv,
            button_text="Export CSV",
        )

        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=0, columnspan=4, sticky="w", pady=(12, 0))
        ttk.Button(buttons, text="Plot Stack Summary", command=self._plot_stack_summary).pack(side="left", padx=(0, 8))

    def _build_heatmap_tab(self, notebook: ttk.Notebook):
        frame = ttk.Frame(notebook, padding=12)
        frame.columnconfigure(1, weight=1)
        notebook.add(frame, text="Emission Matrix")

        self.heatmap_detector_var = tk.StringVar()
        self.heatmap_map_roi_var = tk.StringVar()
        self.heatmap_cmap_var = tk.StringVar(value="magma")
        self.heatmap_log_var = tk.BooleanVar(value=False)
        self.heatmap_vmin_var = tk.StringVar()
        self.heatmap_vmax_var = tk.StringVar()

        ttk.Label(
            frame,
            text=(
                "Use the emission/excitation matrix to choose the emission-channel ROI that best isolates PFY. "
                "If you integrate almost the full emission range, the result becomes TFY-like."
            ),
            justify="left",
            wraplength=900,
        ).grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 12))

        options = ttk.LabelFrame(frame, text="Matrix Options", padding=12)
        options.grid(row=1, column=0, columnspan=3, sticky="ew")
        options.columnconfigure(1, weight=1)

        ttk.Label(options, text="Detector").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.heatmap_detector_combo = ttk.Combobox(
            options,
            textvariable=self.heatmap_detector_var,
            state="readonly",
            values=[],
        )
        self.heatmap_detector_combo.grid(row=0, column=1, sticky="ew", pady=4)

        self._build_entry_row(options, 1, "Map ROI", self.heatmap_map_roi_var, "x1,x2,y1,y2 (optional)")
        self._build_entry_row(options, 2, "Colormap", self.heatmap_cmap_var, "magma")
        self._build_entry_row(options, 3, "vmin", self.heatmap_vmin_var, "optional")
        self._build_entry_row(options, 4, "vmax", self.heatmap_vmax_var, "optional")
        ttk.Checkbutton(options, text="Apply log10 scale", variable=self.heatmap_log_var).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

        ttk.Button(frame, text="Plot Emission / Excitation Matrix", command=self._plot_heatmap).grid(
            row=2, column=0, sticky="w", pady=(12, 0)
        )

    def _build_pca_tab(self, notebook: ttk.Notebook):
        frame = ttk.Frame(notebook, padding=12)
        frame.columnconfigure(1, weight=1)
        notebook.add(frame, text="PCA / K-Means")

        self.pca_detectors_var = tk.StringVar()
        self.pca_channel_roi_var = tk.StringVar(value="80,101")
        self.pca_clusters_var = tk.StringVar(value="4")
        self.pca_components_var = tk.StringVar(value="4")
        self.pca_normalize_var = tk.BooleanVar(value=True)
        self.pca_roll_shift_var = tk.StringVar(value="0")
        self.pca_plot_detectors_var = tk.StringVar()
        self.pca_outfile_var = tk.StringVar()

        options = ttk.LabelFrame(frame, text="Clustering Options", padding=12)
        options.grid(row=0, column=0, columnspan=4, sticky="ew")
        options.columnconfigure(1, weight=1)

        self._build_entry_row(options, 0, "Analysis detectors", self.pca_detectors_var, "sdd1,sdd2 (blank = all)")
        self._build_entry_row(options, 1, "PFY channel ROI", self.pca_channel_roi_var, "start,end")
        self._build_entry_row(options, 2, "K-Means clusters", self.pca_clusters_var, "4")
        self._build_entry_row(options, 3, "PCA components", self.pca_components_var, "4")
        self._build_entry_row(options, 4, "Roll shift for plots", self.pca_roll_shift_var, "0")
        self._build_entry_row(options, 5, "Plot detectors", self.pca_plot_detectors_var, "blank = use analysis detectors")
        ttk.Checkbutton(options, text="Normalize before PCA", variable=self.pca_normalize_var).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

        export_frame = ttk.LabelFrame(frame, text="Optional CSV Dump", padding=12)
        export_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(12, 0))
        export_frame.columnconfigure(1, weight=1)
        self._build_file_picker_row(
            export_frame,
            row=0,
            label="CSV base path",
            variable=self.pca_outfile_var,
            browse_command=lambda: self._browse_save_file(self.pca_outfile_var, [("CSV files", "*.csv"), ("All files", "*.*")]),
            load_command=self._run_pca_analysis,
            button_text="Analyze + Plot",
        )

        ttk.Button(frame, text="Analyze and Plot", command=self._run_pca_analysis).grid(
            row=2, column=0, sticky="w", pady=(12, 0)
        )

    def _build_xeol_tab(self, notebook: ttk.Notebook):
        frame = ttk.Frame(notebook, padding=12)
        frame.columnconfigure(1, weight=1)
        notebook.add(frame, text="XEOL")

        self.xeol_path_var = tk.StringVar()

        ttk.Label(
            frame,
            text="Plot XEOL directly from a scan HDF5 file or from an `xeol*.bin` file.",
            justify="left",
            wraplength=900,
        ).grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 12))

        self._build_file_picker_row(
            frame,
            row=1,
            label="XEOL source",
            variable=self.xeol_path_var,
            browse_command=lambda: self._browse_open_file(
                self.xeol_path_var,
                [("HDF5 files", "*.h5"), ("Binary files", "*.bin"), ("All files", "*.*")],
            ),
            load_command=self._plot_xeol,
            button_text="Plot XEOL",
        )

    def _build_file_picker_row(
        self,
        parent,
        row,
        label,
        variable,
        browse_command,
        load_command,
        button_text="Load",
    ):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="Browse", command=browse_command).grid(row=row, column=2, sticky="ew", padx=8, pady=4)
        ttk.Button(parent, text=button_text, command=load_command).grid(row=row, column=3, sticky="ew", pady=4)

    def _build_entry_row(self, parent, row, label, variable, hint):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        if hint:
            ttk.Label(parent, text=hint).grid(row=row, column=2, sticky="w", padx=(8, 0), pady=4)
        return entry

    def _browse_open_file(self, variable: tk.StringVar, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            variable.set(path)

    def _browse_source_directory(self):
        path = filedialog.askdirectory()
        if path:
            self.source_root_var.set(path)

    def _browse_save_file(self, variable: tk.StringVar, filetypes):
        path = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=".csv")
        if path:
            variable.set(path)

    def _scan_source_directory(self):
        root_dir = self.source_root_var.get().strip()
        if not root_dir:
            self._show_error("Unable to scan directory", ValueError("Choose a root directory first."))
            return
        if not os.path.isdir(root_dir):
            self._show_error("Unable to scan directory", ValueError(f"Directory not found: {root_dir}"))
            return

        records = self._discover_sources(root_dir)
        self.source_tree.delete(*self.source_tree.get_children())
        self.source_records = {}

        root_iid = "root"
        self.source_tree.insert("", "end", iid=root_iid, text=os.path.basename(root_dir) or root_dir, values=("Root", root_dir))
        self.source_tree.item(root_iid, open=True)

        for record in records:
            self._insert_source_record(root_iid, root_dir, record)

        if records:
            raw_stacks = sum(1 for record in records if record["kind"] == "Raw stack dir")
            hdf5_files = sum(1 for record in records if record["kind"] == "HDF5 file")
            self.source_summary_var.set(
                f"Indexed {len(records)} sources under {root_dir}. "
                f"Found {raw_stacks} raw stack directories and {hdf5_files} HDF5 files."
            )
            self._set_status(
                f"Scanned {root_dir}. Select a discovered stack or file instead of hunting through subfolders manually."
            )
        else:
            self.source_summary_var.set(f"No supported sources were found under {root_dir}.")
            self._set_status(f"No stack or scan sources were discovered under {root_dir}.")

    def _discover_sources(self, root_dir):
        records = []

        for current_root, dirnames, filenames in os.walk(root_dir):
            dirnames.sort()
            filenames.sort()

            if self._looks_like_raw_stack_dir(current_root):
                records.append(self._summarize_raw_stack_dir(current_root, root_dir))
                dirnames[:] = []
                continue

            for filename in filenames:
                lower_name = filename.lower()
                if lower_name.endswith(HDF5_EXTENSIONS):
                    full_path = os.path.join(current_root, filename)
                    rel_path = os.path.relpath(full_path, root_dir)
                    records.append(
                        {
                            "path": full_path,
                            "relative_path": rel_path,
                            "label": filename,
                            "kind": "HDF5 file",
                            "details": "Use as map or stack input",
                        }
                    )

        records.sort(key=lambda record: record["relative_path"].lower())
        return records

    def _summarize_raw_stack_dir(self, stack_dir, root_dir):
        energy_dirs = 0
        mcc_count = 0
        sdd_count = 0
        xeol_count = 0
        detectors = set()

        for entry in os.scandir(stack_dir):
            if not entry.is_dir():
                continue
            if not self._looks_like_energy_dir(entry.name):
                continue

            energy_dirs += 1
            for child in os.scandir(entry.path):
                if not child.is_file():
                    continue
                lower_name = child.name.lower()
                if lower_name.startswith("mcc") and lower_name.endswith(".csv"):
                    mcc_count += 1
                elif lower_name.startswith("sdd") and (lower_name.endswith(".bin") or lower_name.endswith(".out")):
                    sdd_count += 1
                    match = re.match(r"(sdd\d+)", lower_name)
                    if match:
                        detectors.add(match.group(1))
                elif lower_name.startswith("xeol") and lower_name.endswith(".bin"):
                    xeol_count += 1

        detector_text = ", ".join(sorted(detectors)) if detectors else "no SDD detector files found"
        details = (
            f"{energy_dirs} energy folders, {mcc_count} MCC CSV, {sdd_count} SDD files, "
            f"{xeol_count} XEOL files, detectors: {detector_text}, no HDF5 metadata found"
        )
        return {
            "path": stack_dir,
            "relative_path": os.path.relpath(stack_dir, root_dir),
            "label": os.path.basename(stack_dir),
            "kind": "Raw stack dir",
            "details": details,
        }

    def _insert_source_record(self, root_iid, root_dir, record):
        relative_path = record["relative_path"]
        parts = relative_path.split(os.sep)
        parent_iid = root_iid
        current_rel = []

        for folder_name in parts[:-1]:
            current_rel.append(folder_name)
            folder_rel = os.sep.join(current_rel)
            folder_iid = f"dir::{folder_rel}"
            if not self.source_tree.exists(folder_iid):
                self.source_tree.insert(parent_iid, "end", iid=folder_iid, text=folder_name, values=("Folder", ""))
            parent_iid = folder_iid

        record_iid = f"src::{len(self.source_records)}"
        self.source_records[record_iid] = record
        self.source_tree.insert(
            parent_iid,
            "end",
            iid=record_iid,
            text=parts[-1],
            values=(record["kind"], record["details"]),
        )

    def _on_source_tree_select(self, _event):
        record = self._get_selected_source_record()
        if record is None:
            return
        self.source_summary_var.set(f"{record['kind']}: {record['path']} | {record['details']}")

    def _on_source_tree_double_click(self, _event):
        record = self._get_selected_source_record()
        if record is None:
            return
        if record["kind"] == "Raw stack dir":
            self._use_selected_source_for_stack()
            return

        lower_path = record["path"].lower()
        if "stack" in lower_path:
            self._use_selected_source_for_stack()
        else:
            self._use_selected_source_for_map()

    def _get_selected_source_record(self):
        selection = self.source_tree.selection()
        if not selection:
            return None
        return self.source_records.get(selection[0])

    def _use_selected_source_for_stack(self):
        record = self._get_selected_source_record()
        if record is None:
            self._show_error("No source selected", ValueError("Select a discovered source first."))
            return

        self.stack_path_var.set(record["path"])
        self.stack_scan = None

        if record["kind"] == "Raw stack dir":
            self.stack_info_var.set(
                f"Selected raw stack directory: {record['path']} | {record['details']} | "
                "Current SGMPython plotting still expects a stack HDF5 metadata file."
            )
            self._set_status(
                f"Selected raw stack directory {record['label']}. The explorer can index it even though no HDF5 file is present."
            )
            return

        self._load_stack_scan()

    def _use_selected_source_for_map(self):
        record = self._get_selected_source_record()
        if record is None:
            self._show_error("No source selected", ValueError("Select a discovered source first."))
            return
        if record["kind"] != "HDF5 file":
            self._show_error(
                "Unsupported map source",
                ValueError("Maps currently need an HDF5 file. Raw stack directories are only valid as stack sources."),
            )
            return

        self.map_path_var.set(record["path"])
        self.map_scan = None
        self._load_map_scan()

    def _load_map_scan(self):
        try:
            scan = self._get_map_scan()
            self._set_status(f"Loaded map scan: {scan.scan_name}")
        except Exception as exc:
            self._show_error("Unable to load map scan", exc)

    def _load_stack_scan(self):
        try:
            stack = self._get_stack_scan()
            self._set_status(f"Loaded stack scan: {stack.scan_name}")
        except Exception as exc:
            self._show_error("Unable to load stack scan", exc)

    def _get_map_scan(self):
        path = self.map_path_var.get().strip()
        if not path:
            raise ValueError("Choose a map HDF5 file first.")
        resolved_path = self._resolve_map_source_path(path)
        if self.map_scan is None or os.path.normcase(self.map_scan.file_path) != os.path.normcase(resolved_path):
            self.map_scan = MapScan(resolved_path)
            self.map_info_var.set(repr(self.map_scan))
        return self.map_scan

    def _get_stack_scan(self):
        path = self.stack_path_var.get().strip()
        if not path:
            raise ValueError("Choose a stack HDF5 file or discovered stack directory first.")
        resolved_path = self._resolve_stack_source_path(path)
        if self.stack_scan is None or os.path.normcase(self.stack_scan.file_path) != os.path.normcase(resolved_path):
            self.stack_scan = StackScan(resolved_path)
            self._refresh_stack_info()
        return self.stack_scan

    def _resolve_map_source_path(self, path):
        if os.path.isfile(path):
            return path
        if not os.path.isdir(path):
            raise ValueError(f"File not found: {path}")

        candidates = self._find_hdf5_candidates(path)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError("This directory contains multiple HDF5 files. Pick one from the Source Explorer.")
        raise ValueError("No HDF5 file was found in the selected directory.")

    def _resolve_stack_source_path(self, path):
        if os.path.isfile(path):
            return path
        if not os.path.isdir(path):
            raise ValueError(f"File or directory not found: {path}")

        candidates = self._find_hdf5_candidates(path, prefer_stack=True)
        if candidates:
            return candidates[0]

        if self._looks_like_raw_stack_dir(path):
            raw_summary = self._summarize_raw_stack_dir(path, os.path.dirname(path))
            raise ValueError(
                "This looks like a raw stack directory rather than a packaged stack HDF5 file. "
                f"{raw_summary['details']}."
            )

        raise ValueError("No stack HDF5 file was found in the selected directory.")

    def _find_hdf5_candidates(self, directory, prefer_stack=False):
        candidates = []
        for current_root, _dirnames, filenames in os.walk(directory):
            for filename in filenames:
                if filename.lower().endswith(HDF5_EXTENSIONS):
                    candidates.append(os.path.join(current_root, filename))
        candidates.sort()
        if prefer_stack:
            candidates.sort(key=lambda path: ("stack" not in path.lower(), path.lower()))
        return candidates

    def _looks_like_raw_stack_dir(self, path):
        if not os.path.isdir(path):
            return False
        if os.path.basename(path).lower().endswith("_stack"):
            return True
        try:
            energy_children = 0
            for entry in os.scandir(path):
                if entry.is_dir() and self._looks_like_energy_dir(entry.name):
                    energy_children += 1
                    if energy_children >= 3:
                        return True
        except OSError:
            return False
        return False

    def _looks_like_energy_dir(self, name):
        return bool(re.search(r"_\d+(?:_\d+)?ev$", name.lower()))

    def _refresh_stack_info(self):
        if self.stack_scan is None:
            self.stack_info_var.set("No stack scan loaded.")
            return

        stack = self.stack_scan
        detectors = sorted(stack.sdd_files.keys())
        detector_text = ", ".join(detectors) if detectors else "none"
        energy_text = "no energies"
        if stack.energies.size:
            energy_text = f"{len(stack.energies)} energies from {stack.energies.min():.2f} to {stack.energies.max():.2f} eV"

        self.stack_info_var.set(
            f"{repr(stack)} | detectors: {detector_text} | {energy_text} | XEOL frames: {len(stack.xeol_data)}"
        )
        self.heatmap_detector_combo["values"] = detectors
        if detectors and self.heatmap_detector_var.get() not in detectors:
            self.heatmap_detector_var.set(detectors[0])

    def _plot_map_overview(self):
        try:
            scan = self._get_map_scan()
            channel_roi = self._parse_int_tuple(self.map_channel_roi_var.get(), size=2, field_name="Channel ROI")
            scan.plot_overview(
                channel_roi=channel_roi,
                roll_shift=self._parse_int(self.map_roll_shift_var.get(), "Roll shift"),
                as_scatter_plot=self.map_scatter_var.get(),
                map_roi=self._parse_float_tuple(self.map_map_roi_var.get(), size=4, allow_empty=True, field_name="Map ROI"),
                contrast=self._parse_float_tuple(self.map_contrast_var.get(), size=2, allow_empty=True, field_name="Contrast"),
                mcc_channels=self._parse_int_list(self.map_mcc_channels_var.get(), allow_empty=True),
            )
            self._set_status(self._roi_status_message(channel_roi, "map overview"))
        except Exception as exc:
            self._show_error("Unable to plot map overview", exc)

    def _plot_stack_summary(self):
        try:
            stack = self._get_stack_scan()
            channel_roi = self._parse_int_tuple(self.stack_channel_roi_var.get(), size=2, field_name="PFY channel ROI")
            stack.plot_summary(
                channel_roi=channel_roi,
                map_roi=self._parse_float_tuple(self.stack_map_roi_var.get(), size=4, allow_empty=True, field_name="Map ROI"),
                roll_shift=self._parse_int(self.stack_roll_shift_var.get(), "Roll shift"),
                as_scatter_plot=self.stack_scatter_var.get(),
                contrast=self._parse_float_tuple(self.stack_contrast_var.get(), size=2, allow_empty=True, field_name="Contrast"),
                mcc_channels=self._parse_int_list(self.stack_mcc_channels_var.get(), allow_empty=True),
                sdd_detectors_to_plot=self._parse_str_list(self.stack_detectors_var.get(), allow_empty=True),
                xeol_roi=self._parse_int_tuple(self.stack_xeol_roi_var.get(), size=2, allow_empty=True, field_name="XEOL ROI"),
            )
            self._set_status(self._roi_status_message(channel_roi, "stack summary"))
        except Exception as exc:
            self._show_error("Unable to plot stack summary", exc)

    def _export_stack_csv(self):
        try:
            stack = self._get_stack_scan()
            output_path = self.stack_export_var.get().strip()
            if not output_path:
                raise ValueError("Choose an output CSV path first.")

            channel_roi = self._parse_int_tuple(self.stack_channel_roi_var.get(), size=2, field_name="PFY channel ROI")
            stack.export_csv(
                filename=output_path,
                channel_roi=channel_roi,
                map_roi=self._parse_float_tuple(self.stack_map_roi_var.get(), size=4, allow_empty=True, field_name="Map ROI"),
                mcc_channels=self._parse_int_list(self.stack_mcc_channels_var.get(), allow_empty=True),
                sdd_detectors=self._parse_str_list(self.stack_detectors_var.get(), allow_empty=True),
                xeol_roi=self._parse_int_tuple(self.stack_xeol_roi_var.get(), size=2, allow_empty=True, field_name="XEOL ROI"),
                roll_shift=self._parse_int(self.stack_roll_shift_var.get(), "Roll shift"),
            )
            self._set_status(f"Exported stack summary CSV to {output_path}")
        except Exception as exc:
            self._show_error("Unable to export stack CSV", exc)

    def _plot_heatmap(self):
        try:
            stack = self._get_stack_scan()
            detector_name = self.heatmap_detector_var.get().strip()
            if not detector_name:
                raise ValueError("Load a stack scan and choose a detector first.")

            stack.plot_emission_excitation_matrix(
                detector_name=detector_name,
                map_roi=self._parse_float_tuple(self.heatmap_map_roi_var.get(), size=4, allow_empty=True, field_name="Map ROI"),
                cmap=self.heatmap_cmap_var.get().strip() or "magma",
                log_scale=self.heatmap_log_var.get(),
                vmin=self._parse_optional_float(self.heatmap_vmin_var.get()),
                vmax=self._parse_optional_float(self.heatmap_vmax_var.get()),
            )
            self._set_status(
                f"Plotted emission/excitation matrix for {detector_name}. Use it to choose the PFY emission ROI."
            )
        except Exception as exc:
            self._show_error("Unable to plot emission matrix", exc)

    def _run_pca_analysis(self):
        try:
            stack = self._get_stack_scan()
            detector_names = self._parse_str_list(self.pca_detectors_var.get(), allow_empty=True)
            if not detector_names:
                detector_names = sorted(stack.sdd_files.keys())
            if not detector_names:
                raise ValueError("No SDD detectors are available in this stack.")

            channel_roi = self._parse_int_tuple(self.pca_channel_roi_var.get(), size=2, field_name="PFY channel ROI")
            results = stack.analyze_pca_kmeans(
                detector_names=detector_names,
                channel_roi=channel_roi,
                n_clusters=self._parse_int(self.pca_clusters_var.get(), "K-Means clusters"),
                n_components=self._parse_int(self.pca_components_var.get(), "PCA components"),
                normalize=self.pca_normalize_var.get(),
            )
            if results is None:
                raise RuntimeError("The PCA/K-Means analysis did not return any results.")

            plot_detectors = self._parse_str_list(self.pca_plot_detectors_var.get(), allow_empty=True)
            outfile = self.pca_outfile_var.get().strip() or None
            stack.plot_pca_kmeans(
                results,
                detector_names=plot_detectors if plot_detectors else None,
                roll_shift=self._parse_int(self.pca_roll_shift_var.get(), "Roll shift for plots"),
                outfile=outfile,
            )
            status = self._roi_status_message(channel_roi, "PCA/K-Means analysis")
            if outfile:
                status += f" Results were also written with base path {outfile}."
            self._set_status(status)
        except Exception as exc:
            self._show_error("Unable to run PCA / K-Means analysis", exc)

    def _plot_xeol(self):
        try:
            source = self.xeol_path_var.get().strip()
            if not source:
                raise ValueError("Choose an HDF5 file or an XEOL .bin file first.")
            plot_xeol(source)
            self._set_status(f"Plotted XEOL from {source}")
        except Exception as exc:
            self._show_error("Unable to plot XEOL", exc)

    def _parse_int(self, text, field_name):
        value = (text or "").strip()
        if value == "":
            raise ValueError(f"{field_name} is required.")
        return int(value)

    def _parse_optional_float(self, text):
        value = (text or "").strip()
        return None if value == "" else float(value)

    def _parse_int_tuple(self, text, size, field_name, allow_empty=False):
        values = self._parse_tuple(text, size=size, cast=int, field_name=field_name, allow_empty=allow_empty)
        return None if values is None else tuple(values)

    def _parse_float_tuple(self, text, size, field_name, allow_empty=False):
        values = self._parse_tuple(text, size=size, cast=float, field_name=field_name, allow_empty=allow_empty)
        return None if values is None else list(values)

    def _parse_tuple(self, text, size, cast, field_name, allow_empty=False):
        cleaned = (text or "").replace(";", ",").replace(" ", ",").strip(", ")
        if not cleaned:
            if allow_empty:
                return None
            raise ValueError(f"{field_name} is required.")
        parts = [part for part in cleaned.split(",") if part]
        if len(parts) != size:
            raise ValueError(f"{field_name} must contain exactly {size} values.")
        return [cast(part) for part in parts]

    def _parse_int_list(self, text, allow_empty=False):
        cleaned = (text or "").replace(";", ",").replace(" ", ",").strip(", ")
        if not cleaned:
            return None if allow_empty else []
        return [int(part) for part in cleaned.split(",") if part]

    def _parse_str_list(self, text, allow_empty=False):
        cleaned = (text or "").replace(";", ",").strip(", ")
        if not cleaned:
            return None if allow_empty else []
        return [part.strip() for part in cleaned.split(",") if part.strip()]

    def _roi_status_message(self, channel_roi, action_name):
        if channel_roi[0] <= 0 and channel_roi[1] >= 255:
            return (
                f"Completed {action_name}. The selected emission ROI spans almost the full detector, "
                "so the result is TFY-like; Tom noted PFY over a narrower ROI is usually more informative."
            )
        return (
            f"Completed {action_name}. The selected emission ROI is being treated as PFY, "
            "while TEY remains the MCC ch2 trace when available."
        )

    def _set_status(self, message):
        self.status_var.set(message)

    def _show_error(self, title, exc):
        self.status_var.set(f"{title}: {exc}")
        messagebox.showerror(title, str(exc))


def main():
    root = tk.Tk()
    SGMAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
