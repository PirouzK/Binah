"""
project_manager.py  —  Save / load .otproj project files.

Format: gzip-compressed JSON, extension  .otproj
Schema version 1.

Typical usage (from main_app.py):
    from project_manager import save_project, load_project
    save_project("/path/to/file.otproj", app)
    doc = load_project("/path/to/file.otproj")
    restore_project(doc, app)
"""

import gzip
import json
import os
import numpy as np
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _arr(a) -> list:
    """numpy array → plain Python list for JSON serialisation."""
    if hasattr(a, "tolist"):
        return a.tolist()
    return list(a)


def _get(var, default=None):
    """Safe .get() for tk Var objects; returns *default* if var is None."""
    try:
        return var.get()
    except Exception:
        return default


def _scan_metadata_for_save(meta: Any) -> dict:
    """Keep only lightweight metadata needed for reloadable scan relationships."""
    if not isinstance(meta, dict):
        return {}
    keep = {}
    for key, val in meta.items():
        if key.startswith("_binah_link_") or key in {
            "merged_from_labels", "merge_source", "merge_count", "merge_overlap_ev"
        }:
            if hasattr(val, "tolist"):
                keep[key] = val.tolist()
            else:
                keep[key] = val
    return keep


# ─────────────────────────────────────────────────────────────────────────────
#  Save
# ─────────────────────────────────────────────────────────────────────────────

def save_project(path: str, app) -> None:
    """Collect all application state and write to a gzip-JSON .otproj file."""
    plot = app._plot
    xas  = app._xas_tab

    # ── 1. Loaded ORCA files ─────────────────────────────────────────────────
    orca_files = []
    file_paths = getattr(app._file_listbox, "_paths", [])
    for fp in file_paths:
        orca_files.append({
            "path":        fp,
            "section_idx": app._file_section_idx.get(fp, 0),
        })

    # ── 2. Experimental scans (data embedded — no re-parse needed on load) ──
    exp_scans = []
    for label, scan, var, style in plot._exp_scans:
        exp_scans.append({
            "panel_label":  label,
            "scan_label":   scan.label,
            "source_file":  scan.source_file,
            "energy_ev":    _arr(scan.energy_ev),
            "mu":           _arr(scan.mu),
            "e0":           float(scan.e0),
            "is_normalized": bool(scan.is_normalized),
            "scan_type":    scan.scan_type,
            "metadata":     _scan_metadata_for_save(getattr(scan, "metadata", {})),
            "enabled":      bool(_get(var, True)),
            "style":        dict(style),
        })

    # ── 3. TDDFT overlays ───────────────────────────────────────────────────
    file_data = getattr(app, "_file_data", {})
    overlays  = []
    for label, sp, var, col in plot._overlay_spectra:
        src_file, src_idx = "", 0
        for fp, spectra_list in file_data.items():
            for si, s in enumerate(spectra_list):
                if s is sp:
                    src_file, src_idx = fp, si
                    break
        overlays.append({
            "label":    label,
            "src_file": src_file,
            "src_idx":  src_idx,
            "enabled":  bool(_get(var, True)),
            "colour":   col or "",
        })

    # ── 4. Plot-widget state ─────────────────────────────────────────────────
    from plot_widget import _EXP_STYLE_DEFAULTS, _TDDFT_STYLE_DEFAULTS
    plot_state = {
        # Broadening / scale
        "fwhm":          _get(plot._fwhm, 1.0),
        "broadening":    _get(plot._broadening, "Gaussian"),
        "shift":         _get(plot._delta_e, 0.0),
        "scale":         _get(plot._tddft_scale, 1.0),
        # Toggles
        "normalise":     _get(plot._normalise, False),
        "show_tddft":    _get(plot._show_tddft, True),
        "show_sticks":   _get(plot._show_sticks, True),
        "show_env":      _get(plot._show_env, True),
        "show_trans":    _get(plot._show_trans, False),
        "show_legend":   _get(plot._show_legend, True),
        "show_grid":     _get(plot._show_grid, False),
        "overlay_mode":  _get(plot._overlay_mode, False),
        # Appearance
        "bg_colour":     plot._bg_colour,
        "custom_title":  _get(plot._custom_title, ""),
        # Axis limits (string — "" = auto)
        "xlim_lo":       _get(plot._xlim_lo, ""),
        "xlim_hi":       _get(plot._xlim_hi, ""),
        "ylim_lo":       _get(plot._ylim_lo, ""),
        "ylim_hi":       _get(plot._ylim_hi, ""),
        # Inset
        "inset_active":       plot._inset_active,
        "inset_pos":          list(plot._inset_pos),
        "inset_xlim":         [x for x in plot._inset_xlim],
        "inset_ylim":         [y for y in plot._inset_ylim],
        "inset_show_labels":  _get(plot._inset_show_labels, True),
        # Current TDDFT style (global)
        "tddft_style":          dict(plot._tddft_style),
        # Module-level defaults (for next session)
        "exp_style_defaults":   dict(_EXP_STYLE_DEFAULTS),
        "tddft_style_defaults": dict(_TDDFT_STYLE_DEFAULTS),
    }

    # ── 5. XAS analysis params ───────────────────────────────────────────────
    xas_params = xas.get_params()

    doc: dict = {
        "version":    1,
        "orca_files": orca_files,
        "exp_scans":  exp_scans,
        "overlays":   overlays,
        "plot_state": plot_state,
        "xas_params": xas_params,
    }

    data = json.dumps(doc, indent=2).encode("utf-8")
    with gzip.open(path, "wb") as fh:
        fh.write(data)


# ─────────────────────────────────────────────────────────────────────────────
#  Load + Restore
# ─────────────────────────────────────────────────────────────────────────────

def load_project(path: str) -> dict:
    """Read and decode a .otproj file.  Returns the raw doc dict."""
    with gzip.open(path, "rb") as fh:
        data = fh.read()
    return json.loads(data.decode("utf-8"))


def restore_project(doc: dict, app) -> list:
    """
    Restore application state from a parsed project doc.

    Returns a list of warning strings (empty = all good).
    """
    warnings = []
    plot = app._plot
    xas  = app._xas_tab
    version = doc.get("version", 1)

    # ── 1. Clear existing state ──────────────────────────────────────────────
    plot._exp_scans.clear()
    plot._overlay_spectra.clear()
    if hasattr(app, "_file_data"):
        app._file_data.clear()
    app._file_section_idx.clear()
    app._file_listbox.delete(0, "end")
    app._file_listbox._paths = []
    app._spectra = []
    app._current_file = ""
    app._file_label.config(text="No file loaded", fg="gray")
    app._section_cb["values"] = []
    app._section_cb.set("")

    # ── 2. Reload ORCA files ─────────────────────────────────────────────────
    for entry in doc.get("orca_files", []):
        fp  = entry.get("path", "")
        idx = entry.get("section_idx", 0)
        if not os.path.exists(fp):
            warnings.append(f"ORCA file not found (skipped): {fp}")
            continue
        app._load_file(fp, switch=False)
        app._file_section_idx[fp] = idx

    # Select the last-loaded orca file in the listbox
    paths = getattr(app._file_listbox, "_paths", [])
    if paths:
        app._file_listbox.selection_clear(0, "end")
        app._file_listbox.selection_set(len(paths) - 1)
        app._current_file = paths[-1]
        app._switch_to_file(paths[-1])

    # ── 3. Restore experimental scans ───────────────────────────────────────
    from experimental_parser import ExperimentalScan
    for entry in doc.get("exp_scans", []):
        try:
            scan = ExperimentalScan(
                label=entry["scan_label"],
                source_file=entry.get("source_file", ""),
                energy_ev=np.array(entry["energy_ev"], dtype=float),
                mu=np.array(entry["mu"], dtype=float),
                e0=float(entry.get("e0", 0.0)),
                is_normalized=bool(entry.get("is_normalized", True)),
                scan_type=entry.get("scan_type", "normalized"),
                metadata=dict(entry.get("metadata", {})),
            )
            style   = dict(entry.get("style", {}))
            enabled = bool(entry.get("enabled", True))
            var     = _BoolVar_from(app, enabled)

            plot._exp_scans.append((entry["panel_label"], scan, var, style))
        except Exception as exc:
            warnings.append(f"Exp scan restore failed ({entry.get('panel_label','?')}): {exc}")

    # ── 4. Restore TDDFT overlays ────────────────────────────────────────────
    file_data = getattr(app, "_file_data", {})
    for entry in doc.get("overlays", []):
        fp  = entry.get("src_file", "")
        idx = entry.get("src_idx", 0)
        spectra = file_data.get(fp, [])
        if not spectra or idx >= len(spectra):
            warnings.append(f"Overlay spectrum not found (skipped): {entry.get('label','?')}")
            continue
        sp  = spectra[idx]
        var = _BoolVar_from(app, bool(entry.get("enabled", True)))
        plot._overlay_spectra.append((entry["label"], sp, var, entry.get("colour", "")))

    # ── 5. Restore plot state ────────────────────────────────────────────────
    ps = doc.get("plot_state", {})
    _restore_plot_state(ps, plot)

    # ── 6. Restore XAS params ────────────────────────────────────────────────
    xp = doc.get("xas_params", {})
    if xp:
        xas.set_params(xp)

    # ── 7. Refresh UI ────────────────────────────────────────────────────────
    plot._refresh_panel_content()
    plot._replot()
    xas.refresh_scan_list()

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _BoolVar_from(app, value: bool):
    """Create a tk.BooleanVar with a given initial value."""
    import tkinter as tk
    v = tk.BooleanVar(master=app, value=value)
    return v


def _restore_plot_state(ps: dict, plot) -> None:
    """Push a plot_state dict into the PlotWidget's tk vars."""
    from plot_widget import _EXP_STYLE_DEFAULTS, _TDDFT_STYLE_DEFAULTS

    def _sv(var, val):
        try:
            var.set(val)
        except Exception:
            pass

    # Broadening
    _sv(plot._broadening,  ps.get("broadening", "Gaussian"))
    fwhm = float(ps.get("fwhm", 1.0))
    _sv(plot._fwhm,        fwhm)
    _sv(plot._fwhm_str,    f"{fwhm:.2f}")
    plot._fwhm_slider.set(min(10.0, max(0.01, fwhm)))

    # Shift
    shift = float(ps.get("shift", 0.0))
    _sv(plot._delta_e,     shift)
    _sv(plot._delta_e_str, f"{shift:+.2f}")
    plot._de_slider_var.set(max(-20.0, min(20.0, shift)))

    # Scale
    scale = float(ps.get("scale", 1.0))
    _sv(plot._tddft_scale,      scale)
    _sv(plot._tddft_scale_str,  f"{scale:.3f}")
    _sv(plot._scale_slider_var, max(0.01, min(5.0, scale)))

    # Toggle checkboxes
    _sv(plot._normalise,    ps.get("normalise",    False))
    _sv(plot._show_tddft,   ps.get("show_tddft",   True))
    _sv(plot._show_sticks,  ps.get("show_sticks",  True))
    _sv(plot._show_env,     ps.get("show_env",     True))
    _sv(plot._show_trans,   ps.get("show_trans",   False))
    _sv(plot._show_legend,  ps.get("show_legend",  True))
    _sv(plot._show_grid,    ps.get("show_grid",    False))
    _sv(plot._overlay_mode, ps.get("overlay_mode", False))

    # Appearance
    bg = ps.get("bg_colour", "#ffffff")
    plot._bg_colour = bg
    try:
        plot._bg_btn.config(bg=bg)
    except Exception:
        pass

    _sv(plot._custom_title, ps.get("custom_title", ""))

    # Axis limits
    _sv(plot._xlim_lo, ps.get("xlim_lo", ""))
    _sv(plot._xlim_hi, ps.get("xlim_hi", ""))
    _sv(plot._ylim_lo, ps.get("ylim_lo", ""))
    _sv(plot._ylim_hi, ps.get("ylim_hi", ""))

    # Inset
    plot._inset_active = bool(ps.get("inset_active", False))
    plot._inset_pos    = list(ps.get("inset_pos", [0.54, 0.52, 0.40, 0.34]))
    plot._inset_xlim   = [ps.get("inset_xlim", [None, None])[0],
                          ps.get("inset_xlim", [None, None])[1]]
    plot._inset_ylim   = [ps.get("inset_ylim", [None, None])[0],
                          ps.get("inset_ylim", [None, None])[1]]
    _sv(plot._inset_show_labels, ps.get("inset_show_labels", True))

    # TDDFT style
    if "tddft_style" in ps:
        plot._tddft_style.update(ps["tddft_style"])

    # Module-level style defaults (persist for new scans loaded this session)
    if "exp_style_defaults" in ps:
        _EXP_STYLE_DEFAULTS.update(ps["exp_style_defaults"])
    if "tddft_style_defaults" in ps:
        _TDDFT_STYLE_DEFAULTS.update(ps["tddft_style_defaults"])
