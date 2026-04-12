# Binah

A desktop application for visualising and analysing TDDFT X-ray absorption spectra computed with [ORCA](https://www.faccts.de/orca/), with support for experimental XAS data overlay.

---

## Features

- Parse ORCA `.out` files and extract TDDFT XAS spectra
- Gaussian / Lorentzian broadening with adjustable FWHM
- Energy shift and intensity scale controls
- Overlay multiple TDDFT spectra for comparison
- Load experimental XAS data (`.dat`, `.prj` Athena files)
- Inset zoom panel with connector lines
- Full font, colour, axis label, and legend control
- Pop-out resizable graph window with auto-update
- Save figure (PNG/SVG/PDF) and export CSV
- Save / reload full sessions as `.otproj` project files
- Recent projects list (last 10)

---

## Requirements

- Python 3.9 or newer
- See `requirements.txt` for full dependency list

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/binah.git
cd binah

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python binah.py
```

> **Note:** This repository now bundles the `sgmanalysis` source used for CLS SGM data support, so you do not need a separate checkout of `SGMPython`.

---

## Pre-built Executables

Download the latest release for your platform from the [Releases page](../../releases):

| Platform | File |
|----------|------|
| Windows  | `Binah.exe` |
| macOS    | `Binah.app` |
| Linux    | `Binah` |

No Python installation required for the pre-built executables.

---

## Building the Executable Yourself

```bash
pip install pyinstaller
pyinstaller binah.spec
# Output: dist/Binah(.exe)
```

---

## Usage

1. **Load an ORCA file** — File → Open, or drag-and-drop a `.out` file
2. **Select the TDDFT section** — use the dropdown if multiple sections exist
3. **Adjust broadening / shift / scale** using the control panel
4. **Load experimental data** — File → Load Experimental Data (`.dat` or `.prj`)
5. **Add TDDFT overlays** — load additional ORCA files and use the Overlays panel
6. **Save your session** — File → Save Project (`.otproj`)

### Superscript / subscript in labels
Use matplotlib mathtext syntax directly in any text field:
- Superscript: `Ni$^{2+}$`
- Subscript: `K$_{edge}$`
- Greek: `$\Delta$E`, `$\mu$(E)`

---

## File Structure

```
binah.py               — Main application window
plot_widget.py         — Core plotting widget
orca_parser.py         — ORCA .out file parser
experimental_parser.py — Experimental XAS data parser
xas_analysis_tab.py    — XAS pre-edge analysis panel
project_manager.py     — Project save / load (.otproj)
sgm_xas_loader.py      — CLS SGM beamline data loader
ledge_normalizer.py    — L-edge normalisation utilities
sgmanalysis/           — Bundled SGMPython analysis package
```

---

## License

MIT License — see `LICENSE` for details.
