# SGM Analysis

A Python library for analyzing and visualizing data from the SGM (Spherical Grating Monochromator) beamline. This package provides tools for processing map scans, stack scans, and performing advanced data analysis like PCA and K-Means clustering.

## Features

- **Map Scan Analysis**: Process individual map scans (`MapScan`) with support for SDD, MCC, and XEOL data.
- **Stack Scan Analysis**: Analyze series of maps at different energies (`StackScan`).
- **Heatmap Visualization**: Generate detailed heatmaps of SDD XRF spectra vs. incident energy.
- **Advanced Analysis**: Integrated PCA (Principal Component Analysis) and K-Means clustering for spectral data.
- **Interactive Dashboards**: Interactive Plotly-based widgets for Jupyter Notebooks to explore maps and spectra in real-time.
- **XEOL Support**: Dedicated plotting for X-ray Excited Optical Luminescence spectra.
- **Data Export**: Export processed results to CSV for further analysis.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd sgmanalysis
```

Ensure you have the following dependencies installed:
- `numpy`, `h5py`, `matplotlib`, `scikit-learn`, `plotly`, `ipywidgets`

## Usage

### 1. Map Scan Analysis

```python
from sgmanalysis import MapScan

# Load a map scan
scan = MapScan("path/to/your/scan.h5")

# Plot an overview of the map
# channel_roi=(start_bin, end_bin)
scan.plot_overview(channel_roi=(80, 101))
```

### 2. Stack Scan Analysis

```python
from sgmanalysis import StackScan

# Load a stack scan
stack = StackScan("path/to/your/stack.h5")

# Plot a summary of the stack
stack.plot_summary(channel_roi=(80, 101))

# Generate an SDD Heatmap (XRF spectra vs. Energy)
stack.plot_sdd_heatmap(detector_name='sdd1')
```

### 3. PCA and K-Means Clustering

```python
# Perform PCA and K-Means analysis on a StackScan
results = stack.analyze_pca_kmeans(
    detector_names=['sdd1', 'sdd2'], 
    channel_roi=(80, 101), 
    n_clusters=4
)

# Visualize the clustering results
stack.plot_pca_kmeans(results)
```

### 4. Interactive Analysis (Jupyter Notebook)

```python
from sgmanalysis import interactive_map_analysis, interactive_stack_analysis

# Interactive analysis for a single map
interactive_map_analysis(scan, detector_name='sdd1')

# Interactive analysis for a stack scan
interactive_stack_analysis(stack)
```

## Heatmap Functionality

The `StackScan.plot_sdd_heatmap` method is specifically designed to visualize how the SDD XRF spectra evolve with incident energy. This is useful for identifying resonance peaks and understanding the electronic structure of your sample.

```python
stack.plot_sdd_heatmap(
    detector_name='sdd1', 
    log_scale=True, 
    cmap='magma'
)
```

## XEOL Visualization

Use the standalone `plot_xeol` function or the built-in scan methods to visualize XEOL spectra.

```python
from sgmanalysis import plot_xeol

plot_xeol("path/to/your/xeol_data.bin")
```

## Project Structure

- `scans.py`: Core classes `MapScan` and `StackScan`.
- `interactive.py`: Plotly/ipywidgets interactive analysis tools.
- `plotting.py`: General plotting utilities.
- `__init__.py`: Package entry point.
