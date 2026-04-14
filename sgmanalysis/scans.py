import h5py
import numpy as np
import os
import re
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

class MapScan:
    """
    Represents a single map scan from the SGM beamline.
    
    This class loads and holds the data from a map scan, including metadata,
    coordinates, and paths to raw data files. It also provides methods
    for plotting the data.
    """
    def __init__(self, file_path):
        """
        Initializes a MapScan object by reading the data from an HDF5 file.

        Args:
            file_path (str): The path to the HDF5 file from a map scan.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")

        self.file_path = file_path
        self.directory = os.path.dirname(file_path)

        # --- Initialize attributes ---
        self.x = np.array([])
        self.y = np.array([])
        self.scan_name = "N/A"
        self.project = "N/A"
        self.energy = -1.0
        self.mcc_file = None
        self._mcc_data = None
        self.mcc_channel_names = []
        self.sdd_files = {}
        self.xeol_file = None
        self._xeol_data = None

        self._load_data()

    def _load_data(self):
        """Internal method to load all data from the HDF5 file and associated files."""
        with h5py.File(self.file_path, 'r') as f:
            # --- Extract Metadata ---
            if 'scan_metadata' in f:
                metadata_attrs = f['scan_metadata'].attrs
                self.scan_name = metadata_attrs.get('scan_name', 'N/A')
                self.project = metadata_attrs.get('project', 'N/A')
                
                energy = metadata_attrs.get('energy', None)
                if energy is None and 'initial_motor_positions' in f and 'all_beamline_motors_snapshot' in f['initial_motor_positions']:
                    motors_snapshot_attrs = f['initial_motor_positions/all_beamline_motors_snapshot'].attrs
                    energy = motors_snapshot_attrs.get('energy', -1.0)
                
                self.energy = float(energy) if energy is not None else -1.0
            else:
                print("Warning: /scan_metadata group not found in HDF5 file.", file=sys.stderr)

            # --- Extract Coordinates ---
            if 'hexapod_waves/x' in f and 'hexapod_waves/y' in f:
                self.x = f['hexapod_waves/x'][:]
                self.y = f['hexapod_waves/y'][:]
            else:
                print("Warning: Coordinate data (hexapod_waves/x or y) not found.", file=sys.stderr)

        # --- Find and prepare Raw Data Files ---
        self.mcc_file = self._find_first_file('mcc*.csv')
        self.xeol_file = self._find_first_file('xeol*.bin')

        sdd_out_files = glob.glob(os.path.join(self.directory, 'sdd*.out'))
        sdd_bin_files = glob.glob(os.path.join(self.directory, 'sdd*_*.bin'))
        for sdd_file_path in sdd_out_files + sdd_bin_files:
            match = re.match(r'(sdd\d+)', os.path.basename(sdd_file_path))
            if match:
                detector_name = match.group(1)
                self.sdd_files[detector_name] = sdd_file_path
    
    def _find_first_file(self, pattern):
        """Finds the first file in the directory matching a pattern."""
        files = glob.glob(os.path.join(self.directory, pattern))
        return files[0] if files else None

    @property
    def mcc_data(self):
        """Lazy-loads MCC data from the CSV file."""
        if self._mcc_data is None and self.mcc_file:
            with open(self.mcc_file, 'r') as mcc_f:
                header = mcc_f.readline().strip()
                if header.startswith('#'):
                    self.mcc_channel_names = [name.strip() for name in header[1:].split(',')]
            self._mcc_data = np.genfromtxt(self.mcc_file, delimiter=',', skip_header=1)
        return self._mcc_data

    @property
    def xeol_data(self):
        """Lazy-loads XEOL data from the binary file."""
        if self._xeol_data is None and self.xeol_file:
            self._xeol_data = np.fromfile(self.xeol_file, dtype=np.uint32)
        return self._xeol_data
        
    def get_sdd_data(self, detector_name):
        """
        Loads the 2D spectrum data for a specific SDD detector.

        Args:
            detector_name (str): The name of the detector (e.g., 'sdd1').

        Returns:
            np.ndarray: A 2D numpy array where each row is a spectrum. 
                        Returns None if the detector or file is not found.
        """
        sdd_filepath = self.sdd_files.get(detector_name)
        if not sdd_filepath or not os.path.exists(sdd_filepath):
            print(f"Error: Data file for detector {detector_name} not found.", file=sys.stderr)
            return None

        try:
            pixels_per_spectrum = 256
            data_1d = np.fromfile(sdd_filepath, dtype=np.uint32)
            
            if data_1d.size == 0:
                print(f"Warning: File for {detector_name} is empty.", file=sys.stderr)
                return None

            num_spectra = len(data_1d) // pixels_per_spectrum
            
            if num_spectra != self.x.size:
                print(f"Warning: Mismatch in number of spectra ({num_spectra}) and scan points ({self.x.size}) for {detector_name}. Truncating.", file=sys.stderr)
                num_spectra = min(num_spectra, self.x.size)

            clean_size = num_spectra * pixels_per_spectrum
            return data_1d[:clean_size].reshape((num_spectra, pixels_per_spectrum))

        except Exception as e:
            print(f"An error occurred while reading data for {detector_name}: {e}", file=sys.stderr)
            return None

    def __repr__(self):
        return (f"MapScan(scan_name='{self.scan_name}', project='{self.project}', "
                f"energy={self.energy:.2f} eV, detectors={list(self.sdd_files.keys())})")

    def plot_overview(self, channel_roi, roll_shift=0, as_scatter_plot: bool = False, map_roi=None, contrast=None, mcc_channels=None):
        """
        Loads SDD data for all available detectors from a map scan, and for each one,
        plots a map based on an ROI sum and a total summed spectrum. Vertical bars
        are added to the spectrum plot to highlight the channel ROI.

        Args:
            channel_roi (tuple): A tuple of two integers defining the start and end of the ROI for the map, e.g., (80, 101).
            roll_shift (int): The number of positions to shift the data for roll correction.
            as_scatter_plot (bool): If True, plots as a scatter plot; otherwise, plots as a heatmap.
            map_roi (list, optional): A list of four coordinates [x1, x2, y1, y2] to define a
                                      rectangular ROI for the summed spectrum. Defaults to None.
            contrast (list, optional): A list of two numbers [vmin, vmax] for the plot contrast.
            mcc_channels (list, optional): A list of MCC channel numbers to plot maps for.
        """
        if not self.sdd_files:
            print("Error: No SDD files found for this scan.", file=sys.stderr)
            return

        if self.x.size == 0 or self.y.size == 0:
            print("Error: Coordinate data not found.", file=sys.stderr)
            return
        
        x_min, x_max = self.x.min(), self.x.max()
        y_min, y_max = self.y.min(), self.y.max()

        if map_roi is None:
            map_roi = [x_min, x_max, y_min, y_max]
        else:
            user_x1, user_x2 = sorted(map_roi[0:2])
            user_y1, user_y2 = sorted(map_roi[2:4])
            map_roi = [
                max(user_x1, x_min), min(user_x2, x_max),
                max(user_y1, y_min), min(user_y2, y_max)
            ]

        num_detectors = len(self.sdd_files)
        
        num_mcc_plots = 0
        if self.mcc_data is not None and mcc_channels:
            num_mcc_plots = len(mcc_channels)

        num_xeol_plots = 1 if self.xeol_data is not None else 0

        if num_detectors == 0 and num_mcc_plots == 0 and num_xeol_plots == 0:
            print("No SDD, MCC, or XEOL detectors found to plot.", file=sys.stderr)
            return

        # --- Create Figure and Title ---
        total_rows = num_detectors + num_mcc_plots + num_xeol_plots
        fig, axes = plt.subplots(total_rows, 2, figsize=(12, 5 * total_rows), squeeze=False)
        
        title = f"Scan: {self.scan_name}  |  Project: {self.project}  |  Energy: {self.energy:.2f} eV"
        fig.suptitle(title, fontsize=14)

        # Sort detectors by name (e.g., sdd1, sdd2, ...)
        sorted_detectors = sorted(self.sdd_files.keys())

        for i, detector_name in enumerate(sorted_detectors):
            ax_map = axes[i, 0]
            ax_spec = axes[i, 1]
            
            spectra_2d = self.get_sdd_data(detector_name)
            if spectra_2d is None:
                ax_map.set_title(f"{detector_name} - Data Not Found")
                continue

            current_x, current_y = self.x[:spectra_2d.shape[0]], self.y[:spectra_2d.shape[0]]

            # --- Data for Map Plot (Left) ---
            intensity = np.sum(spectra_2d[:, channel_roi[0]:channel_roi[1]], axis=1)
            if roll_shift != 0:
                intensity = np.roll(intensity, shift=roll_shift)

            # --- Data for Spectrum Plot (Right) ---
            spectrum_title = f"{detector_name} - Total Spectrum"
            if map_roi:
                x1, x2 = sorted(map_roi[0:2])
                y1, y2 = sorted(map_roi[2:4])
                mask = (current_x >= x1) & (current_x <= x2) & (current_y >= y1) & (current_y <= y2)
                
                if np.any(mask):
                    total_spectrum = np.sum(spectra_2d[mask], axis=0)
                    spectrum_title = f"{detector_name} - Spectrum from Map ROI"
                else:
                    total_spectrum = np.sum(spectra_2d, axis=0)
            else:
                total_spectrum = np.sum(spectra_2d, axis=0)
            
            bins = np.arange(spectra_2d.shape[1])

            # --- Plotting ---
            plot_kwargs = {}
            if contrast and len(contrast) == 2:
                plot_kwargs['vmin'] = contrast[0]
                plot_kwargs['vmax'] = contrast[1]

            if as_scatter_plot:
                scatter = ax_map.scatter(current_x, current_y, c=intensity, cmap='viridis', marker='s', edgecolors='none', **plot_kwargs)
                fig.colorbar(scatter, ax=ax_map, label=f"Counts (ROI: {channel_roi[0]}-{channel_roi[1]})")
            else:
                tripcolor = ax_map.tripcolor(current_x, current_y, intensity, shading='gouraud', **plot_kwargs)
                fig.colorbar(tripcolor, ax=ax_map, label=f"Counts (ROI: {channel_roi[0]}-{channel_roi[1]})")
            
            x1, x2 = sorted(map_roi[0:2])
            y1, y2 = sorted(map_roi[2:4])
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor='r', facecolor='none', linestyle='--')
            ax_map.add_patch(rect)

            ax_map.set_title(f"{detector_name} - ROI Map")
            ax_map.set_xlabel("Hexapod X")
            ax_map.set_ylabel("Hexapod Y")
            ax_map.set_aspect('equal', adjustable='box')

            ax_spec.plot(bins, total_spectrum)
            ax_spec.axvline(x=channel_roi[0], color='red', linestyle=':', linewidth=1.5, label=f'Channel ROI: {channel_roi[0]}-{channel_roi[1]}')
            ax_spec.axvline(x=channel_roi[1], color='red', linestyle=':', linewidth=1.5)
            ax_spec.legend(loc='upper right', fontsize='small')
            ax_spec.set_title(spectrum_title)
            ax_spec.set_xlabel("Bin Number")
            ax_spec.set_ylabel("Total Intensity")
            ax_spec.grid(True)

        # --- Plot MCC Data ---
        if num_mcc_plots > 0:
            for i, mcc_channel in enumerate(mcc_channels):
                ax_map = axes[num_detectors + i, 0]
                ax_spec = axes[num_detectors + i, 1]
                ax_spec.axis('off')

                try:
                    channel_index = self.mcc_channel_names.index(f'ch{mcc_channel}')
                except ValueError:
                    ax_map.set_title(f"MCC Channel {mcc_channel} - Not Found")
                    continue

                intensity = self.mcc_data[:, channel_index]
                if roll_shift != 0:
                    intensity = np.roll(intensity, shift=roll_shift)

                if as_scatter_plot:
                    scatter = ax_map.scatter(self.x, self.y, c=intensity, cmap='viridis', marker='s', edgecolors='none')
                    fig.colorbar(scatter, ax=ax_map, label=f"MCC Channel {mcc_channel} Value")
                else:
                    tripcolor = ax_map.tripcolor(self.x, self.y, intensity, shading='gouraud')
                    fig.colorbar(tripcolor, ax=ax_map, label=f"MCC Channel {mcc_channel} Value")

                ax_map.set_title(f"MCC Channel {mcc_channel} - Map")
                ax_map.set_xlabel("Hexapod X")
                ax_map.set_ylabel("Hexapod Y")
                ax_map.set_aspect('equal', adjustable='box')

        # --- Plot XEOL Data ---
        if num_xeol_plots > 0:
            ax_map = axes[num_detectors + num_mcc_plots, 0]
            ax_spec = axes[num_detectors + num_mcc_plots, 1]
            ax_map.axis('off')

            ax_spec.plot(self.xeol_data)
            ax_spec.set_title("XEOL Spectrum")
            ax_spec.set_xlabel("Bin Number")
            ax_spec.set_ylabel("Total Intensity")
            ax_spec.grid(True)

class StackScan:
    """
    Represents a stack scan from the SGM beamline.
    """
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")

        self.file_path = file_path
        self.stack_dir = os.path.dirname(file_path)

        self.energies = np.array([])
        self.x = np.array([])
        self.y = np.array([])
        self.scan_name = "N/A"
        self.project = "N/A"
        self.mcc_files = {}
        self.mcc_data = {}
        self.mcc_channel_names = []
        self.sdd_files = {}
        self.xeol_files = {}
        self.xeol_data = {}
        
        self._load_data()

    def _load_data(self):
        with h5py.File(self.file_path, 'r', swmr=True) as f:
            if 'stack_metadata' in f:
                metadata_attrs = f['stack_metadata'].attrs
                self.scan_name = metadata_attrs.get('scan_name', 'N/A')
                self.project = metadata_attrs.get('project', 'N/A')
            else:
                print("Warning: /stack_metadata group not found in HDF5 file.", file=sys.stderr)
                return

            if 'map_data/energy' in f:
                self.energies = f['map_data/energy'][:]
            else:
                print("Warning: 'map_data/energy' not found in HDF5 file.", file=sys.stderr)
                return

            if 'hexapod_waves/x' in f and 'hexapod_waves/y' in f:
                self.x = f['hexapod_waves/x'][:]
                self.y = f['hexapod_waves/y'][:]
            else:
                print("Warning: Coordinate data (hexapod_waves/x or y) not found.", file=sys.stderr)

            got_mcc_header = False
            for energy in self.energies:
                energy_str = f"{energy:.2f}".replace('.', '_')
                expected_subdir_name = f"{self.scan_name}_{energy_str}eV"
                en_dir_path = os.path.join(self.stack_dir, expected_subdir_name)

                if not os.path.isdir(en_dir_path):
                    continue

                mcc_file_list = glob.glob(os.path.join(en_dir_path, 'mcc*.csv'))
                if mcc_file_list:
                    mcc_file_path = mcc_file_list[0]
                    self.mcc_files[energy] = mcc_file_path
                    if not got_mcc_header:
                        with open(mcc_file_path, 'r') as mcc_f:
                            header = mcc_f.readline().strip()
                            if header.startswith('#'):
                                self.mcc_channel_names = [name.strip() for name in header[1:].split(',')]
                        got_mcc_header = True
                    self.mcc_data[energy] = np.genfromtxt(mcc_file_path, delimiter=',', skip_header=1)

                sdd_files_list = glob.glob(os.path.join(en_dir_path, 'sdd*.out')) + glob.glob(os.path.join(en_dir_path, 'sdd*_*.bin'))
                for sdd_file_path in sdd_files_list:
                    match = re.match(r'(sdd\d+)', os.path.basename(sdd_file_path))
                    if match:
                        detector_name = match.group(1)
                        if detector_name not in self.sdd_files:
                            self.sdd_files[detector_name] = {}
                        self.sdd_files[detector_name][energy] = sdd_file_path

                xeol_files_list = glob.glob(os.path.join(en_dir_path, 'xeol*.bin'))
                if xeol_files_list:
                    xeol_file_path = xeol_files_list[0]
                    self.xeol_files[energy] = xeol_file_path
                    self.xeol_data[energy] = np.fromfile(xeol_file_path, dtype=np.uint32)

    def _get_marker_size(self, y_coords, plot_width_inches=5):
        """Heuristic to determine scatter plot marker size to form a contiguous grid."""
        if len(y_coords) < 2:
            return 10
        try:
            rounded_y = np.round(y_coords, 3) 
            _, y_counts = np.unique(rounded_y, return_counts=True)
            if len(y_counts) > 1:
                width_in_pixels = int(max(set(y_counts), key=list(y_counts).count))
                if width_in_pixels > 0:
                    marker_side_points = (plot_width_inches * 72) / width_in_pixels
                    return marker_side_points ** 2
        except Exception:
            return 10
        return 10

    def analyze_pca_kmeans(self, detector_names, channel_roi, n_clusters=4, n_components=4, normalize=True):
        """
        Performs PCA and K-Means clustering on the pixel spectra, optionally using multiple detectors.

        Args:
            detector_names (str or list): Name of a single detector or a list of detector names.
            channel_roi (tuple): A tuple of two integers defining the ROI of the spectrum.
            n_clusters (int): The number of clusters for K-Means. Defaults to 4.
            n_components (int): The number of principal components for PCA. Defaults to 4.
            normalize (bool): Whether to normalize the spectra using StandardScaler. Defaults to True.

        Returns:
            dict: Analysis results including PCA components, labels, and cluster PFYs.
        """
        if isinstance(detector_names, str):
            detector_names = [detector_names]

        # Filter for valid detectors present in this stack
        valid_detectors = [d for d in detector_names if d in self.sdd_files]
        if not valid_detectors:
            print(f"Error: None of the specified detectors {detector_names} found.", file=sys.stderr)
            return None

        # Identify energies consistent across ALL selected detectors
        good_energies = []
        num_pixels = -1
        for energy in sorted(self.energies):
            is_energy_valid = True
            current_energy_pixels = -1
            
            for det in valid_detectors:
                sdd_filepath = self.sdd_files[det].get(energy)
                if not sdd_filepath or not os.path.exists(sdd_filepath):
                    is_energy_valid = False
                    break
                
                pixels_per_spectrum = 256
                det_pixels = os.path.getsize(sdd_filepath) // (4 * pixels_per_spectrum)
                if current_energy_pixels == -1:
                    current_energy_pixels = det_pixels
                elif det_pixels != current_energy_pixels:
                    is_energy_valid = False
                    break
            
            if is_energy_valid:
                if num_pixels == -1:
                    num_pixels = current_energy_pixels
                if current_energy_pixels == num_pixels:
                    good_energies.append(energy)
        
        if not good_energies or num_pixels <= 0:
            print(f"Error: No consistent data found across detectors {valid_detectors}.", file=sys.stderr)
            return None

        # Build concatenated pixel spectra matrix: (num_pixels, num_detectors * num_energies)
        num_en = len(good_energies)
        pixel_spectra_concatenated = np.zeros((num_pixels, len(valid_detectors) * num_en))
        
        # We also keep a summed version for the final PFY plot (easier to read)
        pixel_spectra_summed = np.zeros((num_pixels, num_en))

        for d_idx, det in enumerate(valid_detectors):
            for e_idx, energy in enumerate(good_energies):
                spectra_2d = self.get_sdd_data(det, energy)
                if spectra_2d is not None:
                    intensity = np.sum(spectra_2d[:, channel_roi[0]:channel_roi[1]], axis=1)
                    pixel_spectra_concatenated[:, d_idx * num_en + e_idx] = intensity
                    pixel_spectra_summed[:, e_idx] += intensity

        if np.sum(np.nan_to_num(pixel_spectra_concatenated)) == 0:
            print(f"Error: Zero total intensity in the selected ROI.", file=sys.stderr)
            return None

        # --- Data Validation & Preprocessing ---
        # 1. Cast to float64 and replace all non-finite values (NaN, Inf) with 0.0
        data = np.nan_to_num(pixel_spectra_concatenated.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Aggressive clipping to prevent overflows in variance calculations
        # No physical SGM data should reasonably exceed 1e9 counts per pixel/channel
        data = np.clip(data, 0, 1e9)

        # 3. Remove constant or near-constant columns (very low variance)
        # We use a more conservative threshold (1e-6) to avoid division by tiny numbers
        variances = np.var(data, axis=0)
        non_constant_cols = variances > 1e-6 
        
        if not np.any(non_constant_cols):
            print("Error: All data columns are constant. Cannot perform PCA.", file=sys.stderr)
            return None
        
        data = data[:, non_constant_cols]

        # 4. Normalize
        # MinMaxScaler maps everything strictly to [0, 1], which is mathematically 'safe'
        data = MinMaxScaler().fit_transform(data)

        if normalize:
            # Center and scale. StandardScaler handles zero-variance columns 
            # gracefully, but our 1e-6 filter above makes it even safer.
            processed_data = StandardScaler().fit_transform(data)
            # Clip the output of StandardScaler to a reasonable range (e.g., +/- 100 sigma)
            # to prevent any stray huge values from causing overflows in matrix multiplication.
            processed_data = np.clip(processed_data, -100, 100)
        else:
            processed_data = data - np.mean(data, axis=0)
        
        # Final safety check: force any remaining NaNs/Infs to zero before PCA/K-Means
        processed_data = np.nan_to_num(processed_data, nan=0.0, posinf=0.0, neginf=0.0)

        # --- PCA ---
        # Ensure n_components is not larger than number of samples or features
        n_comp = min(n_components, processed_data.shape[0], processed_data.shape[1])
        if n_comp < 1:
            print("Error: Not enough data for PCA.", file=sys.stderr)
            return None
            
        pca = PCA(n_components=n_comp)
        pca_result = pca.fit_transform(processed_data)

        # --- K-Means ---
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(pca_result)

        # --- Re-label clusters by size ---
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        sorted_indices = np.argsort(cluster_sizes)[::-1]
        remapping = np.zeros(n_clusters, dtype=int)
        for i, original_index in enumerate(sorted_indices):
            remapping[original_index] = i
        relabeled_labels = remapping[labels]

        # --- Calculate Cluster PFYs (using the summed data for visualization) ---
        cluster_pfys = {}
        for i in range(n_clusters):
            cluster_indices = np.where(relabeled_labels == i)[0]
            if len(cluster_indices) > 0:
                cluster_pfys[i] = np.mean(pixel_spectra_summed[cluster_indices, :], axis=0)

        return {
            'pca': pca,
            'pca_result': pca_result,
            'labels': relabeled_labels,
            'energies': np.array(good_energies),
            'cluster_pfys': cluster_pfys,
            'detectors_used': valid_detectors,
            'channel_roi': channel_roi
        }

    def plot_pca_kmeans(self, results, detector_names=None, roll_shift=0, outfile=None):
        """
        Plots the results of the PCA and K-Means clustering.

        Args:
            results (dict): The results dictionary returned by analyze_pca_kmeans.
            detector_names (str or list, optional): Detectors to use for the PFY plot. 
                                                   If None, uses the detectors from the analysis.
            roll_shift (int): The number of positions to shift the data for roll correction.
            outfile (str, optional): Base filename to save the results as CSV. 
                                     Saves spatial data to <outfile>.csv and 
                                     spectra to <outfile>_spectra.csv.
        """
        if results is None:
            return

        pca_result = results['pca_result']
        labels = results['labels']
        energies = results['energies']
        pca = results['pca']
        channel_roi = results['channel_roi']
        
        # Apply roll shift for spatial visualization
        if roll_shift != 0:
            labels = np.roll(labels, shift=roll_shift)
            pca_result = np.roll(pca_result, shift=roll_shift, axis=0)

        # Determine which detectors to plot PFY for
        if detector_names is not None:
            if isinstance(detector_names, str):
                detector_names = [detector_names]
            
            # Recalculate PFY for the selected detectors based on the existing labels
            n_clusters = len(np.unique(labels))
            num_pixels = len(labels)
            pixel_spectra_summed = np.zeros((num_pixels, len(energies)))
            
            valid_plot_detectors = []
            for det in detector_names:
                if det in self.sdd_files:
                    valid_plot_detectors.append(det)
                    for e_idx, energy in enumerate(energies):
                        spectra_2d = self.get_sdd_data(det, energy)
                        if spectra_2d is not None:
                            pixel_spectra_summed[:, e_idx] += np.sum(spectra_2d[:, channel_roi[0]:channel_roi[1]], axis=1)
            
            plot_pfys = {}
            for i in range(n_clusters):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) > 0:
                    plot_pfys[i] = np.mean(pixel_spectra_summed[cluster_indices, :], axis=0)
            
            plot_det_label = ", ".join(valid_plot_detectors)
        else:
            plot_pfys = results['cluster_pfys']
            plot_det_label = ", ".join(results.get('detectors_used', []))

        n_clusters = len(plot_pfys)

        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        fig.suptitle(f"PCA & K-Means Analysis: {self.scan_name} (Plotting: {plot_det_label})", fontsize=16)

        plot_x = self.x[:pca_result.shape[0]]
        plot_y = self.y[:pca_result.shape[0]]
        marker_s = self._get_marker_size(plot_y)

        # 1. Cluster Map
        ax_clusters = fig.add_subplot(gs[0, 0])
        scatter = ax_clusters.scatter(plot_x, plot_y, c=labels, cmap='tab10', marker='s', s=marker_s)
        ax_clusters.set_title("K-Means Clusters (0=Largest Area)")
        ax_clusters.set_aspect('equal', adjustable='box')
        fig.colorbar(scatter, ax=ax_clusters, label="Cluster ID")

        # 2. PCA Components
        for i in range(min(3, pca.n_components_)):
            ax_pca = fig.add_subplot(gs[0, i + 1] if i < 2 else gs[1, 0])
            scatter_pca = ax_pca.scatter(plot_x, plot_y, c=pca_result[:, i], cmap='viridis', marker='s', s=marker_s)
            ax_pca.set_title(f"PCA Component {i+1}")
            ax_pca.set_aspect('equal', adjustable='box')
            fig.colorbar(scatter_pca, ax=ax_pca)

        # 3. Scree Plot
        ax_scree = fig.add_subplot(gs[1, 1])
        var_ratio = pca.explained_variance_ratio_
        ax_scree.bar(range(1, len(var_ratio) + 1), var_ratio)
        ax_scree.set_title("Explained Variance Ratio")
        ax_scree.set_xlabel("Principal Component")
        ax_scree.set_ylabel("Variance Ratio")
        ax_scree.set_xticks(range(1, len(var_ratio) + 1))

        # 4. Cluster PFY Spectra
        ax_spec = fig.add_subplot(gs[2, :])
        for i in range(n_clusters):
            if i in plot_pfys:
                spectrum = plot_pfys[i]
                s_min, s_max = spectrum.min(), spectrum.max()
                norm_spec = (spectrum - s_min) / (s_max - s_min) if s_max > s_min else spectrum
                ax_spec.plot(energies, norm_spec, 'o-', label=f"Cluster {i}")
        
        ax_spec.set_title(f"Normalized Mean PFY ({plot_det_label}) per Cluster")
        ax_spec.set_xlabel("Energy (eV)")
        ax_spec.set_ylabel("Normalized Intensity")
        ax_spec.legend()
        ax_spec.grid(True)

        if outfile:
            # 1. Spatial Data (Pixels)
            # Make sure extension is .csv
            base_out = outfile if not outfile.lower().endswith('.csv') else outfile[:-4]
            spatial_outfile = f"{base_out}.csv"
            
            spatial_header = ["x", "y", "cluster_id"]
            for i in range(pca_result.shape[1]):
                spatial_header.append(f"pca_{i+1}")
            
            spatial_data = np.column_stack([plot_x, plot_y, labels, pca_result])
            np.savetxt(spatial_outfile, spatial_data, delimiter=',', header=",".join(spatial_header), comments='')
            
            # 2. Spectral Data (Clusters)
            spectra_outfile = f"{base_out}_spectra.csv"
            spectra_header = ["energy"]
            spectra_cols = [energies]
            for i in range(n_clusters):
                if i in plot_pfys:
                    spectra_header.append(f"cluster_{i}")
                    spectra_cols.append(plot_pfys[i])
            
            spectra_data = np.column_stack(spectra_cols)
            np.savetxt(spectra_outfile, spectra_data, delimiter=',', header=",".join(spectra_header), comments='')
            print(f"PCA/K-Means results dumped to: {spatial_outfile} and {spectra_outfile}")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def __repr__(self):
        return (f"StackScan(scan_name='{self.scan_name}', project='{self.project}', "
                f"energies={len(self.energies)}, detectors={list(self.sdd_files.keys())})")

    def get_sdd_data(self, detector_name, energy):
        sdd_filepath = self.sdd_files.get(detector_name, {}).get(energy)
        if not sdd_filepath or not os.path.exists(sdd_filepath):
            return None
        try:
            pixels_per_spectrum = 256
            data_1d = np.fromfile(sdd_filepath, dtype=np.uint32)
            if data_1d.size == 0:
                return None
            num_spectra = len(data_1d) // pixels_per_spectrum
            if num_spectra != self.x.size:
                num_spectra = min(num_spectra, self.x.size)
            clean_size = num_spectra * pixels_per_spectrum
            return data_1d[:clean_size].reshape((num_spectra, pixels_per_spectrum))
        except Exception as e:
            print(f"Error reading SDD data for {detector_name} at {energy} eV: {e}", file=sys.stderr)
            return None

    def get_data(self, channel_roi, map_roi=None, mcc_channels=None, sdd_detectors=None, xeol_roi=None, roll_shift=0):
        """
        Calculates energy-dependent summary data (intensities in ROIs) for the stack.

        Returns:
            dict: A dictionary containing energies and the calculated intensity arrays.
        """
        all_energies = np.array(sorted(self.energies))
        all_detector_names = sorted(self.sdd_files.keys())
        
        detector_names = [d for d in sdd_detectors if d in all_detector_names] if sdd_detectors else all_detector_names
        
        results = {
            'energies': all_energies,
            'sdd': {det: [] for det in detector_names},
            'mcc': {ch: [] for ch in mcc_channels} if mcc_channels else {},
            'i0': [],
            'tey': [],
            'xeol_total': [],
            'xeol_roi': []
        }

        if self.x.size > 0 and self.y.size > 0:
            x_min, x_max = self.x.min(), self.x.max()
            y_min, y_max = self.y.min(), self.y.max()
            if map_roi is None:
                map_roi = [x_min, x_max, y_min, y_max]
            else:
                user_x1, user_x2 = sorted(map_roi[0:2])
                user_y1, user_y2 = sorted(map_roi[2:4])
                map_roi = [
                    max(user_x1, x_min), min(user_x2, x_max),
                    max(user_y1, y_min), min(user_y2, y_max)
                ]

        x1_map, x2_map = sorted(map_roi[0:2])
        y1_map, y2_map = sorted(map_roi[2:4])
        spatial_mask = (self.x >= x1_map) & (self.x <= x2_map) & (self.y >= y1_map) & (self.y <= y2_map)

        for energy in all_energies:
            for det_name in detector_names:
                spectra_2d = self.get_sdd_data(det_name, energy)
                if spectra_2d is not None:
                    if roll_shift != 0:
                        spectra_2d = np.roll(spectra_2d, shift=roll_shift, axis=0)
                    roi_intensity = np.sum(spectra_2d[spatial_mask, channel_roi[0]:channel_roi[1]])
                    results['sdd'][det_name].append(roi_intensity)
                else:
                    results['sdd'][det_name].append(np.nan)
            
            if self.mcc_data.get(energy) is not None:
                mcc_en_data = self.mcc_data[energy]
                if roll_shift != 0:
                    mcc_en_data = np.roll(mcc_en_data, shift=roll_shift, axis=0)

                def get_mcc_val(ch_name):
                    try:
                        idx = self.mcc_channel_names.index(ch_name)
                        return np.mean(mcc_en_data[spatial_mask, idx])
                    except (ValueError, IndexError):
                        return np.nan

                results['i0'].append(get_mcc_val('ch1'))
                results['tey'].append(get_mcc_val('ch2'))
                if mcc_channels:
                    for ch in mcc_channels:
                        results['mcc'][ch].append(get_mcc_val(f'ch{ch}'))
            else:
                results['i0'].append(np.nan)
                results['tey'].append(np.nan)
                if mcc_channels:
                    for ch in mcc_channels:
                        results['mcc'][ch].append(np.nan)

            if self.xeol_data.get(energy) is not None:
                xeol_spec = self.xeol_data[energy]
                results['xeol_total'].append(np.sum(xeol_spec))
                if xeol_roi:
                    results['xeol_roi'].append(np.sum(xeol_spec[xeol_roi[0]:xeol_roi[1]]))
            else:
                results['xeol_total'].append(np.nan)
                if xeol_roi:
                    results['xeol_roi'].append(np.nan)

        return results

    def export_csv(self, filename, channel_roi, map_roi=None, mcc_channels=None, sdd_detectors=None, xeol_roi=None, roll_shift=0):
        """
        Exports energy-dependent summary data to a CSV file.
        """
        data = self.get_data(channel_roi, map_roi, mcc_channels, sdd_detectors, xeol_roi, roll_shift)
        
        header = ["Energy"]
        data_columns = [data['energies']]
        fmt_list = ['%.4f']

        for det in data['sdd']:
            header.append(det)
            data_columns.append(data['sdd'][det])
            fmt_list.append('%d')

        if any(~np.isnan(data['i0'])):
            header.append("I0")
            data_columns.append(data['i0'])
            fmt_list.append('%.6e')

        if any(~np.isnan(data['tey'])):
            header.append("TEY")
            data_columns.append(data['tey'])
            fmt_list.append('%.6e')

        if mcc_channels:
            for ch in mcc_channels:
                if ch in data['mcc'] and any(~np.isnan(data['mcc'][ch])):
                    header.append(f"MCC_ch{ch}")
                    data_columns.append(data['mcc'][ch])
                    fmt_list.append('%.6e')

        if any(~np.isnan(data['xeol_total'])):
            header.append("XEOL_Total")
            data_columns.append(data['xeol_total'])
            fmt_list.append('%d')

        if xeol_roi and any(~np.isnan(data['xeol_roi'])):
            header.append(f"XEOL_ROI_{xeol_roi[0]}_{xeol_roi[1]}")
            data_columns.append(data['xeol_roi'])
            fmt_list.append('%d')

        output_data = np.array(data_columns).T
        np.savetxt(filename, output_data, delimiter=',', header=','.join(header), comments='', fmt=fmt_list)
        print(f"Summary data exported to {filename}")

    def plot_summary(self, channel_roi, map_roi=None, roll_shift=0, as_scatter_plot: bool = False, contrast=None, mcc_channels=None, sdd_detectors_to_plot=None, xeol_roi=None):
        if not self.sdd_files:
            print("Error: No SDD files found.", file=sys.stderr)
            return

        if self.x.size == 0 or self.y.size == 0:
            print("Error: Coordinate data not found.", file=sys.stderr)
            return

        if map_roi is None:
            map_roi = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]
        
        summary = self.get_data(channel_roi, map_roi, mcc_channels, sdd_detectors_to_plot, xeol_roi, roll_shift)
        summary_energies = summary['energies']
        detector_names = list(summary['sdd'].keys())
        
        x1_map, x2_map = sorted(map_roi[0:2])
        y1_map, y2_map = sorted(map_roi[2:4])
        spatial_mask = (self.x >= x1_map) & (self.x <= x2_map) & (self.y >= y1_map) & (self.y <= y2_map)

        # --- Determine Threshold Energy ---
        # Find first detector with valid data
        threshold_energy = summary_energies[len(summary_energies)//2] # Default
        for det_name in detector_names:
            spec = np.array(summary['sdd'][det_name])
            valid_indices = ~np.isnan(spec)
            if np.sum(valid_indices) > 5:
                en_valid = summary_energies[valid_indices]
                spec_valid = spec[valid_indices]
                smoothed_spec = np.convolve(spec_valid, np.ones(3)/3, mode='valid')
                smoothed_en = en_valid[1:-1]
                if len(smoothed_en) > 1:
                    derivative = np.gradient(smoothed_spec, smoothed_en)
                    threshold_energy = smoothed_en[np.argmax(derivative)]
                    break

        averaged_maps = {det: {'before': {'sum': None, 'count': 0}, 'after': {'sum': None, 'count': 0}} for det in detector_names}
        for energy in summary_energies:
            period = 'before' if energy < threshold_energy else 'after'
            for det_name in detector_names:
                spectra_2d = self.get_sdd_data(det_name, energy)
                if spectra_2d is not None:
                    if roll_shift != 0:
                        spectra_2d = np.roll(spectra_2d, shift=roll_shift, axis=0)
                    map_intensity = np.sum(spectra_2d[:, channel_roi[0]:channel_roi[1]], axis=1)
                    if averaged_maps[det_name][period]['sum'] is None:
                        averaged_maps[det_name][period]['sum'] = map_intensity.astype(np.float64)
                    else:
                        if averaged_maps[det_name][period]['sum'].shape == map_intensity.shape:
                            averaged_maps[det_name][period]['sum'] += map_intensity
                    averaged_maps[det_name][period]['count'] += 1
        
        final_maps = {det: {} for det in detector_names}
        for det_name in detector_names:
            for period in ['before', 'after']:
                if averaged_maps[det_name][period]['count'] > 0:
                    final_maps[det_name][period] = averaged_maps[det_name][period]['sum'] / averaged_maps[det_name][period]['count']
                else:
                    final_maps[det_name][period] = None

        num_detectors = len(detector_names)
        num_summary_plots = 1 + (1 if mcc_channels else 0) + (1 if self.xeol_data else 0) + (1 if xeol_roi else 0)
        fig = plt.figure(figsize=(18, 5 * num_detectors + 5 * num_summary_plots))
        gs = gridspec.GridSpec(num_detectors + num_summary_plots, 3, figure=fig, height_ratios=[1] * num_detectors + [2] * num_summary_plots)
        
        fig.suptitle(f"Scan: {self.scan_name} | Project: {self.project}", fontsize=16)

        for i, det_name in enumerate(detector_names):
            ax_map_before = fig.add_subplot(gs[i, 0])
            ax_map_after = fig.add_subplot(gs[i, 1])
            ax_spec = fig.add_subplot(gs[i, 2])
            
            plot_kwargs = {'vmin': contrast[0], 'vmax': contrast[1]} if contrast else {}

            for ax, period in [(ax_map_before, 'before'), (ax_map_after, 'after')]:
                map_data = final_maps[det_name].get(period)
                if map_data is not None:
                    if as_scatter_plot:
                        ax.scatter(self.x, self.y, c=map_data, cmap='viridis', marker='s', edgecolors='none', **plot_kwargs)
                    else:
                        ax.tripcolor(self.x, self.y, map_data, shading='gouraud', **plot_kwargs)
                    ax.set_title(f"{det_name} - Avg {period.capitalize()} {threshold_energy:.2f} eV")
                else:
                    ax.set_title(f"{det_name} - No data {period} threshold")
                ax.set_xlabel("Hexapod X"); ax.set_ylabel("Hexapod Y")
                ax.set_aspect('equal', adjustable='box')

                x1, x2 = sorted(map_roi[0:2])
                y1, y2 = sorted(map_roi[2:4])
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor='r', facecolor='none', linestyle='--')
                ax.add_patch(rect)

            representative_energy = summary_energies[np.argmin(np.abs(summary_energies - threshold_energy))] if summary_energies.size > 0 else None
            if representative_energy is not None:
                spectra_2d = self.get_sdd_data(det_name, representative_energy)
                if spectra_2d is not None:
                    spectrum_from_roi = np.sum(spectra_2d[spatial_mask], axis=0)
                    ax_spec.plot(np.arange(spectra_2d.shape[1]), spectrum_from_roi)
                    ax_spec.axvline(x=channel_roi[0], color='r', linestyle=':'); ax_spec.axvline(x=channel_roi[1], color='r', linestyle=':')
                    ax_spec.set_title(f"{det_name} - Spectrum at {representative_energy:.2f} eV")
                    ax_spec.grid(True)

        summary_plot_index = num_detectors
        if summary_energies.size > 0:
            ax_summary = fig.add_subplot(gs[summary_plot_index, :])
            summary_plot_index += 1
            for det_name in detector_names:
                ax_summary.plot(summary_energies, summary['sdd'][det_name], 'o-', label=det_name)
            ax_summary.axvline(x=threshold_energy, color='purple', linestyle='--', label=f'Threshold: {threshold_energy:.2f} eV')
            ax_summary.set_title("Energy Dependence of Intensity in Map ROI")
            ax_summary.set_xlabel("Energy (eV)"); ax_summary.set_ylabel("Total Intensity in ROIs")
            ax_summary.legend(); ax_summary.grid(True)

        if mcc_channels:
            ax_mcc_summary = fig.add_subplot(gs[summary_plot_index, :])
            summary_plot_index += 1
            for mcc_channel in mcc_channels:
                if mcc_channel in summary['mcc']:
                    ax_mcc_summary.plot(summary_energies, summary['mcc'][mcc_channel], 'o-', label=f'MCC ch{mcc_channel}')
            
            if any(~np.isnan(summary['i0'])):
                ax_mcc_summary.plot(summary_energies, summary['i0'], 's--', label='I0 (ch1)', alpha=0.7)
            if any(~np.isnan(summary['tey'])):
                ax_mcc_summary.plot(summary_energies, summary['tey'], 'd--', label='TEY (ch2)', alpha=0.7)

            ax_mcc_summary.axvline(x=threshold_energy, color='purple', linestyle='--')
            ax_mcc_summary.set_title("MCC Channel Dependence on Energy in Map ROI")
            ax_mcc_summary.set_xlabel("Energy (eV)"); ax_mcc_summary.set_ylabel("Mean Value in ROI")
            ax_mcc_summary.legend(); ax_mcc_summary.grid(True)

        if self.xeol_data:
            xeol_energies = sorted(self.xeol_data.keys())
            if len(xeol_energies) > 0:
                # --- Calculate summed and averaged XEOL spectra ---
                first_spec_shape = self.xeol_data[xeol_energies[0]].shape
                sum_below, sum_above = np.zeros(first_spec_shape), np.zeros(first_spec_shape)
                count_below, count_above = 0, 0
                
                for energy, spectrum in self.xeol_data.items():
                    if energy < threshold_energy:
                        sum_below += spectrum
                        count_below += 1
                    else:
                        sum_above += spectrum
                        count_above += 1
                
                # Avoid division by zero
                avg_below = sum_below / count_below if count_below > 0 else sum_below
                avg_above = sum_above / count_above if count_above > 0 else sum_above

                # --- Create new layout for XEOL plots ---
                ax_summed_xeol = fig.add_subplot(gs[summary_plot_index, 0])
                ax_xeol_vs_energy = fig.add_subplot(gs[summary_plot_index, 1:])
                summary_plot_index += 1

                # --- Plot averaged spectra (left) ---
                ax_summed_xeol.plot(avg_below, label=f'Avg. Below {threshold_energy:.2f} eV')
                ax_summed_xeol.plot(avg_above, label=f'Avg. Above {threshold_energy:.2f} eV')
                ax_summed_xeol.set_title("Averaged XEOL Spectra")
                ax_summed_xeol.set_xlabel("XEOL Bin Number")
                ax_summed_xeol.set_ylabel("Average Intensity")
                ax_summed_xeol.legend()
                ax_summed_xeol.grid(True)

                # --- Plot XEOL vs Energy Heatmap (right) ---
                xeol_stack = np.array([self.xeol_data[en] for en in xeol_energies])
                # We transpose so X = Excitation Energy, Y = Emission Bin
                im = ax_xeol_vs_energy.imshow(
                    xeol_stack.T, 
                    aspect='auto', 
                    extent=[min(xeol_energies), max(xeol_energies), 0, xeol_stack.shape[1]], 
                    origin='lower', 
                    cmap='magma',
                    interpolation='nearest'
                )
                
                if xeol_roi:
                    ax_xeol_vs_energy.axhline(y=xeol_roi[0], color='cyan', linestyle='--', linewidth=1, label='XEOL ROI')
                    ax_xeol_vs_energy.axhline(y=xeol_roi[1], color='cyan', linestyle='--', linewidth=1)
                
                fig.colorbar(im, ax=ax_xeol_vs_energy, label="Intensity (counts)")
                ax_xeol_vs_energy.set_title("XEOL Emission Heatmap vs. Excitation Energy")
                ax_xeol_vs_energy.set_xlabel("Excitation Energy (eV)")
                ax_xeol_vs_energy.set_ylabel("XEOL Emission Bin")

        if xeol_roi and any(~np.isnan(summary['xeol_roi'])):
            ax_xeol_summary = fig.add_subplot(gs[summary_plot_index, :])
            ax_xeol_summary.plot(summary_energies, summary['xeol_roi'], 'o-')
            ax_xeol_summary.set_title(f"XEOL Intensity vs. Energy (ROI: {xeol_roi[0]}-{xeol_roi[1]})")
            ax_xeol_summary.set_xlabel("Energy (eV)")
            ax_xeol_summary.set_ylabel("Total XEOL Intensity in ROI")
            ax_xeol_summary.grid(True)


        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_sdd_heatmap(self, detector_name, map_roi=None, cmap='magma', log_scale=False, vmin=None, vmax=None):
        """
        Plots a heatmap of the SDD XRF spectra as a function of incident energy.
        
        Args:
            detector_name (str): The name of the detector (e.g., 'sdd1').
            map_roi (list, optional): [x_min, x_max, y_min, y_max] to restrict the spatial area.
            cmap (str): The colormap to use for the heatmap.
            log_scale (bool): Whether to plot the intensity on a log10 scale.
            vmin (float, optional): The minimum value for the color scale.
            vmax (float, optional): The maximum value for the color scale.
        """
        if detector_name not in self.sdd_files:
            print(f"Error: Detector {detector_name} not found.", file=sys.stderr)
            return

        energies = sorted(self.energies)
        if not energies:
            print("Error: No energy data found.", file=sys.stderr)
            return

        # Handle map_roi
        if map_roi is not None:
            x1, x2 = sorted(map_roi[0:2])
            y1, y2 = sorted(map_roi[2:4])
            spatial_mask = (self.x >= x1) & (self.x <= x2) & (self.y >= y1) & (self.y <= y2)
        else:
            spatial_mask = np.ones(self.x.shape, dtype=bool)

        heatmap_data = []
        valid_energies = []
        
        for en in energies:
            spectra_2d = self.get_sdd_data(detector_name, en)
            if spectra_2d is not None:
                current_mask = spatial_mask[:spectra_2d.shape[0]]
                if np.any(current_mask):
                    heatmap_data.append(np.sum(spectra_2d[current_mask], axis=0))
                    valid_energies.append(en)
        
        if not heatmap_data:
            print(f"Error: No SDD data found for {detector_name} within the specified ROI.", file=sys.stderr)
            return
            
        heatmap_data = np.array(heatmap_data) # (Energies, Channels)
        plot_data = heatmap_data.T.astype(np.float64)
        
        if log_scale:
            # Use log10(intensity + 1) to handle zero counts gracefully
            plot_data = np.log10(plot_data + 1)
        
        plt.figure(figsize=(10, 6))
        im = plt.imshow(
            plot_data, 
            aspect='auto', 
            extent=[min(valid_energies), max(valid_energies), 0, heatmap_data.shape[1]], 
            origin='lower', 
            cmap=cmap,
            interpolation='nearest',
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(im, label="Log10(Intensity + 1)" if log_scale else "Intensity (counts)")
        plt.title(f"SDD {detector_name} Emission / Excitation Matrix")
        plt.xlabel("Incident Energy (eV)")
        plt.ylabel("SDD Channel")
        plt.show()

    def plot_emission_excitation_matrix(self, detector_name, map_roi=None, cmap='magma', log_scale=False, vmin=None, vmax=None):
        """
        Alias for plot_sdd_heatmap that uses the beamline's preferred terminology.

        This matrix is useful for choosing the emission-channel ROI for PFY while
        keeping TEY separate from the SDD integration.
        """
        self.plot_sdd_heatmap(
            detector_name=detector_name,
            map_roi=map_roi,
            cmap=cmap,
            log_scale=log_scale,
            vmin=vmin,
            vmax=vmax,
        )




