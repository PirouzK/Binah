import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob

def plot_xeol(file_path):
    """
    Finds and plots the XEOL spectrum from a '.bin' file.

    If the provided path is an HDF5 file, it searches for a '.bin'
    file in the same directory. If the path is a '.bin' file directly,
    it plots that file.

    Args:
        file_path (str): The path to the HDF5 file from a scan or a .bin file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return

    if file_path.endswith('.h5'):
        directory = os.path.dirname(file_path)
        xeol_files = glob.glob(os.path.join(directory, 'xeol*.bin'))
        if not xeol_files:
            print(f"Error: No 'xeol*.bin' file found in {directory}", file=sys.stderr)
            return
        xeol_file_path = xeol_files[0]
    elif file_path.endswith('.bin'):
        xeol_file_path = file_path
    else:
        print(f"Error: Please provide a path to an HDF5 file or a .bin file.", file=sys.stderr)
        return

    print(f"Plotting XEOL file: {xeol_file_path}")

    try:
        xeol_data = np.fromfile(xeol_file_path, dtype=np.uint32)
        if xeol_data.size == 0:
            print("Warning: XEOL file is empty.", file=sys.stderr)
            return
    except Exception as e:
        print(f"Error reading XEOL file: {e}", file=sys.stderr)
        return

    plt.figure(figsize=(10, 6))
    plt.plot(xeol_data)
    plt.title(f"XEOL Spectrum from {os.path.basename(xeol_file_path)}")
    plt.xlabel("Bin Number")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.show()
