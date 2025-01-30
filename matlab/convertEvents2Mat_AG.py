import os
import sys
import numpy as np
from scipy.io import savemat
import h5py  # Added for HDF5 support. Updated on 2025/01/12 by AG
import argparse

parser = argparse.ArgumentParser(description="Converts open-ephys events to mat files")
parser.add_argument('-p', '--path', type=str, help='Path to "events" folder of recording session')
args = parser.parse_args()

path = os.path.expanduser(args.path)
if '.' in path:
    if 'Rhythm_FPGA-100.0' not in path:
        raise ValueError("Please provide the full path, not a relative path.")

# Helper functions to construct folder names
constructFolderName = lambda path, s: s if path[-1] == '/' else constructFolderName(path[:-1], path[-1] + s)
findFolderName = lambda path: constructFolderName(path, '') if path != '/' else findFolderName(path[:-1], '')
findPreviousFolderName = lambda path: findFolderName(path[:-1]) if path[-1] == '/' else findPreviousFolderName(path[:-1])

for r, d, f in os.walk(path):
    data = {}
    if np.sum(['.npy' in n for n in f]) == 0:
        continue
    for file in f:
        if '.npy' in file:
            data[file[:-4]] = np.load(os.path.join(r, file))

    # Save using HDF5 to handle large datasets
    output_file = os.path.join(path, f"{findPreviousFolderName(r)}_{findFolderName(r)}.mat")
    with h5py.File(output_file, 'w') as f:
        f.attrs['MATLAB_compatible'] = True
        f.attrs['MATLAB_fields'] = True
        f.attrs['HDF5_version'] = "MATLAB 7.3"

        for key, value in data.items():
            # Ensure compatibility with MATLAB types
            if isinstance(value, np.ndarray):
                if value.dtype == np.float64:
                    value = value.astype(np.float32)
                elif value.dtype == np.int64:
                    value = value.astype(np.int32)
            f.create_dataset(key, data=value)
        f.create_group('#refs#')  # Add MATLAB-readable group for references
    print(f"Saved: {output_file}")