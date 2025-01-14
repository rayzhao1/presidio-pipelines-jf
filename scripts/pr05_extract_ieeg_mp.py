import sys
import os
from glob import glob
import presidio_pipelines as prespipe
from multiprocessing import Pool
import pandas as pd
import numpy as np

def map_fn(h5_fn: str, output_dir:str, edf_meta_csv: str):
    print(f"Extracting iEEG from {h5_fn}...", flush=True)
    ieeg, times = prespipe.ieeg.extract_ieeg.Pipeline(h5_fn, output_dir, edf_meta_csv)

    # Store down numpy arrays storing this h5's iEEG data and time axis
    ieeg_fn = os.path.join(output_dir, f"{os.path.basename(h5_fn)[:-3]}.npz") # can also do .split('.')[0], but risky if CL arg has "."

    np.savez(ieeg_fn, ieeg_arr=ieeg, times_arr=times)
    # np.save(ieeg_fn, ieeg)

    extract_ieeg_success = ieeg_fn in glob(os.path.join(output_dir, '*'))

    if not extract_ieeg_success:
        raise ValueError(f"Error saving numpy data for file {h5_fn}. {ieeg_fn} not found.")

    print(f"Done processing {h5_fn}.")

if __name__ == '__main__':
    print("Starting...", flush=True)

    # Collect command-line arguments
    assert len(sys.argv) == 5, f'Expected 4, got {sys.argv}'
    h5_input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    edf_meta_csv = sys.argv[3]
    night_idx = int(sys.argv[4])

    # Retrieve files
    input_dir_files = glob(os.path.join(h5_input_dir, '*'))
    print(f'Working in directory {h5_input_dir}')

    h5_files = prespipe.ieeg.edf_merge_pr05.get_night_files(edf_meta_csv, night_idx, item_idx=(8,))
    h5_files = prespipe.ieeg.edf_merge_pr05.basename_intersection(input_dir_files, h5_files)

    proc_inputs = [(fn, output_dir, edf_meta_csv) for fn in h5_files]
    with Pool(6) as p:
        p.starmap(map_fn, proc_inputs)

    # Ensure input directory was not mutated.
    assert input_dir_files == glob(os.path.join(h5_input_dir, '*')), 'Input directly was unexpectedly changed.'
    print("Success")











