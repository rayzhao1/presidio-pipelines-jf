import sys
import os
from glob import glob
import presidio_pipelines as prespipe
from multiprocessing import Pool
import pandas as pd
import numpy as np

TESTING_LIMIT = 5

SERVER_INPUTS = ('~/presidio-pipelines/scripts/pr05_stageone_ieeg.py', '/data_store0/presidio/nihon_kohden/PR05/nkhdf5/edf_to_hdf5/*', '~/presidio-pipelines/out', '/data_store0/presidio/nihon_kohden/PR05/nkhdf5/PR05_edf_catalog.csv')

def map_fn(h5_fn: str, output_dir: str, edf_meta_csv: str, delete=True):

    # Preprocess
    print(f"Preprocessing file {h5_fn}...")
    preprocess_fn = prespipe.ieeg.stage_one_preprocess.Pipeline(h5_fn, output_dir, edf_meta_csv)
    preprocess_success = preprocess_fn in glob(os.path.join(output_dir, '*'))

    # Wavelet transform
    print(f"Performing wavelet transform on {preprocess_fn}")
    wavelettransform_fn = prespipe.ieeg.stage_one_wavelettransform.Pipeline(preprocess_fn, output_dir)
    wavelettransform_success = wavelettransform_fn in glob(os.path.join(output_dir, '*'))
    if not preprocess_success:
        raise ValueError(f"Error performing preprocess on file {h5_fn}")
    os.remove(preprocess_fn)

    # Wavelet means
    print(f"Computing wavelet means on {wavelettransform_fn}")
    waveletpower_fn, start_time = prespipe.ieeg.stage_one_waveletpower.Pipeline(wavelettransform_fn, output_dir)
    waveletpower_success = waveletpower_fn in glob(os.path.join(output_dir, '*'))
    if not wavelettransform_success:
        raise ValueError(f"Error performing wavelet transform on file {h5_fn}")
    os.remove(wavelettransform_fn)

    # Store needed data
    print(f"Extracting data from {waveletpower_fn}")

    waveletpower_arr, times_h5 = prespipe.ieeg.sleep_stage_align.Pipeline(waveletpower_fn) #sleep_stages, edf_meta)

    # Store down numpy arrays storing this h5's wavelet power, sleep states, and 2 time axes
    sleep_state_aligned_fn = f"{waveletpower_fn[:-3]}.npz" # can also do .split('.')[0], but risky if CL arg has "."
    np.savez(sleep_state_aligned_fn, waveletpower_arr=waveletpower_arr, time_h5=times_h5)
    sleep_state_align_success = sleep_state_aligned_fn in glob(os.path.join(output_dir, '*'))
    print(f"Success? ", sleep_state_align_success)
    if not waveletpower_success:
        raise ValueError(f"Error computing wavelet means on file {h5_fn}")
    if delete:
        os.remove(waveletpower_fn)

    if not sleep_state_align_success:
        raise ValueError(f"Error saving numpy data for file {h5_fn}. {sleep_state_aligned_fn} not found.")

    print(f"Done processing {h5_fn}.")

if __name__ == '__main__':
    print("Starting...", flush=True)

    # Collect command-line arguments
    assert len(sys.argv) == 5, f'Expected 5, got {sys.argv}'
    h5_input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    edf_meta_csv = sys.argv[3]
    night_idx = int(sys.argv[4])

    # Retrieve files
    input_dir_files = glob(os.path.join(h5_input_dir, '*'))
    print(f'Working in directory {h5_input_dir}')

    h5_files = prespipe.ieeg.edf_merge_pr05.get_night_files(night_idx, edf_meta_csv, input_dir_files, item_idx=8)

    # Skip files that have already been processed.
    existing_files = set(glob(os.path.join(output_dir, '*')))
    fn_suffix = '_preprocess_wavelettransform_meanwaveletpower.npz'
    proc_inputs = [(fn, output_dir, edf_meta_csv) for fn in h5_files if os.path.join(output_dir, f'{os.path.basename(fn)[:-3]}{fn_suffix}') not in existing_files]
    with Pool(6) as p:
        p.starmap(map_fn, proc_inputs)

    # An arbitrary temporary h5 that stores information to be used later
    random_file = glob(os.path.join(h5_input_dir, '*'))[0]
    map_fn(random_file, output_dir, edf_meta_csv, delete=False)

    # Ensure input directory was not mutated.
    assert input_dir_files == glob(os.path.join(h5_input_dir, '*')), 'Input directly was unexpectedly changed.'
    print("Success")











