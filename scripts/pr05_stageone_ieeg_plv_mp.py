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
    print("delete", delete)

    # Preprocess
    print(f"Preprocessing file {h5_fn}...")
    preprocess_fn = prespipe.ieeg.stage_one_preprocess.Pipeline(h5_fn, output_dir, edf_meta_csv)
    preprocess_success = preprocess_fn in glob(os.path.join(output_dir, '*'))

    if not preprocess_success:
        raise ValueError(f"Error performing preprocess on file {h5_fn}")

    # Wavelet transform
    print(f"Performing wavelet transform on {preprocess_fn}")
    wavelettransform_fn = prespipe.ieeg.stage_one_wavelettransform.Pipeline(preprocess_fn, output_dir)
    wavelettransform_success = wavelettransform_fn in glob(os.path.join(output_dir, '*'))
    os.remove(preprocess_fn)

    if not wavelettransform_success:
        raise ValueError(f"Error performing wavelet transform on file {h5_fn}")

    wavelets, time = prespipe.ieeg.extract_wavelets.Pipeline(wavelettransform_fn)
    fn = f"{wavelettransform_fn[:-3]}.npz" # can also do .split('.')[0], but risky if CL arg has "."
    if delete:
        os.remove(wavelettransform_fn)
    np.savez(fn, wavelets=wavelets, time=time)
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

    h5_files = [tup[0] for tup in prespipe.ieeg.edf_merge_pr05.get_night_files(edf_meta_csv, night_idx, item_idx=(8,))]
    h5_files = prespipe.ieeg.edf_merge_pr05.basename_intersection(input_dir_files, h5_files)

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











