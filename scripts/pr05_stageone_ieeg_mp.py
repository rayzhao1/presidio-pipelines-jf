import sys
import os
import pandas as pd
from glob import glob
import presidio_pipelines as prespipe
from multiprocessing import Pool

sys.stdout.flush()
print("start")

TESTING_LIMIT = 5

SERVER_INPUTS = ('~/presidio-pipelines/scripts/pr05_stageone_ieeg.py', '/data_store0/presidio/nihon_kohden/PR05/nkhdf5/edf_to_hdf5/*', '~/presidio-pipelines/out', '/data_store0/presidio/nihon_kohden/PR05/nkhdf5/PR05_edf_catalog.csv')

def map_fn(h5_fn: str, output_dir: str, edf_meta_csv: str):
    # Preprocess
    print(f"Preprocessing file {h5_fn}...")
    preprocess_fn = prespipe.ieeg.stage_one_preprocess.Pipeline(h5_fn, output_dir, edf_meta_csv)
    preprocess_success = preprocess_fn in glob(os.path.join(output_dir, '*'))
    print(f"Success? ", preprocess_success)

    # Wavelet transform
    print(f"Performing wavelet transform on {preprocess_fn}")
    wavelettransform_fn = prespipe.ieeg.stage_one_wavelettransform.Pipeline(preprocess_fn, output_dir)
    wavelettransform_success = wavelettransform_fn in glob(os.path.join(output_dir, '*'))
    print(f"Success? ", wavelettransform_success)

    # Wavelet means
    print(f"Computing wavelet means on {wavelettransform_fn}")
    waveletpower_fn = prespipe.ieeg.stage_one_waveletpower.Pipeline(wavelettransform_fn, output_dir, edf_meta_csv)
    waveletpower_success = waveletpower_fn in glob(os.path.join(output_dir, '*'))
    print(f"Success? ", waveletpower_success)

    if not preprocess_success or not wavelettransform_success or not waveletpower_success:
        raise Error(f"Something went wrong for file {h5_fn}")

    os.remove(preprocess_fn)
    os.remove(wavelettransform_fn)

    print(f"Done processing {h5_fn} to produce output {waveletpower_fn}")

if __name__ == '__main__':

    """
    h5_input_dir = SERVER_INPUTS[0] # sys.argv[1]

    output_dir = SERVER_INPUTS[1] # sys.argv[2]

    edf_meta_csv = SERVER_INPUTS[2] # sys.argv[3]
    """

    assert len(sys.argv) == 4, f'Expected 4, got {sys.argv}'

    h5_input_dir = sys.argv[1]

    output_dir = sys.argv[2]

    edf_meta_csv = sys.argv[3]

    print(f'[Preprocess] Working in directory {h5_input_dir}')

    print(os.path.join(h5_input_dir, '*'))

    input_dir_files = glob(os.path.join(h5_input_dir, '*'))

    h5_files = [(fn, output_dir, edf_meta_csv) for fn in input_dir_files if 'preprocess' not in fn][:2]

    print(f'[Preprocess] Working on {len(h5_files)} files: {h5_files}')

    with Pool(2) as p:
        p.starmap(map_fn, h5_files)

    assert input_dir_files == glob(os.path.join(h5_input_dir, '*')), 'Input directly was unexpectedly changed.'

    print("Success")











