import sys
import os
from glob import glob
import presidio_pipelines as prespipe
from multiprocessing import Pool
import pandas as pd
import numpy as np

sys.stdout.flush()
print("start")

TESTING_LIMIT = 5

SERVER_INPUTS = ('~/presidio-pipelines/scripts/pr05_stageone_ieeg.py', '/data_store0/presidio/nihon_kohden/PR05/nkhdf5/edf_to_hdf5/*', '~/presidio-pipelines/out', '/data_store0/presidio/nihon_kohden/PR05/nkhdf5/PR05_edf_catalog.csv')

def map_fn(h5_fn: str, output_dir: str, edf_meta_csv: str, sleep_stages_txt):
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

    # Store needed data
    print(f"Creating time aligned dataframe using {waveletpower_fn}")
    edf_meta = pd.read_csv(edf_meta_csv)
    sleep_stages = pd.read_fwf(sleep_stages_txt, names=["Time", "State", "to_drop"]).drop(["to_drop"], axis="columns")
    waveletpower_arr, sleep_states_arr, times_artificial, times_h5 = prespipe.ieeg.sleep_stage_align.Pipeline(waveletpower_fn, sleep_stages, edf_meta)
    sleep_state_aligned_fn = f"{waveletpower_fn.split('.')[0]}.npz"
    np.savez(sleep_state_aligned_fn, waveletpower_arr=waveletpower_arr, sleep_states_arr=sleep_states_arr, times_artificial=times_artificial, time_h5=times_h5)
    sleep_state_align_success = sleep_state_aligned_fn in glob(os.path.join(output_dir, '*'))
    print(f"Success? ", sleep_state_align_success)

    if not preprocess_success or not wavelettransform_success or not waveletpower_success or not sleep_state_align_success:
        raise Error(f"Something went wrong for file {h5_fn}")

    os.remove(preprocess_fn)
    os.remove(wavelettransform_fn)
    os.remove(waveletpower_fn)

    print(f"Done processing {h5_fn} to produce output {waveletpower_fn}")


if __name__ == '__main__':

    """
    h5_input_dir = SERVER_INPUTS[0] # sys.argv[1]

    output_dir = SERVER_INPUTS[1] # sys.argv[2]

    edf_meta_csv = SERVER_INPUTS[2] # sys.argv[3]
    """

    assert len(sys.argv) == 5, f'Expected 5, got {sys.argv}'

    h5_input_dir = sys.argv[1]

    output_dir = sys.argv[2]

    edf_meta_csv = sys.argv[3]

    sleep_stages_txt = sys.argv[4]

    print(f'[Preprocess] Working in directory {h5_input_dir}')

    print(os.path.join(h5_input_dir, '*'))

    input_dir_files = glob(os.path.join(h5_input_dir, '*'))

    h5_files = [(fn, output_dir, edf_meta_csv, sleep_stages_txt) for fn in input_dir_files if 'preprocess' not in fn]

    print(f'[Preprocess] Working on {len(h5_files)} files: {h5_files}')

    with Pool() as p:
        p.starmap(map_fn, h5_files)

    assert input_dir_files == glob(os.path.join(h5_input_dir, '*')), 'Input directly was unexpectedly changed.'

    print("Success")











