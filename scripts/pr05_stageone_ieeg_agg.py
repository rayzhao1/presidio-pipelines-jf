import sys
import os
from glob import glob
import presidio_pipelines as prespipe
from multiprocessing import Pool
import pandas as pd
import numpy as np



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

    edf_dir = sys.argv[4]

    print(f'[Preprocess] Working in directory {h5_input_dir}')

    print(os.path.join(h5_input_dir, '*'))

    input_dir_files = glob(os.path.join(h5_input_dir, '*'))

    h5_file = next(fn for fn in input_dir_files if 'preprocess_wavelettransform_meanwaveletpower' in fn and 'npz' not in fn)

    print("Starting per night DataFrame creation...")
    prespipe.ieeg.sleep_stage_agg.Pipeline(h5_file, output_dir, edf_meta_csv)

    print("Success")











