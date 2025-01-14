import sys
import os
from glob import glob
import presidio_pipelines as prespipe
from multiprocessing import Pool
import pandas as pd
import numpy as np
from collections import Counter

if __name__ == '__main__':
    assert len(sys.argv) == 6, f'Expected 6, got {sys.argv}'

    h5_input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    edf_meta_csv = sys.argv[3]
    sleep_stages_dir = sys.argv[4]
    night_idx = int(sys.argv[5])

    print(f'[Preprocess] Working in directory {h5_input_dir}')

    input_dir_files = glob(os.path.join(h5_input_dir, '*'))
    input_npz_files = [fn for fn in input_dir_files if '.npz' in fn]

    h5_file = next(fn for fn in input_dir_files if 'preprocess_wavelettransform_meanwaveletpower' in fn and 'npz' not in fn)

    fn_prefix = os.path.dirname(h5_file)
    h5_files = prespipe.ieeg.edf_merge_pr05.get_night_files(night_idx, edf_meta_csv, item_idx=8)
    ordered_npz_paths = [os.path.join(fn_prefix, f"{os.path.basename(fn)[:-3]}_preprocess_wavelettransform_meanwaveletpower.npz") for fn in h5_files]

    assert Counter(input_npz_files) == Counter(ordered_npz_paths), 'Did not receive expected npz files.'

    print("Starting single night DataFrame creation...")
    prespipe.ieeg.sleep_stage_agg.Pipeline(output_dir, h5_file, ordered_npz_files, sleep_stages_dir, night_idx)

    assert input_dir_files == glob(os.path.join(h5_input_dir, '*'))
    print("Success")











