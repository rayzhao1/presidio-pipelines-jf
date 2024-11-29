import sys
import os
from glob import glob
import presidio_pipelines as prespipe
from multiprocessing import Pool
import pandas as pd
import numpy as np
from collections import Counter

if __name__ == '__main__':
    assert len(sys.argv) == 6, f'Expected 5, got {sys.argv}'

    h5_input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    edf_meta_csv = sys.argv[3]
    edf_dir = sys.argv[4]
    night_idx = int(sys.argv[5])

    print(f'[Preprocess] Working in directory {h5_input_dir}')

    input_dir_files = glob(os.path.join(h5_input_dir, '*'))

    h5_file = next(fn for fn in input_dir_files if 'preprocess_wavelettransform_meanwaveletpower' in fn and 'npz' not in fn)

    all_edfs = glob(os.path.join(edf_dir, '*'))
    nights = prespipe.ieeg.edf_merge_pr05.parse_find(edf_meta_csv, all_edfs, idx=8)
    ordered_npz_paths = []

    for night_num, night in enumerate(nights):
        for interval_num, interval in enumerate(night.intervals):
            # if `contiguous_interval.t0 is None`, then that contiguous_interval never reached a starting point.
            if len(interval) < 1 or not interval.t0:
                continue
            ordered_npz_paths = ["{0}_preprocess_wavelettransform_meanwaveletpower.npz".format(file[1].split('.')[0]) for file in interval.files]
            if night_num == night_idx:
                break

    input_npz_files = [fn for fn in input_dir_files if '.npz' in fn]
    #assert len(input_npz_files) == len(ordered_npz_paths), f'len(input_npz_files) == {len(input_npz_files)} vs. len(ordered_npz_paths) == {len(ordered_npz_paths)}'
    #assert Counter(input_npz_files) == Counter(ordered_npz_paths), 'Did not receive expected npz files.'

    print("Starting single night DataFrame creation...")
    prespipe.ieeg.sleep_stage_agg.Pipeline(output_dir, h5_file, input_npz_files, night_idx)

    assert input_dir_files == glob(os.path.join(h5_input_dir, '*'))
    print("Success")











