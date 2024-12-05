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

    h5_file = next(fn for fn in input_dir_files if 'preprocess_wavelettransform_meanwaveletpower' in fn and 'npz' not in fn)

    nights = prespipe.ieeg.edf_merge_pr05.parse_find(edf_meta_csv, idx=8)
    ordered_npz_paths = []

    fn_prefix = os.path.dirname(next(fn for fn in input_dir_files if 'preprocess_wavelettransform_meanwaveletpower' in fn and 'npz' not in fn))

    for night_num, night in enumerate(nights):
        for interval_num, interval in enumerate(night.intervals):
            # if `contiguous_interval.t0 is None`, then that contiguous_interval never reached a starting point.
            if len(interval) < 1 or not interval.t0:
                continue
            if night_num == night_idx:
                ordered_npz_paths = [os.path.join(fn_prefix, f"{os.path.basename(file[1])[:-3]}_preprocess_wavelettransform_meanwaveletpower.npz") for file in interval.files]
                break

    input_npz_files = [fn for fn in input_dir_files if '.npz' in fn]
    print(len(input_npz_files), len(ordered_npz_paths))
    assert len(input_npz_files) == len(ordered_npz_paths), f'len(input_npz_files) == {len(input_npz_files)} vs. len(ordered_npz_paths) == {len(ordered_npz_paths)}'
    assert Counter(input_npz_files) == Counter(ordered_npz_paths), 'Did not receive expected npz files.'

    print("Starting single night DataFrame creation...")
    prespipe.ieeg.sleep_stage_agg.Pipeline(output_dir, h5_file, input_npz_files, sleep_stages_dir, night_idx)

    assert input_dir_files == glob(os.path.join(h5_input_dir, '*'))
    print("Success")











