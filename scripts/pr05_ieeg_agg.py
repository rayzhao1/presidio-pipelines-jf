import sys
import os
from glob import glob
import presidio_pipelines as prespipe
from multiprocessing import Pool
import pandas as pd
import numpy as np
from collections import Counter

if __name__ == '__main__':
    print("Starting...", flush=True)

    # Collect command-line arguments
    assert len(sys.argv) == 5, f'Expected 4, got {sys.argv}'
    h5_input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    edf_meta_csv = sys.argv[3]
    night_idx = int(sys.argv[4])

    print(f'Working in directory {h5_input_dir}')

    # Retrieve files
    input_dir_files = glob(os.path.join(h5_input_dir, '*'))
    input_npz_files = [fn for fn in input_dir_files if '.npz' in fn]

    h5_files = prespipe.ieeg.edf_merge_pr05.get_night_files(night_idx, edf_meta_csv, item_idx=(8,))
    ordered_npz_paths = [f'{fn[:-2]}npz' for fn in h5_files]

    assert Counter(input_npz_files) == Counter(ordered_npz_paths), 'Did not receive expected npz files.'

    prespipe.ieeg.ieeg_agg.Pipeline(output_dir, ordered_npz_paths, night_idx)

    # Ensure input directory was not mutated.
    assert input_dir_files == glob(os.path.join(h5_input_dir, '*')), 'Input directly was unexpectedly changed.'
    print("Success")











