import sys
import os
from glob import glob
import presidio_pipelines as prespipe
import numpy as np

if __name__ == '__main__':
    print("Starting...", flush=True)

    # Collect command-line arguments
    assert len(sys.argv) == 6, f'Expected 5, got {sys.argv}'
    ieeg_input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    edf_meta_csv = sys.argv[3]
    sleep_stages_dir = sys.argv[4]
    night_idx = int(sys.argv[5])

    out_fn = prespipe.ieeg.collect_epochs.Pipeline(ieeg_input_dir, output_dir, edf_meta_csv, sleep_stages_dir, night_idx)
    assert out_fn in glob(output_dir), 'Failed to collect epochs.'
    print("Success")
