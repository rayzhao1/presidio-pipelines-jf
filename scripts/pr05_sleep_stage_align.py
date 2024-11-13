import sys
from glob import glob
import os
import presidio_pipelines as prespipe
import pandas as pd

if __name__ == '__main__':
    print(sys.argv)
    assert len(sys.argv) == 4, 'Incorrect command line arguments'

    h5_input_dir = sys.argv[1]
    edf_meta = pd.read_csv(sys.argv[2])
    sleep_stages = pd.read_fwf(sys.argv[3], names=["Time", "State", "to_drop"]).drop(["to_drop"], axis="columns")

    h5_files = glob(os.path.join(h5_input_dir, '*'))
    print(h5_files)
    # assert all(['meanwaveletpower' in fn for fn in h5_files])
    h5_files = [fn for fn in h5_files if 'meanwaveletpower' in fn]
    print(h5_files)
    for h5_fn in h5_files:
        output = prespipe.ieeg.sleep_stage_align.Pipeline(h5_fn, sleep_stages, edf_meta)
        s = f"{h5_fn.split('.')[0]}.pkl"
        print(s)
        output.to_pickle(s)
