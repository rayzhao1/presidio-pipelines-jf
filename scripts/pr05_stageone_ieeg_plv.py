"""pr05_stageone_ieeg_waveletpower.py
"""

import sys
from multiprocessing import Pool
from glob import glob
import presidio_pipelines as prespipe
import os


def map_fn(h5_fn, output_dir):
    prespipe.ieeg.stage_one_plv.Pipeline(h5_fn, output_dir)

def check(fn):
    bn = os.path.basename(fn)
    return 'power' not in bn and 'plv' not in bn and 'wavelet' in bn and 'preprocess' in bn

if __name__ == '__main__':

    assert len(sys.argv) == 3, f'Expected 2 arguments, got {sys.argv}'
    h5_input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    h5_files = glob(os.path.join(h5_input_dir, '*'))
    h5_files = [fn for fn in h5_files if check(fn)]

    proc_inputs = [(fn, output_dir) for fn in h5_files]

    with Pool(1) as p:
        p.starmap(map_fn, proc_inputs)
