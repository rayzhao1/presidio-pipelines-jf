"""pr05_stageone_ieeg_waveletpower.py
"""

import sys
from multiprocessing import Pool
from glob import glob
import presidio_pipelines as prespipe


def _helper(minput):
    prespipe.ieeg.stage_one_waveletpower.Pipeline(minput[0], minput[1], minput[2])


if __name__ == '__main__':

    h5_input_dir = sys.argv[1]
    print(h5_input_dir)

    h5_files = glob(h5_input_dir)
    print(h5_files)
    h5_files = [fn for fn in h5_files if (('power' not in fn) & ('wavelet' in fn) & ('preprocess' in fn))]

    minput = []
    for h5_file in h5_files:
        minput.append((h5_file, sys.argv[2], sys.argv[3]))

    pool = Pool(16)
    pool.map(_helper, minput)
