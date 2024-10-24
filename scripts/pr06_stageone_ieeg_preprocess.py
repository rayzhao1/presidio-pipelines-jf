"""pr05_stageone_ieeg_preprocess.py
"""

import sys
import os
from glob import glob
import presidio_pipelines as prespipe


if __name__ == '__main__':

    h5_input_dir = sys.argv[1]
    #print(h5_input_dir)

    h5_files = glob(h5_input_dir)
    #print(h5_files)
    h5_files = [fn for fn in h5_files if 'preprocess' not in fn]

    for h5_fn in h5_files:
        #print(h5_fn)
        output = prespipe.ieeg.stage_one_preprocess.Pipeline(h5_fn, sys.argv[2])
