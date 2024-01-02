"""pr05_stageone_ieeg_wavelettransform.py
"""

import sys
from multiprocessing import Pool
from glob import glob
import presidio_pipelines as prespipe


if __name__ == '__main__':

    h5_input_dir = sys.argv[1]
    print(h5_input_dir)

    h5_files = glob(h5_input_dir)
    print(h5_files)
    h5_files = [fn for fn in h5_files if (('wavelet' not in fn) & ('preprocess' in fn))]

    pool = Pool(16)
    pool.map(prespipe.ieeg.stage_one_wavelettransform.Pipeline, h5_files)
