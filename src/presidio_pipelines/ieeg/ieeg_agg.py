"""stage_one_preprocess.py
"""

from collections import OrderedDict

import numpy as np
import os
from glob import glob
import sys
import zappy as zp
from nkhdf5 import hdf5nk
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm

from .modules import *

HDF5NK = hdf5nk.HDF5NK_0_1_0
def Pipeline(output_dir, npz_paths, night_idx) -> str:
    # (2) Concatenate numpy arrays
    ieeg_arrs = []
    times_arrs = []

    for npz in tqdm(npz_paths):  # Already ordered.
        arrs = np.load(npz, allow_pickle=True)

        ieeg = arrs['ieeg_arr']
        times = arrs['times_arr']

        assert ieeg.shape == (300000, 160)
        assert times.shape == (300000,)

        ieeg_arrs.append(ieeg)
        times_arrs.append(times)


    night_ieeg_data = np.vstack(ieeg_arrs)
    night_time_axis = np.hstack(times_arrs)

    ieeg_fn = os.path.join(output_dir, f'output-n{night_idx}.npz')
    np.savez(ieeg_fn, ieeg_data=night_ieeg_data, times_axis=night_time_axis)
    assert ieeg_fn in glob(os.path.join(output_dir, '*'))

