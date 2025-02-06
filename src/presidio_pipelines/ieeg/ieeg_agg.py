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

    assert night_ieeg_data.shape == (300000 * 1320, 160), f'night_ieeg_data.shape == {night_ieeg_data.shape}'
    assert night_time_axis.shape == (300000*132,), f'night_time_axis.shape == {night_time_axis.shape}'

    return night_time_axis, night_ieeg_data
