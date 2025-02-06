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
def Pipeline(npz_paths) -> str:
    # (2) Concatenate numpy arrays
    wavelets_arrs = []
    times_arrs = []

    for npz in tqdm(npz_paths):  # Already ordered.
        arrs = np.load(npz, allow_pickle=True)

        wavelets = arrs['wavelets']
        times = arrs['time']

        assert wavelets.shape == (50, 300, 150)
        assert times.shape == (300,)

        wavelets_arrs.append(wavelets)
        times_arrs.append(times)

    night_wavelet_data = np.hstack(wavelets_arrs)
    night_time_axis = np.hstack(times_arrs)

    assert night_wavelet_data.shape == (50, 300 * 1320, 150), f'night_wavelet_data.shape == {night_wavelet_data.shape}'
    assert night_time_axis.shape == (300*132,), f'night_time_axis.shape == {night_time_axis.shape}'

    return night_time_axis, night_wavelet_data

