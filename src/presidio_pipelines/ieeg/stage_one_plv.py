"""stage_one_waveletpower.py
"""

from collections import OrderedDict

from tqdm import tqdm
import numpy as np
import pandas as pd
import zappy as zp
from datetime import datetime
from pyeisen import family, convolve
from presidio_hdf5objects.dataset.files.hdf5waveletdata import HDF5WaveletData_0_1_0
import os
import sys
import h5py

from .modules import *

HDF5WaveletData = HDF5WaveletData_0_1_0

WINDOW_LEN = 30 # seconds

def epoch_plv(arr, n):
    """Compute the phase locking value over an n sample epoch."""
    num_epochs = arr.shape[1]/n
    # Split arr into a list of np arrays (windows), where each window has n samples.
    epochs = np.split(np.swapaxes(arr, 0, 1), num_epochs)
    assert len(epochs) == num_epochs and all([epoch.shape == (n, 50, 150) for epoch in epochs])

    # Compute PLV over each window, and collect them along the n_samples axis.
    return np.stack([zp.sigproc.connectivity.spectral_synchrony(epoch, 'plv') for epoch in epochs], axis=0)

def Pipeline(h5_path, wt_path):
    with h5py.File(h5_path, 'r') as f:
        cfreqs = f['MorletFamily_kernel_axis']['CFreq']
        assert cfreqs.shape == (50,)

    arrs = np.load(wt_path, allow_pickle=True)
    print(arrs.keys(), flush=True)

    spectrogram = arrs['night_wavelet_data']
    night_time_axis = arrs['times_axis']

    #assert spectrogram.shape == (50, 300 * 1320, 150), f'arr.shape is {spectrogram.shape}'
    #assert night_time_axis.shape == (300 * 132,), f'night_time_axis.shape == {night_time_axis.shape}'

    time_axis = night_time_axis[::WINDOW_LEN]
    #assert time_axis.shape == ((300 * 132) // 30,), f'time_axis.shape == {time_axis.shape}'

    plv = epoch_plv(spectrogram, WINDOW_LEN)
    #assert plv.shape == ((300 * 132) // 30, 50, 150, 150), f'plv.shape == {plv.shape}'

    return plv, time_axis, cfreqs

def H5Pipeline(h5_path: str) -> str:
    f_obj = apply_reader(path=h5_path, h5interface=HDF5WaveletData)

    with h5py.File(h5_path, 'r') as f:
        cfreqs = f['MorletFamily_kernel_axis']['CFreq']
        assert cfreqs.shape == (50,)

    spectrogram = f_obj['spectrogram_data'][...].astype(complex)
    assert spectrogram.shape == (50, 300, 150), f'arr.shape is {arr.shape}'

    time_axis = f_obj["spectrogram_data"].axes[1]["time_axis"][::WINDOW_LEN]
    assert time_axis.shape == (10,)

    plv = epoch_plv(spectrogram, WINDOW_LEN)
    assert plv.shape == (10, 50, 150, 150), f'plv.shape == {plv.shape}'

    return plv, time_axis, cfreqs
