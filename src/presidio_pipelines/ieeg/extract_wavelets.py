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
def Pipeline(h5_path: str) -> str:
    print(h5_path)
    f_obj = apply_reader(path=h5_path, h5interface=HDF5WaveletData)

    spectrogram = f_obj['spectrogram_data'][...].astype(complex)
    assert spectrogram.shape == (50, 300, 150), f'arr.shape is {arr.shape}'

    time_axis = f_obj["spectrogram_data"].axes[1]["time_axis"][...]
    assert time_axis.shape == (300,), f'time_axis.shape == {time_axis.shape}'

    return spectrogram, time_axis
