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

from .modules import *

HDF5WaveletData = HDF5WaveletData_0_1_0

def mean_downsample(arr, n):
    return np.nanmean(np.reshape(arr, (-1, n)), 1)

def Pipeline(h5_path: str, output_path: str, return_start_time=True) -> str:
    f_obj = apply_reader(path=h5_path, h5interface=HDF5WaveletData)

    spectrogram = (np.abs(f_obj['spectrogram_data'][...].astype(complex))**2).astype(float)
    spectrogram = np.sqrt(np.apply_along_axis(mean_downsample, 1, spectrogram, n=30))

    #
    out_path = os.path.join(output_path, f'{os.path.basename(h5_path)[:-3]}_meanwaveletpower.h5')
    file = HDF5WaveletData(file=out_path, mode="a", create=True, construct=True)

    #
    file_data = file["morlet_kernel_data"]
    file_data.resize(f_obj["morlet_kernel_data"].shape)
    file_data[:, :] = f_obj["morlet_kernel_data"]
    file_data.axes[1]['time_axis'].resize(f_obj["morlet_kernel_data"].axes[1]['time_axis'].shape)
    file_data.axes[1]['time_axis'][:] = f_obj["morlet_kernel_data"].axes[1]['time_axis'][:]
    file_data.axes[0]["kernel_axis"].resize(f_obj["morlet_kernel_data"].axes[0]['kernel_axis'].shape)
    file_data.axes[0]["kernel_axis"][:] = f_obj["morlet_kernel_data"].axes[0]['kernel_axis'][:]
    file.close()

    #
    file = HDF5WaveletData(file=out_path, mode="a", create=True, construct=True)
    file_data = file["spectrogram_data"]
    file_data.resize(spectrogram.shape)
    file_data[:, :, :] = spectrogram
    file_data.axes[1]["time_axis"].append(f_obj["spectrogram_data"].axes[1]["time_axis"][::30]) # used to be [:1]
    file_data.axes[2]["channellabel_axis"].append(f_obj["spectrogram_data"].axes[2]["channellabel_axis"][...])
    file_data.axes[2]["channelcoord_axis"].append(f_obj["spectrogram_data"].axes[2]["channelcoord_axis"][...])
    morlet_refs = file["morlet_kernel_data"]
    file_data.axes[0]["kerneldata_axis"].resize((morlet_refs.shape[0],))
    for i in range(morlet_refs.shape[0]):
        file_data.axes[0]["kerneldata_axis"][i] = (morlet_refs.ref, morlet_refs.regionref[[i], :])

    file.close()

    if return_start_time:
        return out_path, np.array([datetime.fromtimestamp(t/1e9) for t in f_obj['spectrogram_data'].axes[1]['time_axis'][:]])[0]
    else:
        return out_path
