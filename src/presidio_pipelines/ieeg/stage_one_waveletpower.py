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


def average(arr, n):
    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)

def Pipeline(h5_path: str, output_path: str, path_edf_cat: str = None, path_annot: str = None) -> str:
    #
    print("Flusing stdout...")
    sys.stdout.flush()

    #
    f_obj = apply_reader(path=h5_path, h5interface=HDF5WaveletData)
    spec_time = f_obj['spectrogram_data'].axes[1]['time_axis'][:]
    spec_time = np.array([datetime.fromtimestamp(t/1e9) for t in spec_time])
    spec_time = spec_time[:f_obj['spectrogram_data'].shape[1]]

    #
    df_edfcat = pd.read_csv(path_edf_cat)[['edf_start', 'edf_end']]
    df_edfcat['edf_start'] = pd.to_datetime(df_edfcat['edf_start'], format='ISO8601')
    df_edfcat['edf_end'] = pd.to_datetime(df_edfcat['edf_end'], format='ISO8601')


    #
    spectrogram = (np.abs(f_obj['spectrogram_data'][...].astype(complex))**2).astype(float)

    # for agg_func, agg_name in [(np.nanmean, 'mean'), (np.nanmedian, 'median')]:
    for agg_func, agg_name in [(np.nanmean, 'mean')]:
        print("initial dimensions", spectrogram.shape)

        spectrogram1 = np.sqrt(agg_func(spectrogram, axis=1))[:, None, :]
        print("existing processing", spectrogram1.shape)

        # spectrogram = np.sqrt(np.apply_along_axis(mean_downsample, 1, spectrogram, n=100 * 30))
        spectrogram = np.sqrt(np.apply_along_axis(mean_downsample, 1, spectrogram, n=30))
        print("new (target) dimensions", spectrogram.shape)

        # 10ms -> 1s (because output is 1s resolution), have enough granularity to be able to concatenate across two h5s if needed
        # this intermediate step helps us address misalignmnent. Bins might span two h5 files.
        # goal = 50, 1200/30 = 300, 150
        # don't need to concatenate h5's unless end of data spans to another -> source of truth, if time to end is less than 30 seconds.
        # two at a time - "cascade of shifts" doesn't make sense
        # at any given time, concatenate two files for a 10 minute interval.
        # can we eventually concatenate every night into 1 file

        #
        out_path = os.path.join(output_path, f'{os.path.basename(h5_path)[:-3]}_{agg_name}waveletpower.h5')
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
    return out_path
    # return apply_reader(h5_path=out_path, h5interface=HDF5WaveletData)
