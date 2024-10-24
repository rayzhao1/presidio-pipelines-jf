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

from .modules import *

HDF5WaveletData = HDF5WaveletData_0_1_0

def Pipeline(path: str, path_edf_cat: str = None, path_annot: str = None) -> None:
    #
    f_obj = apply_reader(path=path, h5interface=HDF5WaveletData)
    spec_time = f_obj['spectrogram_data'].axes[1]['time_axis'][:]
    spec_time = np.array([datetime.fromtimestamp(t/1e9) for t in spec_time])
    spec_time = spec_time[:f_obj['spectrogram_data'].shape[1]]

    #
    df_edfcat = pd.read_csv(path_edf_cat)[['edf_start', 'edf_end']]
    df_edfcat['edf_start'] = pd.to_datetime(df_edfcat['edf_start'], format='ISO8601')
    df_edfcat['edf_end'] = pd.to_datetime(df_edfcat['edf_end'], format='ISO8601')


    #
    spectrogram = (np.abs(f_obj['spectrogram_data'][...].astype(complex))**2).astype(float)

    for agg_func, agg_name in [(np.nanmean, 'mean'), (np.nanmedian, 'median')]:
        spectrogram = np.sqrt(agg_func(spectrogram, axis=1))[:, None, :]

        #
        out_path = path.split('.h5')[0] + '_{}waveletpower.h5'.format(agg_name)
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
        file_data.axes[1]["time_axis"].append(f_obj["spectrogram_data"].axes[1]["time_axis"][:1])
        file_data.axes[2]["channellabel_axis"].append(f_obj["spectrogram_data"].axes[2]["channellabel_axis"][...])
        file_data.axes[2]["channelcoord_axis"].append(f_obj["spectrogram_data"].axes[2]["channelcoord_axis"][...])
        morlet_refs = file["morlet_kernel_data"]
        file_data.axes[0]["kerneldata_axis"].resize((morlet_refs.shape[0],))
        for i in range(morlet_refs.shape[0]):
            file_data.axes[0]["kerneldata_axis"][i] = (morlet_refs.ref, morlet_refs.regionref[[i], :])

        file.close()

    return apply_reader(path=out_path, h5interface=HDF5WaveletData)
