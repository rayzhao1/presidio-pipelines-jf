"""sleep_stage_align.py
"""

import h5py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


import sys
#np.set_printoptions(threshold=sys.maxsize)

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 1000)

from .modules import *

#h5file = 'C:\Users\raymo\ray\ucsf\PR05\nkhdf5\New folder\sub-PR05_ses-stage1_task-continuous_acq-20230614_run-102140_ieeg.h5'

res_path = './sub-PR05_ses-stage1_task-continuous_acq-20230614_run-102140_ieeg_preprocess_wavelettransform_meanwaveletpower'

def base_file_name(processed_fn: str) -> str:
    fn = os.path.basename(processed_fn)
    patient = fn.split('_')[0][4:]
    trim_index = [i for i, c in enumerate(fn) if c == '_'][5]
    base_fn = fn[:trim_index]
    return os.path.join(f'/data_store0/presidio/nihon_kohden/{patient}/nkhdf5/edf_to_hdf5/', f'{base_fn}.h5')

def time_sleep_state_df(h5_path: str, edf_metadata: pd.DataFrame, sleep_stages: pd.DataFrame, split=False) -> pd.DataFrame:
    base_name = base_file_name(h5_path)
    h5_start_date = edf_metadata.loc[edf_metadata['h5_path'] == base_name]['edf_start'].item()
    h5_start_date, h5_start_time = h5_start_date.split(" ")

    #h5_start_dt = datetime.strptime(h5_start_time, '%H:%M:%S.%f') # from edf_catalog

    h5_start_dt_range = [h5_start_dt + dt for dt in [timedelta(i) for i in [-2, -1, 1, 2]]]

    formatted_start = datetime.strftime(h5_start_dt, '%H:%M:%S')
    print("ALERT")
    print(formatted_start)
    #formatted_start = '00:26:21'
    start_index = sleep_stages.index[sleep_stages['Time'] in h5_start_dt_range].item()
    #start_index = sleep_stages.index[sleep_stages['Time'] == formatted_start].item()

    sleep_stage_5min = sleep_stages.iloc[start_index: start_index + 10]
    print(sleep_stage_5min.head(5))

    if split:
        return sleep_stage_5min['State'].to_numpy().astype(int), sleep_stage_5min['Time'].to_numpy()

    return sleep_stage_5min


# this file is from 10:21:40AM on 6/14/2023

def get_channel_freq(h5_path: str):
    file_obj = h5py.File(h5_path, 'r')

    assert list(file_obj.keys()) == ['MorletFamily', 'MorletFamily_kernel_axis', 'MorletFamily_time_axis', 'MorletSpectrogram', 'MorletSpectrogram_channelcoord_axis', 'MorletSpectrogram_channellabel_axis', 'MorletSpectrogram_kerneldata_axis', 'MorletSpectrogram_time_axis']

    assert list(file_obj.attrs.keys()) == ['FileType', 'FileVersion', 'map_namespace', 'map_type', 'pipeline_json', 'subject_id']

    # kernel axis is our frequencies, each wavelet corresponds to a frequency. Center freuqnecies for wavelet.s
    wavelets_freqs = file_obj['MorletFamily_kernel_axis']['CFreq']

    assert wavelets_freqs.shape == (50,), f'`wavelets_freqs.shape` is {wavelets_freqs.shape}'

    channel_labels_array = file_obj['MorletSpectrogram_channellabel_axis'][...][0:150]  # channel labels = 0–149
    assert channel_labels_array.shape == (150, 2), f'`channel_labels_array.shape` is {channel_labels_array.shape}'



def Pipeline(h5_path: str, sleep_stages: pd.DataFrame, edf_metadata: pd.DataFrame) -> pd.DataFrame:


    file_obj = h5py.File(h5_path, 'r')

    assert list(file_obj.keys()) == ['MorletFamily', 'MorletFamily_kernel_axis', 'MorletFamily_time_axis', 'MorletSpectrogram', 'MorletSpectrogram_channelcoord_axis', 'MorletSpectrogram_channellabel_axis', 'MorletSpectrogram_kerneldata_axis', 'MorletSpectrogram_time_axis']

    assert list(file_obj.attrs.keys()) == ['FileType', 'FileVersion', 'map_namespace', 'map_type', 'pipeline_json', 'subject_id']

    morelet_spectrogram = file_obj['MorletSpectrogram']

    # Grab spectrogram
    data_array = np.abs(morelet_spectrogram[...].astype(complex))

    morelet_time_axis = file_obj['MorletSpectrogram_time_axis'][...].astype(float)

    assert morelet_time_axis.shape == (10,), f'morelet_time_axis.shape is {morelet_time_axis.shape}'

    assert data_array.shape == (50, 10, 150), f'data_array shape {data_array.shape} is incorrect.'
    # 50 morelet families, 1 ..., 150 channels

    # kernel axis is our frequencies, each wavelet corresponds to a frequency. Center freuqnecies for wavelet.s
    wavelets_freqs = file_obj['MorletFamily_kernel_axis']['CFreq']

    assert wavelets_freqs.shape == (50,), f'`wavelets_freqs.shape` is {wavelets_freqs.shape}'

    channel_labels_array = file_obj['MorletSpectrogram_channellabel_axis'][...][0:150]  # channel labels = 0–149
    assert channel_labels_array.shape == (150, 2), f'`channel_labels_array.shape` is {channel_labels_array.shape}'

    # Swap spectrogram data to be indexed by channel
    data_array = np.swapaxes(data_array, 0, 2) # indexed by channel
    assert data_array.shape == (150, 10, 50)

    data_array = np.swapaxes(data_array, 1, 2)
    assert data_array.shape == (150, 50, 10)

    # Produce time: state data
    states, times = time_sleep_state_df(h5_path, edf_metadata, sleep_stages, True)
    assert len(channel_labels) == 150
    assert len(wavelets_freqs) == 50
    assert len(times) == 10
    assert len(states) == 10

    states_array = np.tile(states, 7500)

    morelet_time_axis = np.apply_along_axis(lambda x: datetime.fromtimestamp(x*1e-9), 1, morelet_time_axis)

    return data_array, states_array, times, morelet_time_axis
