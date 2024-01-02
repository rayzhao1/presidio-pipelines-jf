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
    df_edfcat['edf_start'] = pd.to_datetime(df_edfcat['edf_start'])
    df_edfcat['edf_end'] = pd.to_datetime(df_edfcat['edf_end'])

    #
    df_annot = pd.read_csv(path_annot)
    df_annot = df_annot[[True if ('artifact' in ev.lower()) else False for ev in df_annot['event_name']]]
    df_annot['event_timestamp'] = pd.to_datetime(df_annot['event_timestamp'])
    df_annot = df_annot[(df_annot['event_timestamp'] >= spec_time[0]) & (df_annot['event_timestamp'] < spec_time[-1])]
    df_annot = df_annot.loc[df_annot[['event_name', 'event_timestamp']].drop_duplicates().index]

    df_annot = df_annot.sort_values(by='event_timestamp')
    artifact_time = []
    for ii in range(df_annot.shape[0]):
        if 'on' in df_annot.iloc[ii]['event_name'].lower():
            if ii == (df_annot.shape[0]-1):
                artifact_time.append((df_annot.iloc[ii]['event_timestamp'], df_edfcat[(df_edfcat['edf_end'] >= df_annot.iloc[ii]['event_timestamp'])]['edf_end'].min()))
            else:
                if 'off' in df_annot.iloc[ii+1]['event_name'].lower():
                    artifact_time.append((df_annot.iloc[ii]['event_timestamp'], df_annot.iloc[ii+1]['event_timestamp']))
                else:
                    artifact_time.append((df_annot.iloc[ii]['event_timestamp'], df_edfcat[(df_edfcat['edf_end'] >= df_annot.iloc[ii]['event_timestamp'])]['edf_end'].min()))
        if 'off' in df_annot.iloc[ii]['event_name'].lower():
            if ii == 0:
                artifact_time.append(((df_edfcat[(df_edfcat['edf_start'] <= df_annot.iloc[ii]['event_timestamp'])]['edf_start'].max()), df_annot.iloc[ii]['event_timestamp']))
    print(artifact_time)

    #
    spectrogram = (np.abs(f_obj['spectrogram_data'][...].astype(complex))**2).astype(float)
    for atime in artifact_time:
        nan_ix = np.flatnonzero((spec_time >= atime[0]) & (spec_time <= atime[1]))
        if len(nan_ix) > 0:
            spectrogram[:, nan_ix, :] = np.nan
    spectrogram = np.sqrt(np.nanmean(spectrogram, axis=1))[:, None, :]

    #  
    out_path = path.split('.h5')[0] + '_waveletpower.h5'
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
