"""stage_one_preprocess.py
"""

from collections import OrderedDict

import numpy as np
import os
import sys
import zappy as zp
from nkhdf5 import hdf5nk
from datetime import datetime, timedelta
import pandas as pd

from .modules import *

HDF5NK = hdf5nk.HDF5NK_0_1_0


#TODO: Get HDF5NK class automatically. no need to guess class version
#TODO: Addopt virtualaxismap
#TODO: Cleanup the pipeline function

def from_unix_epoch(t: int):
    return datetime.fromtimestamp(t * 1e-9)

def base_file_name(processed_fn: str) -> str:
    fn = os.path.basename(processed_fn)
    patient = fn.split('_')[0][4:]
    return os.path.join(f'/data_store0/presidio/nihon_kohden/{patient}/nkhdf5/edf_to_hdf5/', f'{fn}')

def round_seconds(dt):
    new_dt = dt + timedelta(seconds=.5)
    return new_dt.replace(microsecond=0)

def Pipeline(h5_path: str, output_path: str, edf_catalog_path: str) -> str:
    #
    print("Flushing stdout...")
    sys.stdout.flush()

    #
    f_obj = apply_reader(path=h5_path, h5interface=HDF5NK)

    # Data validation
    data = f_obj['data_ieeg']
    time_axis = f_obj['data_ieeg'].axes[f_obj['data_ieeg'].attrs['t_axis']]['time_axis'][:]
    assert data.axes[f_obj['data_ieeg'].attrs['t_axis']]['time_axis'].attrs['sample_rate'] == 2000, 'Input h5 has incorrect sampling rate'
    assert data.shape == (600000, 160), 'Input h5 has incorrect iEEG data dimensions.'
    assert time_axis.shape == (600000,)
    assert time_axis[1] - time_axis[0] == 499968

    edf_meta = pd.read_csv(edf_catalog_path)

    edf_start = edf_meta.loc[edf_meta["h5_path"] == base_file_name(h5_path)]['edf_start']
    edf_end = edf_meta.loc[edf_meta["h5_path"] == base_file_name(h5_path)]['edf_end']
    edf_nsample = edf_meta.loc[edf_meta["h5_path"] == base_file_name(h5_path)]['edf_nsample'].item()

    edf_start = datetime.strptime(edf_start.item(), '%Y-%m-%d %H:%M:%S.%f')
    edf_end = datetime.strptime(edf_end.item(), '%Y-%m-%d %H:%M:%S.%f')

    h5_start = from_unix_epoch(time_axis[0])
    h5_end = round_seconds(from_unix_epoch(time_axis[-1]))

    assert edf_start == h5_start, f'edf_start == {edf_start} and h5_start == {h5_start}'
    assert edf_end == h5_end, f'edf_end == {edf_end} and h5_end == {h5_end}'
    assert edf_nsample == data.shape[0], f'edf_nsample == {edf_nsample} and data.shape[0] == {data.shape[0]}'

    #
    vsignal, vchangrp = apply_reference(data, montage='bipolar')

    #

    vsignal_ds, Q = zp.sigproc.filters.decimate(vsignal, f_obj['data_ieeg'].axes[f_obj['data_ieeg'].attrs['t_axis']]['time_axis'].attrs['sample_rate'], 1024.0)
    time_axis_ds = f_obj['data_ieeg'].axes[f_obj['data_ieeg'].attrs['t_axis']]['time_axis'][...][::Q]
    new_sample_rate = f_obj['data_ieeg'].axes[f_obj['data_ieeg'].attrs['t_axis']]['time_axis'].attrs['sample_rate'] / Q
    #

    vsignal_ds = zp.sigproc.filters.high_pass_filter(vsignal_ds, new_sample_rate, corner_freq=0.5, stop_tol=10)

    #
    vsignal_ds = zp.sigproc.filters.notch_line(vsignal_ds, new_sample_rate)

    # Data validation 2
    assert Q == 2, 'Unexpected decimation factor'
    assert new_sample_rate == 1000, 'Unexpected value for new sampling rate'
    assert vsignal_ds.shape == (300000, 150), 'Unexpected dimensions for iEEG data dimensions after preprocessing'

    #
    out_path = os.path.join(output_path, f'{os.path.basename(h5_path)[:-3]}_preprocess.h5')
    data_dict = {"data": vsignal_ds, "time_axis_data": time_axis_ds, "sample_rate": new_sample_rate, "low_pass_filter": 500.0, "high_pass_filter": 0.5, "vchangrp": vchangrp}

    apply_writer(out_path, f_obj, data_dict)

    return out_path
