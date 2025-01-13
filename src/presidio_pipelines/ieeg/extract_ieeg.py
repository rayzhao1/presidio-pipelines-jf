"""extract_ieeg.py
"""

from collections import OrderedDict

import numpy as np
import os
import sys
from nkhdf5 import hdf5nk
from datetime import datetime, timedelta
import zappy as zp

from .modules import *

HDF5NK = hdf5nk.HDF5NK_0_1_0
def Pipeline(h5_path: str, out_path, edf_catalog_path: str) -> str:
    f_obj = apply_reader(path=h5_path, h5interface=HDF5NK)

    # Data validation
    data = f_obj['data_ieeg']
    time_axis = f_obj['data_ieeg'].axes[f_obj['data_ieeg'].attrs['t_axis']]['time_axis'][:]

    assert data.axes[f_obj['data_ieeg'].attrs['t_axis']]['time_axis'].attrs['sample_rate'] == 2000, 'Input h5 has incorrect sampling rate'
    assert data.shape == (600000, 160), f'Input h5 has iEEG data dimensions {data.shape}'
    assert time_axis.shape == (600000,)

    vsignal_ds, Q = zp.sigproc.filters.decimate(data[...], f_obj['data_ieeg'].axes[f_obj['data_ieeg'].attrs['t_axis']]['time_axis'].attrs['sample_rate'], 1024.0)
    time_axis_ds = f_obj['data_ieeg'].axes[f_obj['data_ieeg'].attrs['t_axis']]['time_axis'][...][::Q]
    assert vsignal_ds.shape == (300000, 160)
    assert time_axis_ds.shape == (300000,)
    # return data[...], time_axis
    return vsignal_ds, time_axis_ds
