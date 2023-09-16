"""stage_one_preprocess.py
"""

from collections import OrderedDict

import numpy as np
import zappy as zp
from nkhdf5 import hdf5nk 

from .modules import *

HDF5NK = hdf5nk.HDF5NK_0_1_0 


#TODO: Get HDF5NK class automatically. no need to guess class version
#TODO: Addopt virtualaxismap
#TODO: Cleanup the pipeline function

def Pipeline(path: str):
    #
    f_obj = apply_reader(path=path, h5interface=HDF5NK)

    #
    vsignal, vchangrp = apply_reference(f_obj['data_ieeg'], montage='bipolar')
    
    #
    vsignal_ds, Q = zp.sigproc.filters.decimate(vsignal, f_obj['data_ieeg'].axes[f_obj['data_ieeg'].attrs['t_axis']]['time_axis'].attrs['sample_rate'], 1024.0)
    time_axis_ds = f_obj['data_ieeg'].axes[f_obj['data_ieeg'].attrs['t_axis']]['time_axis'][...][::Q]
    new_sample_rate = f_obj['data_ieeg'].axes[f_obj['data_ieeg'].attrs['t_axis']]['time_axis'].attrs['sample_rate'] / Q

    #
    vsignal_ds = zp.sigproc.filters.high_pass_filter(vsignal_ds, new_sample_rate, corner_freq=0.5, stop_tol=10)

    #
    vsignal_ds = zp.sigproc.filters.notch_line(vsignal_ds, new_sample_rate)

    #
    out_path = path.split('.h5')[0] + '_preprocess.h5'
    data_dict = {"data": vsignal_ds, "time_axis_data": time_axis_ds, "sample_rate": new_sample_rate, "low_pass_filter": 500.0, "high_pass_filter": 0.5, "vchangrp": vchangrp}

    apply_writer(out_path, f_obj, data_dict) 
