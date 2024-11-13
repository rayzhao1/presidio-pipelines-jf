"""stage_one_wavelettransform.py
"""

from collections import OrderedDict

import os
from tqdm import tqdm
import numpy as np
import zappy as zp
from pyeisen import family, convolve
from presidio_hdf5objects.dataset.files.basehdf5processeddata import BaseHDF5ProcessedData_0_1_0
from presidio_hdf5objects.dataset.files.hdf5waveletdata import HDF5WaveletData_0_1_0

from .modules import *

import sys

sys.stdout.flush()

BaseHDF5ProcessedData = BaseHDF5ProcessedData_0_1_0
HDF5WaveletData = HDF5WaveletData_0_1_0

def Pipeline(h5_path: str, output_path: str) -> str:
    #
    print("sanity")
    print("Flusing stdout...")
    sys.stdout.flush()
    print("sanity")

    #
    f_obj = apply_reader(path=h5_path, h5interface=BaseHDF5ProcessedData)

    #
    morlet_fam = family.morlet(
        freqs=np.logspace(np.log10(1),np.log10(250), 50),       # Center frequencies of the children (Cycles/sec (Hz))
        cycles=np.ones(50)*6,                                   # Number of cycles to sample for each child. Governs the width of the Gaussian envelope.
        fs=f_obj['data'].axes[f_obj['data'].attrs['t_axis']]['time_axis'].attrs['sample_rate'],  # Sampling frequency of the wavelet (depends on the sampling frequency of the signal to be analyzed); set arbitrarily large here.
        n_win=6.5                                               # Window support size for the wavelet -- wide enough to sample enough of the envelope roll-off.
    )

    #
    out_path = os.path.join(output_path, f'{os.path.basename(h5_path)[:-3]}_wavelettransform.h5')
    file = HDF5WaveletData(file=out_path, mode="a", create=True, construct=True)

    file_data = file["morlet_kernel_data"]
    file_data.resize(morlet_fam["kernel"].shape)
    file_data[:, :] = morlet_fam["kernel"]
    file_data.axes[1]['time_axis'].resize((morlet_fam["sample"]["time"].shape[0],))
    file_data.axes[1]['time_axis'][:] = morlet_fam["sample"]["time"]
    file_data.axes[0]["kernel_axis"].resize((morlet_fam["kernel"].shape[0],))
    params_tups = [(morlet_fam["params"]["freqs"][i], morlet_fam["params"]["cycles"][i], morlet_fam["params"]["scales"][i]) for i in range(morlet_fam["kernel"].shape[0])]
    file_data.axes[0]["kernel_axis"][:] = params_tups
    file.close()

    #

    _, old_q = zp.sigproc.filters.resample_factor(
        f_obj['data'].axes[f_obj['data'].attrs['t_axis']]['time_axis'].attrs['sample_rate'], 100)
    #

    new_sample_rate = 1
    _, q = zp.sigproc.filters.resample_factor(f_obj['data'].axes[f_obj['data'].attrs['t_axis']]['time_axis'].attrs['sample_rate'], new_sample_rate)

    # Data validation
    assert old_q == 10 and q == 1000, 'Downsample was not successfully completed.'

    #
    file = HDF5WaveletData(file=out_path, mode="a", create=True, construct=True)
    file_data = file["spectrogram_data"]

    # Ensure sampling rate >= 2 * 120hz to satisfy Nyquist theorem for Gamma frequencies
    assert f_obj['data'].axes[f_obj['data'].attrs['t_axis']]['time_axis'].attrs['sample_rate'] >= 240

    fp = "C://Users//raymo//ray//ucsf//res.txt"
    pbar = tqdm(f_obj["data"][...].T, file=open(fp, 'w'))

    #
    for proc_ii, proc_data in enumerate(pbar):
        #
        pbar.write(f"Iteration {proc_ii}:")
        pbar.write(f'Shape of morelet kernel being convolved {morlet_fam["kernel"].T.shape}')
        pbar.write(f'Shape of signal {proc_data[:, None].shape}')
        #

        convolved_signal = convolve.fconv(morlet_fam["kernel"].T, proc_data[:, None]).transpose((1, 0, 2))

        #
        pbar.write(f'Shape of convolved signal is {convolved_signal.shape}')
        #

        convolved_signal = convolve.fconv(morlet_fam["kernel"].T, proc_data[:, None]).transpose((1, 0, 2))[:,::q,:]

        #
        pbar.write(f'Shape of convolved signal after downsample is is {convolved_signal.shape}')
        #

        if file_data.shape == (0, 0, 0):#
            file_data.resize((convolved_signal.shape[0], convolved_signal.shape[1], f_obj["data"].shape[1])) #(file["morlet_kernel_data"].shape[0], f_obj["data"].shape[0] // q, f_obj["data"].shape[1])))

        # ellipses selection fails for some reason:
        #  = convolved_signal[...] -> TypeError: Broadcasting is not supported for complex selections
        file_data[:,:,[proc_ii]] = convolved_signal[:, :, :]
    file_data.axes[1]["time_axis"].append(f_obj["data"].axes[0]["time_axis"][...][::q])
    file_data.axes[2]["channellabel_axis"].append(f_obj["data"].axes[1]["vlabel_axis"][...])
    file_data.axes[2]["channelcoord_axis"].append(f_obj["data"].axes[1]["vcoord_axis"][...])
    morlet_refs = file["morlet_kernel_data"]
    file_data.axes[0]["kerneldata_axis"].resize((morlet_refs.shape[0],))
    for i in range(morlet_refs.shape[0]):
        file_data.axes[0]["kerneldata_axis"][i] = (morlet_refs.ref, morlet_refs.regionref[[i], :])
    file.close()

    return out_path
