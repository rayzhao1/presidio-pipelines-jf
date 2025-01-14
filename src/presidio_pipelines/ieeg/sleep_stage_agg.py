"""sleep_stage_agg.py
"""

import h5py
import pandas as pd
import numpy as np
import os
import scipy
import presidio_pipelines as prespipe
from datetime import datetime, date, timedelta
from glob import glob
import pytz

def stage_mean_power(stages, arr, stage, validation):
    """Given the mean wavelets for one night, produce the mean and SD
       of these wavelets at a specified stage for each channel across
       all frequencies. The time axis is collapsed.
    """
    assert arr.shape == (150, 1320, 50) # 150 channels, 1320 samples, 50 freqs

    # Produce bitmap of indices corresponding to specified stage.
    mask = stages==stage
    assert mask.shape == (1320,)

    # Filter for samples of the desired stage using bitmap
    stage_arrs = arr[:, mask, :]
    print(np.sum(mask))
    assert stage_arrs.shape == (150, np.sum(mask), 50), f'stage_arrs.shape is {stage_arrs.shape}'

    # Compute mean and std of power across all frequencies, killing the second dimension
    stage_means = np.apply_along_axis(np.nanmean, 1, stage_arrs)
    stage_stds = np.apply_along_axis(np.nanstd, 1, stage_arrs)# /np.sum(mask)
    assert stage_means.shape == (150, 50), f'stage_means.shape is {stage_means.shape}'
    assert stage_stds.shape == (150, 50), f'stage_stds.shape is {stage_stds.shape}'

    return stage_means, stage_stds, validation + mask

def convert(s, d):
    """Helper function to convert timestamp 's', ex. '22:12', into a datetime where day is specified by 'd'."""
    if int(s[:2]) in range(20, 24):
        return datetime.strptime(s, '%H:%M:%S').replace(year=d.year, month=d.month, day=d.day)
    return datetime.strptime(s, '%H:%M:%S').replace(year=d.year, month=d.month, day=d.day+1)

def process_stages(sleep_stages_dir, start_dt, night_idx):
    """Process scored output files for a day into np arrays."""
    sleep_stages_fn = os.path.join(sleep_stages_dir, f'PR05_night_{night_idx + 1}.1 Stages_with_file.txt')
    df = pd.read_fwf(sleep_stages_fn, names=["Time", "State", "drop1", "drop2"]).drop(["drop1", 'drop2'],
                                                                                                   axis="columns")
    states = df['State'].values.astype(int)
    times = np.array([convert(t, start_dt) for t in df['Time'].values])
    return times, states

def Pipeline(output_dir, h5_path, npz_paths: list[str], sleep_stages_dir: str, night_idx: int) -> pd.DataFrame:
    file_obj = h5py.File(h5_path, 'r')
    print(h5_path)
    assert list(file_obj.keys()) == ['MorletFamily', 'MorletFamily_kernel_axis', 'MorletFamily_time_axis', 'MorletSpectrogram', 'MorletSpectrogram_channelcoord_axis', 'MorletSpectrogram_channellabel_axis', 'MorletSpectrogram_kerneldata_axis', 'MorletSpectrogram_time_axis']
    assert list(file_obj.attrs.keys()) == ['FileType', 'FileVersion', 'map_namespace', 'map_type', 'pipeline_json', 'subject_id']

    channel_labels_array = file_obj['MorletSpectrogram_channellabel_axis'][...][0:150]  # channel labels = 0–149
    assert channel_labels_array.shape == (150, 2), f'`channel_labels_array.shape` is {channel_labels_array.shape}'

    waveletpower_arrs = []
    time_axis_h5 = []

    for npz in npz_paths: # Already ordered.
        arrs = np.load(npz, allow_pickle=True)

        wavelets = arrs['waveletpower_arr']
        times_h5 = arrs['time_h5']

        assert wavelets.shape == (150, 50, 10)
        assert times_h5.shape == (10,)

        waveletpower_arrs.append(wavelets)
        time_axis_h5.append(times_h5)

    waveletpower = np.dstack(waveletpower_arrs)
    wavelets_freqs = file_obj['MorletFamily_kernel_axis']['CFreq']
    channel_labels = [f'{elem[0].decode()}{elem[1].decode()}' for elem in channel_labels_array]
    times = np.hstack(time_axis_h5)

    morelet_time_axis = [pd.Timestamp(dt).to_pydatetime() for dt in times]

    txt_times, txt_stages = process_stages(sleep_stages_dir, morelet_time_axis[0], night_idx)
    sleep_states = np.tile(txt_stages, 150 * 50)

    assert len(morelet_time_axis) == len(txt_times)
    assert np.all(np.abs(txt_times - morelet_time_axis) <= timedelta(seconds=1))

    assert waveletpower.shape == (150, 50, 10 * 132)
    assert sleep_states.shape == (132*10*150*50,)
    assert wavelets_freqs.shape == (50,), f'`wavelets_freqs.shape` is {wavelets_freqs.shape}'
    assert len(channel_labels) == 150
    assert len(times) == 10*132

    # (1) Produce stage aligned wavelet power DataFrame
    multi_idx = pd.MultiIndex.from_product([channel_labels, wavelets_freqs, times],
                                           names=['Channel', 'Frequency', 'Time'])

    df = pd.DataFrame(index=multi_idx, data={"Power": waveletpower.flatten(), "State": sleep_states})

    assert all(df.columns == pd.Index(['Power', 'State'], dtype='object'))
    assert df.shape == (9900000, 2)

    mat = np.swapaxes(waveletpower, 0, 2)
    assert mat.shape == (1320, 50, 150)
    assert txt_stages.shape == (1320,)

    mat_fn = os.path.join(output_dir, 'pr05_night1')
    print(mat_fn)

    #scipy.io.savemat(mat_fn, {'wavelet_power': mat, 'sleep_stages': txt_stages}, appendmat=True)

    out_fn = os.path.join(output_dir, f"night-{night_idx}-df.pkl")
    df.to_pickle(out_fn)

    # (2) Produce per stage wavelet power DataFrame
    waveletpower = np.swapaxes(waveletpower, 1, 2)
    #np.save(f'{out_fn[:-7]}.npy', waveletpower)
    assert waveletpower.shape == (150, 1320, 50)

    sleep_stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
    sleep_stages_map = {'Wake': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 5}

    stage_wavelet_mean_arrs = []
    stage_wavelet_std_arrs = []

    # A bitmap s = [0, 0, ..., 0] representing how many times the value at each index is used.
    # If the algorithm is correct, each sample is used once and sum(s) == len(times) == 1320
    validation_arr = np.zeros(1320)

    for stage in sleep_stages:
        mu, sd, validation_arr = stage_mean_power(txt_stages, waveletpower, sleep_stages_map[stage], validation_arr)
        stage_wavelet_mean_arrs.append(mu.flatten())
        stage_wavelet_std_arrs.append(sd.flatten())

    stage_wavelet_means = np.hstack(stage_wavelet_mean_arrs)
    stage_wavelet_stds = np.hstack(stage_wavelet_std_arrs)

    assert np.sum(validation_arr) == 1320, f'np.sum(validation_arr) == {np.sum(validation_arr)}'
    assert stage_wavelet_means.shape == (37500,), f'stage_wavelet_means.shape is {stage_wavelet_means.shape}'
    assert stage_wavelet_stds.shape == (37500,), f'stage_wavelet_stds.shape is {stage_wavelet_stds.shape}'

    multi_idx = pd.MultiIndex.from_product([channel_labels, sleep_stages, wavelets_freqs],
                                           names=['Channel', 'Stage', 'Frequency'])

    print(len(multi_idx), multi_idx.array)

    df = pd.DataFrame(index=multi_idx, data={"Power": stage_wavelet_means, "Error": stage_wavelet_stds})
    # print(df.to_string())

    out_fn = os.path.join(output_dir, f"night-{night_idx}-stage-df.pkl")
    df.to_pickle(out_fn)
