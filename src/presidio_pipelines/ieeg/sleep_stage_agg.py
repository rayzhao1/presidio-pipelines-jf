"""sleep_stage_agg.py
"""

import h5py
import pandas as pd
import numpy as np
import os
import presidio_pipelines as prespipe
from datetime import datetime, timedelta
from glob import glob


def Pipeline(output_dir, h5_path, npz_paths: str, night_idx: int) -> pd.DataFrame:
    file_obj = h5py.File(h5_path, 'r')
    assert list(file_obj.keys()) == ['MorletFamily', 'MorletFamily_kernel_axis', 'MorletFamily_time_axis', 'MorletSpectrogram', 'MorletSpectrogram_channelcoord_axis', 'MorletSpectrogram_channellabel_axis', 'MorletSpectrogram_kerneldata_axis', 'MorletSpectrogram_time_axis']
    assert list(file_obj.attrs.keys()) == ['FileType', 'FileVersion', 'map_namespace', 'map_type', 'pipeline_json', 'subject_id']

    channel_labels_array = file_obj['MorletSpectrogram_channellabel_axis'][...][0:150]  # channel labels = 0–149
    assert channel_labels_array.shape == (150, 2), f'`channel_labels_array.shape` is {channel_labels_array.shape}'

    waveletpower_arrs = []
    sleep_states_arrs = []
    time_axis = []
    time_axis_h5 = []

    for npz in npz_paths: # Already ordered.
        arrs = np.load(npz, allow_pickle=True)

        wavelets = arrs['waveletpower_arr']
        states = arrs['sleep_states_arr']
        # times_artificial = arrs['times_artificial']
        times_h5 = arrs['time_h5']

        assert wavelets.shape == (150, 50, 10)
        assert states.shape == (75000,)
        # assert times_artificial.shape == (10,)
        assert times_h5.shape == (10,)

        waveletpower_arrs.append(wavelets)
        sleep_states_arrs.append(states)
        # time_axis.append(times_artificial)
        time_axis_h5.append(times_h5)

    waveletpower = np.dstack(waveletpower_arrs)
    sleep_states = pd.Series(np.tile(np.hstack(sleep_states_arrs), 150*50))
    wavelets_freqs = file_obj['MorletFamily_kernel_axis']['CFreq']
    channel_labels = [f'{elem[0].decode()}{elem[1].decode()}' for elem in channel_labels_array]
    times = np.hstack(time_axis_h5)

    df_state = pd.read_fwf(state_fn, names=["Time", "State", "drop1", "drop2"]).drop(["drop1", 'drop2'], axis="columns")

    txt_times = df_state['Time'].values
    morelet_time_axis = np.apply_along_axis(lambda x: datetime.fromtimestamp(x * 1e-9), 0, times)

    assert len(morelet_time_axis) == len(txt_times)
    assert np.all(np.abs(txt_times - morelet_time_axis) <= timedelta(seconds=1))

    assert waveletpower.shape == (150, 50, 10 * 132)
    assert sleep_states.shape == (132*10*150*50,)
    assert wavelets_freqs.shape == (50,), f'`wavelets_freqs.shape` is {wavelets_freqs.shape}'
    assert len(channel_labels) == 150
    assert len(times) == 10*132

    # Glues together Channel, Freq, and Time as in index.
    multi_idx = pd.MultiIndex.from_product([channel_labels, wavelets_freqs, times],
                                           names=['Channel', 'Frequency', 'Time'])

    res = pd.Series(index=multi_idx, data=waveletpower.flatten())
    df = res.to_frame().rename(columns={0: 'Power'})
    df['State'] = sleep_states.values

    assert all(df.columns == pd.Index(['Power', 'State'], dtype='object'))
    assert df.shape == (9900000, 2)

    out_fn = os.path.join(output_dir, f"night-{night_idx}-df.pkl")
    df.to_pickle(out_fn)
