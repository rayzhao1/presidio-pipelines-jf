"""sleep_stage_agg.py
"""

import h5py
import pandas as pd
import numpy as np
import os
import presidio_pipelines as prespipe
from datetime import datetime, timedelta
from glob import glob


def Pipeline(h5_path: str, edf_path: str, edf_meta_csv: str) -> pd.DataFrame:
    file_obj = h5py.File(h5_path, 'r')

    assert list(file_obj.keys()) == ['MorletFamily', 'MorletFamily_kernel_axis', 'MorletFamily_time_axis', 'MorletSpectrogram', 'MorletSpectrogram_channelcoord_axis', 'MorletSpectrogram_channellabel_axis', 'MorletSpectrogram_kerneldata_axis', 'MorletSpectrogram_time_axis']

    assert list(file_obj.attrs.keys()) == ['FileType', 'FileVersion', 'map_namespace', 'map_type', 'pipeline_json', 'subject_id']

    wavelets_freqs = file_obj['MorletFamily_kernel_axis']['CFreq']

    assert wavelets_freqs.shape == (50,), f'`wavelets_freqs.shape` is {wavelets_freqs.shape}'

    channel_labels_array = file_obj['MorletSpectrogram_channellabel_axis'][...][0:150]  # channel labels = 0–149
    assert channel_labels_array.shape == (150, 2), f'`channel_labels_array.shape` is {channel_labels_array.shape}'
    channel_labels = [f'{elem[0].decode()}{elem[1].decode()}' for elem in channel_labels_array]

    all_edfs = glob(os.path.join(edf_path, '*'))

    nights = parse_find(edf_meta_csv, all_edfs, idx=8)

    for night_num, night in enumerate(nights):
        waveletpower_arrs = []
        sleep_states_arrs = []
        time_axis = []
        time_axis_h5 = []
        for interval_num, interval in enumerate(night.intervals):
            # if `contiguous_interval.t0 is None`, then that contiguous_interval never reached a starting point.
            if len(interval) < 1 or not interval.t0:
                continue
            npz_paths = [f'{file[1].split('.')[0]}.npz' for file in interval.files]

            for npz in npz_paths:
                arrs = np.load(npz, allow_pickle=True)
                waveletpower_arrs.append(arrs['waveletpower_arr'])
                sleep_states_arrs.append(arrs['sleep_states_arr'])
                time_axis.append(arrs['times_artificial'])
                time_axis_h5.append(arrs['times_h5'])

        waveletpower = np.dstack(waveletpower_arrs)
        sleep_states = pd.Series(np.dstack(sleep_states_arrs))

        # Glues together Channel, Freq, and Time as in index. Freq's are flipped because they are descending by default.
        multi_idx = pd.MultiIndex.from_product([channel_labels, np.flip(wavelets_freqs), time_axis],
                                               names=['Channel', 'CFreq', 'Time'])

        res = pd.Series(index=multi_idx, data=waveletpower.flatten())

        df = res.to_frame()

        # want another column of time pulled from h5 for verification
        df['State'] = sleep_states.values

        print(df.to_string())

        df.to_pickle(f'{h5_path.split('.')[0]}_night-{night_num}.pkl')
