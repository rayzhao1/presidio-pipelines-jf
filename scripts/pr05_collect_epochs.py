import sys
import os
from glob import glob
import presidio_pipelines as prespipe
import pandas as pd
import numpy as np
import bisect
import datetime
import h5py

NUM_EPOCHS = 20 # Number of epochs to collect
EPOCH_LEN = 30 # seconds
IEEG_SAMPLE_RATE = 1000 # hz
def find_consecutive_interval(arr, target):
    """Return indices for first size `target` interval [start, end) where consecutive array values
       increase by 1. Pretty there's a way a better way to do this with several np.diff's."""
    if len(arr) == 0:
        return None
    l, r, n = 0, 1, len(arr)
    while r < n and r-l < target:
        if arr[r] - arr[r-1] != 1:
            l = r
        r += 1
    return (arr[l], arr[r-1]+1) if r-l == target else None

if __name__ == '__main__':
    print("Starting...", flush=True)

    # Collect command-line arguments
    assert len(sys.argv) == 6, f'Expected 5, got {sys.argv}'
    ieeg_input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    edf_meta_csv = sys.argv[3]
    sleep_stages_dir = sys.argv[4]
    night_idx = int(sys.argv[5])

    sleep_stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
    sleep_stages_map = {'Wake': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 5}

    time_and_h5 = prespipe.ieeg.edf_merge_pr05.get_night_files(edf_meta_csv, night_idx=night_idx, item_idx=(2, 8))
    time_range = [prespipe.ieeg.edf_merge_pr05.str_to_time(t) for t, _ in time_and_h5]
    h5_files = [h5 for _, h5 in time_and_h5]

    txt_times, txt_stages = prespipe.ieeg.sleep_stage_agg.process_stages(sleep_stages_dir, time_range[0], night_idx)

    res = h5py.File(os.path.join(output_dir, f"out-n{night_idx}.h5"), "w")

    for stage in sleep_stages:
        grp = res.create_group(stage)
        grp.create_dataset("data", (NUM_EPOCHS * IEEG_SAMPLE_RATE,), dtype='f')

        # Find time for the first NUM_EPOCH consecutive epochs for `stage`.
        txt_start_idx: int = find_consecutive_interval(np.where(txt_stages == sleep_stages_map[stage])[0], NUM_EPOCHS)[0]
        start_time: datetime.datetime = txt_times[txt_start_idx]

        # Obtain the h5 files that contain the data for these consecutive epochs
        h5_idx: int = bisect.bisect_left(time_range, start_time)
        target_files: list[str] = h5_files[h5_idx: h5_idx + round(NUM_EPOCHS/10)]

        # Load and concatenate data (potentially more than needed)
        all_data = [np.load(f'{fn[:-3]}.npz') for fn in target_files]
        ieeg_arr = np.vstack([data['ieeg_arr'] for data in all_data])
        time_arr = np.vstack([data['times_arr'] for data in all_data])

        # Extract true data
        start_idx = np.where(np.abs(time_arr-start_time) <= timedelta(seconds=1))[0]
        end_idx = start_idx + NUM_EPOCHS*EPOCH_LEN*IEEG_SAMPLE_RATE
        grp['data'] = ieeg_arr[start_idx: end_idx]
