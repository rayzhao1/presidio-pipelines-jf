import os
import presidio_pipelines as prespipe
import numpy as np
import bisect
import datetime
import h5py

NUM_EPOCHS = 20 # Number of epochs to collect
EPOCH_LEN = 30 # seconds
IEEG_SAMPLE_RATE = 1000 # hz
N = NUM_EPOCHS*EPOCH_LEN*IEEG_SAMPLE_RATE
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

def find_epochs(stages, stage, input_dir):
    data = np.load(os.path.join(input_dir, 'output-n0.npz'), allow_pickle=True)
    ieeg_arr, time_arr = data['ieeg_data'], data['times_axis']
    assert ieeg_arr.shape == (39600000, 160), f'ieeg_arr.shape == {ieeg_arr.shape}'
    assert time_arr.shape == (39600000,), f'time_arr.shape == {time_arr.shape}'

    mask = np.repeat(stages==stage, EPOCH_LEN * IEEG_SAMPLE_RATE)

    assert len(mask) == len(ieeg_arr) == len(time_arr), f'len(mask) == {len(mask)}, len(ieeg_arr) == {len(ieeg_arr)}, len(time_arr) = {len(time_arr)}'

    ieeg, time = ieeg_arr[mask, :][:N, :], time_arr[mask][:N]

    assert len(ieeg) == len(time) == N, f'len(ieeg) == {len(ieeg)}, len(time) == {len(time)}'
    return ieeg, time

def find_consecutive_epochs(edf_meta_csv, night_idx, consecutive_interval_start):
    time_and_h5 = prespipe.ieeg.edf_merge_pr05.get_night_files(edf_meta_csv, night_idx=night_idx, item_idx=(2, 8))
    time_range = [prespipe.ieeg.edf_merge_pr05.str_to_time(t) for t, _ in time_and_h5]
    h5_files = [h5 for _, h5 in time_and_h5]

    start_time: datetime.datetime = txt_times[consecutive_interval_start]

    # Obtain the h5 files that contain the data for these consecutive epochs
    h5_idx: int = bisect.bisect_left(time_range, start_time) - 1
    target_files: list[str] = h5_files[h5_idx: h5_idx + round(NUM_EPOCHS / 10) + 1]

    # Load and concatenate data (potentially more than needed)
    all_data = [np.load(os.path.join(ieeg_input_dir, f'{os.path.basename(fn)[:-3]}.npz')) for fn in target_files]
    ieeg_arr = np.vstack([data['ieeg_arr'] for data in all_data])
    time_arr = np.hstack([data['times_arr'] for data in all_data])
    time_axis = np.array([prespipe.ieeg.stage_one_preprocess.from_unix_epoch(t) for t in time_arr])

    # Extract true data
    start_idx = np.where(np.abs(time_axis - start_time) <= datetime.timedelta(seconds=1))[0][0]
    end_idx = start_idx + N

    return ieeg_arr[start_idx: end_idx], time_arr[start_idx: end_idx]

def Pipeline(ieeg_input_dir: str, output_dir, edf_meta_csv, sleep_stages_dir, night_idx) -> str:
    txt_times, txt_stages = prespipe.ieeg.sleep_stage_agg.process_stages(sleep_stages_dir, time_range[0], night_idx)
    out_fn = os.path.join(output_dir, f"out-n{night_idx}.h5")
    res = h5py.File(out_fn, "w")
    sleep_stages = [('Wake', 0), ('N1', 1), ('N2', 2), ('N3', 3), ('REM', 5)]

    for stage, idx in sleep_stages:
        grp = res.create_group(stage)

        # Find index for the first NUM_EPOCH consecutive epochs for `stage`.
        stage_idx = np.where(txt_stages == idx)[0]
        consecutive_interval_idx: tuple[int, int] = find_consecutive_interval(stage_idx, NUM_EPOCHS)

        # If a consecutive interval exists for this stage, collect contiguous epochs (efficient). Else collect noncontiguous epochs.
        if consecutive_interval_idx:
            ieeg, time = find_consecutive_epochs(edf_meta_csv, night_idx, consecutive_interval_idx[0])
        else:
            ieeg, time = find_epochs(txt_stages, idx, ieeg_input_dir)

        assert ieeg.shape == (N, 160), f'ieeg.shape == {ieeg.shape}'
        assert time.shape == (N,), f'ieeg.shape == {time.shape}'

        grp.create_dataset("ieeg_data", data=ieeg)
        grp.create_dataset("time_axis", data=time)

        print(f"Collected epochs for stage {stage}")

    return out_fn
