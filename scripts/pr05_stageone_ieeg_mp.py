import sys
import os
from glob import glob
import presidio_pipelines as prespipe
from multiprocessing import Pool, Manager
import pandas as pd
import numpy as np

sys.stdout.flush()
print("start")

TESTING_LIMIT = 5

SERVER_INPUTS = ('~/presidio-pipelines/scripts/pr05_stageone_ieeg.py', '/data_store0/presidio/nihon_kohden/PR05/nkhdf5/edf_to_hdf5/*', '~/presidio-pipelines/out', '/data_store0/presidio/nihon_kohden/PR05/nkhdf5/PR05_edf_catalog.csv')

def map_fn(h5_fn: str, output_dir: str, edf_meta_csv: str, sleep_stages_dict, delete=True):
    def get_states_file(time):
        for range in sleep_stages_dict.keys():
            if range[0] <= time <= range[1]:
                return sleep_stages_dict[range]
        raise Error(f'Time {time} not found')

    # Preprocess
    print(f"Preprocessing file {h5_fn}...")
    preprocess_fn = prespipe.ieeg.stage_one_preprocess.Pipeline(h5_fn, output_dir, edf_meta_csv)
    preprocess_success = preprocess_fn in glob(os.path.join(output_dir, '*'))
    print(f"Success? ", preprocess_success)

    # Wavelet transform
    print(f"Performing wavelet transform on {preprocess_fn}")
    wavelettransform_fn = prespipe.ieeg.stage_one_wavelettransform.Pipeline(preprocess_fn, output_dir)
    wavelettransform_success = wavelettransform_fn in glob(os.path.join(output_dir, '*'))
    print(f"Success? ", wavelettransform_success)

    # Wavelet means
    print(f"Computing wavelet means on {wavelettransform_fn}")
    waveletpower_fn, start_time = prespipe.ieeg.stage_one_waveletpower.Pipeline(wavelettransform_fn, output_dir, edf_meta_csv)
    waveletpower_success = waveletpower_fn in glob(os.path.join(output_dir, '*'))
    print(f"Success? ", waveletpower_success)

    # Store needed data
    print(f"Creating time aligned dataframe using {waveletpower_fn}")

    edf_meta = pd.read_csv(edf_meta_csv)
    print("start time is", start_time)
    print("states_file is", get_states_file(start_time))
    sleep_stages = pd.read_fwf(get_states_file(start_time), names=["Time", "State", "drop1", "drop2"]).drop(["drop1", 'drop2'], axis="columns")

    waveletpower_arr, sleep_states_arr, times_artificial, times_h5 = prespipe.ieeg.sleep_stage_align.Pipeline(waveletpower_fn, sleep_stages, edf_meta)

    # Store down numpy arrays storing this h5's wavelet power, sleep states, and 2 time axes
    sleep_state_aligned_fn = f"{waveletpower_fn.split('.')[0]}.npz"
    np.savez(sleep_state_aligned_fn, waveletpower_arr=waveletpower_arr, sleep_states_arr=sleep_states_arr, times_artificial=times_artificial, time_h5=times_h5)
    sleep_state_align_success = sleep_state_aligned_fn in glob(os.path.join(output_dir, '*'))
    print(f"Success? ", sleep_state_align_success)

    if not preprocess_success or not wavelettransform_success or not waveletpower_success or not sleep_state_align_success:
        raise Error(f"Something went wrong for file {h5_fn}")

    os.remove(preprocess_fn)
    os.remove(wavelettransform_fn)
    if delete:
        os.remove(waveletpower_fn)

    print(f"Done processing {h5_fn} to produce output {waveletpower_fn}")


if __name__ == '__main__':

    # Collect command-line arguments
    assert len(sys.argv) == 5, f'Expected 5, got {sys.argv}'
    h5_input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    edf_meta_csv = sys.argv[3]
    sleep_stages_dir = sys.argv[4]

    # Retrieve files
    input_dir_files = glob(os.path.join(h5_input_dir, '*'))
    print(f'Working in directory {h5_input_dir}')
    sleep_stages_dir_files = glob(os.path.join(sleep_stages_dir, '*'))

    # Identify per night information (1) Interval for each night (2) files for each night
    nights = prespipe.ieeg.edf_merge_pr05.parse_find(edf_meta_csv, idx=8)
    file_count = prespipe.ieeg.edf_merge_pr05.verify_pr05_concatenate_ranges(nights)
    night_time_ranges = []
    nights_h5_set = set()

    for night in nights:
        for interval in night.intervals:
            if len(interval) < 1 or not interval.t0:
                continue
            night_time_ranges.append((interval.t0, interval.tf))
            nights_h5_set.update(interval.files)

    assert len(night_time_ranges) == len(sleep_stages_dir_files), f"{len(night_time_ranges)} vs {len(sleep_stages_dir_files)}"

    h5_files = [fn for fn in input_dir_files if fn in nights_h5_set]
    assert file_count == len(h5_files), f'file_count = {file_cout} vs. len(h5_files) == {len(h5_files)}'

    # https://stackoverflow.com/questions/6832554/multiprocessing-how-do-i-share-a-dict-among-multiple-processes

    # Initialize workers with shared dict
    with Manager() as manager:
        time_to_txt = manager.dict()

        for night_num in range(len(sleep_stages_dir_files)):
            fn = os.path.join(sleep_stages_dir, f'PR05_night_{night_num+1}.1 Stages_with_file.txt')
            time_to_txt[night_time_ranges[night_num]] = fn # Map interval tuple -> sleep stage .txt

        # Work on files that have not yet been processed.
        proc_inputs = [(fn, output_dir, edf_meta_csv, time_to_txt) for fn in h5_files if 'preprocess_wavelettransform_meanwaveletpower' not in fn]

        with manager.Pool(4) as p:
            p.starmap(map_fn, proc_inputs)
            p.close()
            p.join()

        # An arbitrary temporary h5 that stores information to be used later
        map_fn(*h5_files[0], delete=False)

    # Ensure input directory was not mutated.
    assert input_dir_files == glob(os.path.join(h5_input_dir, '*')), 'Input directly was unexpectedly changed.'
    print("Success")











