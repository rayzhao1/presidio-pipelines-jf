import os
import shutil
import mne
import sys
import csv
import datetime
import numpy as np
from scipy.signal import detrend


FILE_CONCAT_LIMIT: float = float('inf')
NIGHT_START_HOUR: int = 21  # 9pm
NIGHT_DURATION: datetime.timedelta = datetime.timedelta(hours=11)  # hours=11)

class Night:
    def __init__(self):
        self.intervals: list[Interval] = []

    def add(self, file):
        self.intervals.append(file)


class Interval:
    def __init__(self, t0=None, tf=None):
        self.files: list[str] = []
        self.t0 = t0
        self.tf = tf

    def __len__(self):
        return len(self.files)

    def add(self, file):
        self.files.append(file)


def to_edf(edf_path: str):
    source_path = os.getcwd()
    os.chdir(EDFS_PATH)
    raw_edf = mne.io.read_raw_edf(edf_path, preload=True)
    os.chdir(source_path)
    return raw_edf


def scalp_trim_and_decimate(raw_edf: mne.io.Raw, freq: int) -> mne.io.Raw:
    """Takes an EDF file path and returns the EDF data as an mne.io.Raw object with:
        - Only scalp channels included.
        - Resamples input EDF's frequency to 'freq'.
    """
    rename_dict: dict[str: str] = {name: name[4:] for name in raw_edf.ch_names}
    if "POL EMG1-Ref" in rename_dict:
        rename_dict["POL EMG1-Ref"] = 'L_EMG-Ref'
    if "POL EMG2-Ref" in rename_dict:
        rename_dict["POL EMG2-Ref"] = 'R_EMG-Ref'
    if 'POL L EOG-Ref' in rename_dict:
        rename_dict['POL L EOG-Ref'] = 'L_EOG-Ref'
    if 'POL R EOG-Ref' in rename_dict:
        rename_dict['POL R EOG-Ref'] = 'R_EOG-Ref'

    raw_edf = raw_edf.rename_channels(rename_dict)

    # Remove non scalp-eeg
    channels: list[str] = raw_edf.ch_names
    scalp_start: int = channels.index('Fp1-Ref')
    print('initial', raw_edf.ch_names)
    to_drop = channels[:scalp_start] + ['EKG1-Ref', 'EKG2-Ref']
    raw_scalp = raw_edf.drop_channels(to_drop)
    print('final', raw_scalp.ch_names)

    # Decimate 2000 hz to 200 hz
    raw_scalp = raw_scalp.resample(freq)  # internally uses scipy.signal.decimate
    return raw_scalp


def concatenate(lst: list[mne.io.Raw]) -> mne.io.Raw:
    """Concatenates a list of mne.io.Raw objects and returns result."""

    return mne.concatenate_raws(lst)


def scalp_bipolar_reference(raw_edf: mne.io.Raw) -> mne.io.Raw:
    cathodes = ['Fp1-Ref', 'F7-Ref', 'T7-Ref', 'P7-Ref', 'Fp1-Ref', 'F3-Ref', 'C3-Ref', 'P3-Ref', 'Fz-Ref', 'Cz-Ref',
                'Fp2-Ref', 'F4-Ref', 'C4-Ref', 'P4-Ref', 'Fp2-Ref', 'F8-Ref', 'T8-Ref', 'P8-Ref', 'L_EOG-Ref',  'R_EOG-Ref', 'L_EMG-Ref']
    anodes   = ['F7-Ref', 'T7-Ref', 'P7-Ref', 'O1-Ref', 'F3-Ref', 'C3-Ref', 'P3-Ref', 'O1-Ref', 'Cz-Ref', 'Pz-Ref',
                'F4-Ref', 'C4-Ref', 'P4-Ref', 'O2-Ref', 'F8-Ref', 'T8-Ref', 'P8-Ref', 'O2-Ref', 'A2-Ref', 'A1-Ref', 'R_EMG-Ref']
    names    = ['Fp1_F7', 'F7_T7', 'T7_P7', 'P7_O1', 'Fp1_F3', 'F3_C3', 'C3_P3', 'P3_O1', 'Fz_Cz', 'Cz_Pz',
             'Fp2_F4', 'F4_C4', 'C4_P4', 'P4_O2', 'Fp2_F8', 'F8_T8', 'T8_P8', 'P8_O2', 'L-EOG_A2', 'R-EOG_A1', 'L-EMG_R-EMG']
    assert len(cathodes) == len(anodes) == len(names), 'Incorrect cathodes, anodes, names input to bipolar_reference()'
    return mne.set_bipolar_reference(raw_edf, anodes, cathodes, names)


def average_reference(raw_edf: mne.io.Raw) -> mne.io.Raw:
    return raw_edf.set_eeg_reference()


def export(raw_edf: mne.io.Raw, target_name: str, mode=None, overwrite_existing=True):
    """Export raw object as EDF file"""
    name: str = f'{target_name}.edf'
    match mode:
        case 'bipolar':
            mne.export.export_raw(name, scalp_bipolar_reference(raw_edf), 'edf', overwrite=overwrite_existing)
        case 'common_average':
            mne.export.export_raw(name, average_reference(raw_edf), 'edf', overwrite=overwrite_existing)
        case 'bipolar_common_average':
            mne.export.export_raw(name, average_reference(scalp_bipolar_reference(raw_edf)), 'edf', overwrite=overwrite_existing)
        case _:  # default
            mne.export.export_raw(name, raw_edf, 'edf', overwrite=overwrite_existing)


def print_edf(raw_edf: mne.io.Raw, name: str) -> None:
    """Print basic information about an mne.io.Raw object."""
    # data, time = raw_edf[:, :]
    print(f'\n\n\n\nTesting {name} EDF:\n')
    print(raw_edf.info)
    print('Dim:', raw_edf.get_data().shape[0], 'channels', 'x', raw_edf.get_data().shape[1], 'time points\n\n\n')


def write_txt(*args) -> None:
    with open(os.path.join(os.getcwd(), 'summary.txt'), 'a') as f:
        for txt in args:
            f.write(txt + '\n')
        f.write('\n\n\n')


def str_to_time(time_str: str, time_format='%Y-%m-%d %H:%M:%S') -> datetime.datetime:
    return datetime.datetime.strptime(time_str.split('.')[0], time_format)


def get_first_date(csv_in: str) -> datetime.datetime:
    with open(csv_in) as csv_file:
        csv_reader: csv.reader = csv.reader(csv_file, delimiter=',')
        _, first_row = next(csv_reader), next(csv_reader)
        return str_to_time(first_row[3])


def parse_find(csv_in: str, all_files: set[str], idx=None, margin=datetime.timedelta(minutes=1)) -> list[Night]:
    """Iterate through 'csv_in' and return a list of lists, where each sublist contains an contiguous_interval of EDF file names
       such that each EDF is less than 'margin' away from the previous file in time. This implementation relies on the
       fact that csv_in is sorted in time-chronological order. All returned EDF files are also constrained to be in the
       time range between 'start' and 'end'.
    """
    start: datetime.datetime = get_first_date(csv_in)  # Set start date
    start = start.replace(hour=NIGHT_START_HOUR, minute=0, second=0, microsecond=0)  # Set start date and time
    end: datetime.datetime = start + NIGHT_DURATION  # Set end time

    source_path: str = os.getcwd()
    os.chdir(PATIENT_PATH)
    nights: list[Night] = []
    curr_night: Night = Night()
    curr_interval: Interval = Interval()
    count: int = 0

    with open(csv_in) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # remove header
        new_interval_flag: bool = True

        for row in csv_reader:
            curr_name: str = row[0]
            curr_time_start: datetime.datetime = str_to_time(row[2])
            curr_time_end: datetime.datetime = str_to_time(row[3])
            # Do not record files earlier than start, or file names that cannot be found in folder.
            if curr_name not in all_files or curr_time_start < start - margin:
                new_interval_flag = True
                prev_time_end = curr_time_end
                continue
            if new_interval_flag:
                new_interval_flag = False
                curr_interval.t0 = curr_time_start

            # If reach concat length or the time difference is large, add a new Interval for the current night.
            if curr_time_start - prev_time_end > margin or count >= FILE_CONCAT_LIMIT:
                raise Error("Something is off with the files.")

            # Add to list subsection, if the file exists.
            if idx:
                curr_interval.add((curr_name, row[idx]))
            else:
                curr_interval.add(curr_name)

            prev_time_end = curr_time_end
            count += 1

            # If we exceed the contiguous_interval length, add a new night
            if curr_time_end >= end - margin: #curr_date != curr_time_start.day:
                curr_interval.tf = prev_time_end
                curr_night.add(curr_interval)
                nights.append(curr_night)
                curr_interval = Interval()
                curr_night = Night()
                start = curr_time_start.replace(hour=NIGHT_START_HOUR, minute=0, second=0, microsecond=0)  # Set start date and time
                end = start + NIGHT_DURATION
                count = 0

    # Tail case
    if curr_interval.files:
        curr_interval.tf = prev_time_end
        curr_night.add(curr_interval)
        nights.append(curr_night)

    os.chdir(source_path)
    return nights

def verify_pr05_concatenate_ranges(nights):
    def get_num_str(num):
        if num < 100:
            return f'PR05_00{num}.edf'
        if num < 1000:
            return f'PR05_0{num}.edf'
        else:
            return f'PR05_{num}.edf'

    assert len(nights[0].intervals[0].files) == 132 and nights[0].intervals[0].files == [get_num_str(i) for i in range(129, 261)], 'Night 1 files to be concatenated are wrong.'
    assert len(nights[1].intervals[0].files) == 132 and nights[1].intervals[0].files == [get_num_str(i) for i in range(415, 547)], 'Night 2 files to be concatenated are wrong.'
    assert len(nights[2].intervals[0].files) == 132 and nights[2].intervals[0].files == [get_num_str(i) for i in
                                         range(697, 829)], 'Night 3 files to be concatenated are wrong.'
    assert len(nights[3].intervals[0].files) == 132 and nights[3].intervals[0].files == [get_num_str(i) for i in
                                         range(982, 1114)], 'Night 4 files to be concatenated are wrong.'
    assert len(nights[4].intervals[0].files) == 132 and nights[4].intervals[0].files == [get_num_str(i) for i in
                                         range(1260, 1392)], 'Night 5 files to be concatenated are wrong.'
    assert len(nights[5].intervals[0].files) == 132 and nights[5].intervals[0].files == [get_num_str(i) for i in
                                         range(1545, 1677)], 'Night 6 files to be concatenated are wrong.'
    assert len(nights[6].intervals[0].files) == 132 and nights[6].intervals[0].files == [get_num_str(i) for i in
                                         range(1830, 1962)], 'Night 7 files to be concatenated are wrong.'
    assert len(nights[7].intervals[0].files) == 132 and nights[7].intervals[0].files == [get_num_str(i) for i in
                                         range(2114, 2246)], 'Night 8 files to be concatenated are wrong.'
    assert len(nights[8].intervals[0].files) == 132 and nights[8].intervals[0].files == [get_num_str(i) for i in
                                         range(2399, 2531)], 'Night 9 files to be concatenated are wrong.'


if __name__ == "__main__":  # can get rid of
    # Process command-line args
    argc: int = len(sys.argv)
    limit: float = float('inf')
    name, tag = '', 'pr05-8.31-fix'
    SRC_PATH: str = os.getcwd()
    # Navigate to EDF files
    while os.getcwd() is not os.sep: # for _ in range(1):
        os.chdir('..')
    HOME_PATH: str = os.getcwd()
    PATIENT_PATH: str = os.path.join(HOME_PATH, 'data_store0/presidio/nihon_kohden/PR05')
    CATALOG_PATH: str = os.path.join(PATIENT_PATH, 'nkhdf5')
    os.chdir(PATIENT_PATH)

    # Identify file paths
    PATIENT: str = os.path.basename(os.getcwd())
    EDFS_PATH: str = os.path.join(HOME_PATH, PATIENT_PATH, PATIENT)
    csv_catalog: str = f'{PATIENT}_edf_catalog.csv'

    # Retrieve list of sub lists. Each sublist is a set of continuous, in-range file names.
    os.chdir(EDFS_PATH)
    all_edfs: set[str] = set(os.listdir())
    os.chdir(PATIENT_PATH)
    nights: list[Night] = parse_find(os.path.join(CATALOG_PATH, csv_catalog), all_edfs)
    out_dir = os.path.join(SRC_PATH, f'out-{PATIENT}-{tag}')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    os.chdir(out_dir)

    verify_pr05_concatenate_ranges(nights)

    res: mne.io.Raw = None
    # For each night, export one merged file for each continuous time-contiguous_interval for each night.
    for night_num, night in enumerate(nights):
        for interval_num, interval in enumerate(night.intervals):
            # if `contiguous_interval.t0 is None`, then that contiguous_interval never reached a starting point.
            if len(interval) < 1 or not interval.t0:
                continue

            t0_str: str = interval.t0.strftime("%Y-%m-%d_%H.%M")
            tf_str: str = interval.tf.strftime("%Y-%m-%d_%H.%M")
            out_name: str = f'{PATIENT}_night_{night_num+1}.{interval_num+1}_scalp_{t0_str}--{tf_str}'

            print(f"Night-{night_num} t0_str: ", t0_str)
            print(f"Night-{night_num} tf_str: ", tf_str)
            print(f"Night-{night_num} files: {interval.files}")

            concatenated = concatenate([scalp_trim_and_decimate(to_edf(edf), 200) for edf in interval.files])
            # 1) bandpass for neural data 2) bandstop for electrical noise 3) demean 4) scale
            res = ((concatenated
                    .filter(l_freq=0.5, h_freq=80)
                    .notch_filter(60, notch_widths=4)
                    .apply_function(detrend, channel_wise=True, type="constant"))
                    .apply_function(lambda x: x*1e-6, picks="eeg"))

            export(res, out_name, 'bipolar', True)

    print("PR05 concatenation completed!")
