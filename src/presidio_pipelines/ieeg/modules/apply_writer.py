"""apply_writer.py
"""

from typing import Any
from typing import Dict

import numpy as np

from presidio_hdf5objects.dataset.files.basehdf5processeddata import BaseHDF5ProcessedData_0_1_0

BaseHDF5ProcessedData = BaseHDF5ProcessedData_0_1_0

def apply_writer(path: str, old_obj: Any, data_dict: Dict) -> Any:
    """Opens the interface to the H5 schema containing raw, or very minimally processed data."""

    f_obj = BaseHDF5ProcessedData(file=path, mode="a", create=True, construct=True)
    f_obj.attributes["subject_id"] = old_obj.attributes["subject_id"]
    f_obj.attributes["start"] = old_obj.attributes["start"]
    f_obj.attributes["end"] = old_obj.attributes["end"]

    file_data = f_obj["data"]
    file_data.append(data_dict["data"], component_kwargs={"timeseries": {"data": data_dict["time_axis_data"]}})

    file_data.attributes["filter_lowpass"]  = data_dict["low_pass_filter"]
    file_data.attributes["filter_highpass"] = data_dict["high_pass_filter"]
    file_data.attributes["channel_count"]   = data_dict["data"].shape[1]
    file_data.axes[0]['time_axis'].attrs['sample_rate'] = data_dict["sample_rate"]
    file_data.axes[0]['time_axis'].attrs['time_zone'] = old_obj["data_ieeg"].axes[0]["time_axis"].attrs["time_zone"]

    vs_label = np.array([contact['name'].split('-') for op in data_dict["vchangrp"] for contact in op.electrode_contacts])
    vs_coord = np.array([contact['coord'] for op in data_dict["vchangrp"] for contact in op.electrode_contacts])
    vs_pairs = np.array([(contact['anode_index'], contact['cathode_index']) for op in data_dict["vchangrp"] for contact in op.electrode_contacts])

    file_data.axes[1]['vlabel_axis'].append(vs_label)
    file_data.axes[1]['vcoord_axis'].append(vs_coord)
    file_data.axes[1]['vchannel_axis'].resize((vs_label.shape[0], old_obj["data_ieeg"].shape[1]))
    for ix in range(vs_label.shape[0]):
        for iy in range(old_obj["data_ieeg"].shape[1]):
            if iy in [idx["index"] for idx in vs_pairs[ix][0]]:
                an_val = 1
            else:
                an_val = 0

            if iy in [idx["index"] for idx in vs_pairs[ix][1]]:
                ct_val = 1
            else:
                ct_val = 0

            file_data.axes[1]['vchannel_axis'][ix, iy] = (None, None, an_val, ct_val)
