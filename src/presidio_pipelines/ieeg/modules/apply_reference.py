"""apply_reference.py
"""

from collections.abc import Mapping
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import numpy.typing as npt
from zappy.geometry import electrodes


def construct_zappy_electrodes(channel_labels, channel_coords):
    """Create ElectrodeContact and ElectrodeGroup objects."""

    electrode_groups = [
        electrodes.ElectrodeGroup(name=ch_grp,
                                  type='seeg',
                                  electrode_contacts=[electrodes.ElectrodeContact({'name': '-'.join(channel_labels[idx, :]),
                                                                                   'index': idx,
                                                                                   'coord': channel_coords[idx]})
                                                      for idx in np.flatnonzero(channel_labels[:,0] == ch_grp)])
	for ch_grp in np.unique(channel_labels[:,0])]

    return electrode_groups


        
def apply_reference(data_obj: Any, montage: str) -> Any:
    ch_group = construct_zappy_electrodes(data_obj.axes[data_obj.attrs['c_axis']]['channellabel_axis'][...].astype(str),
                                          data_obj.axes[data_obj.attrs['c_axis']]['channelcoord_axis'][...].astype(float))

    if montage == 'bipolar':
        v_group = electrodes.make_virtual_bipolar_channels(ch_group)
        v_signal = electrodes.make_virtual_signal(data_obj[...], v_group)

    return v_signal, v_group
