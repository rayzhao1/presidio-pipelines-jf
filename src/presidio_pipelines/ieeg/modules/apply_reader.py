"""apply_reader.py
"""

from typing import Any

def apply_reader(path: str, h5interface: Any) -> Any:
    """Opens the interface to the H5 schema containing raw, or very minimally processed data."""

    f_obj = h5interface(file=path, mode="r", load=True, swmr=True)
    return f_obj
