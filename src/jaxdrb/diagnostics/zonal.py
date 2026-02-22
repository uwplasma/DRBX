from __future__ import annotations

import numpy as np


def zonal_mean(field, *, axis: int = 1) -> np.ndarray:
    """Return zonal mean over the binormal axis."""
    arr = np.asarray(field)
    return np.mean(arr, axis=axis)
