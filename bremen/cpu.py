"""Explicit CPU array backend; no implicit GPU fallback."""
import numpy as np
available = False
def get_array_module(*arrays):
    if any(not isinstance(x, np.ndarray) for x in arrays):
        raise TypeError("Only NumPy CPU arrays are supported")
    return np
def to_cpu(array):
    return np.asarray(array)
