import numpy as np


def deg2rad(values):
    return np.pi / 180.0 * np.array(values, copy=None)


def dict2array(obj, dtype=None):
    return np.array(list(v for v in obj.values()), dtype=dtype)
