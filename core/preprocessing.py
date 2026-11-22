import numpy as np


def standardize(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not x.size:
        raise ValueError("需要非空二维特征矩阵")
    centered = x - x.mean(axis=0)
    scale = x.std(axis=0)
    constant = np.ptp(x, axis=0) == 0
    scale[constant | (scale == 0)] = 1
    centered[:, constant] = 0
    return centered / scale
