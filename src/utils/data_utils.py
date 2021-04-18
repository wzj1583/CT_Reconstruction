from struct import *
import numpy as np


def read_data(path, fmt, start, end):
    with open(path, 'rb') as f:
        br = f.read()
        data = unpack(fmt, br[start:end])
        f.close()
    return data


def save_interpolated_data(path, proj_data_array):
    proj_data_array.tofile(path)
    return 0


def read_interpolated_data(path):
    proj_date_array = np.fromfile(path, float)
    return proj_date_array


