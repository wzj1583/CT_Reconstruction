from struct import *
from numpy import np


def read_data(path, fmt, start, end):
    with open(path, 'rb') as f:
        br = f.read()
        data = unpack(fmt, br[start:end])
        f.close()
    return data


def origin2CT(origin_data, n_width, n_proj_num, n_height):
    n_proj_step = n_width / float(n_proj_num)
    proj_data = np.zeros(n_proj_step*n_height, float)
    for i in range(0, n_height):
        for j in range(0, n_proj_num):
            pos = j * n_proj_step
            index = int(pos)
            a = pos - index
            if index < n_width - 1:
                proj_data[j * n_height + i] = origin_data[index * n_height + i] * (1 - a) + origin_data[(index + 1) * n_height + i] * a
            else:
                proj_data[j * n_height + i] = origin_data[index * n_height + i]
