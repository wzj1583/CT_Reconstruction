import numpy as np
from numpy import sin, cos


def conv2_all(buf, h):
    tmp = h[::-1]
    h = np.append(tmp, h[1:])
    buf_len = len(buf)
    h_len = len(h)
    res_len = buf_len + h_len - 1
    res = np.zeros(res_len)
    h = np.array(h)[::-1]
    for n in range(0, res_len):
        for m in range(h_len):
            k = n - h_len + m + 1
            if 0 <= k < buf_len:
                res[n] += buf[k] * h[m]
    if h_len % 2 == 0:
        i = int((h_len - 2) / 2)
    else:
        i = int((h_len - 1) / 2)
    return res[i:i + buf_len]


def conv2(buf, h):
    N = len(buf)
    res = np.zeros(N, float)
    for i in range(0, N):
        for j in range(0, N):
            if i-j < 0:
                res[i] += buf[j]*h[j-i]
            if i-j >= 0:
                res[i] += buf[j]*h[i-j]
    return res


def sincos(proj_num, dbeta):
    sin_list = []
    cos_list = []
    for i in range(0, proj_num):
        sin_list.append(sin(i * dbeta))
        cos_list.append(cos(i * dbeta))
    return sin_list, cos_list


def get_FOV_XY(image_size, f_FOV):
    f_pixel_len = np.float(f_FOV / image_size)
    tmp = np.zeros((image_size, 1), float)
    for i in range(0, image_size):
        tmp[i][0] = i - np.float((image_size - 1)) / 2
    XY_map = tmp
    for i in range(1, image_size):
        XY_map = np.append(XY_map, tmp, axis=1)
    X_0 = f_pixel_len * XY_map.T
    Y_0 = - f_pixel_len * XY_map
    return X_0, Y_0


def _get_FOV_XY(image_size, FOV):
    pixel_len = np.float(FOV) / image_size
    tmp = np.zeros((image_size, 1), float)
    for i in range(0, image_size):
        tmp[i][0] = i - np.float(image_size - 1) / 2
    image2FOV = tmp
    for j in range(1, image_size):
        image2FOV = np.append(image2FOV, tmp, axis=1)
    X = image2FOV * pixel_len
    Y = image2FOV.T * pixel_len
    return X, Y