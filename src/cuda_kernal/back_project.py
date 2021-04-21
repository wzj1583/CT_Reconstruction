import numpy as np
from math import asin, sqrt
from numba import cuda
from utils.core import get_FOV_XY, sincos

@cuda.jit
def cuda_back_proj(proj_data, ct_image, X_0, Y_0, _sin, _cos, det_num, d_det_ang, cents, n_proj_num):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    for k in range(0, n_proj_num):
        offset = i * det_num
        x = X_0[i][j]*_cos[k] - Y_0[i][j]*_sin[k]
        y = X_0[i][j]*_sin[k] - Y_0[i][j]*_cos[k]
        L2 = x*x + y*y
        pixel_ang = asin(x / sqrt(L2))
        pixel_idx = pixel_ang / d_det_ang + cents
        if pixel_idx < 0:
            break
        if pixel_idx > det_num - 1:
            break
        pixel_det_idx = int(pixel_idx)
        pos = pixel_det_idx - pixel_idx
        ct_image[i][j] += ((1 - pos) * proj_data[offset + pixel_det_idx] + pos * proj_data[offset + pixel_det_idx + 1]) / L2
        ct_image[i][j] *= 10


def main_back_proj(proj_data, image_size, det_num, d_proj_ang, d_det_ang, cents, f_FOV, n_proj_num):
    X_0, Y_0 = get_FOV_XY(image_size, f_FOV)
    sin_list, cos_list = sincos(n_proj_num, d_proj_ang)
    sin_array = np.array(sin_list)
    cos_array = np.array(cos_list)

    f_pixel_len = float(f_FOV / image_size)
    dev_proj_data = cuda.to_device(proj_data)
    dev_X_0 = cuda.to_device(X_0)
    dev_Y_0 = cuda.to_device(Y_0)
    dev_sin_tuple = cuda.to_device(sin_array)
    dev_cos_tuple = cuda.to_device(cos_array)
    ct_image = cuda.device_array((image_size, image_size), float)

    res = cuda_back_proj[image_size, image_size](dev_proj_data, dev_X_0, dev_Y_0, det_num, d_det_ang, cents, n_proj_num)

