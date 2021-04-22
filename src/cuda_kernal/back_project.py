import time
import numpy as np
from numpy import pi
from numba import cuda
from math import asin, sqrt

import config
from utils.core import get_FOV_XY, sincos
from utils.filter import get_filter_SL
from utils.fan_adj import adj_fan_fine
from utils.origin_data_single import _DataFetcher
from MIL.fan_CT_core import back_proj_rotate_pixel

from cuda_kernal.cuda_conv import main_conv2


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
            ct_image[i][j] = 0
        elif pixel_idx > det_num - 1:
            ct_image[i][j] = 0
        else:
            pixel_det_idx = int(pixel_idx)
            pos = pixel_det_idx - pixel_idx
            ct_image[i][j] += ((1 - pos) * proj_data[offset + pixel_det_idx] + pos * proj_data[offset + pixel_det_idx + 1]) / L2
        ct_image[i][j] *= 10


@cuda.jit
def cuda_back_proj_modify(proj_data, ct_image, X_0, Y_0, _sin, _cos, det_num, d_det_ang, cents, n_proj_num):
    i = cuda.blockIdx.x
    j = cuda.blockIdx.y
    k = cuda.threadIdx.x
    offset = i * det_num
    x = X_0[i][j]*_cos[k] - Y_0[i][j]*_sin[k]
    y = X_0[i][j]*_sin[k] - Y_0[i][j]*_cos[k]
    L2 = x*x + y*y
    pixel_ang = asin(x / sqrt(L2))
    pixel_idx = pixel_ang / d_det_ang + cents
    if pixel_idx < 0:
        ct_image[i][j] = 0
    elif pixel_idx > det_num - 1:
        ct_image[i][j] = 0
    else:
        pixel_det_idx = int(pixel_idx)
        pos = pixel_det_idx - pixel_idx
        ct_image[i][j] += ((1 - pos) * proj_data[offset + pixel_det_idx] + pos * proj_data[offset + pixel_det_idx + 1]) / L2
    ct_image[i][j] *= 10


def main_back_proj(proj_data, image_size, det_num, d_proj_ang, d_det_ang, cents, f_FOV, n_proj_num):
    X_0, Y_0 = get_FOV_XY(image_size, f_FOV)
    sin_list, cos_list = sincos(n_proj_num, d_proj_ang)
    sin_array = np.array(sin_list)
    cos_array = np.array(cos_list)
    dev_proj_data = cuda.to_device(proj_data)
    dev_X_0 = cuda.to_device(X_0)
    dev_Y_0 = cuda.to_device(Y_0)
    dev_sin_array = cuda.to_device(sin_array)
    dev_cos_array = cuda.to_device(cos_array)
    ct_image = cuda.device_array((image_size, image_size), float)

    #cuda_back_proj[image_size, image_size](dev_proj_data, ct_image, dev_X_0, dev_Y_0, dev_sin_array, dev_cos_array, det_num, d_det_ang, cents, n_proj_num)

    cuda_back_proj_modify[(image_size, image_size), n_proj_num](dev_proj_data, ct_image, dev_X_0, dev_Y_0, dev_sin_array, dev_cos_array, det_num, d_det_ang, cents, n_proj_num)

    res = ct_image.copy_to_host()
    return res


def back_proj_verify(image_size):
    path = config.FIXED_CT_PATH4
    origin_data = _DataFetcher.create_data_fetcher(path)

    n_proj_num = 1000
    f_proj_data = origin_data.data_buff  # 投影数据
    f_zero = origin_data.zero  # 暗场数据
    f_full = origin_data.full  # 亮场数据
    n_width = origin_data.header.p_Width  # 1000
    n_height = origin_data.header.p_Height  # 448
    f_pixel_len = origin_data.pixel_len  # 重建图像像素宽度 0.8mm
    n_pixel_num = origin_data.pixel_num  # 重建图像分辨率 512
    n_det_num = origin_data.det_config.det_num  # 探测器数量 224
    f_cen2src = origin_data.det_config.det2cen  # 射线源-旋转中心距离 858.5mm
    f_det2src = origin_data.det_config.det2scr  # 射线源-探测器距离 1316.0mm
    f_cent_pos = origin_data.det_config.cents  # 中心投影线序号 111.6
    det_lar = origin_data.det_config.det_lar  # 探测器层数 1
    f_det_width = origin_data.det_config.det_width  # 探测器宽度 2.6mm
    f_FOV = f_pixel_len * n_pixel_num  # 视场

    det_ang = f_det_width / f_det2src  # 投影线间角
    proj_ang = 2 * pi / n_proj_num  # 投影角步长
    cent_ang = f_cent_pos * det_ang  # 中心投影线角度
    ang = n_det_num * det_ang  # 扇形束夹角

    proj_float = _DataFetcher.image_pre(f_proj_data, f_zero, f_full, n_width, n_height)
    h = get_filter_SL(det_ang, n_det_num)

    for i in range(0, n_proj_num):
        srt_idx = i * n_det_num
        end_idx = srt_idx + n_det_num
        temp = proj_float[srt_idx:end_idx]
        proj_float[srt_idx:end_idx] = adj_fan_fine(temp, det_ang, cent_ang, f_cen2src, n_det_num)

    proj_float = main_conv2(proj_float, h)


    start = time.time()
    gup_ct_image = main_back_proj(proj_float, image_size, n_det_num, proj_ang, det_ang, f_cent_pos, f_FOV, n_proj_num)
    end = time.time()
    gpu_time = end - start

    '''
    start = time.time()
    cpu_ct_image = np.zeros((image_size, image_size), float)
    X_0, Y_0 = get_FOV_XY(image_size, f_FOV)
    sin_list, cos_list = sincos(n_proj_num, -proj_ang)
    for i in range(0, n_proj_num):
        srt_idx = i * n_det_num
        end_idx = srt_idx + n_det_num
        cpu_ct_image += back_proj_rotate_pixel(proj_float[srt_idx:end_idx], image_size, X_0, Y_0, sin_list[i], cos_list[i], f_cen2src, n_det_num, det_ang, f_cent_pos)
    end = time.time()
    cpu_time = end - start
    
    print('CPU: ' + str(cpu_time))
    '''
    print('GPU: ' + str(gpu_time))

    return 0