import time
import numpy as np
from numpy import pi
from numba import cuda, float64

import config
from utils.core import conv2
from utils.filter import get_filter_SL
from utils.origin_data_single import _DataFetcher


@cuda.jit
def cuda_conv2(buf, h, res, len):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i < len:
        index = cuda.blockIdx.y * len
        tmp = 0
        for j in range(0, i):
            tmp += buf[index + j] * h[i - j]
        for j in range(i, len):
            tmp += buf[index + j] * h[j - i]

        cuda.syncthreads()

        res[index + i] = tmp


def main_conv2(buf, h):
    n_det_num = len(h)
    total = len(buf)
    length = int(len(buf) / n_det_num)
    dev_buf = cuda.to_device(buf)
    dev_h = cuda.to_device(h)
    dev_res = cuda.device_array(total)
    block_dim = (2, length)
    thread_dim = int(n_det_num/2)
    cuda_conv2[block_dim, thread_dim](dev_buf, dev_h, dev_res, n_det_num)
    cuda.synchronize()
    res = dev_res.copy_to_host()
    return res


def conv_verify():
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

    start = time.time()
    gpu_res = main_conv2(proj_float, h)
    end = time.time()
    gpu_time = end - start


    start = time.time()
    cpu_res = np.zeros(proj_float.shape, float)
    for i in range(0, n_proj_num):
        srt_idx = i * n_det_num
        end_idx = srt_idx + n_det_num
        temp = proj_float[srt_idx:end_idx]
        cpu_res[srt_idx:end_idx] = conv2(temp, h)
    end = time.time()
    cpu_time = end - start

    print('CPU: ' + str(cpu_time))


    print('GPU: ' + str(gpu_time))

    return 0





