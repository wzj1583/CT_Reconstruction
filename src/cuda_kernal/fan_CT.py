import config
import time
import numpy as np

from numba import cuda
from numpy import pi
from utils.origin_data_single import _DataFetcher
from utils.image_save import save_image
from utils.filter import get_filter_SL
from utils.fan_adj import adj_fan_fine
from utils.core import get_FOV_XY, sincos
from MIL.helical_180_core import helical_180IL

from cuda_kernal.cuda_conv import cuda_conv2
from cuda_kernal.back_project import cuda_back_proj, cuda_back_proj_modify


def fan_CT_acc(image_size):
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
    fan_ang = n_det_num * det_ang  # 扇形束夹角

    start = time.time()

    proj_float = _DataFetcher.image_pre(f_proj_data, f_zero, f_full, n_width, n_height)
    h = get_filter_SL(det_ang, n_det_num)

    f_proj_data_list = helical_180IL(proj_float, n_det_num, fan_ang, cent_ang, n_proj_num, [pi])

    res_list = []
    for proj_float in f_proj_data_list:
        for i in range(0, n_proj_num):
            srt_idx = i * n_det_num
            end_idx = srt_idx + n_det_num
            temp = proj_float[srt_idx:end_idx]
            proj_float[srt_idx:end_idx] = adj_fan_fine(temp, det_ang, cent_ang, f_cen2src, n_det_num)

        total_length = n_det_num * n_proj_num
        dev_buf = cuda.to_device(proj_float)
        dev_h = cuda.to_device(h)
        dev_proj_data = cuda.device_array(total_length)
        block_dim = (2, n_proj_num)
        thread_dim = int(n_det_num / 2)
        cuda_conv2[block_dim, thread_dim](dev_buf, dev_h, dev_proj_data, n_det_num)
        cuda.synchronize()
        proj_data = dev_proj_data.copy_to_host()

        dev_proj_data = cuda.to_device(proj_data)
        X_0, Y_0 = get_FOV_XY(image_size, f_FOV)
        sin_list, cos_list = sincos(n_proj_num, -proj_ang)
        sin_array = np.array(sin_list)
        cos_array = np.array(cos_list)
        dev_X_0 = cuda.to_device(X_0)
        dev_Y_0 = cuda.to_device(Y_0)
        dev_sin_array = cuda.to_device(sin_array)
        dev_cos_array = cuda.to_device(cos_array)
        '''
        dev_ct_image = cuda.device_array((image_size, image_size), float)
        cuda_back_proj[image_size, image_size](dev_proj_data, dev_ct_image, dev_X_0, dev_Y_0, dev_sin_array,
                                                                    dev_cos_array, n_det_num, f_cen2src,
                                                                    det_ang, f_cent_pos, n_proj_num
                                                                    )
        ct_image = dev_ct_image.copy_to_host()
        '''
        ct_image = np.zeros((image_size, image_size), float)
        dev_ct_image = cuda.to_device(ct_image)
        cuda_back_proj_modify[(image_size, image_size), n_proj_num](dev_proj_data, dev_ct_image, dev_X_0, dev_Y_0,
                                                                    dev_sin_array, dev_cos_array, n_det_num, f_cen2src,
                                                                    det_ang, f_cent_pos, n_proj_num
                                                                    )
        ct_image = dev_ct_image.copy_to_host()

        res_list.append(ct_image)

    save_image(res_list)

    end = time.time()
    gpu_time = end - start
    print(gpu_time)

    return res_list