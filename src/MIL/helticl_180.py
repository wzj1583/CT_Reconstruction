from numpy import pi, arctan, arcsin, sqrt, power
from MIL.helical_180_core import helical_180IL
from MIL.fan_CT_core import back_proj_rotate_pixel, _back_proj, back_proj_rotate_src
from utils.core import _get_FOV_XY
from utils.image_save import *
from utils.core import *
from utils.data_utils import origin2CT
from utils.filter import get_filter_RL, get_filter_SL
from utils.origin_data import DataFetcher
from utils.origin_data_single import _DataFetcher
from utils.fan_adj import adj_fan, adj_fan_fine, adj_fan_fine_with_factor

def helical_180IL_back_project(origin_data: _DataFetcher, n_image_size, n_proj_num):
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
    d_det_ang = f_det_width / f_det2src  # 投影线间角
    d_proj_ang = 2 * pi / n_proj_num  # 投影角步长
    cent_ang = f_cent_pos * d_det_ang  # 中心投影线角度
    fan_ang = n_det_num * d_det_ang  # 扇形束夹角

    f_proj_data = _DataFetcher.image_pre(f_proj_data, f_zero, f_full, n_width, n_height)

    h = get_filter_SL(d_det_ang, n_det_num)

    X_0, Y_0 = get_FOV_XY(n_image_size, f_FOV)  # rotate_pixel
    # X_0, Y_0 = _get_FOV_XY(n_image_size, f_FOV)     # rotate_src
    sin_list, cos_list = sincos(n_proj_num, -d_proj_ang)

    res_list = []

    f_proj_data_list = helical_180IL(f_proj_data, n_det_num, fan_ang, cent_ang, n_proj_num, [pi/4, pi/2, pi*3/4, pi, pi*5/4, pi*6/4, pi*7/4])
    for f_proj_data in f_proj_data_list:
        ct_image = np.zeros((n_image_size, n_image_size), float)
        for i in range(0, n_proj_num):
            srt_idx = i * n_det_num
            end_idx = srt_idx + n_det_num
            temp = f_proj_data[srt_idx:end_idx]
            f_proj_data[srt_idx:end_idx] = adj_fan_fine(temp, d_det_ang, cent_ang, f_cen2src, n_det_num)
            temp = f_proj_data[srt_idx:end_idx]
            f_proj_data[srt_idx:end_idx] = conv2(temp, h)
            # ct_image += back_proj_rotate_src(proj_float[srt_idx:end_idx], n_image_size, X_0, Y_0, sin_list[i], cos_list[i], f_cen2src, n_det_num, det_ang, f_cent_pos)
            ct_image += back_proj_rotate_pixel(f_proj_data[srt_idx:end_idx], n_image_size, X_0, Y_0, sin_list[i],
                                               cos_list[i], f_cen2src, n_det_num, d_det_ang, f_cent_pos)
        #ct_image = smooth_filter(ct_image, 2)
        ct_image = sharpen_filter(ct_image, 2)
        res_list.append(ct_image)
    save_image(res_list)
    return res_list

def helical_180MIL_back_project(prev_data: DataFetcher, rear_data: DataFetcher, n_image_size, n_proj_num):
    f_pixel_len = prev_data.pixel_len  # 重建图像像素宽度 1.5mm
    n_pixel_num = prev_data.pixel_num  # 重建图像分辨率 512
    n_det_num = prev_data.det_config.det_num  # 探测器数量 460
    f_cen2src = prev_data.det_config.det2cen  # 射线源-旋转中心距离 700.0mm
    f_det2src = prev_data.det_config.det2scr  # 射线源-探测器距离 1300.0mm
    f_cent_pos = prev_data.det_config.cents  # 中心投影线序号 229.5
    det_lar = prev_data.det_config.det_lar  # 探测器层数 8
    f_det_width = prev_data.det_config.det_width  # 探测器宽度 3.2mm
    f_FOV = f_pixel_len * n_pixel_num  # 视场
    d_det_ang = f_det_width / f_det2src  # 投影线间角
    d_proj_ang = 2 * pi / n_proj_num  # 投影角步长 1000
    cent_ang = f_cent_pos * d_det_ang  # 中心投影线角度
    fan_ang = n_det_num * d_det_ang  # 扇形束夹角

    f_prev_proj_data = prev_data.data_buff  # 投影数据
    f_prev_zero_a = prev_data.zero_a     # 暗场数据
    f_prev_zero_b = prev_data.zero_b
    f_prev_full_a = prev_data.empty_a    # 亮场数据
    f_prev_full_b = prev_data.empty_b
    n_prev_width = prev_data.header.nWidth  # 1000+
    n_prev_height = prev_data.header.nHeight  # 8*460

    f_rear_proj_data = prev_data.data_buff  # 投影数据
    f_rear_zero_a = prev_data.zero_a  # 暗场数据
    f_rear_zero_b = prev_data.zero_b
    f_rear_full_a = prev_data.empty_a  # 亮场数据
    f_rear_full_b = prev_data.empty_b
    n_rear_width = prev_data.header.nWidth  # 1000+
    n_rear_height = prev_data.header.nHeight  # 8*460

    f_prev_proj_data = DataFetcher.image_pre(f_prev_proj_data, f_prev_zero_a, f_prev_zero_b, f_prev_full_a, f_prev_full_b, n_prev_width, n_prev_height)
    f_rear_proj_data = DataFetcher.image_pre(f_rear_proj_data, f_rear_zero_a, f_rear_zero_b, f_rear_full_a, f_rear_full_b, n_rear_width, n_rear_height)
    if n_prev_width != n_proj_num:
        f_prev_proj_data = origin2CT(f_prev_proj_data, n_prev_width, n_proj_num, n_prev_height)
    if n_rear_width != n_proj_num:
        f_rear_proj_data = origin2CT(f_rear_proj_data, n_rear_width, n_proj_num, n_rear_height)

    h = get_filter_SL(d_det_ang, n_det_num)

    X_0, Y_0 = get_FOV_XY(n_image_size, f_FOV)  # rotate_pixel
    # X_0, Y_0 = _get_FOV_XY(n_image_size, f_FOV)     # rotate_src
    sin_list, cos_list = sincos(n_proj_num, -d_proj_ang)