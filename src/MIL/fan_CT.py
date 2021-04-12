from numpy import pi, arctan, arcsin, sqrt, power
from MIL.fan_CT_core import back_proj_rotate_pixel, _back_proj, back_proj_rotate_src
from utils.core import _get_FOV_XY
from utils.image_save import *
from utils.core import *
from utils.filter import get_filter_RL, get_filter_SL
from utils.origin_data_single import _DataFetcher
from utils.fan_adj import adj_fan, adj_fan_fine, adj_fan_fine_with_factor


def fan_CT_Single(origin_data: _DataFetcher, n_image_size, n_proj_num):
    f_proj_data = origin_data.data_buff  # 投影数据
    f_zero = origin_data.zero  # 暗场数据
    f_full = origin_data.full  # 亮场数据
    n_width = origin_data.header.p_Width    #1000
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


    #sin_list, cos_list = sincos(n_proj_num, proj_ang)
    sin_list, cos_list = sincos(n_proj_num, -proj_ang)

    res_list = []
    ct_image = np.zeros((n_image_size, n_image_size), float)
    for i in range(0, n_proj_num):
        srt_idx = i * n_det_num
        end_idx = srt_idx + n_det_num
        temp = proj_float[srt_idx:end_idx]
        proj_float[srt_idx:end_idx] = adj_fan_fine(temp, det_ang, cent_ang, f_cen2src, n_det_num)

    #proj_float = conv2_all(proj_float, h)

    X_0, Y_0 = get_FOV_XY(n_image_size, f_FOV)     # rotate_pixel
    #X_0, Y_0 = _get_FOV_XY(n_image_size, f_FOV)     # rotate_src
    for i in range(0, n_proj_num):
        srt_idx = i * n_det_num
        end_idx = srt_idx + n_det_num
        temp = proj_float[srt_idx:end_idx]

        proj_float[srt_idx:end_idx] = conv2(temp, h)
        #ct_image += _back_proj(proj_float[srt_idx:end_idx], n_image_size, f_FOV, sin_list[i], cos_list[i], f_cen2src, n_det_num, det_ang, f_cent_pos)

        #ct_image += back_proj_rotate_src(proj_float[srt_idx:end_idx], n_image_size, X_0, Y_0, sin_list[i], cos_list[i], f_cen2src, n_det_num, det_ang, f_cent_pos)

        ct_image += back_proj_rotate_pixel(proj_float[srt_idx:end_idx], n_image_size, X_0, Y_0, sin_list[i], cos_list[i], f_cen2src, n_det_num, det_ang, f_cent_pos)


    res_list.append(ct_image)

    save_image(res_list)
    return res_list






