import os
import config
from numpy import pi, arctan, arcsin, sqrt, power
from MIL.helical_180_core import helical_180IL, helical_180MIL, origin2CT
from MIL.fan_CT_core import back_proj_rotate_pixel, _back_proj, back_proj_rotate_src
from utils.core import _get_FOV_XY
from utils.image_save import *
from utils.core import *
from utils.filter import get_filter_RL, get_filter_SL
from utils.processed_data import ParaTransfer
from utils.origin_data import Header, DetConfig, DataFetcher
from utils.origin_data_single import _DataFetcher
from utils.fan_adj import adj_fan, adj_fan_fine, adj_fan_fine_with_factor
from utils.data_utils import save_interpolated_data, read_interpolated_data
from MIL.fan_CT import fan_CT

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

def helical_180MIL_back_project(n_image_size, n_proj_num):
    path = config.HELICAL_FAN_CT_PATH1 + config.FILE_NAME_PREFIX1 + '1001' + ".DAT"
    origin_data = DataFetcher.create_data_fetcher(path)
    interpolated_layer_num = 8

    f_pixel_len = origin_data.pixel_len  # 重建图像像素宽度 1.5mm
    n_pixel_num = origin_data.pixel_num  # 重建图像分辨率 512
    n_det_num = origin_data.det_config.det_num  # 探测器数量 460
    f_cen2src = origin_data.det_config.det2cen  # 射线源-旋转中心距离 700.0mm
    f_det2src = origin_data.det_config.det2scr  # 射线源-探测器距离 1300.0mm
    f_cent_pos = origin_data.det_config.cents  # 中心投影线序号 229.5
    n_layer_num = origin_data.det_config.det_lar  # 探测器层数 8
    f_det_width = origin_data.det_config.det_width  # 探测器宽度 3.2mm
    f_layer_thick = origin_data.det_config.det_height
    f_pitch_layer = (origin_data.header.fStart - origin_data.header.fend)/n_layer_num
    f_FOV = f_pixel_len * n_pixel_num  # 视场
    d_det_ang = f_det_width / f_det2src  # 投影线间角
    d_proj_ang = 2 * pi / n_proj_num  # 投影角步长 1000
    cent_ang = f_cent_pos * d_det_ang  # 中心投影线角度
    fan_ang = n_det_num * d_det_ang  # 扇形束夹角

    # batch generate interpolated projection data
    origin2CT_batch(n_proj_num)

    # generate 180MIL projection data
    origin_interpolated_path = config.ORIGIN_INTERPOLATED_DATA_PATH + '\\'
    file_num = len(os.listdir(origin_interpolated_path))
    for i in range(1001, 1001 + file_num):
        file_path_prev = config.ORIGIN_INTERPOLATED_DATA_PATH + config.FILE_NAME_PREFIX1 + str(i) + ".bin"
        f_prev_proj_data = read_interpolated_data(file_path_prev)
        file_path_rear = config.ORIGIN_INTERPOLATED_DATA_PATH + config.FILE_NAME_PREFIX1 + str(i + 1) + ".bin"
        f_rear_proj_data = read_interpolated_data(file_path_rear)
        f_proj_data = np.append(f_prev_proj_data, f_rear_proj_data, axis=0)
        for n_cur_slice in range(0, interpolated_layer_num):
            f_180MIL_proj_slice = helical_180MIL(f_proj_data, n_det_num, fan_ang, cent_ang, n_proj_num, n_layer_num,
                                                f_layer_thick, f_pitch_layer, n_cur_slice)
            interpolated_save_path = config._180MIL_INTERPOLATED_DATA_PATH + config.FILE_NAME_PREFIX1 + str(i) + '-' + str(n_cur_slice + 1) + ".bin"
            save_interpolated_data(interpolated_save_path, f_180MIL_proj_slice)

    # read 180MIL interpolated data
    interpolated_save_path = config._180MIL_INTERPOLATED_DATA_PATH + '\\'
    file_num = len(os.listdir(interpolated_save_path))
    for i in range(1001, 1001 + file_num):
        interpolated_save_path = config._180MIL_INTERPOLATED_DATA_PATH + config.FILE_NAME_PREFIX1 + str(i) + ".bin"
        f_proj_data = read_interpolated_data(interpolated_save_path)
        for j in range(0, interpolated_layer_num):
            f_proj_layer_data = f_proj_data[j*n_proj_num*n_det_num: (j+1)*n_proj_num*n_det_num]
            ct_image = fan_CT(f_proj_layer_data, n_image_size, n_proj_num)
            res = []
            res.append(ct_image)
            save_image(res)
    return 0


def helical_180IL_back_project_test(f_proj_data, parameters, n_image_size):
    n_det_num = parameters.n_det_num
    f_cent_pos = parameters.f_cent_pos
    n_proj_num = parameters.n_proj_num
    f_FOV = parameters.f_FOV

    d_det_ang = parameters.d_det_ang
    d_proj_ang = parameters.d_proj_ang
    fan_ang = parameters.fan_ang
    cent_ang = parameters.cent_ang
    f_cen2src = parameters.f_cen2src

    h = get_filter_SL(d_det_ang, n_det_num)

    X_0, Y_0 = get_FOV_XY(n_image_size, f_FOV)  # rotate_pixel
    # X_0, Y_0 = _get_FOV_XY(n_image_size, f_FOV)     # rotate_src
    sin_list, cos_list = sincos(n_proj_num, d_proj_ang)

    res_list = []

    f_proj_data = f_proj_data.reshape(n_proj_num, -1)
    f_proj_data = f_proj_data[:, 3*n_det_num:4*n_det_num]
    f_proj_data = f_proj_data.reshape(-1, 1)
    f_proj_data = np.squeeze(f_proj_data)

    f_proj_data_list = helical_180IL(f_proj_data, n_det_num, fan_ang, cent_ang, n_proj_num, [pi])
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
            if (i + 1) % 100 == 0:
                print(str(i + 1) + '/' + str(n_proj_num))
        # ct_image = smooth_filter(ct_image, 2)
        ct_image = sharpen_filter(ct_image, 2)
        res_list.append(ct_image)
    save_image(res_list)
    return res_list


def origin2CT_batch(n_proj_num):
    data_path = config.HELICAL_FAN_CT_PATH1 + '\\'
    file_num = len(os.listdir(data_path))
    for i in range(1001, 1001 + file_num):
        path = config.HELICAL_FAN_CT_PATH1 + config.FILE_NAME_PREFIX1 + str(i) + ".DAT"
        origin_data = DataFetcher.create_data_fetcher(path)
        f_origin_data = origin_data.data_buff
        f_zero_a = origin_data.zero_a
        f_zero_b = origin_data.zero_b
        f_full_a = origin_data.empty_a
        f_full_b = origin_data.empty_b
        n_width = origin_data.header.nWidth
        n_height = origin_data.header.nHeight
        f_prev_proj_data = DataFetcher.image_pre(f_origin_data, f_zero_a, f_zero_b, f_full_a, f_full_b, n_width,
                                                     n_height)
        f_prev_proj_data = origin2CT(f_prev_proj_data, n_width, n_proj_num, n_height)
        origin_interpolated_path = config.ORIGIN_INTERPOLATED_DATA_PATH + config.FILE_NAME_PREFIX1 + str(i) + ".bin"
        save_interpolated_data(origin_interpolated_path, f_prev_proj_data)
        print("Interpolate: " + str(i) + '/' + str(file_num))
    return 0


def origin2CT_single(index, n_proj_num):
    path = config.HELICAL_FAN_CT_PATH1 + config.FILE_NAME_PREFIX1 + str(1000 + index) + ".DAT"
    origin_data = DataFetcher.create_data_fetcher(path)
    f_origin_data = origin_data.data_buff
    f_zero_a = origin_data.zero_a
    f_zero_b = origin_data.zero_b
    f_full_a = origin_data.empty_a
    f_full_b = origin_data.empty_b
    n_width = origin_data.header.nWidth
    n_height = origin_data.header.nHeight
    f_prev_proj_data = DataFetcher.image_pre(f_origin_data, f_zero_a, f_zero_b, f_full_a, f_full_b, n_width, n_height)
    if n_width != n_proj_num:
        f_prev_proj_data = origin2CT(f_prev_proj_data, n_width, n_proj_num, n_height)
    origin_interpolated_path = config.ORIGIN_INTERPOLATED_DATA_PATH + config.FILE_NAME_PREFIX1 + str(
            1000 + index) + ".bin"
    save_interpolated_data(origin_interpolated_path, f_prev_proj_data)
    return origin_interpolated_path


def helical_180MIL_interpolate_batch(parameters: ParaTransfer, interpolated_layer_num):
    n_det_num = parameters.n_det_num
    f_cent_pos = parameters.f_cent_pos
    n_proj_num = parameters.n_proj_num
    f_FOV = parameters.f_FOV

    d_det_ang = parameters.d_det_ang
    d_proj_ang = parameters.d_proj_ang
    fan_ang = parameters.fan_ang
    cent_ang = parameters.cent_ang
    f_cen2src = parameters.f_cen2src
    f_layer_thick = parameters.f_layer_thick
    n_layer_num = parameters.n_layer_num
    f_pitch_layer = parameters.f_pitch_layer

    origin_interpolated_path = config.ORIGIN_INTERPOLATED_DATA_PATH + '\\'
    file_num = len(os.listdir(origin_interpolated_path))
    for i in range(1001, 1001 + file_num):
        file_path_prev = config.ORIGIN_INTERPOLATED_DATA_PATH + config.FILE_NAME_PREFIX1 + str(i) + ".bin"
        f_prev_proj_data = read_interpolated_data(file_path_prev)
        file_path_rear = config.ORIGIN_INTERPOLATED_DATA_PATH + config.FILE_NAME_PREFIX1 + str(i + 1) + ".bin"
        f_rear_proj_data = read_interpolated_data(file_path_rear)
        f_proj_data = np.append(f_prev_proj_data, f_rear_proj_data, axis=0)
        for n_cur_slice in range(0, interpolated_layer_num):
            offset = n_proj_num * n_cur_slice / n_layer_num
            offset = int(offset * n_det_num * n_layer_num)
            f_180MIL_proj_slice = helical_180MIL(f_proj_data[offset:], n_det_num, fan_ang, cent_ang, n_proj_num, n_layer_num,
                                                 f_layer_thick, f_pitch_layer, n_cur_slice)
            interpolated_save_path = config._180MIL_INTERPOLATED_DATA_PATH + config.FILE_NAME_PREFIX1 + str(
                i) + '-' + str(n_cur_slice + 1) + ".bin"
            save_interpolated_data(interpolated_save_path, f_180MIL_proj_slice)
    return 0


def helical_180MIL_interpolate_single(index, parameters: ParaTransfer, interpolated_layer_num):
    n_det_num = parameters.n_det_num
    f_cent_pos = parameters.f_cent_pos
    n_proj_num = parameters.n_proj_num
    f_FOV = parameters.f_FOV

    d_det_ang = parameters.d_det_ang
    d_proj_ang = parameters.d_proj_ang
    fan_ang = parameters.fan_ang
    cent_ang = parameters.cent_ang
    f_cen2src = parameters.f_cen2src
    f_layer_thick = parameters.f_layer_thick
    n_layer_num = parameters.n_layer_num
    f_pitch_layer = parameters.f_pitch_layer

    origin_interpolated_path = config.ORIGIN_INTERPOLATED_DATA_PATH + '\\'
    file_path_prev = config.ORIGIN_INTERPOLATED_DATA_PATH + config.FILE_NAME_PREFIX1 + str(1000 + index) + ".bin"
    f_prev_proj_data = read_interpolated_data(file_path_prev)
    file_path_rear = config.ORIGIN_INTERPOLATED_DATA_PATH + config.FILE_NAME_PREFIX1 + str(1000 + index + 1) + ".bin"
    f_rear_proj_data = read_interpolated_data(file_path_rear)
    f_proj_data = np.append(f_prev_proj_data, f_rear_proj_data, axis=0)
    f_180MIL_proj = np.zeros(1, float)
    for n_cur_slice in range(0, interpolated_layer_num):
        f_180MIL_proj_slice = helical_180MIL(f_proj_data, n_det_num, fan_ang, cent_ang, n_proj_num, n_layer_num,
                                                 f_layer_thick, f_pitch_layer, n_cur_slice)
        f_180MIL_proj = np.append(f_180MIL_proj, f_180MIL_proj_slice)
    interpolated_save_path = config._180MIL_INTERPOLATED_DATA_PATH + config.FILE_NAME_PREFIX1 + str(1000 + index) + ".bin"
    save_interpolated_data(interpolated_save_path, f_180MIL_proj[1:])
    return interpolated_save_path