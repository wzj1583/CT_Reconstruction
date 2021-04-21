import os
import config
from utils.origin_data import DataFetcher
from utils.data_utils import save_interpolated_data

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


def origin2CT(origin_data, n_width, n_proj_num, n_height):
    n_proj_step = n_width / float(n_proj_num)
    proj_data = np.zeros(n_proj_num*n_height, float)
    for i in range(0, n_height):
        for j in range(0, n_proj_num):
            pos = j * n_proj_step
            index = int(pos)
            a = pos - index
            if index < n_width - 1:
                proj_data[j * n_height + i] = origin_data[index * n_height + i] * (1 - a) + origin_data[(index + 1) * n_height + i] * a
            else:
                proj_data[j * n_height + i] = origin_data[index * n_height + i]
    return proj_data