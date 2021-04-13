import config
import time
import struct
import numpy as np

from utils.origin_data import DataFetcher
from utils.origin_data_single import _DataFetcher
from MIL.fan_CT import fan_CT_Single
from MIL.helticl_180 import helical_180IL_back_project

if __name__ == '__main__':
    '''
    start = time.clock()
    path = config.FIXED_CT_PATH4

    origin_data = _DataFetcher.create_data_fetcher(path)

    # graph_list = fan_CT_Single(origin_data, origin_data.pixel_num, origin_data.header.p_Width)
    graph_list = helical_180IL_back_project(origin_data, origin_data.pixel_num, origin_data.header.p_Width)
    end = time.clock()
    print(end - start)
    '''

    path = config.HELICAL_FAN_CT_PATH1 + config.FILE_NAME_PREFIX1 + "01" + ".DAT"
    origin_data = DataFetcher.create_data_fetcher(path)

    f_origin_data = origin_data.data_buff
    f_zero_a = origin_data.zero_a
    f_zero_b = origin_data.zero_b
    f_full_a = origin_data.empty_a
    f_full_b = origin_data.empty_b
    n_width = origin_data.header.nWidth
    n_height = origin_data.header.nHeight
    f_prev_proj_data = DataFetcher.image_pre(f_origin_data, f_zero_a, f_zero_b, f_full_a, f_full_b, n_width, n_height)

    origin_inter_path = config.ORIGIN_INTERPOLATED_DATA_PATH + config.FILE_NAME_PREFIX1 + "01" + ".bin"
