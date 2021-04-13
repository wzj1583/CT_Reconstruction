import config
import time
import struct
import numpy as np

from utils.origin_data import DataFetcher
from utils.origin_data_single import _DataFetcher
from MIL.fan_CT import fan_CT_Single
from MIL.helticl_180 import helical_180IL_back_project

if __name__ == '__main__':
    start = time.clock()
    path = config.FIXED_CT_PATH4

    origin_data = _DataFetcher.create_data_fetcher(path)

    # graph_list = fan_CT_Single(origin_data, origin_data.pixel_num, origin_data.header.p_Width)
    graph_list = helical_180IL_back_project(origin_data, origin_data.pixel_num, origin_data.header.p_Width)
    end = time.clock()
    print(end - start)
