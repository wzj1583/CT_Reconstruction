import config
import os
import time
import struct
import numpy as np

from utils.origin_data import Header, DataFetcher
from utils.origin_data_single import _DataFetcher
from utils.processed_data import ParaTransfer
from utils.data_utils import save_interpolated_data, read_interpolated_data
from MIL.helticl_180 import helical_180IL_back_project_test, helical_180MIL_interpolate_single, origin2CT_single
from MIL.helical_360 import helical_360MIL_interpolate_single
from MIL.fan_CT import fan_CT

if __name__ == '__main__':
    image_size = 512
    index = 30
    is_origin = 0
    interpolated_layer_num = 8

    origin_path = config.HELICAL_FAN_CT_PATH1 + config.FILE_NAME_PREFIX1
    interpolated_path = config.ORIGIN_INTERPOLATED_DATA_PATH + config.FILE_NAME_PREFIX1
    MIL_path = config.ORIGIN_INTERPOLATED_DATA_PATH + config.FILE_NAME_PREFIX1

    origin_path = config.HELICAL_FAN_CT_PATH1 + config.FILE_NAME_PREFIX1 + str(1000 + index) + ".DAT"
    header, header_content = Header.read_header(origin_path)

    #interpolated_path = origin2CT_single(index, 1000)
    #interpolated_path = origin2CT_single(index + 1, 1000)

    # 180MIL
    f_proj_data, parameters = ParaTransfer.pack_para_transfer(index, is_origin=0)
    MIL_path = helical_180MIL_interpolate_single(index, parameters, interpolated_layer_num)
    MIL_path = MIL_path[0:-8]

    # 360MIL
    #f_proj_data, parameters = ParaTransfer.pack_para_transfer(index, is_origin=0)
    #MIL_path = helical_360MIL_interpolate_single(index, parameters, interpolated_layer_num)
    #MIL_path = MIL_path[0:-8]

    if is_origin == 1:
        f_proj_data, parameters = ParaTransfer.pack_para_transfer(index, is_origin=1)
        res_list = helical_180IL_back_project_test(f_proj_data, parameters, image_size)
        '''
        f_proj_data = f_proj_data.reshape(parameters.n_proj_num, -1)
        f_proj_data = f_proj_data[:, 3 * 460:4 * 460]
        f_proj_data = f_proj_data.reshape(-1, 1)
        f_proj_data = np.squeeze(f_proj_data)
        res_list = fan_CT(f_proj_data, image_size, parameters.n_proj_num)
        '''
    else:
        path = MIL_path
        f_proj_data, parameters = ParaTransfer.pack_para_transfer(index, path=path, is_origin=0)
        if path == interpolated_path:
            res_list = helical_180IL_back_project_test(f_proj_data, parameters, image_size)
        else:
            f_proj_layer_data = f_proj_data[0:460 * 1000]
            res_list = fan_CT(f_proj_layer_data, image_size, 1000)

    print('done')