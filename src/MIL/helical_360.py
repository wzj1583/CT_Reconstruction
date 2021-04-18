import os
import config
from numpy import pi, arctan, arcsin, sqrt, power
from MIL.helical_360_core import helical_360MIL
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

def helical_360MIL_interpolate_single(index, parameters: ParaTransfer, interpolated_layer_num):
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

    file_path_prev = config.ORIGIN_INTERPOLATED_DATA_PATH + config.FILE_NAME_PREFIX1 + str(1000 + index) + ".bin"
    f_prev_proj_data = read_interpolated_data(file_path_prev)
    file_path_rear = config.ORIGIN_INTERPOLATED_DATA_PATH + config.FILE_NAME_PREFIX1 + str(1000 + index + 1) + ".bin"
    f_rear_proj_data = read_interpolated_data(file_path_rear)
    f_proj_data = np.append(f_prev_proj_data, f_rear_proj_data, axis=0)
    f_360MIL_proj = np.zeros(1, float)
    for n_cur_slice in range(0, interpolated_layer_num):
        offset = n_proj_num * n_cur_slice / n_layer_num
        offset = int(offset * n_det_num * n_layer_num)
        f_360MIL_proj_slice = helical_360MIL(f_proj_data[offset:], n_det_num, fan_ang, cent_ang, n_proj_num, n_layer_num, f_layer_thick, f_pitch_layer, n_cur_slice)
        f_360MIL_proj = np.append(f_360MIL_proj, f_360MIL_proj_slice)
    interpolated_save_path = config._360MIL_INTERPOLATED_DATA_PATH + config.FILE_NAME_PREFIX1 + str(1000 + index) + ".bin"
    save_interpolated_data(interpolated_save_path, f_360MIL_proj[1:])
    return interpolated_save_path