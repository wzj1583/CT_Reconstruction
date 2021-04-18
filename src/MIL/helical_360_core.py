import os
import config
import numpy as np
from numpy import pi, fabs
from utils.origin_data import DataFetcher
from utils.data_utils import save_interpolated_data


def helical_360MIL(f_proj_data, n_det_num, fan_ang, cent_ang, n_proj_num, n_layer_num, f_layer_thick, f_pitch_layer,
                   n_cur_slice):
    inpolated_proj_data = np.zeros(n_proj_num * n_det_num, float)

    inter_start_idx = n_cur_slice * n_proj_num / n_layer_num
    f_pitch_proj = f_pitch_layer * n_layer_num
    f_cur_slice_z = (f_pitch_proj + n_layer_num * f_layer_thick) / 2
    for i in range(0, n_proj_num):
        cur_proj_idx = i
        f_cur_bottom_layer_z = f_pitch_proj * cur_proj_idx / n_proj_num + f_layer_thick / 2
        n_cur_inter_start_idx = int(((cur_proj_idx + inter_start_idx) % n_proj_num) * n_det_num)
        cur_orig_start_idx = cur_proj_idx * n_det_num * n_layer_num

        origin_idx = n_layer_num - 1
        for k in range(0, n_layer_num):
            cur_layer_z = f_cur_bottom_layer_z + k*f_layer_thick
            if f_cur_slice_z - cur_layer_z < 0:
                if k > 0:
                    origin_idx = k - 1
                else:
                    origin_idx = 0
                break
        cur_layer_z = f_cur_bottom_layer_z + origin_idx*f_layer_thick
        w = (f_cur_slice_z - cur_layer_z)/f_layer_thick
        if w > 1:
            w = 1
        if w < 0:
            w = 0

        for j in range(0, n_det_num):
            if origin_idx < n_layer_num - 1:
                inpolated_proj_data[n_cur_inter_start_idx + j] = (1-w)*f_proj_data[cur_orig_start_idx+origin_idx*n_det_num+j] + w*f_proj_data[cur_orig_start_idx + (origin_idx+1)*n_det_num+j]
            else:
                inpolated_proj_data[n_cur_inter_start_idx + j] = f_proj_data[cur_orig_start_idx+origin_idx*n_det_num+j]
    return inpolated_proj_data