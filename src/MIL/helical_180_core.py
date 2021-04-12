import numpy as np
from numpy import pi, fabs


def helical_180IL(f_proj_data, n_det_num, fan_ang, cent_ang, n_slice_num):
    d_slice_ang = 2*pi/n_slice_num
    d_det_ang = fan_ang / n_det_num
    res = np.zeros(n_slice_num*n_det_num, float)
    for i in range(0, n_slice_num):
        for j in range(0, n_det_num):
            cur_det_cent_ang = j*d_det_ang - cent_ang
            nxt_det_cent_ang = -cur_det_cent_ang+cent_ang
            cur_proj_ang = i*d_slice_ang
            nxt_proj_ang = cur_proj_ang+pi-2*d_det_ang if cur_proj_ang+pi-2*d_det_ang - 2*pi <= 0 else cur_proj_ang+pi-2*d_det_ang -2*pi
            cur_det_pos = j
            nxt_det_pos = int(nxt_det_cent_ang / d_det_ang)
            cur_proj_pos = i
            nxt_proj_pos = int(nxt_proj_ang / d_slice_ang)
            cur_proj_z = fabs(cur_proj_ang - pi)
            nxt_proj_z = fabs(nxt_proj_ang - pi)
            res[i * n_det_num + j] = (cur_proj_z*f_proj_data[nxt_proj_pos*n_det_num + nxt_det_pos]
                                      + nxt_proj_z*f_proj_data[cur_proj_pos*n_det_num + cur_det_pos])\
                                     /(cur_proj_z + nxt_proj_z)
    return res


def helical_180MIL(f_proj_data, n_det_num, fan_ang, cent_ang, n_slice_num):
    d_slice_ang = 2*pi/n_slice_num
    d_det_ang = fan_ang / n_det_num
    res = np.zeros(n_slice_num*n_det_num, float)
    for i in range(0, n_slice_num):
        for j in range(0, n_det_num):
            cur_det_cent_ang = j*d_det_ang - cent_ang
            nxt_det_cent_ang = -cur_det_cent_ang+cent_ang
            cur_proj_ang = i*d_slice_ang
            nxt_proj_ang = cur_proj_ang+pi-2*d_det_ang if cur_proj_ang+pi-2*d_det_ang - 2*pi <= 0 else cur_proj_ang+pi-2*d_det_ang -2*pi
            cur_det_pos = j
            nxt_det_pos = int(nxt_det_cent_ang / d_det_ang)
            cur_proj_pos = i
            nxt_proj_pos = int(nxt_proj_ang / d_slice_ang)
            cur_proj_z = fabs(cur_proj_ang - pi)
            nxt_proj_z = fabs(nxt_proj_ang - pi)
            res[i * n_det_num + j] = (cur_proj_z*f_proj_data[nxt_proj_pos*n_det_num + nxt_det_pos]
                                      + nxt_proj_z*f_proj_data[cur_proj_pos*n_det_num + cur_det_pos])\
                                     /(cur_proj_z + nxt_proj_z)
    return res