import numpy as np
from numpy import pi, fabs


def helical_180IL(f_proj_data, n_det_num, fan_ang, cent_ang, n_slice_num, plane_pos):
    res_list = []
    d_slice_ang = 2*pi/n_slice_num
    d_det_ang = fan_ang / n_det_num
    for k in range(0, len(plane_pos)):
        res = np.zeros(n_slice_num * n_det_num, float)
        for i in range(0, n_slice_num):
            for j in range(0, n_det_num):
                cur_det_cent_ang = j*d_det_ang - cent_ang
                nxt_det_cent_ang = -cur_det_cent_ang+cent_ang
                cur_proj_ang = i*d_slice_ang
                nxt_proj_ang = cur_proj_ang+pi-2*cur_det_cent_ang
                if nxt_proj_ang - 2*pi > 0:
                    nxt_proj_ang = nxt_proj_ang - 2*pi
                cur_det_pos = j
                nxt_det_pos = int(nxt_det_cent_ang / d_det_ang)
                cur_proj_pos = i
                nxt_proj_pos = int(nxt_proj_ang / d_slice_ang)
                if (nxt_proj_pos > n_slice_num - 1):
                    nxt_proj_pos = n_slice_num - 1
                zr = plane_pos[k]
                cur_proj_z = fabs(cur_proj_ang - zr)
                nxt_proj_z = fabs(nxt_proj_pos*d_slice_ang - zr)
                try:
                    res[i * n_det_num + j] = (cur_proj_z*f_proj_data[nxt_proj_pos*n_det_num + nxt_det_pos] + nxt_proj_z*f_proj_data[cur_proj_pos*n_det_num + cur_det_pos])/(cur_proj_z + nxt_proj_z)
                except Exception as e:
                    print(e)
        res_list.append(res)
    return res_list


def helical_180MIL(f_proj_data, n_det_num, fan_ang, cent_ang, n_proj_num, n_layer_num, f_layer_thick, f_pitch_layer, n_cur_slice):
    inter_start_idx = n_cur_slice*n_proj_num/n_layer_num
    d_proj_ang = 2*pi/n_proj_num
    d_det_ang = fan_ang / n_det_num
    f_pitch_proj = f_pitch_layer*n_layer_num
    f_cur_slice_z = (f_pitch_proj +  n_layer_num*f_layer_thick)/2
    for i in range(0, n_proj_num):
        cur_proj_ang = i*d_proj_ang
        f_cur_bottom_layer_z = f_pitch_proj*i/n_proj_num + f_layer_thick/2
        n_cur_inter_start_idx = ((i + inter_start_idx)%n_proj_num)*n_det_num
        cur_orig_start_idx = i * n_det_num * n_layer_num
        cur_layers_z = np.zeros(n_layer_num, float)
        for k in range(0, n_layer_num):
            cur_layers_z[k] = f_cur_bottom_layer_z + k*f_layer_thick

        for j in range(0, n_det_num):
            cur_det_cent_ang = j * d_det_ang - cent_ang
            nxt_det_cent_ang = -cur_det_cent_ang + cent_ang
            # cur_proj_ang
            nxt_proj_ang = cur_proj_ang + pi -2*cur_det_cent_ang
            if nxt_proj_ang >= 2*pi:
                nxt_proj_ang -=2*pi
            if nxt_proj_ang <0:
                nxt_proj_ang += 2*pi
            cur_det_idx = j
            nxt_det_idx = int(nxt_det_cent_ang/d_det_ang)
            cur_proj_idx = i
            nxt_proj_idx = int(nxt_proj_ang/d_proj_ang)
            f_nxt_bottom_layer_z = f_pitch_proj * nxt_proj_idx / n_proj_num + f_layer_thick / 2
            # cur_layers_z
            nxt_layers_z = np.zeros(n_layer_num, float)
            for k in range(0, n_layer_num):
                nxt_layers_z[k] = f_nxt_bottom_layer_z + k*f_layer_thick
            for k in range(0, n_layer_num):
                if f_cur_slice_z - cur_layers_z[k] < 0:
                    if k > 0:
                        pre_close_layer_idx = k-1
                    else:
                        pre_close_layer_idx = 0
                    break
            for k in range(0, n_layer_num):
                if f_cur_slice_z - nxt_layers_z[k] < 0:
                    if k > 0:
                        pre_close_layer_idx = k-1
                    else:
                        pre_close_layer_idx = 0
                    break

