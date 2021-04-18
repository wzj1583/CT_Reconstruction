import os
import config
import numpy as np
from numpy import pi, fabs
from utils.origin_data import DataFetcher
from utils.data_utils import save_interpolated_data


def helical_180IL(f_proj_data, n_det_num, fan_ang, cent_ang, n_slice_num, plane_pos):
    res_list = []
    d_slice_ang = 2 * pi / n_slice_num
    d_det_ang = fan_ang / n_det_num
    for k in range(0, len(plane_pos)):
        res = np.zeros(n_slice_num * n_det_num, float)
        for i in range(0, n_slice_num):
            for j in range(0, n_det_num):
                cur_det_cent_ang = j * d_det_ang - cent_ang
                nxt_det_cent_ang = -cur_det_cent_ang + cent_ang
                cur_proj_ang = i * d_slice_ang
                nxt_proj_ang = cur_proj_ang + pi + 2 * cur_det_cent_ang
                if nxt_proj_ang - 2 * pi > 0:
                    nxt_proj_ang = nxt_proj_ang - 2 * pi
                cur_det_pos = j
                nxt_det_pos = int(nxt_det_cent_ang / d_det_ang)
                cur_proj_pos = i
                nxt_proj_pos = int(nxt_proj_ang / d_slice_ang)
                if (nxt_proj_pos > n_slice_num - 1):
                    nxt_proj_pos = n_slice_num - 1
                zr = plane_pos[k]
                cur_proj_z = fabs(cur_proj_ang - zr)
                nxt_proj_z = fabs(nxt_proj_pos * d_slice_ang - zr)
                try:
                    res[i * n_det_num + j] = (cur_proj_z * f_proj_data[
                        nxt_proj_pos * n_det_num + nxt_det_pos] + nxt_proj_z * f_proj_data[
                                                  cur_proj_pos * n_det_num + cur_det_pos]) / (cur_proj_z + nxt_proj_z)
                except Exception as e:
                    print(e)
        res_list.append(res)
    return res_list


def helical_180MIL(f_proj_data, n_det_num, fan_ang, cent_ang, n_proj_num, n_layer_num, f_layer_thick, f_pitch_layer,
                   n_cur_slice):
    inpolated_proj_data = np.zeros(n_proj_num * n_det_num, float)

    inter_start_idx = n_cur_slice * n_proj_num / n_layer_num
    d_proj_ang = 2 * pi / n_proj_num
    d_det_ang = fan_ang / n_det_num
    f_pitch_proj = f_pitch_layer * n_layer_num
    f_cur_slice_z = (f_pitch_proj + n_layer_num * f_layer_thick) / 2
    for i in range(0, n_proj_num):
        cur_proj_idx = i
        cur_proj_ang = cur_proj_idx * d_proj_ang
        f_cur_bottom_layer_z = f_pitch_proj * cur_proj_idx / n_proj_num + f_layer_thick / 2
        n_cur_inter_start_idx = int(((cur_proj_idx + inter_start_idx) % n_proj_num) * n_det_num)
        cur_orig_start_idx = cur_proj_idx * n_det_num * n_layer_num
        cur_layers_z = np.zeros(n_layer_num, float)

        for k in range(0, n_layer_num):
            cur_layers_z[k] = f_cur_bottom_layer_z + k * f_layer_thick

        for j in range(0, n_det_num):
            cur_det_cent_ang = j * d_det_ang - cent_ang
            nxt_det_cent_ang = 2 * cent_ang - j * d_det_ang
            # cur_proj_ang
            nxt_proj_ang = cur_proj_ang + pi + 2 * cur_det_cent_ang
            if nxt_proj_ang >= 2 * pi:
                nxt_proj_ang -= 2 * pi
            if nxt_proj_ang < 0:
                nxt_proj_ang += 2 * pi
                #nxt_proj_ang = cur_proj_ang + 2 * pi
            cur_det_idx = j
            nxt_det_idx = int(nxt_det_cent_ang / d_det_ang)
            # cur_proj_idx = i
            nxt_proj_idx = int(nxt_proj_ang / d_proj_ang)
            f_nxt_bottom_layer_z = f_pitch_proj * nxt_proj_idx / n_proj_num + f_layer_thick / 2
            # cur_layers_z
            nxt_layers_z = np.zeros(n_layer_num, float)
            cur_gonge_start_idx = nxt_proj_idx * n_det_num * n_layer_num

            gonge = np.zeros(n_layer_num, float)
            orig = np.zeros(n_layer_num, float)
            for k in range(0, n_layer_num):
                nxt_layers_z[k] = f_nxt_bottom_layer_z + k * f_layer_thick
                gonge[k] = f_proj_data[cur_gonge_start_idx + k * n_det_num + nxt_det_idx]
                orig[k] = f_proj_data[cur_orig_start_idx + k * n_det_num + cur_det_idx]

            prev_origin_idx = n_layer_num - 1
            for k in range(0, n_layer_num):
                if f_cur_slice_z - cur_layers_z[k] < 0:
                    if k > 0:
                        prev_origin_idx = k - 1
                    else:
                        prev_origin_idx = 0
                    break
            prev_gonge_idx = n_layer_num - 1
            for k in range(0, n_layer_num):
                if f_cur_slice_z - nxt_layers_z[k] < 0:
                    if k > 0:
                        prev_gonge_idx = k - 1
                    else:
                        prev_gonge_idx = 0
                    break
            '''
            if prev_origin_idx < 7:
                if cur_layers_z[prev_origin_idx + 1] - f_cur_slice_z < f_cur_slice_z - cur_layers_z[prev_origin_idx]:
                    prev_origin_idx += 1
            if nxt_layers_z[prev_gonge_idx] > cur_layers_z[prev_origin_idx]:
                w1 = abs(nxt_layers_z[prev_gonge_idx] - f_cur_slice_z)
                if prev_origin_idx < 7:
                    w2 = abs(cur_layers_z[prev_origin_idx + 1] - f_cur_slice_z)
                    w = 1 / (w1 + w2)
                    inpolated_proj_data[n_cur_inter_start_idx + cur_det_idx] = w * (
                                w2 * gonge[prev_gonge_idx] + w1 * orig[prev_origin_idx])
                else:
                    inpolated_proj_data[n_cur_inter_start_idx + cur_det_idx] = gonge[prev_gonge_idx]
            else:
                w1 = abs(cur_layers_z[prev_origin_idx] - f_cur_slice_z)
                if prev_gonge_idx < 7:
                    w2 = abs(nxt_layers_z[prev_gonge_idx + 1] - f_cur_slice_z)
                    w = 1 / (w1 + w2)
                    inpolated_proj_data[n_cur_inter_start_idx + cur_det_idx] = w * (
                                w1 * gonge[prev_gonge_idx] + w2 * orig[prev_origin_idx])
                else:
                    inpolated_proj_data[n_cur_inter_start_idx + cur_det_idx] = orig[prev_origin_idx]
            '''


            if cur_layers_z[prev_origin_idx] > nxt_layers_z[prev_gonge_idx]:
                prev_gonge_idx += 1
            else:
                prev_origin_idx += 1
            if nxt_layers_z[prev_gonge_idx] > cur_layers_z[prev_origin_idx]:
                w1 = abs(nxt_layers_z[prev_gonge_idx] - f_cur_slice_z)
                if prev_origin_idx < 7:
                    w2 = abs(cur_layers_z[prev_origin_idx] - f_cur_slice_z)
                    w = 1 / (w1 + w2)
                    inpolated_proj_data[n_cur_inter_start_idx + cur_det_idx] = w * (
                                w2 * gonge[prev_gonge_idx] + w1 * orig[prev_origin_idx])
                else:
                    inpolated_proj_data[n_cur_inter_start_idx + cur_det_idx] = gonge[prev_gonge_idx]
            else:
                w1 = abs(cur_layers_z[prev_origin_idx] - f_cur_slice_z)
                if prev_gonge_idx < 7:
                    w2 = abs(nxt_layers_z[prev_gonge_idx] - f_cur_slice_z)
                    w = 1 / (w1 + w2)
                    inpolated_proj_data[n_cur_inter_start_idx + cur_det_idx] = w * (
                                w1 * gonge[prev_gonge_idx] + w2 * orig[prev_origin_idx])
                else:
                    inpolated_proj_data[n_cur_inter_start_idx + cur_det_idx] = orig[prev_origin_idx]

    return inpolated_proj_data


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