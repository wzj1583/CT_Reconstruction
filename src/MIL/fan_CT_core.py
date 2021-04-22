import numpy as np
from math import asin, sqrt
from numpy import power, sqrt, arcsin, trunc


def back_proj_rotate_pixel(proj_data, image_size, X_0, Y_0, _sin, _cos, det2cen, det_num, dalpha, cents):
    D = det2cen
    ct_image = np.zeros((image_size, image_size), float)
    _X = _cos * X_0 - _sin * Y_0
    _Y = _sin * X_0 + _cos * Y_0
    L2 = power(_X, 2) + power(D + _Y, 2)
    _dtheta = arcsin(_X / sqrt(L2))
    _dtheta = _dtheta / dalpha + cents
    '''
    dtheta = _dtheta
    _det_idx = np.trunc(dtheta).astype(int)
    _det_idx[_det_idx < 0] = 0
    _det_idx[_det_idx > (det_num - 2)] = det_num - 2
    '''
    _dtheta[_dtheta < 0] = 0
    _dtheta[_dtheta > (det_num - 2)] = det_num - 2
    det_idx = np.trunc(_dtheta).astype(int)
    _dtheta[_dtheta == (det_num - 2)] = det_num - 1
    a = _dtheta - det_idx
    ct_image = ct_image + ((1 - a) * proj_data[det_idx] + a * proj_data[det_idx + 1]) / L2
    ct_image = 10 * ct_image
    return ct_image

def _back_proj_rotate_pixel(proj_data, image_size, X_0, Y_0, _sin, _cos, det2cen, det_num, d_det_ang, cents):
    ct_image = np.zeros((image_size, image_size), float)
    for i in range(0, image_size):
        for j in range(0, image_size):
            x = X_0[i][j]*_cos - Y_0[i][j]*_sin
            y = X_0[i][j]*_sin + Y_0[i][j]*_cos
            L2 = x*x + y*y
            pixel_ang = asin(x / sqrt(L2))
            pixel_idx = pixel_ang / d_det_ang + cents
            if np.isnan(pixel_idx):
                print("fuck")
            if pixel_idx < 0:
                ct_image[i][j] = 0
            elif pixel_idx > det_num - 1:
                ct_image[i][j] = 0
            else:
                pixel_det_idx = int(pixel_idx)
                pos = pixel_det_idx - pixel_idx
                ct_image[i][j] += ((1 - pos) * proj_data[pixel_det_idx] + pos * proj_data[pixel_det_idx + 1]) / L2
            ct_image[i][j] *= 10
    return ct_image

def back_proj_rotate_src(proj_data, image_size, X, Y, _sin, _cos, det2cen, det_num, dbeta, cents):
    ct_image = np.zeros((image_size, image_size), float)
    x_src = -(det2cen * _sin)
    y_src = det2cen * _cos
    X1 = X - x_src
    Y1 = Y - y_src
    L2 = X1*X1 + Y1*Y1
    dtheta = (X*_cos + Y*_sin) / sqrt(L2)
    dtheta = arcsin((dtheta))
    dtheta = dtheta/dbeta + cents
    dtheta[dtheta < 0] = 0
    dtheta[dtheta > det_num-2] = det_num - 2
    det_idx = trunc(dtheta).astype(np.int)
    a = dtheta - det_idx
    res = a*proj_data[det_idx+1]+(1-a)*proj_data[det_idx]
    ct_image += 10*res/L2
    return ct_image


def _back_proj(proj_data, image_size, FOV, sin, cos, det2cen, det_num, dang, cents):
    ct_image = np.zeros((image_size, image_size), float)
    x_0 = -(det2cen * sin)
    y_0 = det2cen * cos
    piel_len = np.float(FOV / image_size)
    for i in range(0, image_size):
        x = i*piel_len - FOV/2
        x1 = x - x_0
        for j in range(0, image_size):
            y = j*piel_len - FOV/2
            y1 = y - y_0
            L2 = x1*x1 + y1*y1
            dN = (x*cos+y*sin)/sqrt(L2)
            res = arcsin(dN)
            dN = res/dang + cents
            if dN <= 0:
                dN = 0
            if dN >= det_num - 2:
                dN = det_num - 2
            pos = int(dN)
            dN = dN - pos
            res = dN*proj_data[pos+1]+(1-dN)*proj_data[pos]
            ct_image[i][j] += 10*res/L2
    return ct_image
