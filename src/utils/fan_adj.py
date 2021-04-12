import numpy as np
from numpy import cos


def adj_fan(buf, ang, det2cen, det_num):
    # ang=ang*pi/180;
    dang = ang / det_num
    for i in range(0, det_num):
        buf[i] = buf[i] * det2cen * cos((i - det_num / 2) * dang)
    return buf


def adj_fan_fine(buf, dang, ang0, det2cen, det_num):
    for i in range(0, det_num):
        buf[i] = buf[i] * det2cen * cos(i * dang - ang0)
    return buf


def adj_fan_fine_with_factor(buf, dang, ang0, det2cen, det_num):
    for i in range(0, det_num):
        buf[i] = buf[i] * det2cen * np.power(cos(i * dang - ang0), 3)
    return buf
