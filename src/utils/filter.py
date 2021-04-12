import numpy as np
from numpy import pi, sin

# 单边卷积核
def get_filter_RL(dang, N):
    h = np.zeros(N, float)
    h[0] = 1 / (8 * dang * dang)
    for i in range(1, N, 2):
        h[i] = -1 / (2 * pi * pi * sin(dang * i) * sin(dang * i))
        if i + 1 < N:
            h[i + 1] = 0
    return h


def get_filter_SL(dang, N):
    h = np.zeros(N, float)
    h[0] = 1 / (pi * pi * dang * dang)
    for i in range(1, N):
        h[i] = - ((i * i) / (pi * pi * sin(i * dang) * sin(i * dang) * (4 * i * i - 1)))
    return h
