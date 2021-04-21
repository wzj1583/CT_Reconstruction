import numpy as np
from numba import cuda


@cuda.jit
def cuda_conv2(buf, h, res, len):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i < len:
        index = cuda.blockIdx.y * len
        tmp = 0
        for j in range(0, i):
            tmp += buf[index + j] * h[i - j]
        for j in range(i, len):
            tmp += buf[index + j] * h[j - i]
        res[index + i] = tmp


def main_conv2(buf, h):
    length = len(buf)
    n_det_num = len(h)
    dev_buf = cuda.to_device(buf)
    dev_h = cuda.to_device(h)
    dev_res = cuda.device_array(length)
    res = cuda_conv2[2, length](dev_buf, dev_h, dev_res, n_det_num)
    cuda.synchronize()
    res = dev_res.copy_to_host()
    return res


