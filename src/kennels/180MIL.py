from numba.cuda import *


def cudaConv(buf, h, res, len):
	i = threadIdx.x + blockIdx.x*blockDim.x
	if i > len:
		return
	index = blockIdx.y * len
	t = buf[index:]
	tmp = 0
	for j in range(0, i):
		tmp += t[j] * h[i-j]
	for j in range(i, len):
		tmp += t[j] * h[j-i]
	res[i+index] = tmp