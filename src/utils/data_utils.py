from struct import *


def read_data(path, fmt, start, end):
    with open(path, 'rb') as f:
        br = f.read()
        data = unpack(fmt, br[start:end])
        f.close()
    return data