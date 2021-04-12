import numpy as np
from ctypes import *
from ctypes.wintypes import *
from utils.data_utils import *


DET2SCR = 1300
DET2CEN = 700
CENTS = 229.5
DET_NUM = 460
DET_LAR = 8
DET_WIDTH = 3.2
DET_HEIGHT = 0.5


class DRGIMAGEHEADER(Structure):
    _pack_ = 1
    _fields_ = [('strType', DWORD),
                ('nVersion', DWORD),
                ('nWidth', DWORD),
                ('nHeight', DWORD),
                ('nCompression', BYTE),
                ('nBitsPerPixel', BYTE),
                ('strDate', BYTE * 8),
                ('strTime', BYTE * 6),
                ('nDataOffset', WORD),
                ('bCorrect', BYTE),
                ('nDbtRgnCounts', BYTE),
                ('bChecked', BYTE),
                ('nCompressLength', c_int),
                ('nScanTime', WORD),
                ('nCapSize', WORD),
                ('nZeroPos', DWORD),
                ('fStart', c_double),
                ('fend', c_double),
                ('fSpeed', c_double),
                ('fTemp', c_double),
                ('nTimeCount', UINT),
                ('researve', BYTE * (162 + 9))
                ]


class Header:
    def __init__(self, strType, nVersion, nWidth, nHeight, nCompression, nBitsPerPixel, strDate, strTime, nDataOffset,
                 bCorrect, nDbtRgnCounts, bChecked, nCompressLength, nScanTime, nCapSize, nZeroPos, fStart, fend,
                 fSpeed, fTemp, nTimeCount, researve):
        self.strType = strType
        self.nVersion = nVersion
        self.nWidth = nWidth    # amount of projection data
        self.nHeight = nHeight  # layer * det_num
        self.nCompression = nCompression
        self.nBitsPerPixel = nBitsPerPixel
        self.strDate = strDate
        self.strTime = strTime
        self.nDataOffset = nDataOffset
        self.bCorrect = bCorrect
        self.nDbtRgnCounts = nDbtRgnCounts
        self.bChecked = bChecked
        self.nCompressLength = nCompressLength
        self.nScanTime = nScanTime
        self.nCapSize = nCapSize
        self.nZeroPos = nZeroPos
        self.fStart = fStart
        self.fend = fend
        self.fSpeed = fSpeed
        self.fTemp = fTemp
        self.nTimeCount = nTimeCount
        self.researve = researve

    @staticmethod
    def create_header(data):
        new_header = Header(strType=data[0], nVersion=data[1], nWidth=data[2], nHeight=data[3], nCompression=data[4],
                            nBitsPerPixel=data[5], strDate=data[6:13], strTime=data[14:19], nDataOffset=data[20],
                            bCorrect=data[21], nDbtRgnCounts=data[22], bChecked=data[23], nCompressLength=data[24],
                            nScanTime=data[25], nCapSize=data[26], nZeroPos=data[27], fStart=data[28], fend=data[29],
                            fSpeed=data[30], fTemp=data[31], nTimeCount=data[32], researve=data[33:194])
        return new_header

    @staticmethod
    def read_header(path):
        header_fmt = '<LLLLBB8B6BHBBBiHHLddddI171B'
        data = read_data(path, header_fmt, 0, sizeof(DRGIMAGEHEADER))
        header = Header.create_header(data)
        return header, data


class DetConfig:
    def __init__(self):
        self.det2scr = DET2SCR  # distance from detector to source
        self.det2cen = DET2CEN  # distance from detector to center
        self.cents = CENTS  # ios ray
        self.det_num = DET_NUM  # amount of detectors
        self.det_lar = DET_LAR  # amount of detector layers
        self.det_width = DET_WIDTH  # width of detector
        self.det_height = DET_HEIGHT    # height of detector

    @staticmethod
    def create_det_config():
        det_config = DetConfig()
        return det_config


class DataFetcher:
    def __init__(self, header, det_config, data_buff, zero_a, zero_b, empty_a, empty_b):
        self.header: Header = header
        self.det_config: DetConfig = det_config
        self.data_buff = data_buff  # original projection data
        self.zero_a = zero_a    # light field data A
        self.zero_b = zero_b    # light field data B
        self.empty_a = empty_a  # dark field data A
        self.empty_b = empty_b  # dark field data B

    @staticmethod
    def create_data_fetcher(path):
        header, data = Header.read_header(path)
        det_config = DetConfig()
        nWidth = header.nWidth
        nHeight = header.nHeight
        DataFetcher.print_info(path, nWidth, nHeight)
        data_graph = DataFetcher.read_graph(path, nWidth, nHeight)
        array_list = DataFetcher.read_fields(path, nWidth, nHeight)
        data_fetcher = DataFetcher(header, det_config, data_graph, array_list[0], array_list[1], array_list[2], array_list[3])
        return data_fetcher

    @staticmethod
    def print_info(path, nWidth, nHeight):
        with open(path, 'rb') as f:
            br = f.read()
            bin_size = len(br)
            # header_size = sizeof(DRGIMAGEHEADER)
            header_size = 256
            graph_size = nWidth * nHeight * sizeof(WORD)
            fields_size = 4 * nHeight * sizeof(WORD)
            print('Expecting:\n header_size: ' + str(header_size) +'\n graph_size: ' + str(graph_size) + '\n fields_size: ' + str(fields_size) + '\n total: ' + str(header_size + graph_size + fields_size))
            print('Receiving: \n total: ' + str(bin_size))
            f.close()

    @staticmethod
    def read_graph(path, nWidth, nHeight):
        # start = sizeof(DRGIMAGEHEADER)
        start = 256
        data_graph = np.empty(nHeight)
        row_fmt = '<' + str(nHeight) + 'H'
        for i in range(0, nWidth):
            index_start = start + i * nHeight * sizeof(WORD)
            a = np.array(read_data(path, row_fmt, index_start, index_start + nHeight * sizeof(WORD)))
            if i == 0:
                data_graph = a
            else:
                data_graph = np.append(data_graph, a, axis=0)
        return data_graph

    @staticmethod
    def read_fields(path, nWidth, nHeight):
        start = sizeof(DRGIMAGEHEADER) + nWidth * nHeight * sizeof(WORD)
        row_fmt = '<' + str(nHeight) + 'H'
        array_list = []
        for i in range(0, 4):
            index_start = start + i * nHeight * sizeof(WORD)
            a = np.array(read_data(path, row_fmt, index_start, index_start + nHeight * sizeof(WORD)))
            array_list.append(a)
        return array_list

    @staticmethod
    def image_process(data_graph, zA, zB, eA, eB, width, height):
        zA = (zA + zB) / 2
        eA = (eA + eB) / 2
        data_graph = data_graph.reshape((width, height))
        f_data_graph = np.zeros(data_graph.shape, float)
        temp = 0
        zero_matrix = zA
        full_matrix = eA
        for i in range(1, width):
            zero_matrix = np.append(zero_matrix, zA, axis=1)
            full_matrix = np.append(zero_matrix, eA, axis=1)
        I = data_graph - zero_matrix
        I[I < 1] = 1
        I_0 = full_matrix - zero_matrix
        I_0[I_0 < 1] = I[I_0 < 1]
        f_data_graph = np.log(I_0/I)
        f_data_graph[f_data_graph < 0.000001] = 0

        '''
        for i in range(0, height):
            for j in range(0, width):
                temp = data_graph[j * height + i] - zA[i]
                if temp <= 1:
                    temp = 1
                if eA[i] - zA[i] > 1:
                    temp = np.log((eA[i] - zA[i]) / temp)
                else:
                    temp = 0
                data_graph_float[j * height + i] = temp
                if temp <= 0.000001:
                    data_graph_float[j * height + i] = 0
        '''
        return f_data_graph