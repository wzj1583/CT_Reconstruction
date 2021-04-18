import numpy as np
from ctypes import *
from ctypes.wintypes import *
from config import fan_ct_milt_layers
from utils.data_utils import *


class DRGIMAGEHEADER(Structure):
    _pack_ = 1
    _fields_ = [('strType', DWORD),
                ('nVersion', DWORD),
                ('nWidth', DWORD),
                ('nHeight', DWORD),
                ('nCompression', BYTE),
                ('nBitsPerPixel', BYTE),
                ('strDate', BYTE * 8),
                ('strTime', BYTE * (6 + 7)),
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
                ('researve', BYTE * (164))
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
                            bCorrect=data[21], nDbtRgnCounts=data[22], bChecked=data[23], nCompressLength=data[27],
                            nScanTime=data[28], nCapSize=data[29], nZeroPos=data[30], fStart=data[35], fend=data[36],
                            fSpeed=data[37], fTemp=data[38], nTimeCount=data[39], researve=data[40:203])
        return new_header

    @staticmethod
    def read_header(path):
        header_fmt = '<LLLLBB8B6BHBBB3BiHHL4BddddI164B'
        data = read_data(path, header_fmt, 0, sizeof(DRGIMAGEHEADER))
        header = Header.create_header(data)
        return header, data


class DetConfig:
    def __init__(self):
        self.det2scr = fan_ct_milt_layers["det2src"]  # distance from detector to source
        self.det2cen = fan_ct_milt_layers["det2cen"]  # distance from detector to center
        self.cents = fan_ct_milt_layers["cents"]  # ios ray
        self.det_num = fan_ct_milt_layers["det_num"]  # amount of detectors
        self.proj_num = fan_ct_milt_layers["proj_num"]  # amount of projection per cycle
        self.det_lar = fan_ct_milt_layers["det_lar"]  # amount of detector layers
        self.det_width = fan_ct_milt_layers["det_width"]  # width of detector
        self.det_height = fan_ct_milt_layers["det_height"]    # height of detector
        self.pixel_num = fan_ct_milt_layers["pixel_num"]
        self.pixel_len = fan_ct_milt_layers["pixel_len"]

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
        self.pixel_num = fan_ct_milt_layers["pixel_num"]
        self.pixel_len = fan_ct_milt_layers["pixel_len"]

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
        data_graph = np.empty(nHeight, float)
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
    def image_pre(data_graph, zA, zB, eA, eB, width, height):
        '''
        zA = (zA + zB) / 2
        eA = (eA + eB) / 2
        zA = zA.reshape(1, -1)
        eA = eA.reshape(1, -1)
        data_graph = data_graph.reshape(width, height)
        f_data_graph = np.zeros(data_graph.shape, float)
        temp = 0
        zero_matrix = zA
        full_matrix = eA
        for i in range(1, width):
            zero_matrix = np.append(zero_matrix, zA, axis=0)
            full_matrix = np.append(full_matrix, eA, axis=0)
        I = data_graph - zero_matrix
        I[I < 1] = 1
        I_0 = full_matrix - zero_matrix
        I_0[I_0 < 0] = 0
        tmp = I_0/I
        tmp[tmp < 1] = 1
        f_data_graph = np.log(tmp)
        f_data_graph[f_data_graph < 0.000001] = 0
        f_data_graph = f_data_graph.reshape(-1, 1)
        f_data_graph = f_data_graph.squeeze()
        '''
        f_data_graph = np.zeros(data_graph.shape, float)
        zA = (zA + zB) / 2
        eA = (eA + eB) / 2
        for i in range(0, height):
            for j in range(0, width):
                temp = data_graph[j * height + i] - zA[i]
                if temp <= 1:
                    temp = 1
                if eA[i] - zA[i] > 1:
                    temp = np.log((eA[i] - zA[i]) / temp)
                else:
                    temp = 0
                f_data_graph[j * height + i] = temp
                if temp <= 0.000001:
                    f_data_graph[j * height + i] = 0
        return f_data_graph