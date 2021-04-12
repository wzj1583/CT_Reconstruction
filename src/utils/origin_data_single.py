import numpy as np
from numpy import log
from ctypes import *
from ctypes.wintypes import *
from utils.data_utils import *
from config import fan_ct_single

DET2SCR = fan_ct_single['det2src']
DET2CEN = fan_ct_single['det2cen']
CENTS = fan_ct_single['cents']
DET_NUM = fan_ct_single['det_num']
DET_LAR = fan_ct_single['det_lar']
DET_WIDTH = fan_ct_single['det_width']
DET_HEIGHT = fan_ct_single['det_height']


class ADCFILEHEADER(Structure):
    _pack_ = 1
    _fields_ = [('SerialNo', WORD),
                ('p_Width', WORD),
                ('p_Height', WORD),
                ('MoveMode', BYTE),
                ('ScanMode', BYTE),
                ('SDMode', BYTE),
                ('CaliMode', BYTE),
                ('Wiggle', BYTE),
                ('RotSpeed', c_double),
                ('RiseDpeed', c_double),
                ('MoveSpeed', c_double),
                ('SlicePos', c_double),
                ('IntvlTime', DWORD),
                ('ProNumber', DWORD),
                ('ZeroPos', LONG),
                ('MaxCalibValue', BYTE),
                ('Segments', BYTE),
                ('reserved', 23 * BYTE)
                ]


class _Header:
    def __init__(self, SerialNo, p_Width, p_Height, MoveMode, ScanMode, SDMode, CaliMode, Wiggle, RotSpeed, RiseDpeed,
                 MoveSpeed, SlicePos, IntvlTime, ProNumber, ZeroPos, MaxCalibValue, Segments, reserved):
        self.SerialNo = SerialNo
        self.p_Width = p_Width
        self.p_Height = p_Height
        self.MoveMode = MoveMode
        self.ScanMode = ScanMode
        self.SDMode = SDMode
        self.CaliMode = CaliMode
        self.Wiggle = Wiggle
        self.RotSpeed = RotSpeed
        self.RiseDpeed = RiseDpeed
        self.MoveSpeed = MoveSpeed
        self.SlicePos = SlicePos
        self.IntvlTime = IntvlTime
        self.ProNumber = ProNumber
        self.ZeroPos = ZeroPos
        self.MaxCalibValue = MaxCalibValue
        self.Segments = Segments
        self.reserved = reserved

    @staticmethod
    def create_header(data):
        new_header = _Header(SerialNo=data[0], p_Width=data[1], p_Height=data[2], MoveMode=data[3], ScanMode=data[4],
                            SDMode=data[5], CaliMode=data[6], Wiggle=data[7], RotSpeed=data[8], RiseDpeed=data[9],
                            MoveSpeed=data[10], SlicePos=data[11], IntvlTime=data[12], ProNumber=data[13],
                            ZeroPos=data[14], MaxCalibValue=data[15], Segments=data[16], reserved=data[17:39],
                            )
        return new_header

    @staticmethod
    def read_header(path):
        header_fmt = '<HHHBBBBBddddLLlBB23B'
        print(sizeof(ADCFILEHEADER))
        data = read_data(path, header_fmt, 0, sizeof(ADCFILEHEADER))
        header = _Header.create_header(data)
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


class _DataFetcher:
    def __init__(self, header, det_config, data_buff, zero, full):
        self.header: _Header = header
        self.det_config: DetConfig = det_config
        self.data_buff = data_buff  # original projection data
        self.zero = zero
        self.full = full
        self.pixel_num = fan_ct_single["pixel_num"]
        self.pixel_len = fan_ct_single["pixel_len"]

    @staticmethod
    def create_data_fetcher(path):
        header, data = _Header.read_header(path)
        det_config = DetConfig()
        nWidth = header.p_Width
        nHeight = header.p_Height
        _DataFetcher.print_info(path, nWidth, nHeight)
        fields_list = _DataFetcher.read_fields(path, nWidth, nHeight)
        data_graph = _DataFetcher.read_graph(path, nWidth, nHeight)
        data_fetcher = _DataFetcher(header, det_config, data_graph, fields_list[0], fields_list[1])
        return data_fetcher

    @staticmethod
    def print_info(path, nWidth, nHeight):
        with open(path, 'rb') as f:
            br = f.read()
            bin_size = len(br)
            header_size = sizeof(ADCFILEHEADER) + 8
            fields_size = 2 * nHeight * sizeof(c_double)
            graph_size = nWidth * nHeight * sizeof(WORD)
            print('Expecting:\n header_size: ' + str(header_size)
                  + '\n graph_size: ' + str(graph_size)
                  + '\n fields_size: ' + str(fields_size)
                  + '\n total: ' + str(header_size + graph_size + fields_size))
            print('Receiving: \n total: ' + str(bin_size))
            f.close()

    @staticmethod
    def read_fields(path, nWidth, nHeight):
        start = sizeof(ADCFILEHEADER) + 8
        row_fmt = '<' + str(nHeight) + 'd'
        array_list = []
        for i in range(0, 2):
            index_start = start + i * nHeight * sizeof(c_double)
            a = np.array(read_data(path, row_fmt, index_start, index_start + nHeight * sizeof(c_double)))
            array_list.append(a)
        return array_list

    @staticmethod
    def read_graph(path, nWidth, nHeight):
        start = sizeof(ADCFILEHEADER) + 8 + 2 * nHeight * sizeof(c_double)
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
    def image_pre(data_graph, z, e, width, height):
        data_graph_float = np.zeros(int(height/2) * width, float)
        for i in range(0, width):
            for j in range(0, int(height/2)):
                a = log((e[2*j]-z[2*j])/(data_graph[2000*j+i]-z[j*2]))
                b = log((e[2*j+1]-z[2*j+1])/(data_graph[2000*j+1000+i]-z[j*2+1]))
                data_graph_float[i*int(height/2)+j] = (a+b)/2
                if data_graph_float[i*int(height/2)+j] < 0:
                    data_graph_float[i * int(height / 2) + j] = 0

        '''data_graph_float = np.zeros(data_graph.shape, float)
        for i in range(0, height):
            for j in range(0, width):
                temp = data_graph[j * height + i] - z[i]
                if temp <= 1:
                    temp = 1
                if e[i] - z[i] > 1:
                    temp = np.log((e[i] - z[i]) / temp)
                else:
                    temp = 0
                if temp <= 0.000001:
                    data_graph_float[j * height + i] = 0
                else:
                    data_graph_float[j * height + i] = temp
        data_graph_float = data_graph_float.reshape((width, height), order='F')
        data_graph_float_merged = np.zeros((width, 1), float)
        for i in range(0, int(height / 2)):
            temp = (data_graph_float[:, 2*i] + data_graph_float[:, 2*i+1])/2
            temp = temp.reshape((width, 1))
            data_graph_float_merged = np.append(data_graph_float_merged, temp, axis=1)
        data_graph_float_merged = data_graph_float_merged[:, 1:]
        data_graph_float = data_graph_float_merged.reshape((1, -1), order='C')
        data_graph_float = np.squeeze(data_graph_float)'''
        return data_graph_float
