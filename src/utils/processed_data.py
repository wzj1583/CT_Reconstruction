import config
import numpy as np
from numpy import pi
from utils.origin_data import DetConfig, Header, DataFetcher
from utils.data_utils import read_interpolated_data

class ParaTransfer:
    def __init__(self, f_pixel_len, n_pixel_num, n_det_num, n_width, n_height, n_proj_num, f_cen2src,
                 f_cent_pos, n_layer_num, f_det_width, f_layer_thick, f_pitch_layer, f_FOV, d_det_ang,
                 d_proj_ang, cent_ang, fan_ang):
        self.f_pixel_len = f_pixel_len
        self.n_pixel_num = n_pixel_num
        self.n_det_num = n_det_num
        self.n_width = n_width
        self.n_height = n_height
        self.n_proj_num = n_proj_num
        self.f_cen2src = f_cen2src
        self.f_cent_pos = f_cent_pos
        self.n_layer_num = n_layer_num
        self.f_det_width = f_det_width
        self.f_layer_thick = f_layer_thick
        self.f_pitch_layer = f_pitch_layer
        self.f_FOV = f_FOV
        self.d_det_ang = d_det_ang
        self.d_proj_ang = d_proj_ang
        self.cent_ang = cent_ang
        self.fan_ang = fan_ang

    @staticmethod
    def create_para_transfer(det_config: DetConfig, header: Header):
        f_pixel_len = det_config.pixel_len  # 重建图像像素宽度 1.5mm
        n_pixel_num = det_config.pixel_num  # 重建图像分辨率 512
        n_width = header.nWidth
        n_height = header.nHeight
        n_proj_num = det_config.proj_num    # 插值后的投影数量
        n_det_num = det_config.det_num  # 探测器数量 460
        f_cen2src = det_config.det2cen  # 射线源-旋转中心距离 700.0mm
        f_det2src = det_config.det2scr  # 射线源-探测器距离 1300.0mm
        f_cent_pos = det_config.cents  # 中心投影线序号 229.5
        n_layer_num = det_config.det_lar  # 探测器层数 8
        f_det_width = det_config.det_width  # 探测器宽度 3.2mm
        f_layer_thick = det_config.det_height
        f_pitch_layer = - (header.fend - header.fStart) / n_layer_num
        f_FOV = f_pixel_len * n_pixel_num  # 视场
        d_det_ang = f_det_width / f_det2src  # 投影线间角
        d_proj_ang = 2 * pi / n_proj_num  # 投影角步长 1000
        cent_ang = f_cent_pos * d_det_ang  # 中心投影线角度
        fan_ang = n_det_num * d_det_ang  # 扇形束夹角
        para_transfer = ParaTransfer(f_pixel_len, n_pixel_num, n_det_num, n_width, n_height,
                                     n_proj_num, f_cen2src, f_cent_pos, n_layer_num, f_det_width,
                                     f_layer_thick, f_pitch_layer, f_FOV, d_det_ang, d_proj_ang, cent_ang,
                                     fan_ang)
        return para_transfer

    @staticmethod
    def pack_para_transfer(index, path=None, is_origin=0):
        if is_origin:
            origin_data_path = config.HELICAL_FAN_CT_PATH1 + config.FILE_NAME_PREFIX1 + str(1000 + index) + ".DAT"
            header, header_content = Header.read_header(origin_data_path)
            det_config = DetConfig.create_det_config()
            origin_data = DataFetcher.create_data_fetcher(origin_data_path)
            f_origin_data = DataFetcher.image_pre(origin_data.data_buff, origin_data.zero_a, origin_data.zero_b,
                                                  origin_data.empty_a, origin_data.empty_b,
                                                  origin_data.header.nWidth, origin_data.header.nHeight)
            det_config.proj_num = header.nWidth
            return f_origin_data, ParaTransfer.create_para_transfer(det_config, header)
        else:
            header_path = config.HELICAL_FAN_CT_PATH1 + config.FILE_NAME_PREFIX1 + str(1000 + index) + ".DAT"
            header, header_content = Header.read_header(header_path)
            det_config = DetConfig.create_det_config()
            if path:
                path = path + str(1000 + index) + ".bin"
                f_interpolated_data = read_interpolated_data(path)
            else:
                f_interpolated_data = None
            return f_interpolated_data, ParaTransfer.create_para_transfer(det_config, header)
