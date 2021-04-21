FIXED_CT_PATH1 = r"D:\Study\Graduation Design\CT Reconstruction\data_src\20171103006DAT\201711030060001001.DAT"
FIXED_CT_PATH2 = r"D:\Study\Graduation Design\CT Reconstruction\data_src\20171103008DAT\201711030080001001.DAT"
FIXED_CT_PATH3 = r"D:\Study\Graduation Design\CT Reconstruction\data_src\20201014014DAT\202010140140001001.DAT"
FIXED_CT_PATH4 = r"D:\Study\Graduation Design\CT Reconstruction\data_src\20130613114840DAT\20130613114840.dat"

HELICAL_FAN_CT_PATH1 = r"D:\Study\Graduation Design\CT Reconstruction\data_src\20210225004DAT"
FILE_NAME_PREFIX1 = r"\20210225004000"

ORIGIN_INTERPOLATED_DATA_PATH = r"D:\Study\Graduation Design\CT Reconstruction\src\data repository\origin interpolated projection data"
_180MIL_INTERPOLATED_DATA_PATH = r"D:\Study\Graduation Design\CT Reconstruction\src\data repository\180MIL interpolated projection data"
_360MIL_INTERPOLATED_DATA_PATH = r"D:\Study\Graduation Design\CT Reconstruction\src\data repository\360MIL interpolated projection data"
RECONSTRUCTION_RESULT = r"D:\Study\Graduation Design\CT Reconstruction\src\data repository\reconstruction result"

RECONSTRUCTION_SAVE_PATH1 = r"D:\Study\Graduation Design\CT Reconstruction\src\reconstruction results"

fan_ct_single = {
    "pixel_num": 512,
    "pixel_len": 500/512,
    "det2src": 1316.0,
    "det2cen": 858.5,
    "cents": 111.6,
    "det_num": 224,
    "det_lar": 1,
    "det_width": 2.6,
    "det_height": 0
}

fan_ct_milt_layers = {
    "pixel_num": 512,
    "pixel_len": 1.5,
    "det2src": 1300.0,
    "det2cen": 700.0,
    "cents": 229.5,
    "det_num": 460,
    "det_lar": 8,
    "det_width": 3.2,
    "det_height": 5,
    "proj_num": 1000
}