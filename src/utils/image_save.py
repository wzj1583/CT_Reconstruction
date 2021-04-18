import config

import numpy as np
from PIL import Image


def gray_map(image):
    max = image.max()
    min = image.min()
    image = (image - min) * 255 / (max - min)
    image = np.trunc(image)
    return image


def save_image(res_list):
    for i in range(0, len(res_list)):
        res_list[i] = gray_map(res_list[i])
        path = config.RECONSTRUCTION_SAVE_PATH1 + '\\' + str(i) + '.bmp'
        output_img = Image.fromarray(res_list[i])
        output_img = output_img.convert('L')
        output_img.save(path)
    return 0