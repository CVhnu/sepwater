import numpy as np
from PIL import Image
from .noiseconfig import *


noise_tuple = (moire,
                    brightness, jpeg_compression,
                   rotate,translate,perspective_noise,gaussian_noise,light_distortion,eye_protection,gradient_gray)
noise_tuple_tr = (moire,
                    brightness,  jpeg_compression,
                    rotate,translate,perspective_noise,gaussian_noise,light_distortion,eye_protection,gradient_gray)
noise_tuple_vl = (moire,
                    brightness,  jpeg_compression,
                    rotate,translate,perspective_noise,gaussian_noise,light_distortion,eye_protection,gradient_gray)


noise_dict = {corr_func.__name__: corr_func for corr_func in noise_tuple}


def corrupt(x, severity=1, corruption_name=None, corruption_number=-1):

    if corruption_name:
        x_corrupted = noise_dict[corruption_name](Image.fromarray(x), severity)
    elif corruption_number != -1:
        x_corrupted = noise_tuple[corruption_number](Image.fromarray(x), severity)
    else:
        raise ValueError("Either corruption_name or corruption_number must be passed")

    return np.uint8(x_corrupted)
