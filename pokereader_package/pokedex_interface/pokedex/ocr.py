from pathlib import Path
import numpy as np
import pandas as pd
import cv2

import pyocr
from PIL import Image, ImageEnhance, ImageOps, ImageFilter


def is_groove(img: Image.Image):
    orig_arr = np.array(img, dtype="float32") / 255
    eroded_arr = np.array(img.filter(ImageFilter.GaussianBlur(3)).filter(ImageFilter.MaxFilter(15)), dtype="float32") / 255
    orig_var = orig_arr.var()
    eroded_var = eroded_arr.var()
    return eroded_var - orig_var < 0
