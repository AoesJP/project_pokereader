import pytesseract
import pyocr

import numpy as np

import cv2
from PIL import Image,ImageEnhance,ImageOps

def preproc_clean(data: list):
    _ = np.array(data)
    return np.expand_dims(_, axis=2).astype("uint8")


def get_id_coords(set_id):
    # Default value if set_id doesn't match any conditions
    id_coord = None

    # left sets
    if set_id == ('sv3', 'sv4', 'sv3pt5', 'sv2'):
        id_coord = (285, 75, 550, 125) # checked
    elif set_id in ('swsh9', 'swsh6', 'swsh12pt5', 'swsh10','swsh45'):
        id_coord = (280, 75, 540, 125) # checked
    elif set_id == 'sm4':
        id_coord = (200, 70, 470, 90) # checked

    # right sets
    elif set_id == ('dv1', 'g1'):
        id_coord = (210, 90, 390, 110) # checked
    elif set_id == 'xy1':
        id_coord = (150, 90, 340, 110) # checked
    elif set_id in ('xy2', 'xy3'):
        id_coord = (150, 90, 335, 110) # checked
    elif set_id in ('xy4', 'xy6', 'xy7'):
        id_coord = (150, 90, 335, 115) # checked
    elif set_id in ('dp1', 'dp2'):
        id_coord = (210, 100, 400, 150) # checked

    # Raise an error if id_coord was not set
    if id_coord is None:
        raise ValueError(f"Invalid set_id: {set_id}")

    return id_coord

def ocr_preprocessor(img_input,set_id):
    img_input = preproc_clean(img_input).squeeze()

    # Gray scale
    gray_img = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

    img = Image.fromarray(gray_img)
    side_offset = 20 # we need to fine tune this
    a,b,c,d = get_id_coords(set_id)
    img_contrast = img.crop((a, b-side_offset, c+side_offset, d+ side_offset))

    contrast_enhancer = ImageEnhance.Contrast(img_contrast)
    img_contrast = contrast_enhancer.enhance(1)
    median = np.median(img_contrast)
    mean = np.mean(img_contrast)

    im_offset = np.clip(img_contrast, 0, 255).astype("uint8")
    im_offset = Image.fromarray(im_offset)

    im_offset = ImageOps.invert(im_offset)

    return im_offset

def ocr_text(img):
    img = preproc_clean(img).squeeze()
    tools = pyocr.get_available_tools()
    tool = tools[0]

    builder = pyocr.builders.TextBuilder(tesseract_layout=6)
    # builder.tesseract_configs.append("digits")
    builder.tesseract_configs.append("-c")
    builder.tesseract_configs.append("tessedit_char_whitelist=0123456789/")
    builder.tesseract_configs.append("--psm")
    builder.tesseract_configs.append("6")  # 0~13
    builder.tesseract_configs.append("--oem")
    builder.tesseract_configs.append("3")  # 0~3
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    result = tool.image_to_string(img, lang="eng", builder=builder)

    return result


def get_pokeid(img,set_id):
    """Returns pokemon card number written on the corner"""
    preprocessed_img = ocr_preprocessor(img,set_id)
    return ocr_text(preprocessed_img)
