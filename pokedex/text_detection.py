import pytesseract
import pyocr

import numpy as np

import cv2
from PIL import Image,ImageEnhance,ImageOps

def preproc_clean(data: list):
    _ = np.array(data)
    return np.expand_dims(_, axis=2).astype("uint8")

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


def get_id_coords(set_id):
    # Default value if set_id doesn't match any conditions
    id_coord = None

    # left sets
    if set_id == 'sv3':
        id_coord = (115, 17, 205, 40)
    elif set_id in ('swsh9', 'swsh6', 'swsh12pt5', 'swsh10'):
        id_coord = (110, 17, 190, 40)
    elif set_id in ('sv4', 'sv3pt5', 'sv2'):
        id_coord = (115, 17, 205, 40)
    elif set_id == 'sm4':
        id_coord = (90, 2, 155, 27)

    # right sets
    elif set_id == 'dv1':
        id_coord = (70, 25, 110, 27)
    elif set_id == 'xy1':
        id_coord = (38, 25, 91, 27)
    elif set_id in ('xy2', 'xy3'):
        id_coord = (36, 26, 91, 32)
    elif set_id == 'g1':
        id_coord = (50, 26, 91, 29)
    elif set_id in ('xy4', 'xy6', 'xy7'):
        id_coord = (40, 24, 91, 29)
    elif set_id in ('dp1', 'dp2'):
        id_coord = (90, 45, 130, 50)

    # Raise an error if id_coord was not set
    if id_coord is None:
        raise ValueError(f"Invalid set_id: {set_id}")

    return id_coord

def ocr_preprocessor(img,set_id):
    img = preproc_clean(img).squeeze()
    img= Image.fromarray(img)
    side_offset = 10 #we need to fine tune this
    a,b,c,d = get_id_coords(set_id)

    img = img.crop((a - side_offset, b-side_offset, c+side_offset, d+ side_offset))
    img_contrast = img.resize((img.width * 3, img.height * 3), Image.BICUBIC)
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
    tool
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
    preprocessed_img = ocr_preprocessor(img,set_id)
    return ocr_text(preprocessed_img)
