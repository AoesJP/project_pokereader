import numpy as np

import pytesseract
import pyocr

import cv2
from PIL import Image, ImageEnhance, ImageOps

import re

from pokedex import SETINFO
from pokedex import ocr


def preproc_clean(data: list):
    """
    Converts image data (list) into a NumPy array, adds an extra dimension,
    and converts the data type to uint8.
    """
    _ = np.array(data)
    return np.expand_dims(_, axis=2).astype("uint8")


def get_id_coords(set_id: str):
    """
    Returns the coordinates of the rectangular portion of an image that includes the PokeID.
    Note: Coordinates are hard-coded per Set ID.
    """

    # Default value if set_id doesn't match any conditions
    id_coord = None

    # Poke ID coordinates for
    if set_id in ("sv3", "sv4", "sv3pt5", "sv2"):
        id_coord = (285, 75, 550, 125)
    elif set_id in ("swsh9", "swsh6", "swsh12pt5", "swsh10", "swsh45"):
        id_coord = (280, 75, 540, 125)
    elif set_id == "sm4":
        id_coord = (190, 70, 359, 90)

    # right sets
    elif set_id in ("dv1", "g1"):
        id_coord = (210, 85, 380, 110)
    elif set_id == "xy1":
        id_coord = (130, 80, 330, 100)
    elif set_id in ("xy2", "xy3"):
        id_coord = (150, 90, 335, 110)
    elif set_id in ("xy4", "xy6", "xy7"):
        id_coord = (150, 90, 335, 115)
    elif set_id in ("dp1", "dp2"):
        id_coord = (210, 100, 400, 150)

    # Raise an error if id_coord was not set
    if id_coord is None:
        raise ValueError(f"Invalid set_id: {set_id}")

    return id_coord


def add_contrast(img: Image, low: float =0.1, high: float=0.95):
    """This function enhances the contrast of a PokeID image for improved OCR results.

    The contrast is adjusted by scaling the pixel values based on the specified
    low and high quantile thresholds.
    """
    np_num = np.array(img, dtype=np.float32) / 255
    np_btm = np.quantile(np_num, low)
    np_num -= np_btm
    np_num = np.clip(np_num, 0, 1)
    np_top = np.quantile(np_num, high)
    np_num = np.clip(np_num / np_top, 0, 1) * 255
    return Image.fromarray(np_num.astype(np.uint8))


def ocr_preprocessor(img_input: list, set_id: str):
    """
    This function applies several preprocessing steps to an image to enhance the OCR performance:
    - Cleans the input data dimension and converts it to a NumPy array.
    - Converts the image to grayscale.
    - Crops the image to the rectangular portion that includes only the PokeID.
    - Enhances the image contrast.
    - Inverts the color scale if the text is white, ensuring that text is black for better OCR recognition.
    """
    img_input = preproc_clean(img_input).squeeze()

    # Gray scale
    gray_img = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

    img = Image.fromarray(gray_img)
    side_offset = 20
    a, b, c, d = get_id_coords(set_id)
    img_contrast = img.crop((a, b - side_offset, c + side_offset, d + side_offset))

    contrast_enhancer = ImageEnhance.Contrast(img_contrast)
    img_contrast = contrast_enhancer.enhance(1)

    im_offset = np.clip(img_contrast, 0, 255).astype("uint8")
    im_offset = Image.fromarray(im_offset)

    if ocr.is_groove(im_offset):  # Black Text
        im_offset = ImageOps.invert(im_offset)
        im_offset = add_contrast(im_offset, 0.2, 0.97)

        im_offset = ImageOps.invert(im_offset)
    else:  # White Text
        im_offset = add_contrast(im_offset, 0.2, 0.97)

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

    if not result:
        # Try 10 times to get something from OCR
        brightness_enhancer = ImageEnhance.Brightness(img)
        contrast_enhancer = ImageEnhance.Contrast(img)
        # sharp_enhancer = ImageEnhance.Sharpness(img)
        for i in range(10):
            rand_floats = 2 * np.random.rand(3) - 1
            img_bright = brightness_enhancer.enhance(1 + rand_floats[2] * 0.2)
            result = tool.image_to_string(img_bright, lang="eng", builder=builder)
            if result:
                break
            img_enhance = contrast_enhancer.enhance(1 + rand_floats[0] * 0.2)
            result = tool.image_to_string(img_enhance, lang="eng", builder=builder)
            if result:
                break
            # img_sharp = sharp_enhancer.enhance(1 + rand_floats[1] * 0.2)
            # result = tool.image_to_string(img_sharp, lang="eng", builder=builder)
            # if result:
            #     break

    return result


def extract_number_before_slash_or_cardtotal(text: str, set_id: str):
    """
    Extracts the number before a slash or the card total sequence from a given text.

    This function processes the input text to extract a number that appears before a slash or a specific
    card total sequence, which is associated with the provided set ID.
    It follows these steps:
    1. Retrieves the card total sequence for the specified set ID from the SETINFO
    2. Removes all special characters from the text except digits, slashes, and the specific sequence.
    3. Searches for the number before a slash if it exists.
    4. If no slash is found, searches for the number before the card total sequence.
    5. If neither is found, extracts the first sequence of digits from the cleaned text.

    """
    # storing the number that should show on right side of slash
    # (i.e., total card number for particular set)
    sequence = SETINFO[SETINFO[:, 0] == set_id][0, 4]

    # Remove all special characters except digits, slash, and the specific sequence
    cleaned_text = re.sub(r"[^0-9/]", "", text)

    # Check if a slash exists in the cleaned text
    if "/" in cleaned_text:
        # Regular expression to find numbers before a slash
        match = re.search(r"(\d+)(?=/)", cleaned_text)
        if match:
            return match.group(1)
        else:
            # If no digits are found before the slash, return an empty string
            return ""
    else:
        # Check if the sequence exists in the cleaned text and is followed by a non-digit or end of string
        pattern = rf"(\d+)(?={sequence}(?!\d))"
        match = re.search(pattern, cleaned_text)
        if match:
            return match.group(1)[:-1]
        else:
            # If no slash and no sequence, extract all digits in the cleaned text
            number_match = re.search(r"\d+", cleaned_text)
            if number_match:
                return number_match.group(0)
            else:
                # If no digits are found, return an empty string or handle it as needed
                return ""


def clean_pokeid(pokeid: str, set_id: str):
    """Cleans the PokeID detected through OCR to ensure a valid PokeID is returned.

    Below steps are performed:
    1. Removes any leading text before a space character.
    2. Extracts the number before a slash or a card total specific to the set ID.
    3. Removes any non-alphanumeric characters.
    4. Adjusts the PokeID based on the specific rules for different set IDs.

    """

    if " " in pokeid:
        pokeid = pokeid.split(" ", 1)[1]

    pokeid = extract_number_before_slash_or_cardtotal(pokeid, set_id)

    pokeid = re.sub(r"[^A-Za-z0-9]", "", pokeid)
    if pokeid != "":
        if set_id == "dv1":
            if len(pokeid) > 2:
                pokeid = pokeid[:2]
            if len(pokeid) == 2 and pokeid[0] == "7":
                pokeid = "1" + pokeid[1:]

        elif set_id in ("swsh9", "swsh45", "swsh6", "swsh12pt5", "swsh10"):
            if len(pokeid) > 3:
                pokeid = pokeid[-3:]
            if len(pokeid) == 3 and pokeid[0] == "7":
                pokeid = "1" + pokeid[1:]

        elif set_id in ("xy1", "xy2", "xy3", "xy4", "xy6", "xy7"):
            if len(pokeid) > 3:
                pokeid = pokeid[-3:]
            if len(pokeid) == 3 and pokeid[0] == "7":
                pokeid = "1" + pokeid[1:]

        elif set_id == "g1":
            if len(pokeid) > 3:
                pokeid = pokeid[-3:]
            if len(pokeid) == 3 and pokeid[0] == "7":
                pokeid = "1" + pokeid[1:]

        elif set_id in (
            "dp1",
            "dp2",
        ):
            if len(pokeid) > 3:
                pokeid = pokeid[-3:]
            if len(pokeid) == 3 and pokeid[0] == "7":
                pokeid = "1" + pokeid[1:]

        elif set_id == "sm4":
            if len(pokeid) > 3:
                pokeid = pokeid[-3:]
            if len(pokeid) == 3 and pokeid[0] == "7":
                pokeid = "1" + pokeid[1:]

        elif set_id in ("sv4", "sv3pt5", "sv3", "sv2"):
            if len(pokeid) > 3:
                pokeid = pokeid[-3:]
            if len(pokeid) == 3 and pokeid[0] == "7":
                pokeid = "1" + pokeid[1:]

    return pokeid


def get_pokeid(img, set_id: str):
    """
    Extracts the PokeID from an image using the provided set ID.

    This function processes the input image to enhance its quality for OCR
    and extracts the PokeID by following these steps:
    1. The corner image is preprocessed for better OCR results.
    2. The OCR model reads the values from the preprocessed image.
    3. The OCR results are cleaned to extract only the characters before the '/'
    """

    preprocessed_img = ocr_preprocessor(img, set_id)
    pokeid = ocr_text(preprocessed_img)
    pokeid_cleaned = clean_pokeid(pokeid, set_id)

    return pokeid_cleaned
