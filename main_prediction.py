from pokedex.prediction import card_prediction_processing, card_ocr_crop, get_card_info
from pokedex.edges.deformer import deform_card
from pokedex.text_detection import get_pokeid

import numpy as np


# Get the photo location
photo_path = '../../raw_data/PokemonCards/xy1-94.jpg'

# Process photo to detect edges and cut corners
card_image = deform_card(photo_path) # outputs a card with (HIRES_HEIGHT, HIRES_WIDTH) size


## ---- SET ID ---- ##
# resize card to (INITIAL_WIDTH, INITIAL_HEIGHT)
# crops the corners and grayscale
graybottomleft, graybottomright = card_prediction_processing(card_image)

# Predict the set id for each corner
model_right, label_encoder_right = load_model("right")
model_left, label_encoder_left = load_model("left")

pred_right = model_right.predict(graybottomright)
set_right = label_encoder_right.classes_[np.argmax(pred_right)]

pred_left = model_left.predict(graybottomleft)
set_left = label_encoder_left.classes_[np.argmax(pred_left)]

if set_right == 'no' and set_left == 'no':
    print("Please try again!")
elif set_right != 'no' and set_left != 'no':
    ### User input
    print("User input")

elif set_right == 'no':
    set_id = set_left
elif set_left == 'no':
    set_id = set_right


## ---- NUMBER ID ---- ##
# Input is HIGH RES picture
# Crops the bottome corners
bottomcorner = card_ocr_crop(card_image, set_id)
# OCR reads corner number #
im_text = get_pokeid(bottomcorner,set_id)

# im_text cleaning

## ---- API call ---- ##
(rarity, market_price, image_url) = get_card_info(set_id, poke_id)
