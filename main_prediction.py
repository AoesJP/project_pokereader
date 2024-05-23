from pokedex.prediction import card_prediction_processing, card_ocr_crop
from pokedex.edges.deformer import deform_card

import numpy as np


# Get the photo location
photo_path = '../../raw_data/PokemonCards/xy1-94.jpg'

# Process photo to detect edges and cut corners
card_image = deform_card(photo_path)


## ---- SET ID ---- ##
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
# OCR on high res picture
set_id
bottomleft, bottomright = card_ocr_crop(card_image)
# -- add OCR model -- #

number_id



## ---- API call ---- ##
set_id
number_id
