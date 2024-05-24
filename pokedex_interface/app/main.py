import streamlit as st

import numpy as np
import pandas as pd

import requests
from PIL import Image
from io import BytesIO

from app_utils import get_logo
from pokedex.edges.deformer import deform_card
from pokedex.prediction import card_prediction_processing, get_card_info
from pokedex.modeling import load_model

st.set_page_config(
    page_title="Pokereader streamlit", # => Quick reference - Streamlit
    page_icon="🐍",
    layout="wide", # wide
    initial_sidebar_state="auto") # collapsed

# Displaying the logo
logo = get_logo()
st.image(logo)

# User input to upload pic
uploaded_file = st.file_uploader("### Upload your image here ... ", type=['jpg', 'jpeg', 'png'])

columns = st.columns(2)

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    if bytes_data:
        try:
            image = Image.open(BytesIO(bytes_data))
            columns[0].image(image, caption='Uploaded Image.', use_column_width=True)
        except IOError:
            st.error("Cannot identify image file. Please check the file format and try again.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Uploaded file is empty. Please upload a valid image file.")
else:
    st.warning("Please upload an image file.")

# edge detection
card_image = deform_card(image)
columns[1].image(card_image, caption='Cut Image.', use_column_width=True)


# cutting the corners
graybottomleft, graybottomright = card_prediction_processing(card_image)
st.image(graybottomleft, caption='left Image.', use_column_width=True)


# get left and right models
model_right, label_encoder_right = load_model("right")
model_left, label_encoder_left = load_model("left")

pred_right = model_right.predict(graybottomright)
set_right = label_encoder_right.classes_[np.argmax(pred_right)]

st.write(set_right)

pred_left = model_left.predict(graybottomleft)
set_left = label_encoder_left.classes_[np.argmax(pred_left)]

st.write(set_left)

if set_right == 'no' and set_left == 'no':
    st.warning("Please try again!")
elif set_right != 'no' and set_left != 'no':
    st.warning("User input")

elif set_right == 'no':
    set_id = set_left
elif set_left == 'no':
    set_id = set_right


# set_id = 'swsh10'
poke_id = 8

rarity, market_price, image_url = get_card_info(set_id, poke_id)

# Define your CSS styles
st.markdown("""
<style>
.big-red {
    font-size: 25px;
    font-weight: bold;
    color: #FF0000;  # Red color
}
.right-align {
    text-align: right;
}
</style>
""", unsafe_allow_html=True)
# Use the styles in your Markdown
st.markdown("## Your pokemon card has rarity of...")
st.markdown(f"""<h2 class='right-align'>...{rarity.capitalize()}!!!!</h2>""", unsafe_allow_html=True)

st.markdown(f"""<h3 class='general-text'>Trading at <span class='market-price'>{market_price}</span> USD today!</h3>""", unsafe_allow_html=True)
# Printing the image

st.image(image_url)
