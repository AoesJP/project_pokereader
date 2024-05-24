import streamlit as st

import numpy as np

from PIL import Image
from io import BytesIO
import cv2

from app_utils import get_logo
from pokedex.edges.deformer import deform_card
from pokedex import INITIAL_HEIGHT, INITIAL_WIDTH
from pokedex.prediction import get_card_info

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

### ---------- ###
# SEND card_image TO THE API
# API RETURNS set_id and poke_id
### ---------- ###

set_id = 'swsh10'
poke_id = 8

# Get the info about the card!
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
