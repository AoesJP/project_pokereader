import streamlit as st
import app_utils

import numpy as np
import pandas as pd

import requests

from PIL import Image
from io import BytesIO
import cv2
from matplotlib import pyplot as plt

import app_utils

from app_utils import get_logo

# Displaying the logo
logo = get_logo()
st.image(logo)

# User input to upload pic
uploaded_file = st.file_uploader("### Upload your image here ... ", type=['jpg', 'jpeg', 'png'])


if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    if bytes_data:
        try:
            image = Image.open(BytesIO(bytes_data))
            st.image(image, caption='Uploaded Image.', use_column_width=True)
        except IOError:
            st.error("Cannot identify image file. Please check the file format and try again.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Uploaded file is empty. Please upload a valid image file.")
else:
    st.warning("Please upload an image file.")

if uploaded_file is not None:
    #swsh9-81
    #dv1-2
    s_id = 'dv1'
    p_id = '5'
    url = f'https://api.pokemontcg.io/v2/cards/{s_id}-{p_id}'
    response = requests.get(url).json()

    #Printing the rarity
    rarity = response['data']['rarity']
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

    # Printing the market price in big bold red font
    market_price = response['data']['tcgplayer']['prices']['holofoil']['market']
    st.markdown(f"""<h3 class='general-text'>Trading at <span class='market-price'>{market_price}</span> USD today!</h3>""", unsafe_allow_html=True)


    # Printing the image
    image_url = response['data']['images']['large']
    st.image(image_url)
