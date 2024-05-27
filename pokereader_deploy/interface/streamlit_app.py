import streamlit as st
import cv2

import requests
from PIL import Image
from io import BytesIO
import base64

from pokereader_deploy.interface.app_utils import get_logo,show_rarity
from pokedex.edges.deformer import deform_card
from pokedex import INITIAL_HEIGHT,INITIAL_WIDTH
from pokedex.prediction import get_card_info


def main():
    st.set_page_config(
        page_title="Pokereader streamlit", # => Quick reference - Streamlit
        page_icon="🐍",
        layout="wide", # wide
        initial_sidebar_state="auto") # collapsed

    # Displaying the logo
    logo = get_logo()
    st.image(logo)



    # Camera input
    uploaded_file = st.camera_input("Take a pic of a Pokemon card!")
    # st.write(type(pic))

    # User input to upload pic
    # uploaded_file = st.file_uploader("### Upload your image here ... ", type=['jpg', 'jpeg', 'png'])

    # Displaying the card pic
    columns = st.columns(2)
    # columns[0].image(pic)

    uploaded = True
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        if bytes_data:
            try:
                image = Image.open(BytesIO(bytes_data))
                columns[0].image(image, caption='Uploaded Image.', use_column_width=True)
                uploaded = True
                st.write("Test test")
            except IOError:
                st.error("Cannot identify image file. Please check the file format and try again.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("Uploaded file is empty. Please upload a valid image file.")
    else:
        st.warning("Please upload an image file.")

    # edge detection
    if uploaded == True:
        card_image = deform_card(image)
        columns[1].image(card_image, caption='Cut Image.', use_column_width=True)

    set_id = 'swsh10'
    poke_id = 300

    # ### ----- MODEL API REQUEST ----- ###
    # if st.button("Predict") and card_image is not None:
    #     encoded_image = base64.b64encode(card_image).decode("utf-8")
    #     response = requests.post("http://localhost:8000/predict", json={"image": encoded_image})

    #     if response.status_code == 200:
    #         set_id = response.json()["set_id"]
    #         poke_id = response.json()["poke_id"]
    #         st.success(f"Set ID: {set_id}, Poke ID: {poke_id}")
    #     else:
    #         st.error("Failed to get prediction")
    ### ---------- ###

    # USE THOSE ONES WHILE THE API IS NOT UP



    # Get the info about the card!
    # rarity, market_price, image_url = get_card_info(set_id, poke_id)
    rarity, market_price, image_url = get_card_info(set_id, poke_id)

    st.image(image_url)

    if st.button('click me'):
        # print is visible in the server output, not in the page
        print('button clicked!')
        st.write('I was clicked 🎉')
        st.write('Further clicks are not visible but are executed')
    else:
        st.write('I was not clicked 😞')

    #show_rarity(rarity)


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