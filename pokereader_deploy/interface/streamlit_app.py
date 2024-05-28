import streamlit as st
import matplotlib.pyplot as plt

import requests
from PIL import Image
from io import BytesIO

from interface.app_utils import get_logo,show_rarity,rarity_emoji,price_hype,get_teamrocket,get_corners

from pokedex.edges.deformer import deform_card
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
    #uploaded_file = st.camera_input("Take a pic of a Pokemon card!")

    # User input to upload pic
    uploaded_file = st.file_uploader("### Upload a picture of your pokemon card here ... ", type=['jpg', 'jpeg', 'png'])

    # Displaying the card pic
    #columns = st.columns(2)

    uploaded = False
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        if bytes_data:
            try:
                image = Image.open(BytesIO(bytes_data))
                #columns[0].image(image, caption='Uploaded Image.', use_column_width=True)
                # st.image(image, caption='Uploaded Image.', use_column_width=True)
                uploaded = True
            except IOError:
                st.error("Cannot identify image file. Please check the file format and try again.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("Uploaded file is empty. Please upload a valid image file.")

    # edge detection
    edge_detection = False
    if uploaded == True:
        try:
            card_image = deform_card(image)
            st.image(card_image, use_column_width=True) # caption='Cut Image.'
            edge_detection = True
        except:
            "We could not recognize your card. Please try to upload another image."

        ### ----- MODEL API REQUEST ----- ###
        predicted = False
        if card_image is not None:
            buf = BytesIO()
            plt.imsave(buf, card_image, format="png")
            encoded_image_bytes = buf.getvalue()
            file = {"file": encoded_image_bytes}
            response = requests.post("https://poke-api-cloud-instance1-yjefrbroka-an.a.run.app/predict", files=file)

            if response.status_code == 200:
                set_id = response.json()["set_id"]
                poke_id = response.json()["poke_id"]
                st.success(f"Set ID: {set_id}, Poke ID: {poke_id}")
                predicted = True
            else:
                st.error("Failed to get prediction")
        ### ---------- ###

        correct_card = False
        if edge_detection == True and predicted == True:
            # Get the info about the card!
            if poke_id == "":
                st.write('Poke ID could not be retrieved.')
                imcorners = get_corners()
                st.image(imcorners, use_column_width=True)
                poke_id = st.number_input('Please input Poke ID by hand as shown above:', step=1, placeholder="Poke ID...")

            if poke_id != "" and poke_id != 0:
                rarity, market_price, image_url = get_card_info(set_id, int(poke_id))

                # Create a dropdown menu with options 'Yes' and 'No'
                user_input = st.radio('## Is this the correct card?', ('Absolutely :)', 'Not Quite :('))
                if user_input == 'Not Quite :(':
                    st.write("Please try uploading another pic... Sorry!")
                    team_rocket = get_teamrocket()
                    left_co, cent_co,last_co = st.columns(3)
                    with cent_co:
                        st.image(team_rocket, use_column_width=True)
                elif user_input == 'Absolutely :)':
                    correct_card = True
                    left_co, cent_co,last_co = st.columns(3)
                    with cent_co:
                        st.image(image_url, use_column_width=True)

        if correct_card == True and edge_detection == True:
        # Display the API pokemon card
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
            emoji = rarity_emoji(rarity)
            price_emoji = price_hype(market_price)
            st.markdown(f"## Your pokemon card has rarity of...{rarity.upper()} {emoji}")
            st.markdown(f"## It is worth ${market_price} today {price_emoji}")
            show_rarity(rarity)
