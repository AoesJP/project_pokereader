import streamlit as st
import matplotlib.pyplot as plt

import requests
from PIL import Image
from io import BytesIO
import base64

from interface.app_utils import get_logo, get_teamrocket, get_corners, SETS, show_rarity, rarity_emoji, price_hype

from pokedex.edges.deformer import deform_card
from pokedex.prediction import get_card_info


def main():
    """
    Main function for Streamlit app.
    - Set the page configuration
    - Display the PokeReader Logo
    - Ask user to upload picture of Pokemon card and store as image
    - Run edge detection to crop card image from background
    -
    """
    # Setting page configuration
    st.set_page_config(
        page_title="Pokereader streamlit",
        page_icon="🐍",
        layout="wide",
        initial_sidebar_state="auto",
    )

    # Displaying PokeReader logo
    logo = get_logo()
    st.image(logo)

    # User input to upload Pokemon card picture
    uploaded_file = st.file_uploader(r"$\textsf{\large Upload a Pokemon card picture...}$", type=["jpg", "jpeg", "png"])

    # Store uploaded picture as image
    uploaded = False
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        if bytes_data:
            try:
                image = Image.open(BytesIO(bytes_data))
                uploaded = True
            except IOError:
                st.error("Cannot identify image file. Please check the file format and try again.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("Uploaded file is empty. Please upload a valid image file.")

    edge_detection = False
    ## START RUNNING ONCE IMAGE IS UPLOADED ##
    if uploaded == True:
        try:
            # Run edge detection on image to isolate card from background
            card_image = deform_card(image)
            co = st.columns(3)
            co[1].image(card_image)
            edge_detection = True
        except:
            st.warning("We could not recognize your card. Please try to upload another image.")

        # Store running Pikachu GIF to be used during loading time
        pikachu = "https://media.tenor.com/SH31iAEWLT8AAAAi/pikachu-running.gif"
        co = st.columns(3)
        with co[1].image(pikachu):
            ### ----- MODEL API REQUEST ----- ###
            predicted = False
            if card_image is not None:
                buf = BytesIO()
                plt.imsave(buf, card_image, format="png")
                encoded_image_bytes = buf.getvalue()
                file = {"file": encoded_image_bytes}

                response = requests.post("https://poke-api-cloud-na-yjefrbroka-uw.a.run.app/predict", files=file)
                ### STORE THE SET_ID AND POKE_ID FROM PREDICTION ##
                if response.status_code == 200:
                    set_id = response.json()["set_id"]
                    poke_id = response.json()["poke_id"]
                    st.success("Set name: %s, Poke ID: %s" % (SETS[set_id], poke_id))
                    predicted = True
                else:
                    st.error("Failed to get prediction")
            ### ---------- ###

        correct_card = False

        ## START RUNNING IF EDGE_DETECTION and PREDICTION STEP IS COMPLETE ##
        if edge_detection == True and predicted == True:
            if poke_id == "": # If PokeID is not detected, ask for manual input
                st.write("Poke ID could not be retrieved.")
                imcorners = get_corners()
                co = st.columns(3)
                co[1].image(imcorners)
                poke_id = st.number_input("Please input Poke ID by hand as shown above:", step=1, value=0)

            if poke_id != "" and poke_id != 0:
                # If PokeID is detected, obtain rarity/market price/image_url from Pokemon API
                rarity, market_price, image_url = get_card_info(set_id, int(poke_id))

                # Ask user if it is the correct card
                user_input = st.radio("Is this the correct card?", ("Absolutely :)", "Not Quite :("))
                if user_input == "Not Quite :(":
                    st.write("Please try uploading another pic... Sorry!")
                    team_rocket = get_teamrocket()
                    left_co, cent_co, last_co = st.columns(3)
                    cent_co.image(team_rocket)
                elif user_input == "Absolutely :)":
                    correct_card = True
                    left_co, cent_co, last_co = st.columns(3)
                    cent_co.image(image_url)

        if correct_card == True and edge_detection == True:
            # Display the API pokemon card
            st.markdown(
                """
            <style>
            .big-red {
                font-size: 24px;
                font-weight: bold;
                color: #FF0000;  # Red color
            }
            .right-align {
                text-align: right;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )
            # Use the styles in your Markdown
            emoji = rarity_emoji(rarity)
            price_emoji = price_hype(market_price)
            st.markdown(f"## Card value is ${market_price} today {price_emoji}")
            st.markdown(f"## Your card is {rarity.upper()} {emoji}")

            if st.checkbox("Display Rarity table"):
                show_rarity(rarity)
