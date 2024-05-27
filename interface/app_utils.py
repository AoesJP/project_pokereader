import cv2
from pathlib import Path
import streamlit as st

HERE = Path(__file__).parent

rarities =  [
      "Amazing Rare",
      "Common",
      "LEGEND",
      "Promo",
      "Rare",
      "Rare ACE",
      "Rare BREAK",
      "Rare Holo",
      "Rare Holo EX",
      "Rare Holo GX",
      "Rare Holo LV.X",
      "Rare Holo Star",
      "Rare Holo V",
      "Rare Holo VMAX",
      "Rare Prime",
      "Rare Prism Star",
      "Rare Rainbow",
      "Rare Secret",
      "Rare Shining",
      "Rare Shiny",
      "Rare Shiny GX",
      "Rare Ultra",
      "Uncommon"
  ]

def lol():
    print('lol')


#LOGO_PATH = '/Users/emiliasato/code/AoesJP/project_pokereader/pokedex_interface/PokeReader_Logo.png'
LOGO_PATH = str(HERE / 'PokeReader_Logo.png')

def get_logo():
    # logo_bgr = cv2.imread('PokeReader_Logo.png')
    # logo = cv2.cvtColor(logo_bgr, cv2.COLOR_BGR2RGB)

    logo_rgba = cv2.imread(LOGO_PATH, cv2.IMREAD_UNCHANGED)
    logo_rgb = cv2.cvtColor(logo_rgba, cv2.COLOR_BGRA2RGBA)

    cropped_logo = logo_rgb [400:700, 180:1800]

    return cropped_logo

def show_rarity(spotlight_rarity):

    num_columns = 4

    num_rows = len(rarities) // num_columns + (len(rarities) % num_columns > 0)

    for row in range(num_rows):
        cols = st.columns(num_columns)
        for col_index, col in enumerate(cols):
            rarity_index = row * num_columns + col_index
            if rarity_index < len(rarities):
                rarity = rarities[rarity_index]
                if rarity == spotlight_rarity:
                    with col:
                        st.markdown(f'<div style="background-color: yellow; padding: 10px; border: 1px solid black; font-family: Arial;"><b>{rarity}</b></div>', unsafe_allow_html=True)
                else:
                    with col:
                        st.markdown(f'<div style="padding: 10px; border: 1px solid black; font-family: Arial;">{rarity}</div>', unsafe_allow_html=True)
