import cv2
from pathlib import Path
import streamlit as st

HERE = Path(__file__).parent

RARITIES = [
    "Common",
    "Uncommon",
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
    "Amazing Rare",
    "LEGEND",
    "Promo"
]

LOGO_PATH = str(HERE / 'PokeReader_Logo.png')

def lol():
    '''functino to print lol!'''
    print('lol')

def get_logo():
    '''function to return cropped logo for streamlit UI'''
    logo_rgba = cv2.imread(LOGO_PATH, cv2.IMREAD_UNCHANGED)
    logo_rgb = cv2.cvtColor(logo_rgba, cv2.COLOR_BGRA2RGBA)

    cropped_logo = logo_rgb [400:700, 180:1800]

    return cropped_logo

def show_rarity(spotlight_rarity):
    '''function to display rarity of card in colored box'''
    num_rows = 5
    num_columns = len(RARITIES) // num_rows + (len(RARITIES) % num_rows > 0)

    for row in range(num_rows):
        cols = st.columns(num_columns)
        for col_index, col in enumerate(cols):
            rarity_index = row + col_index * num_rows
            if rarity_index < len(RARITIES):
                rarity = RARITIES[rarity_index]
                if rarity == spotlight_rarity:
                    with col:
                        st.markdown(f'<div style="background-color: #6B00FF; color: white; padding: 10px; border: 2px solid black; font-family: Arial;"><b>{rarity}</b></div>', unsafe_allow_html=True)
                else:
                    with col:
                        st.markdown(f'<div style="padding: 10px; border: 1px solid black; font-family: Arial;">{rarity}</div>', unsafe_allow_html=True)

def rarity_emoji(spotlight_rarity):
    '''function to display different emoji depending on card rarity'''
    if spotlight_rarity in RARITIES[0]:
        return '💩...'
    elif spotlight_rarity in RARITIES[1]:
        return '🔥!'
    elif spotlight_rarity in RARITIES[2:14]:
        return '🍾🔥!!'
    elif spotlight_rarity in RARITIES[14]:
        return '🌈🙌 !!!'
    elif spotlight_rarity in RARITIES[15:-1]:
        return '💃🕺🙌 !!!!'
    elif spotlight_rarity in RARITIES[-1]:
        return '🎉🎉🎉🙌 !!!!!'

def price_hype(price):
    '''function to display different emoji depending on card price'''
    if price < 0.5:
        return "...😭"
    elif price < 1:
        return "🫠"
    elif price < 2:
        return "🤩!!"
    else:
        return "🤑!!!"
