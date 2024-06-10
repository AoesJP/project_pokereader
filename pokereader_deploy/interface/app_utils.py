import cv2
from pathlib import Path
import streamlit as st

HERE = Path(__file__).parent

# List of Pokemon card Rarities
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

# Dictionary of pokemon card Set ID and Set Name
SETS = {
    'dv1': 'Dragon Vault',
    'swsh9': 'Brilliant Stars',
    'swsh45': 'Shining Fates',
    'swsh6': 'Chilling Reign',
    'swsh12pt5': 'Crown Zenith',
    'xy1': 'XY',
    'xy2': 'Flashfire',
    'xy3': 'Furious Fists',
    'g1': 'Generations',
    'xy4': 'Phantom Forces',
    'xy6': 'Roaring Skies',
    'xy7': 'Ancient Origins',
    'dp1': 'Diamond & Pearl',
    'dp2': 'Mysterious Treasures',
    'sm4': 'Crimson Invasion',
    'swsh10': 'Astral Radiance',
    'sv4': 'Paradox Rift',
    'sv3pt5': '151',
    'sv3': 'Obsidian Flames',
    'sv2': 'Paldea Evolved'
    }

LOGO_PATH = str(HERE / 'PokeReader_Logo.png')
TEAMROCKET_PATH = str(HERE / 'Team_Rocket.png')
CORNERS_PATH = str(HERE / 'corners.jpeg')

@st.cache_data
def get_logo():
    """Function to return PokeReader Logo.
    Displayed at the top of the Streamlit UI.
    """
    logo_rgba = cv2.imread(LOGO_PATH, cv2.IMREAD_UNCHANGED)
    logo_rgb = cv2.cvtColor(logo_rgba, cv2.COLOR_BGRA2RGBA)

    cropped_logo = logo_rgb [400:700, 180:1800]

    return cropped_logo

@st.cache_data
def get_corners():
    """
    Function to return image of Pokemon card corners with Poke IDs.
    Used to guide users in locating the PokeID manually in case not detected
    by Pokereader.
    """
    corners_rgba = cv2.imread(CORNERS_PATH, cv2.IMREAD_UNCHANGED)
    corners = cv2.cvtColor(corners_rgba, cv2.COLOR_BGRA2RGBA)

    return corners

@st.cache_data
def get_teamrocket():
    """
    Function to return image of Team Rocket.
    Image is displayed on UI if correct Set ID is not idenitified by PokeReader.
    """
    rocket_rgba = cv2.imread(TEAMROCKET_PATH, cv2.IMREAD_UNCHANGED)
    team_rocket = cv2.cvtColor(rocket_rgba, cv2.COLOR_BGRA2RGBA)

    return team_rocket

def show_rarity(spotlight_rarity):
    """
    Function to display rarities of pokemon cards.
    The rarity of the card detected by PokeReader will be highlighted in purple.
    """
    num_rows = len(RARITIES)
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
    """
    Function to return different emojis depending on level of rarity.
    More exciting emojis are returned the rarer the card is.
    """

    if spotlight_rarity in RARITIES[0]:
        return '💩'
    elif spotlight_rarity in RARITIES[1]:
        return '🔥'
    elif spotlight_rarity in RARITIES[2:14]:
        return '🍾🔥'
    elif spotlight_rarity in RARITIES[14]:
        return '🌈🙌'
    elif spotlight_rarity in RARITIES[15:-1]:
        return '💃🕺🙌'
    elif spotlight_rarity in RARITIES[-1]:
        return '🎉🎉🎉🙌 '

def price_hype(price):
    """
    Function to return different emojis depending on card price.
    More exciting emojis are returned the more expensive the card is.
    """
    if price < 0.5:
        return "😭"
    elif price < 1:
        return "🫠"
    elif price < 2:
        return "🤩"
    else:
        return "🤑"
