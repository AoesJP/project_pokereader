import cv2

def lol():
    print('lol')

LOGO_PATH = '/Users/emiliasato/code/AoesJP/project_pokereader/raw_data/PokeReader_Logo.png'

def get_logo():
    logo_bgr = cv2.imread(LOGO_PATH)
    logo = cv2.cvtColor(logo_bgr, cv2.COLOR_BGR2RGB)
    logo_rgba = cv2.imread(LOGO_PATH, cv2.IMREAD_UNCHANGED)
    logo_rgb = cv2.cvtColor(logo_rgba, cv2.COLOR_BGRA2RGBA)
    cropped_logo = logo_rgb [400:700, 180:1800]
    return cropped_logo
