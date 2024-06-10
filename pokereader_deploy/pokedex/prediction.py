from pokedex import HARD_CODED_WIDTH, HARD_CODED_HEIGHT, INITIAL_HEIGHT, INITIAL_WIDTH
from pokedex import SETINFO, RATIO, HIRES_WIDTH, HIRES_HEIGHT
import cv2
import numpy as np
import requests


def card_prediction_processing(card: np.array):
    """
    Import full size card
    Slice the card to get bottom left and right corners
    convert to gray scale, resize to add a dimension for processing, normalize

    returns left and right processed bottom corner
    """
    card_image = cv2.resize(card, (INITIAL_WIDTH, INITIAL_HEIGHT))

    h, w, d = card_image.shape
    bottomleft = card_image[h-HARD_CODED_HEIGHT:, :HARD_CODED_WIDTH, :]
    bottomright = card_image[h-HARD_CODED_HEIGHT:, w-HARD_CODED_WIDTH:, :]

    graybottomleft = cv2.cvtColor(np.array(bottomleft), cv2.COLOR_BGR2GRAY)
    graybottomright = cv2.cvtColor(np.array(bottomright), cv2.COLOR_BGR2GRAY)

    graybottomleft = np.expand_dims(graybottomleft, -1)
    graybottomleft = np.expand_dims(graybottomleft, 0)

    graybottomright = np.expand_dims(graybottomright, -1)
    graybottomright = np.expand_dims(graybottomright, 0)

    graybottomleft = graybottomleft/255
    graybottomright = graybottomright/255

    if len(graybottomright.shape) == 4 and len(graybottomleft.shape) == 4:
        return graybottomleft, graybottomright
    else:
        print("Size of the cropped images is not fit to be used for model prediction.")
        return None


def card_ocr_crop(card: np.array, set_id: str) -> np.array:
    """
    Get full size card, resize it with high res values
    Crop the bottom corners for ocr detection
    """
    side = SETINFO[SETINFO[:,0] == set_id][0,3]

    card = cv2.resize(card, (HIRES_WIDTH, HIRES_HEIGHT))

    h, w, d = card.shape
    if side == 'left':
        bottomleft = card[h-HARD_CODED_HEIGHT*RATIO:, :HARD_CODED_WIDTH*RATIO, :]
        return bottomleft
    elif side == 'right':
        bottomright = card[h-HARD_CODED_HEIGHT*RATIO:, w-HARD_CODED_WIDTH*RATIO:, :]
        return bottomright


def get_card_info(set_id: str, poke_id: str | int):
    """
    Get the card info: image url, rarity, price
    from set id and card number
    using the Pokemon TCG API
    """
    url = f'https://api.pokemontcg.io/v2/cards/{set_id}-{str(poke_id)}'
    response = requests.get(url)
    if response.status_code == 200:
        result = response.json()

        rarity = result['data']['rarity']
        market_price = result['data']['cardmarket']['prices']['averageSellPrice']
        image_url = result['data']['images']['large']

        return rarity, market_price, image_url
    else:
        print(f"Failed to retrieve info. HTTP Status code: {response.status_code}")
        return None
