from pokedex import HARD_CODED_WIDTH, HARD_CODED_HEIGHT, INITIAL_HEIGHT, INITIAL_WIDTH
import cv2
import numpy as np


def card_prediction_processing(card):
    """
    Input is a numpy array of size (825, 600)
    gets cropped to (72, 200)
    add dimensions to fit the model imput size
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


def card_ocr_crop(card):
    h, w, d = card.shape
    bottomleft = card[h-HARD_CODED_HEIGHT*3:, :HARD_CODED_WIDTH*3, :]
    bottomright = card[h-HARD_CODED_HEIGHT*3:, w-HARD_CODED_WIDTH*3:, :]

    return bottomleft, bottomright
