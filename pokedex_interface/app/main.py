from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel

import numpy as np

from pokedex.prediction import card_prediction_processing, card_ocr_crop
from pokedex.text_detection import get_pokeid


app = FastAPI()

app.state.model_left = joblib.load("pokereader_left99.pkl")
app.state.model_right = joblib.load("pokereader_right99.pkl")

app.state.label_encoder_left
app.state.label_encoder_right

# Define request/response models
class Prediction(BaseModel):
    set_id: str
    poke_id: str

# Define prediction endpoint
@app.post("/predict", response_model=Prediction)
def predict(image_string: str):
    try:
        # Convert the image string back to binary data
        card_image = image_string.encode('utf-8')

        # Perform any necessary preprocessing on the image data
        # cutting the corners
        graybottomleft, graybottomright = card_prediction_processing(card_image)

        # Perform prediction using the models
        model_left = app.state.model_left
        assert model_left is not None
        model_right = app.state.model_right
        assert model_right is not None

        pred_left = model_left.predict(graybottomleft)
        pred_right = model_right.predict(graybottomright)

        lel = app.state.label_encoder_left
        ler = app.state.label_encoder_right
        set_left = lel.classes_[np.argmax(pred_left)]
        set_right = ler.classes_[np.argmax(pred_right)]

        if set_right == 'no' and set_left == 'no':
            print("Please try again!")
        elif set_right != 'no' and set_left != 'no':
            ### Maybe user input
            print("Too many sets detected.")
        elif set_right == 'no':
            set_id = set_left
        elif set_left == 'no':
            set_id = set_right

        bottomcorner = card_ocr_crop(card_image, set_id)
        poke_id = get_pokeid(bottomcorner, set_id)

        return {"set_id": set_id, "poke_id": poke_id}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
