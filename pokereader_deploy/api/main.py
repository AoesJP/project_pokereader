from fastapi import FastAPI, HTTPException, UploadFile, File
import joblib
from pydantic import BaseModel
import os

from PIL import Image
import io

import numpy as np
import tensorflow

from pokedex.prediction import card_prediction_processing, card_ocr_crop
from pokedex.text_detection import get_pokeid

model_path_right = os.path.join(os.getcwd(), 'pokedex/models', 'pokereader_right99.pkl')
model_path_left = os.path.join(os.getcwd(), 'pokedex/models', 'pokereader_left99.pkl')
app = FastAPI()

app.state.model_left = joblib.load(model_path_left)
app.state.model_right = joblib.load(model_path_right)

app.state.label_encoder_left = np.array(['no', 'sm4', 'sv2', 'sv3', 'sv3pt5', 'sv4', 'swsh10', 'swsh12pt5', 'swsh45', 'swsh6', 'swsh9'])
app.state.label_encoder_right = np.array(['dp1', 'dp2', 'dv1', 'g1', 'no', 'xy1', 'xy2', 'xy3', 'xy4', 'xy6', 'xy7'])

# Define request/response models
class Prediction(BaseModel):
    set_id: str
    poke_id: str

# Define prediction endpoint
@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        card_image = np.array(Image.open(io.BytesIO(contents)))  # Import io module and use BytesIO to open image

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
        set_left = lel[np.argmax(pred_left)]
        set_right = ler[np.argmax(pred_right)]

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
