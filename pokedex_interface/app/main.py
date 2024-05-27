from typing import Annotated
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.exceptions import RequestValidationError
import joblib
from pydantic import BaseModel
import base64
import pickle
from io import BytesIO
from PIL import Image
import asyncio

import numpy as np

from pokedex.prediction import card_prediction_processing, card_ocr_crop
from pokedex.text_detection import get_pokeid


app = FastAPI()

app.state.model_left = joblib.load("pokereader_left99.pkl")
app.state.model_right = joblib.load("pokereader_right99.pkl")

app.state.label_encoder_left = np.array(
    ["no", "sm4", "sv2", "sv3", "sv3pt5", "sv4", "swsh10", "swsh12pt5", "swsh45", "swsh6", "swsh9"], dtype=object
)
app.state.label_encoder_right = np.array(["dp1", "dp2", "dv1", "g1", "no", "xy1", "xy2", "xy3", "xy4", "xy6", "xy7"], dtype=object)


# Define request/response models
class Prediction(BaseModel):
    set_id: str
    poke_id: str


# Define prediction endpoint
# @app.post("/predict", response_model=Prediction)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("RUNNING PREDICT")
    # print(file.filename)
    try:
        # Convert the image string back to binary data
        # print("Encoded", image.encode("utf-8"))
        # data = await request.body()
        data = await file.read()
        num_byteio = BytesIO(data)
        with Image.open(num_byteio) as img:
            num_numpy = np.asarray(img, dtype="uint8")
        # return {"set_id": "test_id", "poke_id": 99999}
        # print(type(encoded_test))
        # card_image = np.frombuffer(encoded_test, dtype="uint8")
        card_image = num_numpy
        print(card_image.shape)

        # Perform any necessary preprocessing on the image data
        # cutting the corners
        try:
            graybottomleft, graybottomright = card_prediction_processing(card_image)
            print(graybottomleft.shape)
        except Exception as e:
            print(e)
            return {"set_id": "failed", "poke_id": 99999}

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

        if set_right == "no" and set_left == "no":
            print("Please try again!")
        elif set_right != "no" and set_left != "no":
            ### Maybe user input
            print("Too many sets detected.")
        elif set_right == "no":
            set_id = set_left
        elif set_left == "no":
            set_id = set_right

        bottomcorner = card_ocr_crop(card_image, set_id)
        poke_id = get_pokeid(bottomcorner, set_id)

        return {"set_id": set_id, "poke_id": poke_id}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
