import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models


from pokedex.augmentation import get_augment_data
from pokedex import HARD_CODED_WIDTH, HARD_CODED_HEIGHT, BUCKET_NAME


def preprocessing(path):
    """
    Preprocess the data to extract x and y, with train, val and test sets
    path = '../../raw_data/dict_reduceddataset_left.json'
    """
    df = get_augment_data(path)
    df = shuffle(df).reset_index(drop=True)

    # Encoding the target
    label_encoder = LabelEncoder()
    label_encoder.fit(df['set_id']) # Fit LabelEncoder to your column
    df['target'] = label_encoder.transform(df['set_id']) # Transform the column
    categories = to_categorical(df['target'])

    # Split into train, val, and test
    nb_split1 = int(df.shape[0]*0.7)
    nb_split2 = int(df.shape[0]*0.85)

    X_train = df.loc[:nb_split1-1, 'corner']
    X_val = df.loc[nb_split1:nb_split2-1, 'corner']
    X_test = df.loc[nb_split2:, 'corner']

    # Normalizing the images
    XX_train = np.empty((nb_split1, HARD_CODED_HEIGHT, HARD_CODED_WIDTH, 1))
    for i, x in enumerate(X_train):
        XX_train[i,:,:, :] = x/255
    XX_val = np.empty((nb_split2 - nb_split1, HARD_CODED_HEIGHT, HARD_CODED_WIDTH, 1))
    for i, x in enumerate(X_val):
        XX_val[i,:,:,:] = x/255
    XX_test = np.empty((df.shape[0] - nb_split2, HARD_CODED_HEIGHT, HARD_CODED_WIDTH, 1))
    for i, x in enumerate(X_test):
        XX_test[i,:,:,:] = x/255

    y_train = categories[:nb_split1,:]
    y_val = categories[nb_split1:nb_split2,:]
    y_test = categories[nb_split2:,:]

    return XX_train, y_train, XX_val, y_val, XX_test, y_test, label_encoder


def load_model(side):

    # creating label encoder for testing
    df = pd.read_json('/Users/estelle/code/AoesJP/project_pokereader/raw_data/dict_reduceddataset_right.json')
    label_encoder = LabelEncoder()
    label_encoder.fit(df['set_id']) # Fit LabelEncoder to your column

    # client = storage.Client()
    # blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="Model"))

    # try:
    #     latest_blob = max(blobs, key=lambda x: x.updated)
    #     latest_model_path_to_save = '../raw_data/models/'

    #     latest_blob.download_to_filename(latest_model_path_to_save)
    #     latest_model = models.load_model(latest_model_path_to_save)

    #     print("✅ Model downloaded from cloud storage")

    #     return latest_model, label_encoder

    # except:
    #     print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

    #     return None

    path = '/Users/estelle/code/AoesJP/project_pokereader/raw_data/models/Models_set_right_slim-19.keras'

    latest_model = models.load_model(path)

    print("✅ Model loaded from local disk")

    return latest_model, label_encoder
