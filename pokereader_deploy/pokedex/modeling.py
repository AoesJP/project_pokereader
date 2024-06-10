import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical

from pokedex.augmentation import get_augment_data
from pokedex import HARD_CODED_WIDTH, HARD_CODED_HEIGHT, BUCKET_NAME


def preprocessing(path: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, LabelEncoder]:
    """
    After data augmentation

    Preprocess the data to extract X features and Y targets:
    - Label encoder for set_id (target)
    - normalizing the bottom corner images (features)

    and split into train, val and test sets

    Returns train, val and test sets + the label encoder to reconvert the target into set names.
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
