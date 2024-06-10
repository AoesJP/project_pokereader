import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
import tensorflow_datasets as tfds

from pokedex.augmentation import get_augment_data
from pokedex import HARD_CODED_WIDTH, HARD_CODED_HEIGHT


def preprocessing(path: str):
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

def symbols_model(data_path: str):
    """
    Preprocesses the dataset and trains a convolutional neural network (CNN) model for symbol classification.

    This function takes the path to the dataset as input and performs the following steps:
    - Splits the data into train, validation, and test sets.
    - Preprocesses the images using a preprocessing function.
    - Trains a CNN model to classify symbols/categories present in the images.

    Parameters:
        data_path (str): The path to the dataset.

    Returns:
        tuple: A tuple containing the following elements:
            - model (tensorflow.keras.Model): The trained CNN model.
            - history (dict): A dictionary containing training/validation metrics.
            - confusion_matrix (numpy.ndarray): The confusion matrix of the model's predictions.
            - label_encoder (sklearn.preprocessing.LabelEncoder): The label encoder used for encoding symbols.

    Notes:
        - Batch size of 32 is commonly chosen for efficiency and noise reduction.
        - The number of classes (nb_classes) is set to 11, including one additional category of "no symbol."
        - The number of epochs (nb_epochs) is initially set to 20.
        - The neuron sizes are increased significantly in each iteration.
        - ReLU and Softmax activations are used for multi-class classification.
        - Conv2D layers extract features from the images.
        - MaxPool2D layers reduce spatial dimensions.
        - Dropout layers help reduce overfitting.
        - L1 (Lasso) regularization is tried but heavily limits model training.
        - The model is compiled using the Adam optimizer, categorical crossentropy loss, and accuracy metric.

    Examples:
        # Example usage of symbols_model function
        model, history, confusion_matrix, label_encoder = symbols_model('/path/to/dataset')

    """

    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = preprocessing(data_path)

    batch_size = 32
    nb_classes =11
    nb_epochs = 20

    model = Sequential([
        layers.Conv2D(32, (4,4), activation="relu", input_shape=(72, 200, 1)),
        layers.MaxPool2D(pool_size=(2,2)),

        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Dropout(0.1),

        layers.Conv2D(128, (3,3), activation="relu"),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Dropout(0.1),

        layers.Conv2D(128, (2,2), activation="relu"),
        layers.MaxPool2D(pool_size=(2,2)),

        layers.Flatten(),

        layers.Dense(64, activation='relu'),
        layers.Dense(nb_classes, activation='softmax')
])

    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=nb_epochs,
        verbose=1,
        validation_data=(X_val, y_val)
    )

    y_pred_probabilities = model.predict(X_test)
    y_true_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred_probabilities, axis=1)
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)

    return model, history, conf_matrix, label_encoder
