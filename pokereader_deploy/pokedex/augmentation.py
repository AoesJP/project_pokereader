import random
import numpy as np
import pandas as pd

import cv2
import tensorflow as tf
from pokedex import NB_CARDS_PER_SET


def apply_blur(img: np.array) -> np.array:
    """
    Function to add blur to be used within generate_augmented_image function
    """
    img = cv2.convertScaleAbs(img)
    kernel_values = [5, 7]
    selected_kernel = random.choice(kernel_values)
    blurred = cv2.GaussianBlur(img, (selected_kernel, selected_kernel), 0)
    width = img.shape[0]
    height = img.shape[1]
    blurred = blurred.reshape((width,height,1))

    return blurred


def generate_augmented_image(image_np: np.array) -> np.array:
    """
    Generate augmented image with:
    - rotating
    - shifting horizontal and vertical
    - blurring
    """
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=7,
        width_shift_range=3,
        height_shift_range=2,
        preprocessing_function=apply_blur
    )

    it = datagen.flow(image_np, batch_size=1)
    augmented_image = next(it)[0].astype(np.uint8)
    return augmented_image


def transform_array(image_np: np.array) -> np.array:
    """
    Reshape bottom corner images to 3D arrays
    and expand dimension (before) by 1 to match batch size
    """
    width = image_np.shape[0]
    height = image_np.shape[1]
    reshaped_image = image_np.reshape((width, height, 1))
    # Expand dimensions to add batch size
    expanded_image = np.expand_dims(reshaped_image, axis=0)

    return expanded_image


def squeeze_photo(im_array: np.array) -> np.array:
    """
    Remove the extra dimension that had been added from processing reasons
    """
    if im_array.ndim > 3:
        return np.squeeze(im_array, axis=0)

    return im_array


def get_augment_data(dataset_path_name: str) -> pd.DataFrame:
    """
    Combines the functions above to augment the number of
    bottom corner images in each set to NB_CARDS_PER_SET cards
    """
    df = pd.read_json(dataset_path_name)
    # turn all images into np.array
    df['corner'] = [np.array(v) for v in df['corner']]
    # add dimensions to images
    df['corner'] = df['corner'].apply(transform_array)

    set_size = pd.DataFrame(df[['set_id']].value_counts())
    set_size.reset_index(inplace=True)
    # How many cards will need to be added to reach NB_CARDS_PER_SET
    set_size['num_of_aug'] = NB_CARDS_PER_SET-set_size['count']

    for index, row in set_size.iterrows():
        min_idx = min(df[df['set_id'] == set_size.loc[index, 'set_id']].index)
        max_idx = max(df[df['set_id'] == set_size.loc[index, 'set_id']].index)

        for i in range(set_size.loc[index, 'num_of_aug']):
            # randomly picks images out to augment them
            i_rand = np.random.randint(min_idx, high=max_idx)
            image_ = df.loc[i_rand, 'corner']
            augmented_image = generate_augmented_image(image_)

            new_df = df.iloc[[i_rand]].copy()
            # add newly augmented image to dataframe
            new_df['corner'] = [augmented_image]

            df = pd.concat([df, new_df], axis=0, ignore_index=True)

    df['corner'] = df['corner'].apply(squeeze_photo)

    return df
