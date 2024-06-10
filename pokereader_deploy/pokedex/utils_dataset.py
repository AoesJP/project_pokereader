import pandas as pd
import numpy as np
import requests
import cv2
from PIL import Image
from io import BytesIO
from pokedex import HARD_CODED_WIDTH, HARD_CODED_HEIGHT, INITIAL_HEIGHT, INITIAL_WIDTH
from pokedex import SETINFO, REDUCED_SET

def create_dataset() -> pd.DataFrame:
    '''
    Creates the first dataset
    From list of sets that we want to use (SETINFO), we retrieve:
        - all the cards in each set
        - set id for each card (consistent within one set)
        - side of the set id for each card (consistent within one set)
    Each card will have both bottom corners cut:
        - the side with the info will have real set id name
        - the side without the info will have 'no' as set id and full set name

    Returns the following dataframe:
    Bottom corner image | corner side | set id name | full set name
    '''

    # Target array that will contain all the info we will need for training
    dataset_df = pd.DataFrame(columns=['corner', 'position', 'set_id', 'set_name'], index=[0])
    k = 0 # index within the final dataframe that will be incremented for every corner

    # Loop over the sets
    for j in range(SETINFO.shape[0]):
        s_id = SETINFO[j,0]
        print(f'On-going set: {s_id}')

        # Loop over each image in the current set
        for i in range(1,int(SETINFO[j,1])+1):
            # Get the image directly from the link with set id and poke id
            image_url = f'https://images.pokemontcg.io/{s_id}/{str(i)}_hires.png'

            # Send a GET request to the image URL to retrieve the image
            response_card = requests.get(image_url)
            # Check if the request was successful
            if response_card.status_code == 200:
                # Open the image using PIL
                image = Image.open(BytesIO(response_card.content))

                # # Save the pokemon card image
                # image.save(f"../raw_data/pokemon_cards_api/card_{s_id}_{str(i)}.png")

                card_image = np.array(image)
                new_card = cv2.resize(card_image, (INITIAL_WIDTH, INITIAL_HEIGHT))

                # crop bottom corners of the card
                h, w, d = new_card.shape
                bottomleft = new_card[h-HARD_CODED_HEIGHT:, :HARD_CODED_WIDTH, :]
                bottomright = new_card[h-HARD_CODED_HEIGHT:, w-HARD_CODED_WIDTH:, :]
                graybottomleft = cv2.cvtColor(bottomleft, cv2.COLOR_BGR2GRAY)
                graybottomright = cv2.cvtColor(bottomright, cv2.COLOR_BGR2GRAY)

                # add the cropped corners to dataframe with other info
                # depending if the card info is on the left or right side
                if SETINFO[j,3] == 'left':
                    dataset_df.loc[k] = [graybottomleft, SETINFO[j,3], SETINFO[j,0], SETINFO[j,2]]
                    k+=1
                    dataset_df.loc[k] = [graybottomright, 'right', 'no', 'no']
                    k+=1
                elif SETINFO[j,3] == 'right':
                    dataset_df.loc[k] = [graybottomleft, 'left', 'no', 'no']
                    k+=1
                    dataset_df.loc[k] = [graybottomright, SETINFO[j,3], SETINFO[j,0], SETINFO[j,2]]
                    k+=1
            else:
                print(f"Failed to retrieve image. HTTP Status code: {response_card.status_code}")

    return dataset_df



def reduce_dataset(path: str) -> None:
    """
    Creates left and right datasets with max 500 (=REDUCED_SET) cards per set
    (necessary for the artificial 'no' set that has too many cards in it)

    This step is done before data augmentation.

    It saves both the datasets as json. Return nothing.
    """

    # Imports the full dataset created from create_dataset()
    df = pd.read_json(path)

    def df_side(df: pd.DataFrame, setinfo: np.array, side: str) -> pd.DataFrame:
        """
        For chosen side value (left of right), creates the dataset with decided max number of cards
        - finds all the sets that have chosen side, and add 'no' set to the list
        - finds the sets that have more than REDUCED_SET number of cards
        - randomly drop rows to reach that number
        """
        setinfo = setinfo[setinfo[:,3] == side]

        set_list = setinfo[:,0]
        set_list = np.append(set_list, 'no')

        nb_cards = np.array([sum((df['set_id'] == set_list[i]) & (df['position'] == side)) for i in range(set_list.shape[0])])
        set_to_dropcards = np.vstack((set_list[nb_cards > REDUCED_SET], nb_cards[nb_cards > REDUCED_SET]))

        df_small = pd.DataFrame()

        for i in range(set_to_dropcards.shape[1]):
            idx = np.random.randint(0, high=int(set_to_dropcards[1,i]), size=REDUCED_SET, dtype=int)
            df_small = pd.concat([ df_small, df[(df['set_id'] == set_to_dropcards[0,i]) & (df['position'] == side)].iloc[idx] ], axis=0, ignore_index=True)

        for i in range(len(set_list[nb_cards <= REDUCED_SET])):
            df_small = pd.concat([ df_small, df[df['set_id'] == set_list[nb_cards <= REDUCED_SET][i]] ], axis=0, ignore_index=True)

        return df_small

    df_small_left = df_side(df, SETINFO, side = 'left')
    df_small_left.to_json('../../raw_data/dict_reduceddataset_left.json')

    df_small_right = df_side(df, SETINFO, side = 'right')
    df_small_right.to_json('../../raw_data/dict_reduceddataset_right.json')

    return None
