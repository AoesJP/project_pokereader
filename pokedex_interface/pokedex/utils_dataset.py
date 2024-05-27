import pandas as pd
import numpy as np
import requests
import cv2
from PIL import Image
from io import BytesIO
from pokedex import HARD_CODED_WIDTH, HARD_CODED_HEIGHT, INITIAL_HEIGHT, INITIAL_WIDTH
from pokedex import SETINFO, REDUCED_SET

def create_dataset():
    '''
    This function creates the full working dataset we will use for training the data.
    It does not save it though. Returns the dataframe
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


            # url = f'https://api.pokemontcg.io/v2/cards/{s_id}-{str(i)}'
            # # get info about card to get url
            # response = requests.get(url)
            # if response.status_code == 200:
            #     image_url = response.json()['data']['images']['large']


            image_url = f'https://images.pokemontcg.io/{s_id}/{str(i)}_hires.png'

            # Send a GET request to the image URL to retrieve the image
            response_card = requests.get(image_url)
            # Check if the request was successful
            if response_card.status_code == 200:
                # Open the image using PIL
                image = Image.open(BytesIO(response_card.content))

                # Save the image to a file
                # image.save(f"../raw_data/pokemon_cards_api/card_{s_id}_{str(i)}.png")
                # width, height = image.size

                # crop bottom corners of the card
                card_image = np.array(image)

                new_card = cv2.resize(card_image, (INITIAL_WIDTH, INITIAL_HEIGHT))

                h, w, d = new_card.shape
                bottomleft = new_card[h-HARD_CODED_HEIGHT:, :HARD_CODED_WIDTH, :]
                bottomright = new_card[h-HARD_CODED_HEIGHT:, w-HARD_CODED_WIDTH:, :]
                graybottomleft = cv2.cvtColor(bottomleft, cv2.COLOR_BGR2GRAY)
                graybottomright = cv2.cvtColor(bottomright, cv2.COLOR_BGR2GRAY)

                # bottomleft = image.crop((width*0.75, height*0.93, width, height))
                # bottomright = image.crop((0, height*0.93, width*0.25, height))
                # graybottomleft = cv2.cvtColor(np.array(bottomleft), cv2.COLOR_BGR2GRAY)
                # graybottomright = cv2.cvtColor(np.array(bottomright), cv2.COLOR_BGR2GRAY)

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
        # else:
        #     print(f"Failed to retrieve image url. HTTP Status code: {response.status_code}")

    return dataset_df



def reduce_dataset(path):
    """
    Imports the full dataset and drops as many rows as necessary to have maximum 150 cards per set
    before data augmentation.
    Separate left and right dataset.
    It saves both the datasets as json. Return nothing.
    """

    df = pd.read_json(path) # '../raw_data/dict_dataset_full.json'

    # setinfo_left = setinfo[setinfo[:,3] == 'left']
    # setinfo_right = setinfo[setinfo[:,3] == 'right']

    def df_side(setinfo, side):
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

    df_small_left = df_side(SETINFO, side = 'left')
    df_small_left.to_json('../../raw_data/dict_reduceddataset_left.json')

    df_small_right = df_side(SETINFO, side = 'right')
    df_small_right.to_json('../../raw_data/dict_reduceddataset_right.json')

    return None
