import pandas as pd
import numpy as np
import requests
import cv2
from PIL import Image
from io import BytesIO

def create_dataset():
    '''
    This function creates the working dataset we will use for training the data.

    '''

    # Array with the info about the sets of interest
    setinfo = np.array(
        [['dv1', '21', 'Dragon Vault', 'right'],
         ['swsh9', '186', 'Brilliant Stars', 'left'],
         ['swsh45', '73', 'Shining Fates', 'left'],
         ['swsh6', '233', 'Chilling Reign', 'left'],
         ['swsh12pt5', '160', 'Crown Zenith', 'left'],
         ['xy1', '146', 'XY', 'right'],
         ['xy2', '110', 'Flashfire', 'right'],
         ['xy3', '114', 'Furious Fists', 'right'],
         ['g1', '117', 'Generations', 'right'],
         ['xy4', '124', 'Phantom Forces', 'right'],
         ['xy6', '112', 'Roaring Skies', 'right'],
         ['xy7', '100', 'Ancient Origins', 'right'],
         ['dp1', '130', 'Diamond & Pearl', 'right'],
         ['dp2', '124', 'Mysterious Treasures', 'right'],
         ['sm4', '126', 'Crimson Invasion', 'left'],
         ['swsh10', '216', 'Astral Radiance', 'left'],
         ['sv4', '266', 'Paradox Rift', 'left'],
         ['sv3pt5', '207', '151', 'left'],
         ['sv3', '230', 'Obsidian Flames', 'left'],
         ['sv2', '279', 'Paldea Evolved', 'left']])

    # Target array that will contain all the info we will need for training
    dataset_df = pd.DataFrame(columns=['corner', 'position', 'set_id', 'set_name'], index=[0])
    k = 0 # index within the final dataframe that will be incremented for every corner

    # Loop over the sets
    for j in range(setinfo.shape[0]):
        s_id = setinfo[j,0]
        print(f'On-going set: {s_id}')

        # Loop over each image in the current set
        for i in range(1,int(setinfo[j,1])+1):


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
                image.save(f"../raw_data/pokemon_cards_api/card_{s_id}_{str(i)}.png")

                width, height = image.size
                # crop bottom corners of the card
                bottomleft = image.crop((width*0.75, height*0.93, width, height))
                bottomright = image.crop((0, height*0.93, width*0.25, height))
                graybottomleft = cv2.cvtColor(np.array(bottomleft), cv2.COLOR_BGR2GRAY)
                graybottomright = cv2.cvtColor(np.array(bottomright), cv2.COLOR_BGR2GRAY)
                # add the cropped corners to dataframe with other info
                # depending if the card info is on the left or right side
                if setinfo[j,3] == 'left':
                    dataset_df.loc[k] = [graybottomleft, setinfo[j,3], setinfo[j,0], setinfo[j,2]]
                    k+=1
                    dataset_df.loc[k] = [graybottomright, 'right', 'no', 'no']
                    k+=1
                elif setinfo[j,3] == 'right':
                    dataset_df.loc[k] = [graybottomleft, 'left', 'no', 'no']
                    k+=1
                    dataset_df.loc[k] = [graybottomright, setinfo[j,3], setinfo[j,0], setinfo[j,2]]
                    k+=1
            else:
                print(f"Failed to retrieve image. HTTP Status code: {response_card.status_code}")
        # else:
        #     print(f"Failed to retrieve image url. HTTP Status code: {response.status_code}")

    return dataset_df



def reduce_dataset(path):
    setinfo = np.array(
        [['dv1', '21', 'Dragon Vault', 'right'],
         ['swsh9', '186', 'Brilliant Stars', 'left'],
         ['swsh45', '73', 'Shining Fates', 'left'],
         ['swsh6', '233', 'Chilling Reign', 'left'],
         ['swsh12pt5', '160', 'Crown Zenith', 'left'],
         ['xy1', '146', 'XY', 'right'],
         ['xy2', '110', 'Flashfire', 'right'],
         ['xy3', '114', 'Furious Fists', 'right'],
         ['g1', '117', 'Generations', 'right'],
         ['xy4', '124', 'Phantom Forces', 'right'],
         ['xy6', '112', 'Roaring Skies', 'right'],
         ['xy7', '100', 'Ancient Origins', 'right'],
         ['dp1', '130', 'Diamond & Pearl', 'right'],
         ['dp2', '124', 'Mysterious Treasures', 'right'],
         ['sm4', '126', 'Crimson Invasion', 'left'],
         ['swsh10', '216', 'Astral Radiance', 'left'],
         ['sv4', '266', 'Paradox Rift', 'left'],
         ['sv3pt5', '207', '151', 'left'],
         ['sv3', '230', 'Obsidian Flames', 'left'],
         ['sv2', '279', 'Paldea Evolved', 'left']])

    df = pd.read_json(path) # '../raw_data/dict_dataset_full.json'

    # setinfo_left = setinfo[setinfo[:,3] == 'left']
    # setinfo_right = setinfo[setinfo[:,3] == 'right']

    def df_side(setinfo, side):
        setinfo = setinfo[setinfo[:,3] == side]

        set_list = setinfo[:,0]
        set_list = np.append(set_list, 'no')

        nb_cards = np.array([sum((df['set_id'] == set_list[i]) & (df['position'] == side)) for i in range(set_list.shape[0])])
        set_to_dropcards = np.vstack((set_list[nb_cards > 150], nb_cards[nb_cards > 150]))

        df_small = pd.DataFrame()

        for i in range(set_to_dropcards.shape[1]):
            idx = np.random.randint(0, high=int(set_to_dropcards[1,i]), size=150, dtype=int)
            df_small = pd.concat([ df_small, df[(df['set_id'] == set_to_dropcards[0,i]) & (df['position'] == side)].iloc[idx] ], axis=0, ignore_index=True)

        for i in range(len(set_list[nb_cards <= 150])):
            df_small = pd.concat([ df_small, df[df['set_id'] == set_list[nb_cards <= 150][i]] ], axis=0, ignore_index=True)

        return df_small

    df_small_left = df_side(setinfo, side = 'left')
    df_small_left.to_json('../raw_data/dict_reduceddataset_left.json')

    df_small_right = df_side(setinfo, side = 'right')
    df_small_right.to_json('../raw_data/dict_reduceddataset_right.json')

    return None
