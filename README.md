# PokeReader
Created by [Alex C](https://github.com/AoesJP), [Emilia S](https://github.com/emiliasato), [Estelle G](https://github.com/EstelleGqln), [Yuri S](https://github.com/teddy8193) at Le Wagon Tokyo during the 2-week project.

## Summary
PokeReader is a web app allowing a user to get the price and rarity of a Pokemon card from a picture that they would have taken or uploaded.

<img src="PokeReader_app.png" alt="PokeReader_app" width="200"/>

- App 🔗 https://pokereader.streamlit.app/
- [View GoogleSlides presentation](https://docs.google.com/presentation/d/1Pb1OAsDZ5j1nHlwInYsBzDwfNHyLLF8TBN1zhLPsFgI/edit?usp=sharing)


## Workflow
We have use the Pokemon TCG API to retrieve the price and rarity of a given Pokemon card. To access this information, we need to provide to the API the **set ID** and the **card number** of the Pokemon card we are interested in. 

Each Pokemon card has a set ID and a card number located either at the bottom left, bottom right or middle right part of the card. This location is consistent for all the cards within each set. In our project, we have focused on the Pokemon sets with the ID located at the bottom left or right of the card.

### Set ID retrieval
Here is a list of the sets we have used:
| set ID | set name | location | number of cards |
|--------|----------|----------|-----------------|
| dp1 | Diamond & Pearl | right | 130 | 
| dp2 | Mysterious Treasures | right | 124 | 
| dv1 | Dragon Vault | right | 21 | 
| g1 | Generations | right | 83 |
| sm4 | Crimson Invasion | left | 124 |
| sv2 | Paldea Evolved | left | 279 |
| sv3 | Obsidian Flames | left | 230 |
| sv3pt5 | 151 | left | 207 |
| sv4 | Paradox Rift | left | 266 |
| swsh6 | Chilling Reign | left | 233 | 
| swsh9 | Brilliant Stars | left | 186 | 
| swsh10 | Astral Radiance | left | 216 |
| swsh12pt5 | Crown Zenith | left | 160 | 
| swsh45 | Shining Fates | left | 73 | 
| xy1 | XY | right | 146 | 
| xy2 | Flashfire | right | 109 | 
| xy3 | Furious Fists | right | 113 |
| xy4 | Phantom Forces | right | 122 |
| xy6 | Roaring Skies | right | 110 |
| xy7 | Ancient Origins | right | 100 |

List of sets and get all the cards in each set
create dataset of bottom corners
data augmenation
set id detection with CNN

Card
normalizing with edge detection

bottom cropping
detect set id from CNN model

text detection through OCR

From set id and card number -> API





## Limitations
This project serves as a proof of concept, showcasing the effectiveness of Edge Detection, Convolutional Neural Networks (CNN), and Optical Character Recognition (OCR) techniques in identifying Pokemon cards from images. However, there are several limitations to acknowledge:
- **Picture limitation:** The picture taken by the used needs to have a white background as that is how the edge detection model was trained.
- **Limited Set Data:** While there exist around 100 sets of Pokemon cards, our model is trained on a subset of 20 sets. 
- **Limited Language Support:** The current model is trained solely on English cards.


