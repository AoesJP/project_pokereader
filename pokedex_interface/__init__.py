import numpy as np

RATIO = 3

INITIAL_WIDTH = 600
INITIAL_HEIGHT = 825

HIRES_WIDTH = INITIAL_WIDTH*RATIO
HIRES_HEIGHT = INITIAL_HEIGHT*RATIO

HARD_CODED_WIDTH = 200
HARD_CODED_HEIGHT = 72

NB_CARDS_PER_SET = 1000
REDUCED_SET = 500

SETINFO = np.array(
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
