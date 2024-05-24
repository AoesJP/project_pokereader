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
        [['dv1', '21', 'Dragon Vault', 'right',20],
         ['swsh9', '186', 'Brilliant Stars', 'left',172],
         ['swsh45', '73', 'Shining Fates', 'left',72],
         ['swsh6', '233', 'Chilling Reign', 'left',198],
         ['swsh12pt5', '160', 'Crown Zenith', 'left',159],
         ['xy1', '146', 'XY', 'right',146],
         ['xy2', '110', 'Flashfire', 'right',106],
         ['xy3', '114', 'Furious Fists', 'right',111],
         ['g1', '117', 'Generations', 'right',83],
         ['xy4', '124', 'Phantom Forces', 'right',119],
         ['xy6', '112', 'Roaring Skies', 'right',108],
         ['xy7', '100', 'Ancient Origins', 'right',98],
         ['dp1', '130', 'Diamond & Pearl', 'right',130],
         ['dp2', '124', 'Mysterious Treasures', 'right',123],
         ['sm4', '126', 'Crimson Invasion', 'left',111],
         ['swsh10', '216', 'Astral Radiance', 'left',189],
         ['sv4', '266', 'Paradox Rift', 'left',182],
         ['sv3pt5', '207', '151', 'left',165],
         ['sv3', '230', 'Obsidian Flames', 'left',197],
         ['sv2', '279', 'Paldea Evolved', 'left',193]])
