from __future__ import print_function

from dimensions.bowls.dimensions import BowlDimensions

CUP = 'cup'

# blue cup: scale = 0.01

THICKNESS = 0.5 # 0.4 - 0.5 cm

# All dimensions strictly increasing
CUP_PROPERTIES = {
    'teal': BowlDimensions(4.9, 6.9, 10.3, THICKNESS, (0, 80, 115)),

    'orange':       BowlDimensions(4.2, 5.2, 6.0, THICKNESS, (210, 75, 25)),
    'blue':         BowlDimensions(4.7, 6.0, 6.4, THICKNESS, (0, 80, 150)),
    'green':        BowlDimensions(5.2, 6.4, 6.7, THICKNESS, (70, 150, 30)),
    'yellow':       BowlDimensions(5.8, 7.0, 6.9, THICKNESS, (180, 140, 30)),
    'red':          BowlDimensions(6.3, 7.4, 7.0, THICKNESS, (200, 25, 20)),
    'purple':       BowlDimensions(6.7, 8.0, 7.2, THICKNESS, (100, 50, 125)),
    'large_orange': BowlDimensions(7.2, 8.5, 7.3, THICKNESS, (200, 65, 0)),

    'olive1': BowlDimensions(4.2, 5.6, 7.7, THICKNESS, (70, 70, 25)),
    'olive2': BowlDimensions(4.5, 6.0, 7.8, THICKNESS, (70, 70, 25)),
    'olive3': BowlDimensions(4.8, 6.4, 7.9, THICKNESS, (70, 70, 25)),
    'olive4': BowlDimensions(5.1, 6.6, 8.0, THICKNESS, (70, 70, 25)),

    # -1e-3 due to interpolation
    'blue3D':   BowlDimensions(4.9 - 1e-3, 4.9, 4.6, THICKNESS, (7, 9, 33)),
    'cyan3D':   BowlDimensions(5.4 - 1e-3, 5.4, 5.0, THICKNESS, (86, 113, 135)),
    'green3D':  BowlDimensions(5.7 - 1e-3, 5.7, 5.4, THICKNESS, (90, 110, 50)),
    'yellow3D': BowlDimensions(6.2 - 1e-3, 6.2, 5.7, THICKNESS, (215, 175, 45)),
    'orange3D': BowlDimensions(6.5 - 1e-3, 6.5, 6.1, THICKNESS, (225, 115, 10)),
    'red3D':    BowlDimensions(7.0 - 1e-3, 7.0, 6.5, THICKNESS, (190, 45, 15)),
}

# TODO: apply suffix
OLIVE_CUPS = ['{}_{}'.format(name, CUP) for name in ('olive1', 'olive2', 'olive3', 'olive4')]
THREE_D_CUPS = ['{}_{}'.format(name, CUP) for name in ('blue3D', 'cyan3D', 'green3D', 'yellow3D', 'orange3D', 'red3D')]
