from __future__ import print_function

from collections import namedtuple

# brown bowl: scale = 0.0155 m

BOWL = 'bowl'

BowlDimensions = namedtuple('BowlDimensions', [
    'bottom_diameter', 'top_diameter', 'height', 'thickness', 'color'])

# TODO: make a cheat sheet that displays each bowl

THICKNESS = 1.0 # 0.8 - 1.0 cm

BOWL_PROPERTIES = {
    #'purple':       BowlDimensions(9.5, 12.0, 4.8, THICKNESS, (200, 70, 130)), # cannot detect
    'white':        BowlDimensions(9.7, 10.0, 5.4, THICKNESS, (165, 160, 155)),
    'red':          BowlDimensions(5.9, 11.8, 5.0, THICKNESS, (175, 60, 35)), # orange/red
    'red_speckled': BowlDimensions(9.4, 16.0, 5.2, THICKNESS, (145, 40, 40)), # red
    'brown':        BowlDimensions(7.6, 13.6, 8.1, THICKNESS, (110, 40, 30)),
    'yellow':       BowlDimensions(7.7, 15.4, 6.6, THICKNESS, (170, 110, 20)),
    'blue_white':   BowlDimensions(7.5, 15.1, 8.5, THICKNESS, (90, 100, 135)),
    'green':        BowlDimensions(8.4, 18.9, 8.0, THICKNESS, (105, 105, 50)),
    #'steel':        BowlDimensions(4.9, 20.0, 8.9, THICKNESS, (150, 130, 110)), # cannot detect
    'blue':         BowlDimensions(10.5, 22.3, 9.5, THICKNESS, (25, 30, 60)),
    'tan':          BowlDimensions(12.3, 25.9, 10.9, THICKNESS, (180, 160, 130)),
    # TODO: stripes -> striped
    #'stripes':      BowlDimensions(8.0, 15.5, 6.3, THICKNESS, (200, 200, 200)), # cannot detect

    'small_green':  BowlDimensions(5.3, 11.6, 5.2, THICKNESS, (53, 96, 53)),
    'orange':       BowlDimensions(6.9, 15.4, 6.7, THICKNESS, (200, 100, 23)),
    'small_blue':   BowlDimensions(7.9, 18.8, 8.1, THICKNESS, (20, 93, 131)),
    'purple':       BowlDimensions(9.9, 22.2, 9.5, THICKNESS, (150, 50, 38)),
    'lime':         BowlDimensions(11.9, 25.8, 11.0, THICKNESS, (170, 155, 38)),
    'large_red':    BowlDimensions(13.9, 30.0, 12.5, THICKNESS, (188, 65, 38)),
}
