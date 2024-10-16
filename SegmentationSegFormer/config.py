ALL_CLASSES = [
    'background',
    'blood/saline',
    'dura',
    'pia',
    'csf',
    'spinal cord',
    'hematoma',
    'dura/pia complex',
    'extradural space',
    'dura/extradural space', 
]

LABEL_COLORS_LIST = [
    [0, 0, 0], # background
    [128, 0, 128], # blood 
    [128, 0, 0], # dura 
    [0, 128, 0], # pia 
    [128, 128, 0], # csf 
    [0, 0, 128],  # spinal cord 
    [0, 128, 128], # hematoma 
    [64, 0, 0], # dura pia complex 
    [128, 128, 128], # extradural 
    [192, 0, 0] # dura bone complex 
]

VIS_LABEL_MAP = [
    [0, 0, 0],
    [128, 0, 128],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [0, 128, 128],
    [64, 0, 0], 
    [128, 128, 128],
    [192, 0, 0]
]