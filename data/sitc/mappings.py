kitchen_sensors = [1,1,1,2,2,2,1,1,1,1,2,2,1,2,2,1]

activity_names = [
    "B1_BED_OUT",
    "B2_BED_OUT",
    "B1_JACKET_ON",
    "B2_JACKET_ON",
    "_FRIDGE_OPEN",
    "_FRIDGE_CLOSE",
    "_PREPARE_MEAL",
    "D1_WATER",
    "D2_WATER",
    "D1_EAT",
    "D2_EAT",
    "L1_SIT_DOWN",
    "L2_SIT_DOWN",
    "L1_WATCH_TV",
    "L2_WATCH_TV",
    "L1_STAND_UP",
    "L2_STAND_UP",
    "L1_FALL_DOWN",
    "L2_FALL_DOWN",
    "E1_SHOES_ON",
    "E1_LEAVE_HOUSE",
    "E1_ENTER_HOUSE",
    "E1_SHOES_OFF",
    "W1_BRUSH_TEETH",
    "F1_TAKE_BATH",
    "F1_CLEAN_BATH",
    "B1_JACKET_OFF",
    "B2_JACKET_OFF",
    "B1_BED_IN",
    "B2_BED_IN",
]

activity_mappings = {
    "BED_OUT": 1,
    "JACKET_ON": 2,
    "FRIDGE_OPEN": 3,
    "FRIDGE_CLOSE": 4, 
    "PREPARE_MEAL": 5,
    "WATER": 6, 
    "EAT": 7, 
    "SIT_DOWN": 8,
    "WATCH_TV": 9,
    "STAND_UP": 10,
    "FALL_DOWN": 11, 
    "SHOES_ON": 12, #
    "LEAVE_HOUSE": 13, 
    "ENTER_HOUSE": 14, 
    "SHOES_OFF": 15, 
    "BRUSH_TEETH": 16,
    "TAKE_BATH": 17, 
    "CLEAN_BATH": 18,
    "JACKET_OFF": 19, 
    "BED_IN": 20
}

camera_mappings = {
    'B1': 1,
    'B2': 2,
    'D1': 3,
    'D2': 4,
    'E1': 5,
    'F1': 6,
    'K1': 7,
    'K2': 8,
    'L1': 9,
    'L2': 10,
    'W1': 11,
}

coco2NTU = {
    0 : 4,
    6 : 9,
    8 : 10,
    10 : 11,
    5 : 5,
    7 : 6,
    9 : 7,
    12 : 17,
    14 : 18,
    16 : 19,
    11 : 13,
    13 : 14,
    15 : 15,
}

body25toCOCO = {
    0 : 0,
    2 : 6,
    3 : 8,
    4 : 10,
    5 : 5,
    6 : 7,
    7 : 9,
    9 : 12,
    10 : 14,
    11 : 16,
    12 : 11,
    13 : 13,
    14 : 15,
    15 : 2,
    16 : 1,
    17 : 4,
    18 : 3,
    19 : 14,
    20 : 14,
    21 : 14,
    22 : 11,
    23 : 11,
    24 : 11,
}

data_dir = "/path/to/data/dir"
nr_persons = 16
base_folder = "/path/to/base/folder"
subsequences_csv = "path/to/skeleton-cloud/data/sequences/sequences.csv"