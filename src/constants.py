from datetime import date

DF_PATH = '../test_data/SP 7-6 FED3 Refeed Trial #1/FED3_Trial_SP 7-6 _FP2021-10-22T11_07_15.csv'
KEY_PATH = '../test_data/SP 7-6 FED3 Refeed Trial #1/FED3_Trial_SP 7-6 _Pellet RetrivalTimestamp 2021-10-22T11_07_14.csv'

OUTPUT_PLOTS_DIR = '../output/plots'
OUTPUT_CSV_DIR = '../output/csv'

START_DEINTERLEAVE = 20404
OUTPUT_PLOTS = True
OUTPUT_CSVS = True

COLUMNS_IN_USE = [
    "FrameCounter",
    "Timestamp",
    "LedState",
    "Stimulation",
    "Output0",
    "Output1",
    "Region0G",
    "Region1G",
    "Region2R",
    "Region3R",
]

WAVELENGTH_LEDSTATE_MAP = {
    0: "system_start0",
    1: "reference",
    2: "green",
    4: "red",
    7: "system_start7",
}

REGION_MAP = {
    "green_right": "Region0G",
    "red_right": "Region2R",
    "green_left": "Region1G",
    "red_left": "Region3R",
}

SMOOTH_WIN = 20

INITIAL_GUESS = {
    "p1": [2.42174821e-01, -1.38588594e04, 1.50226649e00, -2.47104465e02],
    "p2": [2.45457828e-01, -1.22225553e04, 1.70376797e00, -2.14603333e02],
    "p3": [6.02598768e00, -4.55640002e02, 1.38891091e-02, 5.88312168e03],
    "p4": [2.56144082, -1660.1336067, 5.65778637, 53.25176018],
}

OUTPUT_CSV_PREFIX = f'SP {date.today()}'