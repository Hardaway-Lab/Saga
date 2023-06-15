import pandas as pd
import constants

def load_csv(path: str, usecols: list = []) -> pd.DataFrame:
    """
    Simply using pd.read_csv

    Arguments:
        :path The File path
        :usecols Which cols we want to retrieve
    """
    if len(usecols) > 0:
        return pd.read_csv(path, sep=",", usecols=usecols, index_col=False)
    else:
        return pd.read_csv(path, sep=",", index_col=False)
    

def save_csv(all_trials_dict: dict):
    all_trials_dict['green_right'].to_csv(f"{constants.OUTPUT_CSV_DIR}/{constants.OUTPUT_CSV_PREFIX} GCaMP Left.csv")
    all_trials_dict['green_left'].to_csv(f"{constants.OUTPUT_CSV_DIR}/{constants.OUTPUT_CSV_PREFIX} GCaMP Right.csv")
    all_trials_dict['red_right'].to_csv(f"{constants.OUTPUT_CSV_DIR}/{constants.OUTPUT_CSV_PREFIX} RCaMP Left.csv")
    all_trials_dict['red_left'].to_csv(f"{constants.OUTPUT_CSV_DIR}/{constants.OUTPUT_CSV_PREFIX} RCaMP Right.csv")
