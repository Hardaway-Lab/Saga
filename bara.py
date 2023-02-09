"""
    Parse the data
X       1.  Open a CSV
X       2.  Deinterleive the data
X       3.  Smooth Data
X       4.  Bi-exponential Fit
X       5.  "Correct" the data
X       6.  Extract Fed Events
X       7.  Split on Events
X       8.  Normalize Event Data
X       9.  Average Across the Data
        10. SEM
        11. Stats???
"""

import numpy as np
import pandas as pd
import scipy


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Loads a file given the path,
    returns a data_frame
    """
    data_frame = pd.read_csv(file_path)
    return data_frame


def deinterleive(
    data_frame: pd.DataFrame, column_label: str, keys: list = None
) -> tuple[pd.DataFrame]:
    """
    given a DataFrame, will split into DataFrames in the given column (label given as string)
    by either the key values specified, or by unique values in sorted order
    returns a tuple of these data frames
    """
    output = dict()
    if keys == None:
        keys = sorted(set(data_frame[column_label]))

    for key in keys:
        output[key] = data_frame[data_frame[column_label] == key].reset_index(drop=True)
    return tuple(output.values())


def truncate(data_frame: pd.DataFrame, size: int, from_front=True) -> pd.DataFrame:
    """
    Trims a data_frame to be a specific number of rows long,
    By default will trim off the front of the data ie
    [1, 2, 3, 4], 3, from_front=True
       [2, 3, 4]
    returns a DataFrame
    """
    if from_front:
        return data_frame.iloc[-size:].reset_index(drop=True)
    return data_frame[:size]


def smallest_size(frames: list[pd.DataFrame]) -> int:
    """
    Takes a list of frames and returns the length of the smallest data_frame
    """
    return min(frame.shape[0] for frame in frames)


def smooth(
    data_frame: pd.DataFrame,
    column_labels: list[str],
    window_size: int = 20,
    window: list[float] = None,
) -> pd.DataFrame:
    """
    smoothes the data given a window size or a specific window to convole with
    the default type of window is a flat (ie equal weights)
    returns a DataFrame with the smoothed values

    """
    data_frame = data_frame.copy()

    if window == None:
        window = np.ones(window_size) / window_size

    window_size = len(window)

    for column in column_labels:
        index = data_frame.columns.get_loc(column)
        values = np.convolve(window, data_frame[column], mode="valid")
        data_frame.iloc[window_size // 2 - 1 : -window_size // 2, index] = values

    return data_frame.iloc[window_size // 2 - 1 : -window_size // 2].reset_index(
        drop=True
    )


def bi_exponental(
    x_data: list[float], a: float, b: float, c: float, d: float
) -> list[float]:
    """
    Bi-exponential function used to correct for photobleaching
    Takes the x_data (timestamps) and parameters in the form
    A * e^(B/x) + C * e^(D/x)
    """
    return a * np.exp(b / x_data) + c * np.exp(d / x_data)


def fit_curve(
    x_data: list[float], signal: list[float], function, maxfev=50000, p0=None
) -> tuple[float]:
    """
    takes x_data (time stamps), signal, function (model to fit with)
    optional arguments maxfev (how many iterations to run the optimize for), p0(inital guess)
    retuns the best matching parameters
    """
    best, *_ = scipy.optimize.curve_fit(function, x_data, signal, maxfev=maxfev, p0=p0)
    return best


def correct_photobleach(
    data_frame: pd.DataFrame,
    time_column: str,
    signal_columns: list[str],
    function=bi_exponental,
    maxfev: int = 50000,
    p0: list[float] = None,
) -> pd.DataFrame:
    """
    Given a DataFrame, a label to the time data's column, and the signal columns,
    fits a fucntion to the data and will divide by this correction to fix the photobleaching
    """
    data_frame = data_frame.copy()

    for column in signal_columns:
        best = fit_curve(data_frame[time_column], data_frame[column], function)
        data_frame[column] = data_frame[column] / bi_exponental(
            data_frame[time_column], *best
        )
    return data_frame


def correct_reference(
    data_frame: pd.DataFrame, refernce_frame: pd.DataFrame, signal_columns: list[str]
) -> pd.DataFrame:
    """
    Divides by the refernce signal for each signal
    returns a data frame with the correct signal
    """
    data_frame = data_frame.copy()
    for column in signal_columns:
        data_frame[column] = data_frame[column] / refernce_frame[column]
    return data_frame


def time_to_index(
    data_frame: pd.DataFrame, time_column_label: str, time_stamp: float
) -> int:
    """
    Finds the closest index to a timestamps
    """
    return np.abs(data_frame[time_column_label] - time_stamp).argmin()


def fed_events(
    data_frame: pd.DataFrame,
    time_column_label: str,
    event_column_label: str,
    time_span: float,
) -> list[pd.DataFrame]:
    """
    Given a dataframe and a columnn,
    splits the data based on the rising edge of the fed event with a total time of time_span(s)
    half the time is before and half the time is after the fed event
    returns a list of these split segments
    """
    events = list(
        map(
            lambda a, b: a < b,
            data_frame[event_column_label].iloc[:-1],
            data_frame[event_column_label].iloc[1:],
        )
    ) + [False]

    time_stamps = data_frame[events][time_column_label]
    output = []
    for time in time_stamps:
        start = time_to_index(data_frame, time_column_label, time - time_span / 2)
        end = time_to_index(data_frame, time_column_label, time + time_span / 2)
        output.append(data_frame.iloc[start:end].reset_index(drop=True))
    return output



def normalize_df(epochs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
        Returns the normalized signal for each epoch based on
        (signal - baseline_mean) / baseline_mean
    """
    normalize_epochs = []
    for epoch in epochs:
        baseline_mean = epoch.Signal.iloc[:epoch.Signal.shape[0]//2].mean()
        normalize_epoch = epoch.copy()
        normalize_epoch.Signal = (epoch.Signal - baseline_mean) / baseline_mean
        normalize_epochs.append(normalize_epoch)
    return normalize_epochs


def normalize_z(epochs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
        Returns the normalized signal for each epoch based on
        (signal - baseline_mean) / baseline_std
    """
    normalize_epochs = []
    for epoch in epochs:
        baseline_mean = epoch.Signal.iloc[:epoch.Signal.shape[0]//2].mean()
        baseline_std = epoch.Signal.iloc[:epoch.Signal.shape[0]//2].std()
        normalize_epoch = epoch.copy()
        normalize_epoch.Signal = (epoch.Signal - baseline_mean) / baseline_std
        normalize_epochs.append(normalize_epoch)
    return normalize_epochs

def signal_sems(data_frame:pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Mean":data_frame.mean(axis=1),
            "SEM":data_frame.sem(axis=1)
        }
    )
