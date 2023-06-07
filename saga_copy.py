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


def load_events(*files: str) -> pd.DataFrame:
    """
    (<file_path1>,<file_path2>,...,<file_path-n>)

    return
    FED_ID | EVENT_FLAG | TIME
        0  |     2      | 1720.1213
        1  |     4      | 1720.2213

    """

    dataframe = pd.DataFrame({"FED_ID": [], "EVENT_FLAG": [], "TIME": []})
    for index, file in enumerate(files):
        temp_file = pd.read_csv(file)
        temp_file = pd.DataFrame(
            {
                "FED_ID": [index] * temp_file.shape[0],
                "EVENT_FLAG": temp_file["EventFlag"],
                "TIME": temp_file["Timestamp"],
            }
        )
        dataframe = pd.concat((dataframe, temp_file))
    return dataframe.sort_values("TIME").reset_index(drop=True)


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
        best = fit_curve(data_frame[time_column].iloc[::10], data_frame[column].iloc[::10], function, maxfev=maxfev)
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


def label_events(singal_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the matching time for every dataframe event and stiches it to the dataframe
    """
    dataframe = singal_df.copy()
    dataframe.insert(dataframe.shape[1], "FED_ID", [np.nan] * dataframe.shape[0])
    dataframe.insert(dataframe.shape[1], "EVENT_FLAG", [np.nan] * dataframe.shape[0])

    for (fed_id, event, time) in events_df.iloc:
        index = time_to_index(dataframe, "Timestamp", time)
        dataframe.at[index, "FED_ID"] = fed_id
        dataframe.at[index, "EVENT_FLAG"] = event
    return dataframe


def get_duration(
    data_frame: pd.DataFrame, time_column_label: str, duration: float, time: float
) -> pd.DataFrame:
    """
    Given a duration time,
    will get a segment of signal around a given time
    """
    start = time_to_index(data_frame, time_column_label, time - duration / 2.0)
    end = time_to_index(data_frame, time_column_label, time + duration / 2.0)
    return data_frame.iloc[start:end].reset_index(drop=True)


def split_events(
    signal_frames: [pd.DataFrame],
    duration: float,
    event_label: str,
    fed_id_label: str,
    time_column: str,
    signal_info,
):
    """
    Splits the dataframes into new frames
    Structure expeceted of input, [{ledState == 2}, {ledState == 4}]
    Structure of output
    [
        Region: {
            fed_id : {
                event_type: [
                    event_data
                ]
            }
        }
    ]
    Uses the duration on either end of the event to establish a before and after period for each event.
    """
    output = {}
    for signal_region in signal_info:
        index, label = signal_info[signal_region]
        # index is which led state is active for that specific signal, label is the human readible name for that signal region combo
        signal = signal_frames[index]
        
        
        fed_ids = set(filter(lambda fed: ~np.isnan(fed), signal[fed_id_label]))
          
        temp = {}
        for fed_id in fed_ids:
            temp[fed_id] = {index: [] for index in range(1, 13)}
        # Creates the placeholder

        events = signal[
            ~np.isnan(signal[event_label]) & signal[event_label] != 0.0
        ]

        for event in events.iloc:
            fed_id, event_id, time = (
                event[fed_id_label],
                int(event[event_label]),
                event[time_column],
            )
            span = get_duration(signal, time_column, duration, time)[signal_region]
            temp[fed_id][event_id].append(span)

        output[label] = temp
    return output



"""
Depriciated function left for hold over
"""
# def fed_events(
#     data_frame: pd.DataFrame,
#     time_column_label: str,
#     event_column_label: str,
#     time_span: float,
# ) -> list[pd.DataFrame]:
#     """
#     Given a dataframe and a columnn,
#     splits the data based on the rising edge of the fed event with a total time of time_span(s)
#     half the time is before and half the time is after the fed event
#     returns a list of these split segments
#     """
#     events = list(
#         map(
#             lambda a, b: a < b,
#             data_frame[event_column_label].iloc[:-1],
#             data_frame[event_column_label].iloc[1:],
#         )
#     ) + [False]

#     time_stamps = data_frame[events][time_column_label]
#     output = []
#     for time in time_stamps:
#         output.append(get_duration(data_frame,time_column_label,time_span,time))
#     return output


def normalize_df(epochs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    Returns the normalized signal for each epoch based on
    (signal - baseline_mean) / baseline_mean
    """
    normalize_epochs = []
    for epoch in epochs:
        baseline_mean = epoch.iloc[: epoch.shape[0] // 2].mean()
        normalize_epoch = epoch.copy()
        normalize_epoch = (epoch - baseline_mean) / baseline_mean
        normalize_epochs.append(normalize_epoch)
    return normalize_epochs


def normalize_z(epochs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    Returns the normalized signal for each epoch based on
    (signal - baseline_mean) / baseline_std
    """
    normalize_epochs = []
    for epoch in epochs:
        baseline_mean = epoch.iloc[: epoch.shape[0] // 2].mean()
        baseline_std = epoch.iloc[: epoch.shape[0] // 2].std()
        normalize_epoch = epoch.copy()
        normalize_epoch = (epoch - baseline_mean) / baseline_std
        normalize_epochs.append(normalize_epoch)
    return normalize_epochs


def signal_sems(data_frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"Mean": data_frame.mean(axis=1), "SEM": data_frame.sem(axis=1)}
    )
