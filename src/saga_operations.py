import pandas as pd
import constants
import saga_io as io
import numpy as np
import scipy


def deinterleave(
    input_df: pd.DataFrame,
    start: int = 0,
    column_label: str = "LedState",
    keys: list = [],
) -> dict:
    """
    Splitting a dataframe on unique values in a column

    Arguments:
        :input_df The dataframe to split
        :start From which row to start splitting
        :column_label (optinal) The column to split on
        :keys (optional) The row values to split on

    Output:
        :dict The splitted dataframe
    """
    input_df = input_df.truncate(before=start)
    if len(keys) == 0:
        keys = input_df[column_label].drop_duplicates().to_list()
    dict = {
        constants.WAVELENGTH_LEDSTATE_MAP[keys[index]]: input_df[
            input_df[column_label] == key
        ]
        for index, key in enumerate(keys)
    }
    smallest_size = min(x.shape[0] for x in dict.values())
    for df in dict.values():
        df.truncate(after=smallest_size)
        df.reset_index(inplace=True)

    return dict


def give_me_relevant_regions(dict: dict) -> dict:
    """
    Returning series of all the various regions in the dfs

    Arguments:
        :input_df The dictionary containing all the dfs

    Output:
        :dict Looks like <g/r_l/r/ref>: corresponding Series
    """
    return {
        "green_right": dict["green"][constants.REGION_MAP["green_right"]],
        "red_right": dict["red"][constants.REGION_MAP["red_right"]],
        "green_reference_right": dict["reference"][constants.REGION_MAP["green_right"]],
        "red_reference_right": dict["reference"][constants.REGION_MAP["red_right"]],
        "green_left": dict["green"][constants.REGION_MAP["green_left"]],
        "red_left": dict["red"][constants.REGION_MAP["red_left"]],
        "green_reference_left": dict["reference"][constants.REGION_MAP["green_left"]],
        "red_reference_left": dict["reference"][constants.REGION_MAP["red_left"]],
    }


def smooth_signals(
    regions_dict: dict, window_len: int = constants.SMOOTH_WIN, window="flat"
) -> dict:
    """
    Going through the regions and smoothing all the signals according to some
    mathematics I don't particularly understand.

    Arguments:
        :regions_dict The relevant regions from the method above
        :window_len (optional) Defaults to 10
        :window (optional) Defaults to flat

    Output:
        :dict: The same input dictionary but with smoothed signals
    """
    output_dict = dict()
    for k in regions_dict.keys():
        input_series = regions_dict[k]
        if input_series.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if input_series.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")

        if window_len < 3:
            return input_series

        if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
            raise ValueError(
                "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'",
            )

        s = np.r_[
            input_series[window_len - 1 : 0 : -1],
            input_series,
            input_series[-2 : -window_len - 1 : -1],
        ]

        if window == "flat":  # Moving average
            w = np.ones(window_len, "d")
        else:
            w = eval("np." + window + "(window_len)")

        y = np.convolve(w / w.sum(), s, mode="valid")

        output_dict[k] = y[(int(window_len / 2) - 1) : -int(window_len / 2)]
    return output_dict


def correct_signals(smooth_dict: dict) -> dict:
    """
    Dividing the smoothed signals by their references

    Arguments:
        :smooth_dict The dictionary from above

    Output:
        A dictionary of the form {<g/r l/r>: Corrected series}
    """
    return {
        "green_right": pd.Series(
            smooth_dict["green_right"] / smooth_dict["green_reference_right"]
        ),
        "green_left": pd.Series(
            smooth_dict["green_left"] / smooth_dict["green_reference_left"]
        ),
        "red_right": pd.Series(
            smooth_dict["red_right"] / smooth_dict["red_reference_right"]
        ),
        "red_left": pd.Series(
            smooth_dict["red_left"] / smooth_dict["red_reference_left"]
        ),
    }


def give_me_pellets(reference_df: pd.DataFrame, key_df: pd.DataFrame) -> list:
    """
    Uses the reference dataframe and the key dataframe to find the pellets

    Arguments:
        :reference_df The reference df from the deinterleave dict
        :key_df The keys dataframe loaded at the start

    Output:
        :list A list of all the pellets
    """
    timestamps = key_df.loc[key_df["Value.Value"] == True, "Timestamp"]
    return [
        (np.abs(reference_df["Timestamp"] - float(timestamps.iloc[i]))).argmin()
        for i in range(len(timestamps))
    ]


def bi_exp(x, a, b, c, d):
    """
    Utility function for fit
    """
    return a * np.exp(b / x) + c * np.exp(d / x)


def give_me_xy(df_dict, corrected_dict: dict) -> tuple:
    """ "
    Utility function for fit
    """
    return (
        df_dict["reference"]["Timestamp"].values,
        corrected_dict["green_right"].values,
        corrected_dict["green_left"].values,
        corrected_dict["red_right"].values,
        corrected_dict["red_left"].values,
    )


def fit(df_dict: dict, corrected_dict: dict) -> dict:
    """
    Fit Corrected signals using biexponential function

    Arguments:
        :df_dict dict with "reference" df for x
        :corrected_dic: to get y1, y2, y3, y4 for
        green_right, green_left, red_right, red_left

    Output:
        :dict Dictionary of the form {<g/r_l/r>: fit}
    """
    x, y1, y2, y3, y4 = give_me_xy(df_dict, corrected_dict)
    poptgcamp_right, _ = scipy.optimize.curve_fit(
        bi_exp, x, y1, constants.INITIAL_GUESS["p1"], maxfev=7500
    )
    poptgcamp_left, _ = scipy.optimize.curve_fit(
        bi_exp, x, y2, constants.INITIAL_GUESS["p2"], maxfev=5000
    )
    poptrcamp_right, _ = scipy.optimize.curve_fit(
        bi_exp, x, y3, constants.INITIAL_GUESS["p3"], maxfev=7000
    )
    poptrcamp_left, _ = scipy.optimize.curve_fit(
        bi_exp, x, y4, constants.INITIAL_GUESS["p4"], maxfev=5000
    )
    return {
        "green_right": poptgcamp_right,
        "green_left": poptgcamp_left,
        "red_right": poptrcamp_right,
        "red_left": poptrcamp_left,
    }


def subtract_fits(df_dict: dict, fit_dict: dict, corrected_dict: dict) -> dict:
    """
    Subtract the fits from each corrected signal

    Arguments:
        :df_dict dict with "reference" df for x
        :fit_dict The dictionary returned above
        :corrected_dict Another dictionary returned above

    Output:
        :dict Dictionary of the form {<g/r_l/r>: sub_fit}
    """
    x = df_dict["reference"]["Timestamp"].values
    gcampright_fit = bi_exp(x, *fit_dict["green_right"])
    gcampleft_fit = bi_exp(x, *fit_dict["green_left"])
    rcampright_fit = bi_exp(x, *fit_dict["red_right"])
    rcampleft_fit = bi_exp(x, *fit_dict["red_left"])
    gcampright_fitsub = corrected_dict["green_right"] / gcampright_fit
    gcampleft_fitsub = corrected_dict["green_left"] / gcampleft_fit
    rcampright_fitsub = corrected_dict["red_right"] / rcampright_fit
    rcampleft_fitsub = corrected_dict["red_left"] / rcampleft_fit
    return {
        "green_right": gcampright_fitsub,
        "green_left": gcampleft_fitsub,
        "red_right": rcampright_fitsub,
        "red_left": rcampleft_fitsub,
    }


def chop_up(fitsub_dict: dict, pellets: list):
    """
    Chop up data points

    Arguments:
        :fitsub_dict The dictionary returned above
        :pellets From give_me_pellets above

    Output:
        :dict Dictionary of the form {<g/r_l/r_[1->len(pellets)]>: chop}
    """
    ret_val = dict()
    for index in range(len(pellets)):
        gr = fitsub_dict["green_right"][
            (pellets[index] - 600) : (pellets[index] + 600)
        ].reset_index()[0]
        ret_val[f"green_right_{index}"] = gr
        gl = fitsub_dict["green_left"][
            (pellets[index] - 600) : (pellets[index] + 600)
        ].reset_index()[0]
        ret_val[f"green_left_{index}"] = gl
        rr = fitsub_dict["red_right"][
            (pellets[index] - 600) : (pellets[index] + 600)
        ].reset_index()[0]
        ret_val[f"red_right_{index}"] = rr
        rl = fitsub_dict["red_left"][
            (pellets[index] - 600) : (pellets[index] + 600)
        ].reset_index()[0]
        ret_val[f"red_left_{index}"] = rl
    return ret_val


def avg_dff(chopped_dict: dict) -> dict:
    """
    Calculate the baseline average and df/f for each period

    Arguments:
        :chopped_dict The dictionary returned above

    Output:
        :dict Dictionary of the form {<g/r_l/r_[1->len(pellets)]_averageF/STD>: avgdff}
    """
    ret_val = dict()
    for index in range(len(chopped_dict) // 4):
        ret_val[f"green_right_{index}_averageF"] = (
            chopped_dict[f"green_right_{index}"].iloc[0:600].mean()
        )
        ret_val[f"green_right_{index}_STD"] = (
            chopped_dict[f"green_right_{index}"].iloc[0:600].std()
        )
        ret_val[f"green_left_{index}_averageF"] = (
            chopped_dict[f"green_left_{index}"].iloc[0:600].mean()
        )
        ret_val[f"green_left_{index}_STD"] = (
            chopped_dict[f"green_left_{index}"].iloc[0:600].std()
        )
        ret_val[f"red_right_{index}_averageF"] = (
            chopped_dict[f"red_right_{index}"].iloc[0:600].mean()
        )
        ret_val[f"red_right_{index}_STD"] = (
            chopped_dict[f"red_right_{index}"].iloc[0:600].std()
        )
        ret_val[f"red_left_{index}_averageF"] = (
            chopped_dict[f"red_left_{index}"].iloc[0:600].mean()
        )
        ret_val[f"red_left_{index}_STD"] = (
            chopped_dict[f"red_left_{index}"].iloc[0:600].std()
        )
    return ret_val


def normalized_signal(chopped_dict: dict, avg_dff_dict: dict) -> tuple:
    """
    Calculate normalized signal for each time range

    Arguments:
        :chopped_dict The dictionary returned above
        :avg_dff_dict Another dictionary returned above

    Output:
        :tuple Contains two dictionary for all trials and errors
    """
    norm_dict = dict()
    for index in range(len(chopped_dict) // 4):
        norm_dict[f"green_right_{index}_normalized"] = (
            chopped_dict[f"green_right_{index}"]
            - avg_dff_dict[f"green_right_{index}_averageF"]
        ) / avg_dff_dict[f"green_right_{index}_STD"]
        norm_dict[f"green_left_{index}_normalized"] = (
            chopped_dict[f"green_left_{index}"]
            - avg_dff_dict[f"green_left_{index}_averageF"]
        ) / avg_dff_dict[f"green_left_{index}_STD"]
        norm_dict[f"red_right_{index}_normalized"] = (
            chopped_dict[f"red_right_{index}"]
            - avg_dff_dict[f"red_right_{index}_averageF"]
        ) / avg_dff_dict[f"red_right_{index}_STD"]
        norm_dict[f"red_left_{index}_normalized"] = (
            chopped_dict[f"red_left_{index}"]
            - avg_dff_dict[f"red_left_{index}_averageF"]
        ) / avg_dff_dict[f"red_left_{index}_STD"]
    # test = norm_dict['green_right_1_normalized']
    # test2 = norm_dict['green_right_1_normalized'][0]
    # test3 = norm_dict['green_right_1_normalized'].loc(axis=0)[:,0]
    # test4 = norm_dict['green_right_1_normalized'].iloc[0,]
    # test5 = norm_dict['green_right_1_normalized'].iloc[:,0]

    gra = pd.DataFrame()
    gla = pd.DataFrame()
    rra = pd.DataFrame()
    rla = pd.DataFrame()
    for val in range(len(norm_dict) // 4):
        gra[val] = norm_dict[f'green_right_{val}_normalized']
        gla[val] = norm_dict[f'green_left_{val}_normalized']
        rra[val] = norm_dict[f'red_right_{val}_normalized']
        rla[val] = norm_dict[f'red_left_{val}_normalized']
    
    all_trials_dict = {
        'green_right': gra,
        'green_left': gla,
        'red_right': rra,
        'red_left': rla
    }

    if constants.OUTPUT_CSVS:
        io.save_csv(all_trials_dict)
    # Calculate Mean and SEM values for all
    all_trials_dict["green_right"]["Mean"] = all_trials_dict["green_right"].mean(axis=1)
    all_trials_dict["green_right"]["SEM"] = all_trials_dict["green_right"].sem(axis=1)
    all_trials_dict["green_left"]["Mean"] = all_trials_dict["green_left"].mean(axis=1)
    all_trials_dict["green_left"]["SEM"] = all_trials_dict["green_left"].sem(axis=1)
    all_trials_dict["red_right"]["Mean"] = all_trials_dict["red_right"].mean(axis=1)
    all_trials_dict["red_right"]["SEM"] = all_trials_dict["red_right"].sem(axis=1)
    all_trials_dict["red_left"]["Mean"] = all_trials_dict["red_left"].mean(axis=1)
    all_trials_dict["red_left"]["SEM"] = all_trials_dict["red_left"].sem(axis=1)

    error_dict = {}
    error_dict["green_right_lower"] = (
        all_trials_dict["green_right"]["Mean"]
        + (all_trials_dict["green_right"]["SEM"] * -1)
    )
    error_dict["green_right_upper"] = (
        all_trials_dict["green_right"]["Mean"]
        + (all_trials_dict["green_right"]["SEM"] * 1)
    )
    error_dict["green_left_lower"] = (
        all_trials_dict["green_left"]["Mean"]
        + (all_trials_dict["green_left"]["SEM"] * -1)
    )
    error_dict["green_left_upper"] = (
        all_trials_dict["green_left"]["Mean"] + (all_trials_dict["green_left"]["SEM"] * 1)
    )

    error_dict["red_right_lower"] = (
        all_trials_dict["red_right"]["Mean"] + (all_trials_dict["red_right"]["SEM"] * -1)
    )
    error_dict["red_right_upper"] = (
        all_trials_dict["red_right"]["Mean"] + (all_trials_dict["red_right"]["SEM"] * 1)
    )
    error_dict["red_left_lower"] = (
        all_trials_dict["red_left"]["Mean"] + (all_trials_dict["red_left"]["SEM"] * -1)
    )
    error_dict["red_left_upper"] = (
        all_trials_dict["red_left"]["Mean"] + (all_trials_dict["red_left"]["SEM"] * 1)
    )
    return (all_trials_dict, error_dict)
