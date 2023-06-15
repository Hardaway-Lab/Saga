import constants
import matplotlib.pyplot as plt
import saga_operations as so
import numpy as np


def plot_rs_data(plot_dict: dict, title_keyword: str):
    """
    Plotting the data and optinally saving to output pngs
    (RS stands for "Raw" or "Smooth")

    Arguments:
        :plot_dict The dictionary containing the regions to plot
        :title_keyword Either "raw" or "smooth"
    """
    fig = plt.figure(figsize=(40, 35))
    plot_rs_double(
        fig,
        plot_dict["green_reference_right"],
        plot_dict["green_right"],
        subplot_pos=411,
        label=f"GCamp {title_keyword} Right",
    )
    plot_rs_double(
        fig,
        plot_dict["green_reference_left"],
        plot_dict["green_left"],
        subplot_pos=412,
        label=f"GCamp {title_keyword} Left",
    )
    plot_rs_double(
        fig,
        plot_dict["red_reference_right"],
        plot_dict["red_right"],
        subplot_pos=413,
        label=f"RCamp {title_keyword} Right",
    )
    plot_rs_double(
        fig,
        plot_dict["red_reference_left"],
        plot_dict["red_left"],
        subplot_pos=414,
        label=f"RCamp {title_keyword} Left",
    )
    if constants.OUTPUT_PLOTS:
        plt.savefig(f"output/plots/rs_charts_{title_keyword.lower()}.png")


def plot_rs_double(
    fig, plot1_data, plot2_data, subplot_pos=411, label="GCamp Raw Right"
):
    """
    Utility function for plot_rs_data
    """
    ax1 = fig.add_subplot(subplot_pos, label=f"{label} Signal")
    ax1.plot(
        plot1_data,
        "blue",
        linewidth=1.5,
        label=f"{label} Reference",
    )
    ax1.plot(plot2_data, "purple", linewidth=1.5, label=f"{label} Signal")
    ax1.set_xlabel("Timestamp")
    plt.title(f"{label} Signal")
    plt.legend()


def plot_corrected(corrected_dict: dict, pellets: list, title_keyword="Corrected"):
    fig = plt.figure(figsize=(40, 35))
    plot_corrected_single(
        fig,
        corrected_dict["green_right"],
        pellets,
        subplot_pos=221,
        label=f"GCaMP {title_keyword} Signal Right",
    )
    plot_corrected_single(
        fig,
        corrected_dict["green_left"],
        pellets,
        subplot_pos=222,
        label=f"GCaMP {title_keyword} Signal Left",
    )
    plot_corrected_single(
        fig,
        corrected_dict["red_right"],
        pellets,
        subplot_pos=223,
        label=f"RCaMP {title_keyword} Signal Right",
    )
    plot_corrected_single(
        fig,
        corrected_dict["red_left"],
        pellets,
        subplot_pos=224,
        label=f"RCaMP {title_keyword} Signal Left",
    )
    if constants.OUTPUT_PLOTS:
        plt.savefig(f"output/plots/{title_keyword.lower()}_charts.png")


def plot_corrected_single(
    fig,
    plot_data,
    pellets,
    subplot_pos=221,
    label="GCamp Corrected Signal Right",
):
    ax1 = fig.add_subplot(subplot_pos, label=label)
    ax1.plot(plot_data, "black", linewidth=1.5)
    ax1.set_xlabel("Timestamp")
    for pellet in pellets:
        plt.axvline(pellet, 0, 1, linewidth=2, color="c", linestyle="--")
    plt.title(label)


def plot_fit(df_dict, corrected_dict, fit_dict):
    x, y1, y2, y3, y4 = so.give_me_xy(df_dict, corrected_dict)
    fig = plt.figure(figsize=(40, 35))
    plot_fit_single(fig, 221, x, y1, fit_dict["green_right"], "GCamp_Right")
    plot_fit_single(fig, 223, x, y2, fit_dict["green_left"], "GCamp_Left")
    if constants.OUTPUT_PLOTS:
        plt.savefig(f"output/plots/plot_fit_green.png")
    fig2 = plt.figure(figsize=(40, 35))
    plot_fit_single(fig2, 221, x, y3, fit_dict["red_right"], "RCamp_Right")
    plot_fit_single(fig2, 223, x, y4, fit_dict["red_left"], "RCamp_Left")
    if constants.OUTPUT_PLOTS:
        plt.savefig(f"output/plots/plot_fit_red.png")


def plot_fit_single(fig, subplot_pos, x, y, popt, title):
    ax1 = fig.add_subplot(subplot_pos)
    ax1.set_title(f"{title} with fit")
    ax1.plot(x, y, linewidth=1.5, label="Observed")
    ax1.plot(x, so.bi_exp(x, *popt), "red", linewidth=1.5, label="Fit")
    plt.legend()
    residualerror_gcampright = y - so.bi_exp(x, *popt)
    ax2 = fig.add_subplot(subplot_pos + 1)
    ax2.set_title(f"{title} Residual Error")
    ax2.plot(x, residualerror_gcampright, "blue")


def plot_normalized(all_trials_dict, error_dict):
    x = np.linspace(-30, 30, 1200)
    fig = plt.figure(figsize=(32, 20))
    ax1 = fig.add_subplot(221)
    ax1.plot(x, all_trials_dict["green_right"]["Mean"], "crimson", linewidth=1)
    ax1.plot(x, all_trials_dict["green_left"]["Mean"], "blue", linewidth=1)

    ax1.plot(x, all_trials_dict["red_right"]["Mean"], "darkorange", linewidth=1)
    ax1.plot(x, all_trials_dict["red_left"]["Mean"], "forestgreen", linewidth=1)

    ax1.set_xlabel("Time(Sec)")
    ax1.set_ylabel("Z-score dF/F")
    ax1.fill_between(
        x,
        error_dict["green_right_lower"],
        error_dict["green_right_upper"],
        facecolor="pink",
    )
    ax1.fill_between(
        x,
        error_dict["green_left_lower"],
        error_dict["green_left_upper"],
        facecolor="lightsteelblue",
    )

    ax1.fill_between(
        x,
        error_dict["red_right_lower"],
        error_dict["red_right_upper"],
        facecolor="bisque",
    )
    ax1.fill_between(
        x,
        error_dict["red_left_lower"],
        error_dict["red_left_upper"],
        facecolor="springgreen",
    )

    plt.axhline(0, 0, 1, linewidth=1, color="black", linestyle="--")
    plt.axvline(
        0, 0, 1, linewidth=1, color="c", linestyle="--"
    )  # Reference keydown to find spot to place line
    plt.legend(["GCaMP Right", "GCaMP Left", "RCaMP Right", "RCaMP Left"])
    plt.title("SP 7-6 FED3 Refeed")
    plt.xticks(np.arange(-30, 30, step=5))
    if constants.OUTPUT_PLOTS:
        plt.savefig("output/plots/SP 7-7_38pellet_Refeed.png")
