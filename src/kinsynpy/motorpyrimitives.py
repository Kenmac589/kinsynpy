"""Motorpyrimitives
The main goal of these functions is to assist with the analysis of
electromyographic data.

This includes

Non-Negative Matrix factorization -> @nnmf_factorize
Full width half maximum           -> @fwhm
Center of Activity                -> @coa

author: Kenzie MacKinnon
email (Personal): kenziemackinnon5@gmail.com
email (Work): kenzie.mackinnon@dal.ca
"""

# %%

import os

# from scipy.interpolate import InterpolatedUnivariateSpline
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from scipy import signal
from sklearn.decomposition import NMF

# from statsmodels.nonparametric.kernel_regression import KernelReg


# %%
def read_all_csv(directory_path):
    data_dict = {}  # Initialize an empty dictionary to store the data

    if not os.path.isdir(directory_path):
        print(f"{directory_path} is not a valid directory.")
        return

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            data = pd.read_csv(file_path)
            data_dict[filename] = data

    return data_dict


def nnmf_factorize(A, k):
    """Non-Negative Matrix Factorization for Muscle Synergy Extraction

    Parameters
    ----------
    A: numpy.ndarray
        input matrix
    k: int
        number of components (muscle channels)

    Returns
    -------
    W: numpy.ndarray
        motor primitives
    H: numpy.ndarray
        motor modules
    C: numpy.ndarray
        factorized matrix
    """
    nmf = NMF(n_components=k, init="random", random_state=0)
    W = nmf.fit_transform(A)
    H = nmf.components_
    C = np.dot(W, H)
    return W, H, C


def syn_sel(norm_emg, auto_select=True, input_selection=False):
    """Automatically selects synergies to use based on 95% rule
    Parameters
    ----------
    norm_emg: pandas dataframe
        dataframe containing normalized EMG channels

    Returns
    -------
    out: matplotlib.figure
        returns the figure for the synergy selection
    syn_selection: int
        The amount of synergies to use for Non-Negative matrix factorization

    """

    A = norm_emg.to_numpy()

    # Defining set of components to use
    num_components = np.array([2, 3, 4, 5, 6, 7])
    R2All = np.zeros(len(num_components))

    for i in range(len(R2All)):
        W, H, C = nnmf_factorize(A, num_components[i])
        R2All[i] = np.corrcoef(C.flatten(), A.flatten())[0, 1] ** 2
        print("$R^2$ =", i + 2, ":", R2All[i])

    # Calculating Number of synergies to use
    if auto_select is True:
        percent_cutoff = 0.95
        counter = 0
        current_val = 0
        while current_val < percent_cutoff:
            current_val = R2All[counter]
            counter = counter + 1

        syn_selection = counter + 1
    else:
        syn_selection = R2All[0]

    corrcoef = np.zeros(len(num_components))
    for i in range(len(R2All)):
        corrcoef[i] = np.corrcoef(num_components[0 : i + 2], R2All[0 : i + 2])[0, 1]
        # print("r =", i + 2, ":", corrcoef[i])

    # Plotting Both Methods for determining number of components
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set(style="white", font_scale=1.3, rc=custom_params)
    plt.style.use("~/sync/lab-analysis/configs/default_plot.mplstyle")
    fig, axs = plt.subplots(2)

    axs[0].set_title("Synergy Selection by Percentage")
    axs[0].plot(num_components, R2All)
    axs[0].axhline(y=0.95, color="r", linestyle="-", label="95%")
    axs[0].set_xlabel("Number of Components")
    axs[0].set_ylabel("$R^2$ of $C^x$ fit to original matrix")

    axs[1].set_title("Synergy Selection by Linearity")
    axs[1].scatter(num_components, corrcoef)
    axs[1].set_xlabel("Number of Components")
    axs[1].set_ylabel("Correlation Coefficient")

    # # Plotting Both Methods overlapping
    # plt.plot(num_components, R2All)
    # plt.axhline(y=0.95, color="r", linestyle="-")
    # plt.xlabel("Number of Components")
    # plt.ylabel("$R^2$ of $C^x$ fit to original matrix")
    # plt.title("Muscle Synergy Determinance by Percentage")
    # plt.scatter(num_components, corrcoef)
    # plt.title("Muscle Synergy Determinance by Linearity and $R^2$")
    # plt.show()

    if input_selection is True:
        plt.show()
        syn_selection = int(input("How many components will be used?: "))

    else:
        out = fig.get_figure()

    return out, syn_selection


def synergy_extraction(data_input, synergy_selection):
    """Synergy Extraction from factorized matrices

    Parameters
    ----------
    data_input: pandas.core.frame.Dataframe
        input containing normalized EMG channels
    chosen_synergies: int
        selected synergies to use

    Returns
    -------
    W: numpy.ndarray
        motor primitives array
    H: numpy.ndarray
        motor modules
    """

    # Load Data
    A = data_input.to_numpy()

    # Choosing best number of components
    chosen_synergies = synergy_selection
    W, H, C = nnmf_factorize(A, chosen_synergies)

    motor_modules = H
    motor_primitives = W

    return motor_primitives, motor_modules


def full_width_half_abs_min(motor_p_full, synergy_selection):
    """Full width half maximum calculation
    @param: motor_p_full_full: full length numpy array of selected motor
    primitives

    @return: mean_fwhm: Mean value for the width of the primitives
    """

    number_cycles = len(motor_p_full) // 200

    # Save
    fwhl = np.array([])
    half_width_height_array = np.array([])
    fwhl_start_stop = np.empty((number_cycles, 0))

    for i in range(number_cycles):
        current_primitive = motor_p_full[i * 200 : (i + 1) * 200, synergy_selection - 2]

        primitive_mask = current_primitive > 0.0

        # applying mask to exclude values which were subject to rounding errors
        mcurrent_primitive = np.asarray(current_primitive[primitive_mask])

        # getting maximum
        max_ind = np.argmax(mcurrent_primitive)

        # abs_min_ind = np.argmin(mcurrent_primitive)
        # getting the minimum before
        min_ind_before = np.argmin(mcurrent_primitive[:max_ind])

        # getting the minimum index after maximum
        # Making sure to include the max after so the index for the whole array
        min_ind_after = np.argmin(mcurrent_primitive[max_ind + 1 :]) + (max_ind - 1)

        # Determing the smaller minimum to use
        if mcurrent_primitive[min_ind_before] < mcurrent_primitive[min_ind_after]:
            # print("First minimum used!")

            # Half Width formula
            half_width_height = (
                mcurrent_primitive[max_ind] - mcurrent_primitive[min_ind_before]
            ) / 2

            half_width_start = (
                np.argmax(mcurrent_primitive[::max_ind] < half_width_height)
                + min_ind_before
            )
            half_width_end = np.argmax(mcurrent_primitive[:max_ind] > half_width_height)
        else:
            # print("Second minimum used")
            half_width_height = (
                mcurrent_primitive[max_ind] - mcurrent_primitive[min_ind_after]
            ) / 2

        # largest_index = np.argmax(arr[np.logical_and(arr > 2, arr < 8)])
        # Getting the closest indices on either side of the max closest to half width
        # half_width_start = np.argmax(mcurrent_primitive[::max_ind] > half_width_height)
        # half_width_end = np.argmax(mcurrent_primitive[:max_ind] > half_width_height)
        # half_width_height = (max_ind - abs_min_ind) / 2

        area_above_half = [
            i
            for i in range(len(mcurrent_primitive))
            if mcurrent_primitive[i] > half_width_height
        ]
        half_width_start = area_above_half[0]
        half_width_end = area_above_half[-1]

        # Adding start and stop coordinates appropriate to array
        half_width_height_array = np.append(
            half_width_height_array, [half_width_height]
        )
        # fwhl_height = fwhl_start_stop_list.reshape((len(fwhl_start_stop_list) // 2), 2)
        fwhl_start_stop = np.append(
            fwhl_start_stop, [[half_width_start, half_width_end]]
        )
        fwhl_start_stop = fwhl_start_stop.reshape((len(fwhl_start_stop) // 2), 2)

        # Determing length for primitive and appending
        full_width_length = half_width_end - half_width_start
        fwhl = np.append(fwhl, [full_width_length])

        # print("Start of half width line", half_width_start)
        # print("End of half width line", half_width_end)

        # # # print("Half width height", half_width_height)

        # print("before max min index", min_ind_before, "value", mcurrent_primitive[min_ind_before])
        # print("half width height", half_width_height)
        # print("max value", max_ind, "value", mcurrent_primitive[max_ind])
        # print("after max min value", min_ind_after, "value", mcurrent_primitive[min_ind_after])
        # print("Length", full_width_length)
        # print(mcurrent_primitive[min_ind_after])
        # print()

    return fwhl, fwhl_start_stop, half_width_height_array


def full_width_half_abs_min_scipy(motor_p_full, synergy_selection):
    """Full width half maximum calculation
    @param: motor_p_full_full: full length numpy array of selected motor
    primitives

    @return: mean_fwhm: Mean value for the width of the primitives
    """

    # Save
    fwhl = []
    number_cycles = len(motor_p_full) // 200

    for i in range(number_cycles):
        current_primitive = motor_p_full[i * 200 : (i + 1) * 200, synergy_selection - 1]

        # Find peaks
        peaks, properties = sp.signal.find_peaks(
            current_primitive, distance=40, width=2
        )
        max_ind = np.argmax(peaks)
        # min_ind = np.argmin(mcurrent_primitive[0:max_ind])

        # half_width_height = (mcurrent_primitive[max_ind] - mcurrent_primitive[min_ind]) / 2

        # print("Manually Calculated", half_width_height)
        max_width = properties["widths"][max_ind]
        fwhl.append(max_width)
        # fwhl_start = properties["left_ips"][max_ind]
        # fwhl_stop = properties["right_ips"][max_ind]
        # half_width_height = properties["width_heights"][max_ind]

        print("Scipy calculated", properties["widths"][max_ind])
        # print(peaks[max_ind])
    fwhl = np.asarray(fwhl)

    return fwhl


def fwhm_calc(motor_p_sel, cycle_constant=200):
    """full width half maximum calculation

    Parameters
    ----------
    motor_p_sel: numpy.ndarray
        full length numpy array of selected motor primitives

    Returns
    -------
    fwhm:
        Mean value for the width of the primitives
    """

    # Save
    fwhm = np.array([])
    fwhm_index = [[]]

    prim_sweep_count = int(motor_p_sel.size / cycle_constant)

    # reshaping primitive into
    motor_p_split = motor_p_sel.reshape(prim_sweep_count, cycle_constant)

    avg_trace = np.empty(cycle_constant)
    avg_trace_std = np.empty(cycle_constant)
    for i in range(cycle_constant):
        time_avg = np.mean(motor_p_split[:, i])
        time_avg_std = np.std(motor_p_split[:, i])
        avg_trace[i] = time_avg
        avg_trace_std[i] = time_avg_std

    for i in range(prim_sweep_count):
        current_cycle = motor_p_split[i]

        abs_min_ind = np.argmin(current_cycle)

        # Getting maximum value
        max_ind = np.argmax(current_cycle)

        # Getting half-width height
        half_width_height = (current_cycle[max_ind] - current_cycle[abs_min_ind]) * 0.5

        # Getting all values along curve that fall below half width height
        count_above = np.nonzero(current_cycle > half_width_height)

        fwhm_index.append(count_above)
        fwhm = np.append(fwhm, [len(count_above[0])])

    fwhm = np.asarray(fwhm)

    return fwhm


def _fwhm_calc(motor_p_full, synergy_selection):
    """full width half maximum calculation
    @param: motor_p_full: full length numpy array of selected motor
    primitives

    @return: mean_fwhm: Mean value for the width of the primitives
    """

    number_cycles = len(motor_p_full) // 200

    # Save
    fwhm = np.array([])
    fwhm_index = [[]]

    for i in range(number_cycles):
        current_primitive = motor_p_full[i * 200 : (i + 1) * 200, synergy_selection - 1]

        primitive_mask = current_primitive > 0.0

        # applying mask to exclude values which were subject to rounding errors
        mcurrent_primitive = np.asarray(current_primitive[primitive_mask])

        # Getting minimum value
        abs_min_ind = np.argmin(mcurrent_primitive)

        # Getting maximum value
        max_ind = np.argmax(mcurrent_primitive)

        # Getting half-width height
        half_width_height = (
            mcurrent_primitive[max_ind] - mcurrent_primitive[abs_min_ind]
        ) * 0.5

        # Getting all values along curve that fall below half width height
        count_above = np.nonzero(mcurrent_primitive > half_width_height)

        fwhm_index.append(count_above)
        fwhm = np.append(fwhm, [len(count_above[0])])

    fwhm = np.asarray(fwhm)

    return fwhm


def coa(motor_p_sel, cycle_constant=200):
    """Center of Activity

    Parameters
    ----------
    motor_p: numpy.ndarray
        factorized array from arrays
    chosen_synergies: Selected muscle synergy

    Returns
    -------
    coa_values: numpy.ndarray
        array of center activities for the step cycles in the trial
    """

    # Make selection of synergy and bin primitives by step cycle
    prim_sweep_count = int(motor_p_sel.size / cycle_constant)
    binned_primitives = motor_p_sel.reshape(prim_sweep_count, cycle_constant)

    coa_values = np.empty(len(binned_primitives))
    for i in range(len(binned_primitives)):
        current_step = binned_primitives[i]

        alpha = np.linspace(0, 2 * np.pi, len(current_step))

        a_matrix = np.sum(current_step * np.cos(alpha))
        b_matrix = np.sum(current_step * np.sin(alpha))

        coa_t = np.atan2(b_matrix, a_matrix) * 180 / np.pi

        # Negative degree filter
        is_neg = coa_t < 0
        coa_t = np.where(is_neg, coa_t + 360, coa_t)

        coa_val = coa_t * len(current_step) / 360
        coa_values[i] = coa_val

    return coa_values


def show_modules(data_input, chosen_synergies, modules_filename="./output.png"):
    """
    Make sure you check the channel order!!

    """

    # =======================================
    # Presenting Data as a mutliplot figure |
    # =======================================
    motor_primitives, motor_modules = synergy_extraction(data_input, chosen_synergies)
    channel_order = ["GM", "Ip", "BF", "VL", "St", "TA", "Gs", "Gr"]

    fig, axs = plt.subplots(chosen_synergies, 1, figsize=(4, 10))

    # Calculate the average trace for each column
    # samples = np.arange(0, len(motor_primitives))
    # samples_binned = np.arange(200)
    # number_cycles = len(motor_primitives) // 200

    for col in range(chosen_synergies):
        # primitive_trace = np.zeros(200)  # Initialize an array for accumulating the trace values

        # Begin Presenting Motor Modules

        # Get the data for the current column
        motor_module_column_data = motor_modules[
            col, :
        ]  # Select all rows for the current column

        # Set the x-axis values for the bar graph
        x_values = np.arange(len(motor_module_column_data))

        # Plot the bar graph for the current column in the corresponding subplot
        axs[col].bar(x_values, motor_module_column_data)

        # Remove top and right spines of each subplot

        # Remove x and y axis labels and ticks from the avg_trace subplot
        axs[col].set_xticks([])
        axs[col].set_yticks([])
        axs[col].set_xlabel("")
        axs[col].set_ylabel("")
        axs[col].spines["top"].set_visible(False)
        axs[col].spines["right"].set_visible(False)

        # Remove x and y axis labels and ticks from the motor module subplot
        axs[col].set_xticks(x_values, channel_order)
        axs[col].set_yticks([])
        # axs[1, col].set_xlabel('')
        # axs[1, col].set_ylabel('')

    # Adjust spacing between subplots
    plt.tight_layout()
    # fig.suptitle(synergies_title, fontsize=16, fontweight='bold')
    # plt.savefig(modules_filename, dpi=300)
    # plt.subplots_adjust(top=0.9)
    plt.show()


def _show_synergies(
    data_input,
    chosen_synergies,
    channel_order=["GM", "Ip", "BF", "VL", "St", "TA", "Gs", "Gr"],
    synergies_name="./output.png",
    cycle_length=200,
):
    """
    Make sure you check the channel order!!

    Parameters
    ----------
    data_input: pandas.core.frame.Dataframe
        input containing normalized EMG channels
    chosen_synergies: int
        selected synergies to use
    channel_order: list
        the order that the channels show up in the recording file


    """

    # =======================================
    # Presenting Data as a mutliplot figure |
    # =======================================
    motor_primitives, motor_modules = synergy_extraction(data_input, chosen_synergies)

    # fwhm_line = fwhm(motor_primitives, chosen_synergies)
    cycle_length = 200

    samples = np.arange(0, len(motor_primitives))
    samples_binned = np.arange(cycle_length)

    fig, axs = plt.subplots(chosen_synergies, 2, figsize=(12, 8))
    # Calculate the average trace for each column
    number_cycles = (
        len(motor_primitives) // cycle_length
    )  # Calculate the number of 200-value bins

    for col in range(chosen_synergies):
        primitive_trace = np.zeros(
            cycle_length
        )  # Initialize an array for accumulating the trace values

        # Iterate over the binned data by the number of cycles
        for i in range(number_cycles):
            # Get the data for the current bin in the current column
            time_point_average = motor_primitives[
                i * cycle_length : (i + 1) * cycle_length, col
            ]

            # Accumulate the trace values
            primitive_trace += time_point_average

        # Calculate the average by dividing the accumulated values by the number of bins
        primitive_trace /= number_cycles

        # Plot the average trace in the corresponding subplot
        smooth_sample = signal.savgol_filter(
            samples[samples_binned], window_length=50, polyorder=3, mode="interp"
        )
        axs[col, 1].plot(
            smooth_sample, primitive_trace, color="red", label="Average Trace"
        )
        axs[col, 1].set_title("Synergy {}".format(col + 1))

        # Iterate over the bins again to plot the individual bin data
        for i in range(number_cycles):
            # Get the data for the current bin in the current 0, column
            time_point_average = motor_primitives[
                i * cycle_length : (i + 1) * cycle_length, col
            ]

            smooth_sample = signal.savgol_filter(samples[samples_binned], 40, 3)
            # Plot the bin data
            axs[col, 1].plot(
                smooth_sample,
                time_point_average,
                label="Bin {}".format(i + 1),
                color="black",
                alpha=0.1,
            )

        # Add vertical lines at the halfway point in each subplot
        axs[col, 1].axvline(x=100, color="black")

        # Begin Presenting Motor Modules

        # Get the data for the current column
        motor_module_column_data = motor_modules[
            col, :
        ]  # Select all rows for the current column

        # Set the x-axis values for the bar graph
        x_values = np.arange(len(motor_module_column_data))

        # Plot the bar graph for the current column in the corresponding subplot
        axs[col, 0].bar(x_values, motor_module_column_data)

        # Remove top and right spines of each subplot
        axs[col, 1].spines["top"].set_visible(False)
        axs[col, 1].spines["right"].set_visible(False)
        axs[col, 0].spines["top"].set_visible(False)
        axs[col, 0].spines["right"].set_visible(False)

        # Remove labels on x and y axes
        axs[col, 0].set_xticklabels([])
        axs[col, 1].set_yticklabels([])

        # Remove x and y axis labels and ticks from the avg_trace subplot
        axs[col, 1].set_xticks([])
        axs[col, 1].set_yticks([])
        axs[col, 1].set_xlabel("")
        axs[col, 1].set_ylabel("")
        axs[col, 1].set_ylim(np.min(primitive_trace), np.max(primitive_trace))

        axs[col, 1].text(
            50, -0.3 * np.max(primitive_trace), "Swing", ha="center", va="center"
        )
        axs[col, 1].text(
            150, -0.3 * np.max(primitive_trace), "Stance", ha="center", va="center"
        )

        # Remove x and y axis labels and ticks from the motor module subplot
        axs[col, 0].set_xticks(x_values, channel_order)
        # axs[1, col].set_xticks([])
        axs[col, 0].set_yticks([])
        # axs[1, col].set_xlabel('')
        # axs[1, col].set_ylabel('')

    # Adjust spacing between subplots
    plt.tight_layout()
    fig.suptitle(synergies_name, fontsize=16, fontweight="bold")
    plt.subplots_adjust(top=0.9)
    # plt.savefig(synergies_filename, dpi=300)

    # Show all the plots
    # plt.show(block=True)


def show_sel_primitive(motor_p, chosen_synergies, cycle_constant=200):
    """Creates plot of motor primitive for selected synergy

    Parameters
    ----------
    motor_p: numpy.ndarray
        input containing normalized EMG channels
    syn_selection: int
        selected synergies to use
    cycle_constant: int, default=`200`
        how many samples per step cycle

    Returns
    -------
    out: mpl.gcf
        returns the plot in the event it is apart of figure

    """

    sel_motor_p = motor_p[:, chosen_synergies - 1]

    # smoothen primitive trace
    motor_p_smooth = sp.signal.savgol_filter(
        x=sel_motor_p, window_length=50, polyorder=3, mode="interp"
    )
    # dividing into step cycle bins
    prim_sweep_count = int(motor_p_smooth.size / cycle_constant)

    # reshaping primitive into
    motor_p_split = motor_p_smooth.reshape(prim_sweep_count, cycle_constant)

    # Getting average from each trace at given time point as well as Standard dev
    avg_trace = np.empty(cycle_constant)
    avg_trace_std = np.empty(cycle_constant)
    for i in range(cycle_constant):
        time_avg = np.mean(motor_p_split[:, i])
        time_avg_std = np.std(motor_p_split[:, i])
        avg_trace[i] = time_avg
        avg_trace_std[i] = time_avg_std

    # Plotting individual step primitives in the background
    for i in range(prim_sweep_count):
        plt.plot(
            motor_p_split[i], label="Bin {}".format(i + 1), color="black", alpha=0.1
        )

    plt.plot(avg_trace, color="red", label="Average Trace")
    plt.ylim((np.min(avg_trace), np.max(avg_trace)))

    step_cycle = ["Swing", "Stance"]
    plt.axvline(x=cycle_constant / 2, color="black")
    plt.xticks([cycle_constant * 0.25, cycle_constant * 0.75], step_cycle)

    plt.yticks([np.min(avg_trace), np.max(avg_trace)], ["Min", "Max"])

    out = mpl.pyplot.gcf()

    return out


def show_synergies(
    data_input,
    chosen_synergies,
    channel_order=["GM", "Ip", "BF", "VL", "St", "TA", "Gs", "Gr"],
    synergies_name="./output.png",
    cycle_constant=200,
    smooth_wind=50,
):
    """
    Make sure you check the channel order!!

    Parameters
    ----------
    data_input: pandas.core.frame.Dataframe
        input containing normalized EMG channels
    chosen_synergies: int
        selected synergies to use
    channel_order: list
        the order that the channels show up in the recording file
    smooth_wind: int
        smoothing window size for savgol filter from scipy

    Returns
    -------
    out: mpl.gcf
        the current figure

    """

    # =======================================
    # Presenting Data as a mutliplot figure |
    # =======================================
    motor_p, motor_m = synergy_extraction(data_input, chosen_synergies)

    # Set plot
    fig, axs = plt.subplots(chosen_synergies, 2, figsize=(12, 8))

    for col in range(chosen_synergies):
        sel_motor_p = motor_p[:, col]

        motor_p_smooth = sp.signal.savgol_filter(
            x=sel_motor_p, window_length=smooth_wind, polyorder=3, mode="interp"
        )
        # dividing into step cycle bins
        prim_sweep_count = int(motor_p_smooth.size / cycle_constant)

        # reshaping primitive into
        motor_p_split = motor_p_smooth.reshape(prim_sweep_count, cycle_constant)

        # Getting average from each trace at given time point as well as Standard dev
        avg_trace = np.empty(cycle_constant)
        avg_trace_std = np.empty(cycle_constant)
        for i in range(cycle_constant):
            time_avg = np.mean(motor_p_split[:, i])
            time_avg_std = np.std(motor_p_split[:, i])
            avg_trace[i] = time_avg
            avg_trace_std[i] = time_avg_std

        for i in range(prim_sweep_count):
            axs[col, 1].plot(
                motor_p_split[i], label="Bin {}".format(i + 1), color="black", alpha=0.1
            )
        # Plotting individual step primitives in the background
        cycle_phases = ["Swing", "Stance"]

        axs[col, 1].plot(avg_trace, color="red", label="Average Trace")

        # Add vertical lines at the halfway point in each subplot
        axs[col, 1].axvline(x=cycle_constant / 2, color="black")

        # Begin Presenting Motor Modules

        # Get the data for the current column
        motor_module_column_data = motor_m[
            col, :
        ]  # Select all rows for the current column

        # Set the x-axis values for the bar graph
        x_values = np.arange(len(motor_module_column_data))

        # Plot the bar graph for the current column in the corresponding subplot
        axs[col, 0].bar(x_values, motor_module_column_data)

        # Remove top and right spines of each subplot
        axs[col, 1].spines["top"].set_visible(False)
        axs[col, 1].spines["right"].set_visible(False)
        axs[col, 0].spines["top"].set_visible(False)
        axs[col, 0].spines["right"].set_visible(False)

        # Remove labels on x and y axes
        axs[col, 0].set_xticklabels([])
        axs[col, 1].set_yticklabels([])

        # Remove x and y axis labels and ticks from the avg_trace subplot
        axs[col, 1].set_xticks(
            [cycle_constant * 0.25, cycle_constant * 0.75], cycle_phases
        )
        axs[col, 1].set_yticks([np.min(avg_trace), np.max(avg_trace)], ["Min", "Max"])
        axs[col, 1].set_xlabel("")
        axs[col, 1].set_ylabel("")
        axs[col, 1].set_ylim(np.min(avg_trace), np.max(avg_trace))

        # Remove x and y axis labels and ticks from the motor module subplot
        axs[col, 0].set_xticks(x_values, channel_order)
        axs[col, 0].set_yticks([])

    # Adjust spacing between subplots
    fig.suptitle(synergies_name, fontsize=12, fontweight="bold")
    plt.subplots_adjust(top=0.9)

    out = fig.get_figure()

    return out


def show_modules_dtr(data_input, chosen_synergies, modules_filename="./output.png"):
    """
    Make sure you check the channel order!!

    """

    # =======================================
    # Presenting Data as a mutliplot figure |
    # =======================================
    motor_primitives, motor_modules = synergy_extraction(data_input, chosen_synergies)
    channel_order_dtr = ["GM", "Ip", "BF", "VL", "Gs", "TA", "St", "Gr"]
    print(data_input)
    print(motor_modules)

    fig, axs = plt.subplots(chosen_synergies, 1, figsize=(4, 10))

    # Calculate the average trace for each column
    # samples = np.arange(0, len(motor_primitives))
    # samples_binned = np.arange(200)
    # number_cycles = len(motor_primitives) // 200

    for col in range(chosen_synergies):
        # Begin Presenting Motor Modules

        # Get the data for the current column
        motor_module_column_data = motor_modules[
            col, :
        ]  # Select all rows for the current column

        # Set the x-axis values for the bar graph
        x_values = np.arange(len(motor_module_column_data))

        # Plot the bar graph for the current column in the corresponding subplot
        axs[col].bar(x_values, motor_module_column_data)

        # Remove top and right spines of each subplot

        # Remove x and y axis labels and ticks from the avg_trace subplot
        axs[col].set_xticks([])
        axs[col].set_yticks([])
        axs[col].set_xlabel("")
        axs[col].set_ylabel("")
        axs[col].spines["top"].set_visible(False)
        axs[col].spines["right"].set_visible(False)

        # Remove x and y axis labels and ticks from the motor module subplot
        axs[col].set_xticks(x_values, channel_order_dtr)
        axs[col].set_yticks([])
        # axs[1, col].set_xlabel('')
        # axs[1, col].set_ylabel('')

    # Adjust spacing between subplots
    plt.tight_layout()
    # fig.suptitle(synergies_title, fontsize=16, fontweight='bold')
    # plt.savefig(modules_filename, dpi=300)
    # plt.subplots_adjust(top=0.9)
    plt.show()


def show_synergies_dtr(
    data_input, refined_primitives, chosen_synergies, synergies_name="./output.png"
):
    """
    Make sure you check the channel order!!

    """

    motor_p_data = pd.read_csv(refined_primitives, header=0)
    # =======================================
    # Presenting Data as a mutliplot figure |
    # =======================================
    motor_primitives, motor_modules = synergy_extraction(data_input, chosen_synergies)
    motor_primitives = motor_p_data.to_numpy()
    channel_order_dtr = ["GM", "Ip", "BF", "VL", "Gs", "TA", "St", "Gr"]

    trace_length = 200

    samples = np.arange(0, len(motor_primitives))
    samples_binned = np.arange(trace_length)

    fig, axs = plt.subplots(chosen_synergies, 2, figsize=(12, 8))
    # Calculate the average trace for each column
    number_cycles = (
        len(motor_primitives) // trace_length
    )  # Calculate the number of 200-value bins

    for col in range(chosen_synergies):
        primitive_trace = np.zeros(
            trace_length
        )  # Initialize an array for accumulating the trace values

        # Iterate over the binned data by the number of cycles
        for i in range(number_cycles):
            # Get the data for the current bin in the current column
            time_point_average = motor_primitives[
                i * trace_length : (i + 1) * trace_length, col
            ]

            # Accumulate the trace values
            primitive_trace += time_point_average

        # Calculate the average by dividing the accumulated values by the number of bins
        primitive_trace /= number_cycles

        # Plot the average trace in the corresponding subplot
        axs[col, 1].plot(
            samples[samples_binned], primitive_trace, color="red", label="Average Trace"
        )
        axs[col, 1].set_title("Synergy {}".format(col + 1))

        # Iterate over the bins again to plot the individual bin data
        for i in range(number_cycles):
            # Get the data for the current bin in the current 0, column
            time_point_average = motor_primitives[
                i * trace_length : (i + 1) * trace_length, col
            ]

            # Plot the bin data
            axs[col, 1].plot(
                samples[samples_binned],
                time_point_average,
                label="Bin {}".format(i + 1),
                color="black",
                alpha=0.1,
            )

        # Add vertical lines at the halfway point in each subplot
        axs[col, 1].axvline(x=100, color="black")

        # Begin Presenting Motor Modules

        # Get the data for the current column
        motor_module_column_data = motor_modules[
            col, :
        ]  # Select all rows for the current column

        # Set the x-axis values for the bar graph
        x_values = np.arange(len(motor_module_column_data))

        # Plot the bar graph for the current column in the corresponding subplot
        axs[col, 0].bar(x_values, motor_module_column_data)

        # Remove top and right spines of each subplot
        axs[col, 1].spines["top"].set_visible(False)
        axs[col, 1].spines["right"].set_visible(False)
        axs[col, 0].spines["top"].set_visible(False)
        axs[col, 0].spines["right"].set_visible(False)

        # Remove labels on x and y axes
        axs[col, 0].set_xticklabels([])
        axs[col, 1].set_yticklabels([])

        # Remove x and y axis labels and ticks from the avg_trace subplot
        axs[col, 1].set_xticks([])
        axs[col, 1].set_yticks([])
        axs[col, 1].set_xlabel("")
        axs[col, 1].set_ylabel("")

        # Remove x and y axis labels and ticks from the motor module subplot
        axs[col, 0].set_xticks(x_values, channel_order_dtr)
        # axs[1, col].set_xticks([])
        axs[col, 0].set_yticks([])
        # axs[1, col].set_xlabel('')
        # axs[1, col].set_ylabel('')

    # Adjust spacing between subplots
    plt.tight_layout()
    fig.suptitle(synergies_name, fontsize=16, fontweight="bold")
    plt.subplots_adjust(top=0.9)
    # plt.savefig(synergies_filename, dpi=300)

    # Show all the plots
    plt.show()


def sel_primitive_trace_with_fwhm(
    motor_primitives, synergy_selection, selected_primitive_title="Output"
):
    """This will plot the selected motor primitives
    @param data_input: path to csv data file
    @param synergy_selection: how many synergies you want

    @return null
    """

    # motor_primitives, motor_modules = synergy_extraction(data_input, synergy_selection)

    # Smoothen the data

    fwhm_line = fwhm_calc_dep(motor_primitives, synergy_selection)

    samples = np.arange(0, len(motor_primitives))
    samples_binned = np.arange(200)
    number_cycles = len(motor_primitives) // 200

    # Plot
    primitive_trace = np.zeros(200)

    # Plotting Primitive Selected Synergy Count

    # Iterate over the bins
    for i in range(number_cycles):
        # Get the data for the current bin

        time_point_average = motor_primitives[
            i * 200 : (i + 1) * 200, synergy_selection - 1
        ]

        # Accumulate the trace values
        current_primitive = motor_primitives[
            i * 200 : (i + 1) * 200, synergy_selection - 1
        ]
        plt.plot(samples[samples_binned], current_primitive, color="black", alpha=0.2)
        plt.hlines(
            y=fwhm_line[i],
            xmin=0,
            xmax=len(current_primitive),
            color="black",
            alpha=0.2,
        )

        primitive_trace += time_point_average

    # Calculate the average by dividing the accumulated values by the number of bins
    primitive_trace /= number_cycles

    plt.plot(samples[samples_binned], primitive_trace, color="blue")

    # Plotting individual primitives in the background
    # selected_primitive = motor_primitives[:, synergy_selection - 2]

    # Using the order F so the values are in column order
    # binned_primitives_raw = selected_primitive.reshape((200, -1), order='F')
    # binned_primitives = ndimage.median_filter(binned_primitives_raw, size=3)
    # plt.plot(binned_primitives_raw[:, i], color='black', alpha=0.2)
    # plt.plot(binned_primitives_raw, color='black', alpha=0.2)
    # print(fwhl_start_stop[3, 1])

    # Removing axis values
    plt.xticks([])
    plt.yticks([])

    # Add a vertical line at the halfway point
    plt.axvline(x=100, color="black")

    # Adding a horizontal line for fwhl

    # fwhl_line_start = np.mean(fwhl_start_stop[:, 0])
    # fwhl_line_stop = np.mean(fwhl_start_stop[:, 1])
    # plt.hlines(y=np.mean(fwhl_height), xmin=fwhl_line_start, xmax=fwhl_line_stop, color='red')

    # Add labels for swing and stance
    plt.text(50, -0.2 * np.max(primitive_trace), "Swing", ha="center", va="center")
    plt.text(150, -0.2 * np.max(primitive_trace), "Stance", ha="center", va="center")

    # Removing top and right spines of the plot
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(True)
    plt.gca().spines["left"].set_visible(True)
    plt.title(selected_primitive_title, fontsize=12, fontweight="bold")
    # plt.savefig(selected_primitive_title, dpi=300)
    plt.show()


# Plotting Section
def sel_primitive_trace(
    motor_primitives, synergy_selection, selected_primitive_title="Output"
):
    """This will plot the selected motor primitives
    @param data_input: path to csv data file
    @param synergy_selection: how many synergies you want

    @return null
    """

    motor_p_data = pd.read_csv(motor_primitives, header=0)

    motor_primitives = motor_p_data.to_numpy()
    print(motor_primitives)
    # motor_primitives, motor_modules = synergy_extraction(data_input, synergy_selection)

    # Smoothen the data

    # fwhl, fwhl_start_stop, fwhl_height = full_width_half_abs_min_scipy(motor_primitives, synergy_selection)
    # fwhm = []

    # fwhm, half_values = fwhm(motor_primitives, synergy_selection)

    # applying mask to exclude values which were subject to rounding errors
    # mcurrent_primitive = np.asarray(current_primitive[primitive_mask])

    samples = np.arange(0, len(motor_primitives))
    samples_binned = np.arange(200)
    number_cycles = len(motor_primitives) // 200

    # Plot
    primitive_trace = np.zeros(200)

    # Plotting Primitive Selected Synergy Count

    # Iterate over the bins
    for i in range(number_cycles):
        # Get the data for the current bin

        time_point_average = motor_primitives[
            i * 200 : (i + 1) * 200, synergy_selection - 1
        ]

        # fwhl_line_start = fwhl_start_stop[i, 0]
        # fwhl_line_stop = fwhl_start_stop[i, 1]
        # plt.hlines(fwhl_height[i], fwhl_line_start, fwhl_line_stop, color='black', alpha=0.2)
        # Accumulate the trace values
        current_primitive = motor_primitives[
            i * 200 : (i + 1) * 200, synergy_selection - 1
        ]
        # current_primitive = sp.signal.savgol_filter(current_primitive, , 3)

        primitive_mask = current_primitive > 0.0
        # primitive_mask = interpolate_primitive(primitive_mask)
        # applying mask to exclude values which were subject to rounding errors
        mcurrent_primitive = np.asarray(current_primitive[primitive_mask])

        plt.plot(samples[samples_binned], current_primitive, color="black", alpha=0.2)
        # peaks, properties = signal.find_peaks(current_primitive, distance=40, width=10)
        # max_ind = np.argmax(peaks)
        # print(properties['widths'][max_ind])
        # max_width = properties['widths'][max_ind]
        # fwhl.append(max_width)

        # plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
        # xmax=properties["right_ips"], color='black', alpha=0.2)
        primitive_trace += time_point_average

    # Calculate the average by dividing the accumulated values by the number of bins
    primitive_trace /= number_cycles

    plt.plot(samples[samples_binned], primitive_trace, color="blue")

    # Plotting individual primitives in the background
    selected_primitive = motor_primitives[:, synergy_selection - 2]

    # Using the order F so the values are in column order
    binned_primitives_raw = selected_primitive.reshape((200, -1), order="F")
    # binned_primitives = ndimage.median_filter(binned_primitives_raw, size=3)
    plt.plot(binned_primitives_raw[:, i], color="black", alpha=0.2)
    plt.plot(binned_primitives_raw, color="black", alpha=0.2)

    # Removing axis values
    plt.xticks([])
    plt.yticks([])

    # Add a vertical line at the halfway point
    plt.axvline(x=100, color="black")

    # Adding a horizontal line for fwhl

    # fwhl_line_start = np.mean(fwhl_start_stop[:, 0])
    # fwhl_line_stop = np.mean(fwhl_start_stop[:, 1])
    # plt.hlines(y=np.mean(fwhl_height), xmin=fwhl_line_start, xmax=fwhl_line_stop, color='red')

    # Add labels for swing and stance
    # plt.text(50, -0.2 * np.max(primitive_trace), 'Swing', ha='center', va='center')
    # plt.text(150, -0.2 * np.max(primitive_trace), 'Stance', ha='center', va='center')

    # Removing top and right spines of the plot
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(True)
    plt.gca().spines["left"].set_visible(True)
    plt.title(selected_primitive_title, fontsize=16, fontweight="bold")
    # plt.savefig(selected_primitive_title, dpi=300)
    plt.show()

    # fwhl = np.asarray(fwhl)
    # return fwhllab


# %%


def main():
    # raw_file = pd.read_csv("../../data/emg/")

    raw_file = pd.read_csv("../../data/emg/com-7-egr3-full.txt", header=0)

    raw_file_adjusted = raw_file.set_index("Time")

    channel_order = ["GM", "Ip", "BF", "VL", "St", "TA", "Gs", "Gr"]
    channel_order_dtr = ["GM", "Ip", "BF", "VL", "Gs", "TA", "St", "Gr"]
    # print(raw_file_adjusted["12 Synch"])

    sync_counts = raw_file_adjusted["12 Synch"].value_counts().items()
    count_sync = sync_counts["1"]

    # motor_p, motor_m = synergy_extraction(test_norm_emg, 3)

    # Title Names
    # title_names = [
    #     "Synergies for DTR-M5 preDTX without perturbation",
    #     "Synergies for DTR-M5 preDTX with perturbation",
    #     "Synergies for DTR-M5 postDTX without perturbation",
    #     "Synergies for DTR-M5 postDTX with perturbation",
    # ]
    #
    # # Normalized Data List
    # conditions_normalized_dtr = [
    #     "../../emg/synergies/norm-emg-preDTX-100.csv",
    #     "../../emg/synergies/norm-emg-preDTX-per.csv",
    #     "../../emg/synergies/norm-postdtx-non.csv",
    #     "../../emg/synergies/norm-postdtx-per.csv",
    # ]
    #
    # # Cleaned up Primitives
    # conditions_primitives_dtr = [
    #     "../../emg/synergies/predtx-non-primitives-test.txt",
    #     "../../emg/synergies/predtx-per-primitives-test.txt",
    #     "../../emg/synergies/postdtx-non-primitives.txt",
    #     "../../emg/synergies/postdtx-per-primitives.txt",
    # ]
    #
    # show_synergies('./norm-wt-m1-non.csv', './wt-m1-non-primitives.txt', synergy_selection, "Synergies for WT-M1 without perturbation")
    # show_synergies('./norm-wt-m1-per.csv', './wt-m1-per-primitives.txt', synergy_selection, "Synergies for WT-M1 with perturbation")
    #
    # # for i in range(len(conditions_normalized_dtr)):
    # #     show_synergies_dtr(conditions_normalized_dtr[i], conditions_primitives_dtr[i], synergy_selection, title_names[i])
    # motor_p, motor_m = synergy_extraction(
    #     conditions_normalized_dtr[0], synergy_selection
    # )
    #
    # interp(motor_p[:, 0])
    #
    # print(motor_p[:, 0])

    # motor_p = [-5,-4.19,-3.54,-3.31,-2.56,-2.31,-1.66,-0.96,-0.22,0.62,1.21,3]


if __name__ == "__main__":
    main()
# %%
