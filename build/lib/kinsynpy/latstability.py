import csv
import os
from typing import Union

import dlc2kinematics as dlck
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns


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


def step_duration(input_dataframe, swonset_ch):
    """Calculates step duration based on swing onsets
    :param input_dataframe: Exported channels from spike most importantly swing onset

    :return adjusted_time_differences:
    :return adjusted_treadmill_speeds:
    """
    # Define the value and column to search for
    value_to_find = 1
    column_to_search = swonset_ch
    column_for_time = "Time"
    column_for_treadmill = "2 Trdml"

    # Store time values and treadmill speed when the specified value is found
    time_values = []
    treadmill_speed = []

    # Iterate through the DataFrame and process matches
    for index, row in input_dataframe.iterrows():
        if row[column_to_search] == value_to_find:
            time_value = row[column_for_time]
            time_values.append(time_value)
            treadmill_value = row[column_for_treadmill]
            treadmill_speed.append(treadmill_value)

    # Calculate the differences between consecutive time values
    time_differences = []
    for i in range(len(time_values)):
        time_diff = time_values[i] - time_values[i - 1]
        time_differences.append(time_diff)

    # Finding the average value for the list
    time_differences_array = np.array(time_differences)
    treadmill_speed_array = np.array(treadmill_speed)

    # Creating masks to filter any values above 1 as this would be between distinct recordings
    recording_cutoff_high = 0.6
    recording_cutoff_low = 0.000
    cutoff_high = time_differences_array <= recording_cutoff_high
    cutoff_low = time_differences_array >= recording_cutoff_low
    combined_filter = np.logical_and(cutoff_low, cutoff_high)

    # Applying the filter to the arrays
    adjusted_time_differences = time_differences_array[combined_filter]
    adjusted_treadmill_speeds = treadmill_speed_array[combined_filter]
    adj_time_xaxis = np.arange(0, len(adjusted_time_differences))

    # Finding average step cylce for this length
    average_step_difference = np.mean(adjusted_time_differences)
    print(f" Average step cycle duration for this trial: {average_step_difference}")

    return adjusted_time_differences, adjusted_treadmill_speeds


def swing_estimation(input_dataframe, x_channel, width_threshold=40):
    """This approximates swing onset and offset from kinematic data
    :param input_dataframe: Exported channels from spike most importantly the x values for a channel

    :return swing_onset: A list of indices where swing onset occurs
    :return swing_offset: A list of indices where swing offet occurs
    """

    foot_cord = input_dataframe[x_channel].to_numpy(dtype=float)

    swing_offset, _ = sp.signal.find_peaks(foot_cord, distance=width_threshold)
    swing_onset, _ = sp.signal.find_peaks(-foot_cord, width=width_threshold)

    return swing_onset, swing_offset


def step_cycle_est(input_dataframe, x_channel, width_threshold=40):
    """This approximates swing onset and offset from kinematic data
    :param input_dataframe: Exported channels from spike most importantly the x values for a channel

    :return cycle_durations: A numpy array with the duration of each cycle
    :return average_step: A list of indices where swing offet occurs
    """

    time = input_dataframe["Time"].to_numpy(dtype=float)

    swing_onset, _ = swing_estimation(
        input_dataframe=input_dataframe, x_channel=x_channel
    )
    onset_timing = time[swing_onset]

    cycle_durations = np.array([])
    for i in range(len(onset_timing) - 1):
        time_diff = onset_timing[i + 1] - onset_timing[i]
        cycle_durations = np.append(cycle_durations, time_diff)

    avg_cycle_period = np.mean(cycle_durations)

    return cycle_durations, avg_cycle_period


def manual_marks(related_trace, title="Select Points"):
    """Manually annotate points of interest on a given trace
    :param related_trace: Trace you want to annotate

    :return manual_marks_x: array of indices to approx desired value in original trace
    :return manual_marks_y: array of selected values
    """

    # Open interface with trace
    plt.plot(related_trace)
    plt.title(title)

    # Go through and label regions desired
    manual_marks_pair = plt.ginput(0, 0)

    # Store x coordinates as rounded off ints to be used as indices
    manual_marks_x = np.asarray(list(map(lambda x: x[0], manual_marks_pair)))
    manual_marks_x = manual_marks_x.astype(np.int32)

    # Store y coordinates as the actual value desired
    manual_marks_y = np.asarray(list(map(lambda x: x[1], manual_marks_pair)))
    plt.show()

    return manual_marks_x, manual_marks_y


def extract_cycles(input_dataframe, swonset_channel="44 sw onset"):
    """Get cycle periods
    @param input_dataframe: spike file input as *.csv
    @param swonset_channel: the channel with swing onsets

    @return step_cycles: the step cycles
    """
    # Define the value and column to search for
    value_to_find = 1
    column_to_search = swonset_channel
    column_for_time = "Time"
    column_for_treadmill = "2 Trdml"

    # Store time values and treadmill speed when the specified value is found
    time_values = []
    treadmill_speed = []

    # Iterate through the DataFrame and process matches
    for index, row in input_dataframe.iterrows():
        if row[column_to_search] == value_to_find:
            time_value = row[column_for_time]
            time_values.append(time_value)
            treadmill_value = row[column_for_treadmill]
            treadmill_speed.append(treadmill_value)

    # Calculate the differences between consecutive time values
    time_differences = []
    for i in range(len(time_values)):
        time_diff = time_values[i] - time_values[i - 1]
        time_differences.append(time_diff)

    # Finding the average value for the list
    time_differences_array = np.array(time_differences)

    # Creating masks to filter any values above 1 as this would be between distinct recordings
    recording_cutoff_high = 0.6
    recording_cutoff_low = 0.000
    cutoff_high = time_differences_array <= recording_cutoff_high
    cutoff_low = time_differences_array >= recording_cutoff_low
    combined_filter = np.logical_and(cutoff_low, cutoff_high)

    # Applying the filter to the arrays
    step_cycles = time_differences_array[combined_filter]

    return step_cycles


def stance_duration(
    input_dataframe, swonset_channel="44 sw onset", swoffset_channel="45 sw offset"
):
    """Stance duration during step cycle
    @param input_dataframe: spike file input as *.csv
    @param swonset_channel: the channel with swing onsets
    @param swoffset_channel: the channel with swing offsets

    @return stance_duration_lengths: How long each stance duration is
    @return stance_duration_timings: List of timings where stance duration begins
    """

    # Define the value and column to search for
    value_to_find = 1
    stance_begin = swoffset_channel
    # stance_end = swonset_channel
    # column_to_search = swonset_channel
    column_for_time = "Time"
    # column_for_treadmill = "2 Trdml"

    # Store time values and treadmill speed when the specified value is found
    time_values = []

    # Find the first stance phase to start tracking time duration
    first_stance = (
        input_dataframe[stance_begin]
        .loc[input_dataframe[stance_begin] == value_to_find]
        .index[0]
    )

    # Iterate through the DataFrame and process matches
    for index, row in input_dataframe.iloc[first_stance:].iterrows():
        if row[swoffset_channel] == value_to_find:
            # print("swoff found at", row[column_for_time])
            time_value = row[column_for_time]
            time_values.append(time_value)
        elif row[swonset_channel] == value_to_find:
            # print("swon found at", row[column_for_time])
            time_value = row[column_for_time]
            time_values.append(time_value)
            # treadmill_value = row[column_for_treadmill]
            # treadmill_speed.append(treadmill_value)

    # Calculate the differences between consecutive time values
    time_differences = []
    for i in range(len(time_values)):
        time_diff = time_values[i] - time_values[i - 1]
        time_differences.append(time_diff)

    # Finding the average value for the list
    time_differences_array = np.array(time_differences)

    # Creating masks to filter any values above 1 as this would be between distinct recordings
    recording_cutoff_high = 0.6
    recording_cutoff_low = 0.000
    cutoff_high = time_differences_array <= recording_cutoff_high
    cutoff_low = time_differences_array >= recording_cutoff_low
    combined_filter = np.logical_and(cutoff_low, cutoff_high)
    # Applying the filter to the arrays
    stance_duration_lengths = time_differences_array[combined_filter]

    stance_duration_timings = time_values

    return stance_duration_lengths, stance_duration_timings


def weighted_slope(input_dataframe, p, comy="37 CoMy (cm)"):
    data = input_dataframe[comy].values

    n = len(data)
    slope = [0] * n  # initialize with zeros

    for i in range(p, n - p):
        past = data[i - p : i]
        future = data[i + 1 : i + p + 1]

        # calculate means of past and future points
        mean_past = np.mean(past)
        mean_future = np.mean(future)

        # update slope at time i using the calculated means
        slope[i] = (mean_future - mean_past) / 2

    slope = np.array(slope)

    return slope


def median_filter(arr, k):
    """
    :param arr: input numpy array
    :param k: is the size of the window you want to slide over the array (kernel).

    :return filtarr: An array of the same length where each element is the median of
    a window centered around the index in the array.
    """
    # Initialize output array
    result = []

    # Iterate over every index in arr
    for i in range(len(arr)):
        if i < (k // 2) or i > len(arr) - (k // 2) - 1:
            # Add a placeholder for the indices before k//2 and after length of array - k//2 - 1
            result.append(np.nan)
        else:
            # Calculate median within window and append to result list
            result.append(np.median(arr[i - (k // 2) : i + (k // 2) + 1]))

    filter = np.isnan(comy_values)
    filtarr = np.asarray(result)
    filtarr = filtarr[~filter]

    return filtarr


def spike_slope(comy, p):
    """
    :param comy: numpy array of the y coordinate of the center of mass
    :param p: How many values you want in either direction to be included


    """

    n = len(comy)
    slope = [0] * n  # initialize with zeros

    for i in range(p, n - p):
        past = comy[i - p : i]
        future = comy[i + 1 : i + p + 1]

        # calculate means of past and future points
        mean_past = np.mean(past)
        mean_future = np.mean(future)

        # update slope at time i using the calculated means
        slope[i] = (mean_future - mean_past) / 2

    slope = np.array(slope)

    return slope


def fir_filter(data, taps):
    """
    data : input signal
    taps : number of filter taps
    """

    # Define the FIR filter using a low-pass prototype.
    taps = min(len(data), taps)
    cutoff_freq = 0.3  # Cutoff frequency is set to 30% of the Nyquist rate.
    # This means that only frequencies below this will pass through the filter,
    # and all others will be attenuated.

    nyquist = len(data) // 2
    freqs = np.arange(taps) / taps
    gains = 0.5 - 0.5 * np.cos(2 * np.pi * (freqs - cutoff_freq))

    # Apply the filter using a convolution, which is equivalent to multiplication in the frequency domain.
    filtered = sp.signal.lfilter([1], gains, data)

    return filtered


# TODO: Need visit documentation to understand how to get slope same as spike2
def slope(input_dataframe, time_constant, comy="37 CoMy (cm)"):

    # Converting time constant to meaningful indices
    time_factor = int(time_constant / 2)

    # Bring in data and convert to numpy array
    comy_values = input_dataframe[comy].values

    # Remove missing values
    # filter = comy_values[np.logical_not(np.isnan(comy_values))]
    filter = np.isnan(comy_values)
    comy_values = comy_values[~filter]

    # Calculate slope based on spike 2 specifications

    slope = []

    for i in range(len(comy_values)):
        if i <= 5:
            values_to_consider = np.array(comy_values[0:time_factor])
            current_slope = np.mean(values_to_consider)
            slope.append(current_slope)
        else:
            previous_values = np.array(comy_values[i - time_factor : i])
            next_values = np.array(comy_values[i : i + time_factor])
            current_slope = (np.mean(previous_values) + np.mean(next_values)) / 2
            slope.append(current_slope)

    return slope


# TODO: Center of pressure calculation needs to get done
def copressure(input_dataframe, ds_channel, hl_channel, fl_channel):
    """Calculation for center of pressure
    :param input_dataframe: spike file input as *.csv

    """
    input_dataframe_subset = input_dataframe.loc[
        :, ["Time", ds_channel, hl_channel, fl_channel]
    ]

    print(input_dataframe_subset)


def step_width(
    input_dataframe: pd.DataFrame,
    rl_swoff: str,
    ll_swoff: str,
    rl_y: str,
    ll_y: str,
) -> np.array:
    """Step width during step cycle
    :param input_dataframe: spike file input as *.csv
    :param rl_swoff: channel containing swoffset events
    :param ll_swon: channel containing swoffset events
    :param rl_y: spike channel with y coordinate for the right limb
    :param ll_y: spike channel with y coordinate for the right limb

    :return step_widths: numpy array of step width values for each step cycle
    """
    value_to_find = 1

    # Filtering whole dataframe down to values we are considering
    input_dataframe_subset = input_dataframe.loc[
        :, ["Time", rl_swoff, ll_swoff, rl_y, ll_y]
    ]
    input_dataframe_subset = input_dataframe_subset.set_index("Time")

    rl_swoff_marks = input_dataframe_subset.loc[
        input_dataframe_subset[rl_swoff] == value_to_find
    ].index.tolist()
    ll_swoff_marks = input_dataframe_subset.loc[
        input_dataframe_subset[ll_swoff] == value_to_find
    ].index.tolist()

    # Testing with swon method
    rl_step_placement = input_dataframe_subset.loc[rl_swoff_marks, :][rl_y].values
    ll_step_placement = input_dataframe_subset.loc[ll_swoff_marks, :][ll_y].values

    # Dealing with possible unequal amount of recorded swoffsets for each limb
    comparable_steps = 0
    if rl_step_placement.shape[0] >= ll_step_placement.shape[0]:
        comparable_steps = ll_step_placement.shape[0]
    else:
        comparable_steps = rl_step_placement.shape[0]

    step_widths = []

    # Compare step widths for each step
    for i in range(comparable_steps):
        new_width = np.abs(rl_step_placement[i] - ll_step_placement[i])
        step_widths.append(new_width)

    step_widths = np.asarray(step_widths)

    return step_widths


def step_width_est(
    input_dataframe: pd.DataFrame,
    rl_x: str,
    ll_x: str,
    rl_y: str,
    ll_y: str,
) -> np.array:
    """Step width during step cycle
    :param input_dataframe: spike file input as *.csv
    :param rl_swoff: channel containing swoffset events
    :param ll_swon: channel containing swoffset events
    :param rl_y: spike channel with y coordinate for the right limb
    :param ll_y: spike channel with y coordinate for the right limb

    :return step_widths: numpy array of step width values for each step cycle
    """

    # Filtering whole dataframe down to values we are considering
    rl_y_cords = input_dataframe[rl_y].to_numpy(dtype=float)
    ll_y_cords = input_dataframe[ll_y].to_numpy(dtype=float)

    _, rl_swoff = swing_estimation(input_dataframe, rl_x)
    _, ll_swoff = swing_estimation(input_dataframe, ll_x)

    rl_step_placement = rl_y_cords[rl_swoff]
    ll_step_placement = ll_y_cords[ll_swoff]

    # Dealing with possible unequal amount of recorded swoffsets for each limb
    comparable_steps = 0
    if rl_step_placement.shape[0] >= ll_step_placement.shape[0]:
        comparable_steps = ll_step_placement.shape[0]
    else:
        comparable_steps = rl_step_placement.shape[0]

    step_widths = []

    # Compare step widths for each step
    for i in range(comparable_steps):
        new_width = np.abs(rl_step_placement[i] - ll_step_placement[i])
        step_widths.append(new_width)

    step_widths = np.asarray(step_widths)

    return step_widths


def hip_height(
    input_dataframe,
    toey="24 toey (cm)",
    hipy="16 Hipy (cm)",
    manual=False,
    prominence=0.004,
):
    """Approximates Hip Height
    :param input_dataframe: spike file input as *.csv
    :param toey: spike channel with y coordinate for the toe
    :param hipy: spike channel with y coordinate for the hip
    :param manual: (Boolean) whether to manually label regions where foot is on ground

    :return hip_height: returns hip height in meters (cm)
    """

    # Bringing in the values for toey and hipy
    toey_values = input_dataframe[toey].to_numpy(dtype=float)
    hipy_values = input_dataframe[hipy].to_numpy(dtype=float)

    # Remove missing values
    toey_values = toey_values[np.logical_not(np.isnan(toey_values))]
    hipy_values = hipy_values[np.logical_not(np.isnan(hipy_values))]

    # Either manually mark regions foot is on the ground or go with proxy
    if manual is False:

        # Getting lower quartile value of toey as proxy for the ground
        toey_cutoff = np.percentile(toey_values, q=75)
        toey_values[toey_values > toey_cutoff] = np.nan
        toey_peaks, properties = sp.signal.find_peaks(
            -toey_values, prominence=(None, prominence)
        )
        toey_lower = toey_values[toey_peaks]
        toey_lower = np.mean(toey_lower)

        average_hip_value = np.mean(hipy_values)
        hip_height = average_hip_value - toey_lower

        # plt.plot(toey_values, label="toey")
        # plt.plot(hipy_values, label="hipy")
        # plt.plot(toey_peaks, toey_values[toey_peaks], "x")
        # plt.show()

    elif manual is True:
        # Selection of regions foot would be on the ground
        on_ground_regions, _ = manual_marks(
            toey_values, title="Select Regions foot is on the ground"
        )

        toe_to_consider = np.array([])
        hip_to_consider = np.array([])
        stance_begin = on_ground_regions[0::2]
        swing_begin = on_ground_regions[1::2]

        for i in range(len(stance_begin)):
            # Get regions to consider
            begin = stance_begin[i]
            end = swing_begin[i]

            relevant_toe = toey_values[begin:end]
            relevant_hip = hipy_values[begin:end]

            toe_to_consider = np.append(toe_to_consider, relevant_toe)
            hip_to_consider = np.append(hip_to_consider, relevant_hip)

        # Calculate hip height from filtered regions
        average_hip_value = np.mean(hip_to_consider)
        average_toe_value = np.mean(toe_to_consider)
        hip_height = average_hip_value - average_toe_value

    else:
        print("The `manual` variable must be a boolean")

    return hip_height


def froud_number(
    input_dataframe, trdm="2 Trdml", toey="24 toey (cm)", hipy="16 Hipy (cm)"
):
    # Need to get hip height and convert into centimeters
    hip = hip_height(input_dataframe, toey, hipy) / 100

    # Getting average treadmill speed
    treadmill_speed = input_dataframe[trdm].values
    avg_treadmill_speed = np.round(np.mean(treadmill_speed), 2)

    # Calculating number
    froud_number = np.power(avg_treadmill_speed, 2) / (9.81 * hip)

    return froud_number


# NOTE: Depending on completion of `spike_slope`
def xcom(input_dataframe, hip_height, comy="37 CoMy (cm)"):

    # Bring in data
    comy_values = input_dataframe[comy].values
    comy_values = comy_values[np.logical_not(np.isnan(comy_values))]

    # Getting slope of values
    vcom = slope(comy_values, 6)

    # Get xCoM in (cm)
    xcom = comy_values + (vcom / np.sqrt(981 / hip_height))

    # print("comy", comy_values.size)
    # print("vcom", vcom.size)
    # print("xcom", xcom)

    # x_axis = np.arange(len(comy_values))

    # Testing output
    # rbf = sp.interpolate.Rbf(x_axis, vcom, function="thin_plate", smooth=2)
    # xnew = np.linspace(x_axis.min(), x_axis.max(), num=100, endpoint=True)
    # ynew = rbf(xnew)

    # fig, axs = plt.subplots(4, 1, layout="constrained")
    # axs[0].set_title("CoMy")
    # axs[0].legend(loc="best")
    # axs[0].plot(x_axis, comy_values)
    # axs[1].set_title("vCoM")
    # axs[1].plot(x_axis, vcom)
    # axs[2].set_title("Radial basis funtion interpolation of vCoM")
    # axs[2].plot(xnew, ynew)
    # axs[0].plot(x_axis, xcom)

    # plt.show()

    return xcom


def mos_marks(related_trace, leftcop, rightcop, title="Select Points"):
    """Manually annotate points of interest on a given trace
    :param related_trace: Trace you want to annotate

    :return manual_marks_x: array of indices to approx desired value in original trace
    :return manual_marks_y: array of selected values
    """

    # Removing 0 values
    rightcop = np.where(rightcop == 0.0, np.nan, rightcop)
    leftcop = np.where(leftcop == 0.0, np.nan, leftcop)

    # Correcting to DS regions are close to label
    left_adjustment = np.mean(related_trace) + 0.5
    right_adjustment = np.mean(related_trace) - 0.5

    rightcop = rightcop * right_adjustment
    leftcop = leftcop * left_adjustment

    # Open interface with trace
    plt.plot(related_trace)
    plt.plot(leftcop)
    plt.plot(rightcop)
    plt.title(title)

    # Go through and label regions desired
    manual_marks_pair = plt.ginput(0, 0)

    # Store x coordinates as rounded off ints to be used as indices
    manual_marks_x = np.asarray(list(map(lambda x: x[0], manual_marks_pair)))
    manual_marks_x = manual_marks_x.astype(np.int32)

    # Store y coordinates as the actual value desired
    manual_marks_y = np.asarray(list(map(lambda x: x[1], manual_marks_pair)))
    plt.show()

    return manual_marks_x, manual_marks_y


def double_support_est(
    input_dataframe, fl_channel, hl_channel, manual_peaks=False, width_threshold=40
):
    """Calculates double support phases from step cycle estimations
    :param input_dataframe: Exported channels from spike most importantly x coordinates of limbs
    :param fl_channel: Channel for x coordinate for the forelimb
    :param hl_channel: Channel for x coordinate for the hindlimb
    :param manual_peaks: Label peaks manual or by swing_extimation function
    :param width_threshold: Width threshold used by automatic estimation

    :return double_support: Region where both limb are on the ground
    """

    fl_cord = input_dataframe[fl_channel].to_numpy(dtype=float)
    hl_cord = input_dataframe[hl_channel].to_numpy(dtype=float)

    time = input_dataframe["Time"].to_numpy(dtype=float)

    # Important here are fl_swoff and hl_swon
    fl_swoff, _ = swing_estimation(
        input_dataframe=input_dataframe, x_channel=fl_channel
    )
    _, hl_swon = swing_estimation(input_dataframe=input_dataframe, x_channel=hl_channel)

    # Making sure to start from correct spot in case there's hl_swon before
    first_index = fl_swoff[0]

    mask = hl_swon > first_index
    hl_swon = hl_swon[mask]

    fl_swoff_timings = time[fl_swoff]
    hl_swon_timings = time[hl_swon]
    ds_timings = np.array([])

    if len(fl_swoff) < len(hl_swon):
        for i in range(len(fl_swoff)):
            ds_timings = np.append(ds_timings, fl_swoff_timings[i])
            ds_timings = np.append(ds_timings, hl_swon_timings[i])
    else:
        for i in range(len(hl_swon)):
            ds_timings = np.append(ds_timings, fl_swoff_timings[i])
            ds_timings = np.append(ds_timings, hl_swon_timings[i])

    double_support = ds_timings

    return double_support


def mos(
    xcom, leftcop, rightcop, leftds, rightds, manual_peaks=False, width_threshold=40
):

    # Remove periods where it is not present or not valid
    # left_band = np.percentile(xcom, q=50)
    rightcop = np.where(rightcop == 0.0, np.nan, rightcop)
    leftcop = np.where(leftcop == 0.0, np.nan, leftcop)
    # rightcop[rightcop < right_band] = np.nan
    # leftcop[leftcop < left_band] = np.nan

    # Optional manual point selection
    if manual_peaks is False:
        # Getting peaks and troughs
        xcom_peaks, _ = sp.signal.find_peaks(xcom, width=width_threshold)
        xcom_troughs, _ = sp.signal.find_peaks(-xcom, width=width_threshold)
    elif manual_peaks is True:
        xcom_peaks, _ = mos_marks(xcom, leftds, rightds, title="Select Peaks")
        xcom_troughs, _ = mos_marks(xcom, leftds, rightds, title="Select Troughs")
    else:
        print("The `manual` variable must be a boolean")

    lmos_values = np.array([])
    rmos_values = np.array([])

    for i in range(len(xcom_peaks) - 1):
        # Getting window between peak values
        beginning = xcom_peaks[i]
        end = xcom_peaks[i + 1]
        region_to_consider = leftcop[beginning:end]

        # Getting non-nan values from region
        value_cop = region_to_consider[~np.isnan(region_to_consider)]

        # Making sure we are actually grabbing the last meaningful region of center of pressure
        if value_cop.shape[0] >= 2:
            cop_point = np.mean(value_cop)
            lmos = cop_point - xcom[beginning]
            lmos_values = np.append(lmos_values, lmos)

    for i in range(len(xcom_troughs) - 1):
        # Getting window between peak values
        beginning = xcom_troughs[i]
        end = xcom_troughs[i + 1]
        region_to_consider = rightcop[beginning:end]

        # Getting non-nan values from region
        value_cop = region_to_consider[~np.isnan(region_to_consider)]
        if value_cop.shape[0] >= 2:
            cop_point = np.mean(value_cop)
            rmos = xcom[beginning] - cop_point
            rmos_values = np.append(rmos_values, rmos)

    return lmos_values, rmos_values, xcom_peaks, xcom_troughs


def cycle_period_summary(directory_path):

    trial_list = read_all_csv(directory_path)

    cycle_results = {}
    for key in trial_list:
        cycle_results[key] = None

    # Now, you can access the data from each file like this:
    for filename, data in trial_list.items():
        step_duration_array, treadmill_speed = step_duration(data)
        cycle_results[filename] = np.mean(step_duration_array), np.mean(treadmill_speed)

    # Saving results to csv
    cycle_results_csv = "cycle_analysis.csv"

    with open(cycle_results_csv, "w", newline="") as file:
        writer = csv.writer(file)

        # Write the header row (optional)
        writer.writerow(["Data Point", "Mean", "Standard Deviation"])

        # Write data from the dictionary
        for key, (mean, std_dev) in cycle_results.items():
            writer.writerow([key, mean, std_dev])

    print(f"Data has been saved to {cycle_results_csv}")


# Main Code Body
def main():
    # print("Currently no tests in main")

    # print("Step Width for M1 without Perturbation")

    wt1nondf = pd.read_csv("./wt_data/wt-1-non-all.txt")
    wt5nondf = pd.read_csv("./wt_data/wt-5-non-all.txt")

    # Getting stance duration for all 4 limbs
    lhl_st_lengths, lhl_st_timings = stance_duration(
        wt1nondf, swonset_channel="51 HLl Sw on", swoffset_channel="52 HLl Sw of"
    )
    lfl_st_lengths, lfl_st_timings = stance_duration(
        wt1nondf, swonset_channel="55 FLl Sw on", swoffset_channel="56 FLl Sw of"
    )
    rhl_st_lengths, rhl_st_timings = stance_duration(
        wt1nondf, swonset_channel="53 HLr Sw on", swoffset_channel="54 HLr Sw of"
    )
    rfl_st_lengths, rfl_st_timings = stance_duration(
        wt1nondf, swonset_channel="57 FLr Sw on", swoffset_channel="58 FLr Sw of"
    )

    # For forelimb
    wt1_fl_step_widths = step_width(
        wt1nondf,
        rl_swoff="58 FLr Sw of",
        ll_swoff="56 FLl Sw of",
        rl_y="35 FRy (cm)",
        ll_y="33 FLy (cm)",
    )

    print(f"Manually done step width {len(wt1_fl_step_widths)}")
    # wt1_hl_step_widths = step_width(
    #     wt1nondf,
    #     rl_swoff="54 HLr Sw of",
    #     ll_swoff="52 HLl Sw of",
    #     rl_y="30 HRy (cm)",
    #     ll_y="28 HLy (cm)",
    # )

    # wt1_swingon, wt1_swingoff = swing_estimation(wt1nondf, x_channel="34 FRx (cm)")
    # wt1_cycle_dur, wt1_avg_cycle_period = step_cycle_est(
    #     wt1nondf, x_channel="34 FRx (cm)"
    # )

    wt1_fl_stwi_est = step_width_est(
        wt1nondf,
        rl_x="34 FRx (cm)",
        ll_x="32 FLx (cm)",
        rl_y="35 FRy (cm)",
        ll_y="33 FLy (cm)",
    )
    print(f"Estimated step width {len(wt1_fl_stwi_est)}")

    wt5_fl_stwi_est = step_width_est(
        wt5nondf,
        rl_x="34 FRx (cm)",
        ll_x="32 FLx (cm)",
        rl_y="35 FRy (cm)",
        ll_y="33 FLy (cm)",
    )
    print(f"Estimated step width {len(wt5_fl_stwi_est)}")

    right_ds = double_support_est(
        wt1nondf, fl_channel="34 FRx (cm)", hl_channel="29 HRx (cm)", manual_peaks=False
    )

    # Hip height test

    hiph_test_auto = hip_height(wt1nondf, "24 toey (cm)", "16 Hipy (cm)", manual=False)
    # hiph_test_manual = hip_height(wt1nondf, "24 toey (cm)", "16 Hipy (cm)", manual=True)

    print(f"hip height test {hiph_test_auto}")
    # print(f"hip height test {hiph_test_manual}")

    # print(len(wt1_cycle_dur))
    #
    # print(len(right_ds))

    # x_cord = wt1nondf["34 FRx (cm)"].to_numpy(dtype=float)
    #
    # custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    # sns.set(style="white", font_scale=1.0, rc=custom_params)
    #
    # swing_legend = [
    #     "Limb X cord",
    #     "Swing offset",
    #     "Swing onset",
    # ]
    #
    # # For plotting figure demonstrating how calculation was done
    # plt.title("Swing Estimation")
    # plt.plot(x_cord)
    # plt.plot(wt1_swingoff, x_cord[wt1_swingoff], "^")
    # plt.plot(wt1_swingon, x_cord[wt1_swingon], "v")
    # plt.legend(swing_legend, bbox_to_anchor=(1, 0.7))
    #
    # # Looking at result
    # # axs[1].set_title("MoS Result")
    # # axs[1].bar(0, np.mean(lmos), yerr=np.std(lmos), capsize=5)
    # # axs[1].bar(1, np.mean(rmos), yerr=np.std(rmos), capsize=5)
    # # axs[1].legend(mos_legend, bbox_to_anchor=(1, 0.7))
    #
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
# %%
