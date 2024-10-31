import csv

import dlc2kinematics as dlck
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal


def change_source(statement):
    print(statement)
    return statement


def frame_to_time(frame_index, fps=500):
    """Converts frames of video to time based on fps

    Parameters
    ----------
    frame_index:
        The index of the frame in question
    fps:
        The frames per second the video is shot at

    Returns
    -------
    time_seconds:
        Returns the time in seconds the frame is taken at

    """

    # Convert to miliseconds
    frame_mili = (frame_index / fps) * 1000
    # Convert to seconds
    time_seconds = frame_mili / 1000

    return time_seconds


def dlc_calibrate(df, bodyparts, scorer, calibration_markers):
    """
    Calibrate DLC output to physical distance via calibration marker estimation.
    This is redundant in ways as it's likely this will be loaded in for other things.

    Parameters
    ----------
    df: pandas.core.frame.Dataframe
        h5 file from DeepLabCut
    bodyparts: list
        This is output from the loaded dataset
    scorer: list
        This is output from the loaded dataset
    calibration_markers: list
        List of names for calibration markers

    Returns
    -------
    calibration_factor:
        float value to convert given pixel values to (cm)
    """

    avg_cal_cords = {}

    # TODO: Hard coded to shit plz fix later
    # This is done explicitly based on how I label
    top_row = calibration_markers[0:3]
    bottom_row = calibration_markers[3:6]

    for i in range(len(calibration_markers)):
        # Loading in Respective Dataframe for each marker
        calib_shown = df[scorer][calibration_markers[i]]
        calib_shown = calib_shown.drop(columns=["likelihood"])

        # Get average position
        avg_cord_np = np.array([])
        avg_x, avg_y = calib_shown["x"].mean(), calib_shown["y"].mean()
        avg_cord_np = np.append(avg_cord_np, [avg_x, avg_y])
        entry = calibration_markers[i]
        avg_cal_cords[entry] = avg_cord_np

    # Getting average x movement factor (2 cm across)
    two_cm_difs = np.array([])
    for i in range(2):
        top_dif = avg_cal_cords[top_row[i + 1]][0] - avg_cal_cords[top_row[i]][0]
        bottom_dif = (
            avg_cal_cords[bottom_row[i + 1]][0] - avg_cal_cords[bottom_row[i]][0]
        )
        two_cm_difs = np.append(two_cm_difs, top_dif)
        two_cm_difs = np.append(two_cm_difs, bottom_dif)

    x_factor = np.mean(two_cm_difs) / 2
    # print(x_factor)

    # Getting average y movement factor (2.5 cm across)
    # Noting y values towards bottom of image are higher than the top
    two_p5_cm_difs = np.array([])
    for i in range(3):
        y_dif = avg_cal_cords[bottom_row[i]][1] - avg_cal_cords[top_row[i]][1]
        two_p5_cm_difs = np.append(two_p5_cm_difs, y_dif)

    y_factor = np.mean(two_p5_cm_difs) / 2.5
    # print(y_factor)

    calibration_factor = np.mean([x_factor, y_factor])
    # print(calibration_factor)

    return calibration_factor


def manual_marks(related_trace, title="Select Points"):
    """Manually annotate points of interest on a given trace

    Parameters
    ----------
    related_trace:
        Trace you want to annotate

    Returns
    -------
    manual_marks_x:
        array of indices to approx desired value in original trace
    manual_marks_y:
        array of selected values
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


def analysis_reg_sel(mirror_y, com_y):

    time = np.arange(0, len(com_y), 1)

    # Open interface with trace
    plt.plot(time, mirror_y, label="Mirror Y")
    plt.plot(time, com_y, label="CoM Y")
    plt.title("Label Start and Stop of Desired Regions")

    # Go through and label regions desired
    manual_marks_pair = plt.ginput(0, 0)

    # Store x coordinates as rounded off ints to be used as indices
    manual_marks_x = np.asarray(list(map(lambda x: x[0], manual_marks_pair)))
    manual_marks_x = manual_marks_x.astype(np.int32)

    plt.show()

    region_start = manual_marks_x[0]
    region_stop = manual_marks_x[1]

    return region_start, region_stop


def mark_process(df, scorer, marker, cord, calib, smooth_wind=40):
    """
    Extracts out and smoothens a marker coordinate to an numpy array

    Parameters
    ----------
    df: pandas.core.frame.Dataframe
        input h5 data file to get marker from
    scorer: str
        Who scored the that trained the model, which analyzed the videos
    marker: str
        Name of marker you want to extract
    cord: str
        Whether you want the `x` or `y` coordinate of the marker
    calib: numpy.float64
        Calibration factor calculated from `dlc_calibrate`
    smooth_wind: int, default=`40`
        Just the smoothening window of `scipy.signal.savgol_filter`

    Returns
    -------
    marker_np:
        Smoothened 1D numpy array of the given marker
    """

    mark_df = df[scorer][marker]

    # Get single dimension out and calibrate to (cm)
    marker_np = pd.array(mark_df[cord])
    marker_np = marker_np / calib

    # Smoothen
    marker_np = signal.savgol_filter(marker_np, smooth_wind, 3)

    return marker_np


def swing_estimation(foot_cord, manual=False, width_threshold=40):
    """This approximates swing onset and offset from kinematic data.

    Parameters
    ----------
    foot_cord: numpy.ndarray
        Numpy array of the `x` coordinate of the limb
    manual: boolean, default=`False`
        Whether or not you want to manually annotate the trace
    width_threshold: int, default=`40`
        The width cutoff used by `scipy.signal.find_peaks`

    Returns
    -------
    swing_onset:
        A list of indices where swing onset occurs
    swing_offset:
        A list of indices where swing offset occurs
    """

    if manual is False:
        # Getting peaks and troughs
        swing_offset, _ = signal.find_peaks(foot_cord, width=width_threshold)
        swing_onset, _ = signal.find_peaks(-foot_cord, width=width_threshold)
    elif manual is True:
        swing_offset, _ = manual_marks(foot_cord, title="Select Swing offsets")
        swing_onset, _ = manual_marks(-foot_cord, title="Select Swing onsets")
    else:
        print("The `manual` variable must be a boolean")

    return swing_onset, swing_offset


def step_cycle_est(foot_cord, manual=False, width_threshold=40):
    """This approximates swing onset and offset from kinematic data
    Parameters
    ----------
    foot_cord: numpy.ndarray
        Numpy array of the `x` coordinate of the limb
    manual: boolean, default=`False`
        Whether or not you want to manually annotate the trace
    width_threshold: int, default=`40`
        The width cutoff used by `scipy.signal.find_peaks`

    Returns
    -------
    cycle_durations: numpy.ndarray
        A numpy array with the duration of each cycle
    """

    # Calculating swing estimations
    swing_onset, _ = swing_estimation(
        foot_cord, manual=manual, width_threshold=width_threshold
    )

    # Converting Output to time in seconds
    time_conversion = np.vectorize(frame_to_time)
    onset_timing = time_conversion(swing_onset)

    cycle_durations = np.array([])
    for i in range(len(onset_timing) - 1):
        time_diff = onset_timing[i + 1] - onset_timing[i]
        cycle_durations = np.append(cycle_durations, time_diff)

    return cycle_durations


# Custom median filter from
def median_filter(arr, k):
    """Median filter attempting to be faithful to Spike2's channel process

    Parameters
    ----------
    arr: numpy.ndarray
        input numpy array
    k:
        The size of the window you want to slide over the array.
        This is also considered the kernel

    Returns
    -------
    result:
        A numpy array of the same length where each element is the median of
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

    return np.array(result)


def spike_slope(comy, p):
    """Getting slope of curve attempting to be faithful to Spike2's channel process

    Parameters
    ----------
    comy: numpy.ndarray
        numpy array of the y coordinate of the center of mass
    p: numpy.ndarray
        How many values you want in either direction to be included

    Returns
    -------
    slope: numpy.ndarray
        An array with the slope values
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


def hip_height(toey_values, hipy_values, manual=False):
    """Approximates Hip Height

    Parameters
    ----------
    toey_values: numpy.ndarray
        numpy array of values for y coordinate of toe marker
    hipy_values: numpy.ndarray
        numpy array of values for y coordinate of hip marker
    manual: boolean, default=`False`
        Whether or not you want to manually annotate the trace

    Returns
    -------
    hip_height:
        returns hip height in centimeters (cm)
    """

    # Either manually mark regions foot is on the ground or go with proxy
    if manual is False:

        # Getting lower quartile value of toey as proxy for the ground
        toey_lowerq = np.percentile(toey_values, q=25)
        average_hip_value = np.mean(hipy_values)
        hip_height = toey_lowerq - average_hip_value

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
        hip_height = average_toe_value - average_hip_value

    else:
        print("The `manual` variable must be a boolean")

    return hip_height


def xcom(comy, vcom, hip_height):
    """
    Calculates extrapolated Center-of-mass (xCoM)

    Parameters
    ----------
    comy:
        numpy array of values for y coordinate of toe marker
    hip_height:
        average hip height

    Returns
    -------
    xcom:
        A 1-D array representing the extrapolated CoM in cm
    """

    # Get xCoM in (cm)
    xcom = comy + vcom / np.sqrt(9.81 / hip_height)

    return xcom


def double_support(fl_x, hl_x, manual_analysis=False, filt_window=40):
    """Finds double support phases from forlimb movement
    :param fl_x: forelimb x coordinate
    :param hl_x: hindlimb x coordinate
    :param manual_analysis: Will allow for manual annotation of traces to pick phases.
    :param filt_window: Smoothening factor for traces.

    :return ds_phases: Same length 1-D array as input either 1 or np.nan values indicating phases
    """

    # Smoothen Traces
    fl_x = signal.savgol_filter(fl_x, filt_window, 3)
    hl_x = signal.savgol_filter(hl_x, filt_window, 3)

    fl_swon, fl_swoff = swing_estimation(fl_x, manual=False)
    hl_swon, hl_swoff = swing_estimation(fl_x, manual=False)

    # return ds_phases


def cop(fl_y, hl_y):

    return (fl_y + hl_y) / 2


def step_width_est(
    rl_x: np.array,
    ll_x: np.array,
    rl_y: np.array,
    ll_y: np.array,
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
    rl_y_cords = rl_y
    ll_y_cords = ll_y

    _, rl_swoff = swing_estimation(rl_x)
    _, ll_swoff = swing_estimation(ll_x)

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
    # plt.plot(leftcop)
    # plt.plot(rightcop)
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
        xcom_peaks, _ = signal.find_peaks(xcom, width=width_threshold)
        xcom_troughs, _ = signal.find_peaks(-xcom, width=width_threshold)
    elif manual_peaks is True:
        xcom_peaks, _ = mos_marks(xcom, leftds, rightds, title="Select Peaks")
        xcom_troughs, _ = mos_marks(xcom, leftds, rightds, title="Select Troughs")
    else:
        print("The `manual` variable must be a boolean")

    lmos_values = np.array([])
    rmos_values = np.array([])

    lcop_points = leftcop[xcom_troughs]
    rcop_points = rightcop[xcom_peaks]

    for i in range(len(xcom_peaks)):
        # Getting window between peak values
        xcom_index = xcom_peaks[i]
        cop_point = rcop_points[i]

        # Getting non-nan values from region

        # Making sure we are actually grabbing the last meaningful region of center of pressure
        rmos = cop_point - xcom[xcom_index]
        # print(f"L COP {cop_point}")
        # print(f"xCoM {xcom[xcom_index]}")
        rmos_values = np.append(rmos_values, rmos)

    for i in range(len(xcom_troughs)):
        # Getting window between peak values
        xcom_index = xcom_troughs[i]
        cop_point = lcop_points[i]

        # Getting non-nan values from region
        lmos = xcom[xcom_index] - cop_point
        # print(f"R COP {cop_point}")
        # print(f"xCoM {xcom[xcom_index]}")
        lmos_values = np.append(lmos_values, lmos)

    return lmos_values, rmos_values, xcom_peaks, xcom_troughs


def limb_measurements(
    input_skeleton,
    skeleton_list,
    calibration,
    save_as_csv=False,
    csv_filename="./tmp-limb_measurmets.csv",
):
    """Estimates lengths of limbs coordinates based on skeleton reconstruction

    Parameters
    ----------
    input_skeleton:
        raw dataframe from spike no cleaning necessary
    skeleton_list:
        list of components in skeleton to look for in dataframe
    calibration:
        calibration factor calculated from recording itself

    Returns
    -------
    calibrated_measurements:
        Dictionary of the average distance for each joint
    """
    calibrated_measurments = {}

    input_skeleton = input_skeleton[skeleton_list[:]]
    sk_df = input_skeleton.drop([0])  # removes row saying length
    # print(sk_df)

    for key in skeleton_list:
        column_values = pd.array(sk_df[key], dtype=np.dtype(float)) / calibration
        calibrated_measurments[key] = np.mean(column_values)

    if save_as_csv is True:
        w = csv.writer(open(csv_filename, "w"))
        for key, val in calibrated_measurments.items():
            w.writerow([key, val])

    return calibrated_measurments


# TODO: Create a function for selecting a region for analyzing to feed into rest of analysis
def main():

    # NOTE: Very important this is checked before running
    video = "00"
    mouse_number = 2
    manual_analysis = False
    save_auto = False
    select_region = False
    show_plots = False

    # Settings before running initial workup from DeepLabCut
    figure_title = f"Step Cycles for level-test-M{mouse_number}-vid-{video}"
    figure_filename = f"../tests/dlctools/m{mouse_number}-{video}.pdf"
    step_cycles_filename = f"../tests/dlctools/m{mouse_number}-step-cycles-{video}.csv"

    # Some things to set for plotting/saving
    lmos_filename = f"../tests/dlctools/m{mouse_number}-lmos-{video}.csv"
    rmos_filename = f"../tests/dlctools/m{mouse_number}-rmos-{video}.csv"
    mos_figure_title = (
        f"Measurement of Stability For Level Test M{mouse_number}-{video}"
    )
    mos_figure_filename = f"../tests/dlctools/m{mouse_number}-mos-{video}.pdf"

    # Loading in a dataset
    df, bodyparts, scorer = dlck.load_data(
        f"../data/kinematics/EMG-test-1-pre-emg_0000{video}DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5"
    )

    # Loading in skeleton
    sk_df = pd.read_csv(
        f"../data/kinematics/EMG-test-1-pre-emg_0000{video}DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered_skeleton.csv"
    )
    limb_names = [
        "iliac_crest_hip",
        "hip_knee",
        "knee_ankle",
        "ankle_metatarsal",
        "metatarsal_toe",
    ]

    calib_markers = [
        "calib_1",
        "calib_2",
        "calib_3",
        "calib_4",
        "calib_5",
        "calib_6",
    ]

    calib_factor = dlc_calibrate(df, bodyparts, scorer, calib_markers)
    print(type(calib_factor))
    limb_diffs = limb_measurements(
        sk_df,
        limb_names,
        calib_factor,
        save_as_csv=True,
        csv_filename="../tests/dlctools/limb_measure-test.csv",
    )
    print(f"Length of limb coordinates in cm\n{limb_diffs}")

    # Grabbing marker data
    toex_np = mark_process(df, scorer, "toe", "x", calib_factor)
    toey_np = mark_process(df, scorer, "toe", "y", calib_factor)
    hipy_np = mark_process(df, scorer, "hip", "y", calib_factor)
    comy_np = mark_process(df, scorer, "mirror_com", "y", calib_factor)
    rfly_np = mark_process(df, scorer, "mirror_rfl", "y", calib_factor)
    rhly_np = mark_process(df, scorer, "mirror_rhl", "y", calib_factor)
    lfly_np = mark_process(df, scorer, "mirror_lfl", "y", calib_factor)
    lhly_np = mark_process(df, scorer, "mirror_lhl", "y", calib_factor)
    rflx_np = mark_process(df, scorer, "mirror_rfl", "x", calib_factor)
    rhlx_np = mark_process(df, scorer, "mirror_rhl", "x", calib_factor)
    lflx_np = mark_process(df, scorer, "mirror_lfl", "x", calib_factor)
    lhlx_np = mark_process(df, scorer, "mirror_lhl", "x", calib_factor)
    miry_np = mark_process(df, scorer, "mirror", "y", calib_factor, smooth_wind=70)

    # Selecting a Given Region
    if select_region is True:
        reg_start, reg_stop = analysis_reg_sel(mirror_y=miry_np, com_y=comy_np)
        toex_np = toex_np[reg_start:reg_stop]
        toey_np = toey_np[reg_start:reg_stop]
        hipy_np = hipy_np[reg_start:reg_stop]
        comy_np = comy_np[reg_start:reg_stop]
        rfly_np = rfly_np[reg_start:reg_stop]
        rhly_np = rhly_np[reg_start:reg_stop]
        lfly_np = lfly_np[reg_start:reg_stop]
        lhly_np = lhly_np[reg_start:reg_stop]
        rflx_np = rflx_np[reg_start:reg_stop]
        rhlx_np = rhlx_np[reg_start:reg_stop]
        lflx_np = lflx_np[reg_start:reg_stop]
        lhlx_np = lhlx_np[reg_start:reg_stop]
    else:
        print("Looking at entire recording")

    # Getting a time adjusted array of equal length for time
    time = np.arange(0, len(comy_np), 1)
    time = frame_to_time(time)

    # Center of pressures
    com_slope = spike_slope(comy_np, 30)
    hip_h = hip_height(toey_np, hipy_np)
    xcom_trimmed = xcom(comy_np, com_slope, hip_h)

    # Experimental Estimation of CoP considering the standards used
    rightcop = cop(rfly_np, rhly_np)
    rightcop = signal.savgol_filter(rightcop, 40, 3)
    leftcop = cop(lfly_np, lhly_np)
    leftcop = signal.savgol_filter(leftcop, 40, 3)
    right_DS = rightcop
    left_DS = leftcop

    # Step cycle Estimation
    toe_swing_onset, toe_swing_offset = swing_estimation(toex_np)
    step_cyc_durations = step_cycle_est(toex_np)
    rfl_swon, rfl_swoff = swing_estimation(foot_cord=rflx_np)
    rhl_swon, rhl_swoff = swing_estimation(foot_cord=rhlx_np)
    lfl_swon, lfl_swoff = swing_estimation(foot_cord=lflx_np)
    lhl_swon, lhl_swoff = swing_estimation(foot_cord=lhlx_np)

    # Some of my default plotting parameters I like
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set(
        style="white",
        font_scale=1.6,
        font="serif",
        palette="colorblind",
        rc=custom_params,
    )

    # Plot Legend
    swing_legend = [
        "Limb X cord",
        "Swing offset",
        "Swing onset",
    ]
    filtest_legend = [
        # "Original",
        # "Median",
        "xCoM",
        "CoMy",
        "L CoP",
        "R CoP",
        # "Slope",
    ]

    fig, axs = plt.subplots(2)
    fig.suptitle(figure_title)

    # Showing results for step cycle timing
    axs[0].set_title("Filter test")
    # axs[0].plot(comy_np)
    # axs[0].plot(com_med)
    axs[0].plot(time, xcom_trimmed)
    axs[0].plot(time, comy_np)
    axs[0].plot(time, leftcop)
    axs[0].plot(time, rightcop)
    # axs[0].plot(time_trimmed, com_slope)
    axs[0].legend(filtest_legend, loc="best")
    # axs[0].bar(0, np.mean(step_cyc_durations), yerr=np.std(step_cyc_durations), capsize=5)

    # For plotting figure demonstrating how swing estimation was done
    axs[1].set_title("Swing Estimation")
    axs[1].plot(toex_np)
    axs[1].plot(toe_swing_offset, toex_np[toe_swing_offset], "^")
    axs[1].plot(toe_swing_onset, toex_np[toe_swing_onset], "v")
    axs[1].legend(swing_legend, loc="best")

    # Saving Figure in same folder
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(19.8, 10.80)

    if manual_analysis is True:
        # Saving plot and results
        np.savetxt(step_cycles_filename, step_cyc_durations, delimiter=",")
        plt.savefig(figure_filename, dpi=300)
        print("Kinematic results saved")
    elif manual_analysis is False and save_auto is True:
        np.savetxt(step_cycles_filename, step_cyc_durations, delimiter=",")
        plt.savefig(figure_filename, dpi=300)
        print("Kinematic results saved")
    elif manual_analysis is False and save_auto is False and show_plots is True:
        print("Kinematic results not saved")
        plt.show()
    else:
        print("Kinematic results not saved")

    # Now onto Lateral stability

    lmos, rmos, xcom_peaks, xcom_troughs = mos(
        xcom_trimmed,
        leftcop,
        rightcop,
        left_DS,
        right_DS,
        manual_peaks=manual_analysis,
        width_threshold=50,
    )
    lmos = np.where(lmos < 0.0, np.nan, lmos)
    rmos = np.where(rmos < 0.0, np.nan, rmos)

    mos_comb = pd.DataFrame(columns=["Limb", "MoS (cm)"])
    for i in range(len(lmos)):
        condition = "Left"
        fixed_array = lmos.ravel()
        mos_entry = [[condition, fixed_array[i]]]
        mos_comb = mos_comb._append(
            pd.DataFrame(mos_entry, columns=["Limb", "MoS (cm)"]),
            ignore_index=True,
        )

    for i in range(len(rmos)):
        condition = "Right"
        fixed_array = rmos.ravel()
        mos_entry = [[condition, fixed_array[i]]]
        mos_comb = mos_comb._append(
            pd.DataFrame(mos_entry, columns=["Limb", "MoS (cm)"]),
            ignore_index=True,
        )

    xcom_legend = [
        "xCoM",
        "xCoM peaks",
        "xCoM troughs",
        "L COP",
        "R COP",
    ]
    fig, axs = plt.subplots(2)
    fig.suptitle(mos_figure_title)

    # For plotting figure demonstrating how calculation was done
    axs[0].set_title("How MoS is Derived")
    axs[0].plot(xcom_trimmed)
    axs[0].plot(xcom_peaks, xcom_trimmed[xcom_peaks], "^")
    axs[0].plot(xcom_troughs, xcom_trimmed[xcom_troughs], "v")
    axs[0].plot(leftcop)
    axs[0].plot(rightcop)
    axs[0].legend(xcom_legend, bbox_to_anchor=(1, 0.7))

    # Looking at result
    axs[1].set_title("MoS Result")
    sns.barplot(data=mos_comb, x="Limb", y="MoS (cm)", ci=95, capsize=0.05, ax=axs[1])
    fig = plt.gcf()
    fig.set_size_inches(8.27, 11.7)  # A4 formatting 8.27” by 11.7”
    fig.tight_layout()

    # Saving results
    if manual_analysis is True:
        np.savetxt(lmos_filename, lmos, delimiter=",")
        np.savetxt(rmos_filename, rmos, delimiter=",")
        plt.savefig(mos_figure_filename, dpi=300)
        print("Mos results saved!")
    elif manual_analysis is False and save_auto is True:
        np.savetxt(lmos_filename, lmos, delimiter=",")
        np.savetxt(rmos_filename, rmos, delimiter=",")
        plt.savefig(mos_figure_filename, dpi=300)
        print("Mos results saved!")
    elif manual_analysis is False and save_auto is False and show_plots is True:
        print("Mos results not saved")
        plt.show()
    else:
        print("Mos results not saved")


if __name__ == "__main__":
    main()
