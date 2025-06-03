import os
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from sklearn.decomposition import NMF

import kinsynpy.dlctools as dlt


def folder_selector():

    root = Tk()
    root.withdraw()
    analysis_path = filedialog.askdirectory()

    return analysis_path


def nnmf_factorize(A, k):
    """Non-Negative Matrix Factorization for Muscle Synergy Extraction
    @param A: input matrix
    @param k: number of components (muscle channels)

    @return W: motor primitives
    @return H: motor modules
    @return C: factorized matrix
    """
    nmf = NMF(n_components=k, init="random", random_state=0)
    W = nmf.fit_transform(A)
    H = nmf.components_
    C = np.dot(W, H)
    return W, H, C


def synergy_extraction(data_input, synergy_selection):
    """Synergy Extraction from factorized matricies

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


def get_video_info(video_path, rec_name):

    # Ascertain length of recordings from videos
    video_lengths = {}

    # If video path is not valid
    if not os.path.isdir(video_path):
        print(f"{video_path} is not a valid directory.")
        return

    for filename in os.listdir(video_path):
        #
        if filename.endswith(".avi"):
            file_path = os.path.join(video_path, filename)
            video_name = Path(file_path).stem
            video_number = video_name.replace(f"{rec_name}_0000", "")
            video_number = int(video_number)

            # Creating Video Capture object
            video = cv2.VideoCapture(file_path)

            # Count number of frames
            frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = video.get(cv2.CAP_PROP_FPS)

            # calculate duration of the video
            seconds = frames / fps
            video_lengths.update({video_number: seconds})

            # print(f"Video {video_number} is {seconds} long")

    return video_lengths


def seg_raw_emg(raw_emg, emg_ch, sync, video_path, rec_name):
    """Splices out individual recordings EMG data"""

    # Preprocessing to ensure dataframe is in correct format
    emg_ch.append(sync)
    raw_emg = raw_emg.set_index("Time")
    raw_emg = raw_emg.loc[:, emg_ch]
    sync_occurences = raw_emg.loc[raw_emg[sync] == 1].index.tolist()

    # Ascertain length of recordings from videos
    video_lengths = {}

    # If video path is not valid
    if not os.path.isdir(video_path):
        print(f"{video_path} is not a valid directory.")
        return

    for filename in os.listdir(video_path):
        #
        if filename.endswith(".avi"):
            file_path = os.path.join(video_path, filename)
            video_name = Path(file_path).stem
            video_number = video_name.replace(f"{rec_name}_0000", "")
            video_number = int(video_number)

            # Creating Video Capture object
            video = cv2.VideoCapture(file_path)

            # Count number of frames
            frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = video.get(cv2.CAP_PROP_FPS)

            # calculate duration of the video
            seconds = frames / fps
            video_lengths.update({video_number: seconds})

            # print(f"Video {video_number} is {seconds} long")

    for i in range(len(video_lengths.keys())):
        # Getting timing for given recording
        recording_end = round(sync_occurences[i], 3)
        recording_begin = round(recording_end - video_lengths[i], 3)
        print(f"Video {i} begins at {recording_begin} and ends at {recording_end}")
        # Selecting region from raw trace and saving as csv
        seg_recording = raw_emg[recording_begin:recording_end]
        seg_recording.to_csv(f"{video_path}/{video_name}-{i}-emg.csv")

    segmented_recording = sync_occurences

    return segmented_recording


def show_synergies(
    data_input,
    chosen_synergies,
    channel_order=["GM", "Ip", "BF", "VL", "St", "TA", "Gs", "Gr"],
    synergies_name="./output.png",
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
        smooth_sample = signal.savgol_filter(samples[samples_binned], 30, 3)
        axs[col, 1].plot(
            smooth_sample, primitive_trace, color="red", label="Average Trace"
        )
        axs[col, 1].set_title("Synergy {}".format(col + 1))

        # Iterate over the bins again to plot the individual bin data
        for i in range(number_cycles):
            # Get the data for the current bin in the current 0, column
            time_point_average = motor_primitives[
                i * trace_length : (i + 1) * trace_length, col
            ]

            smooth_sample = signal.savgol_filter(samples[samples_binned], 30, 3)
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


def batch_step_cycle(rec_path):

    # If video path is not valid
    if not os.path.isdir(rec_path):
        print(f"{rec_path} is not a valid directory.")
        return

    for filename in os.listdir(rec_path):

        if filename.endswith(".h5"):
            file_path = os.path.join(rec_path, filename)
            df, bodyparts, scorer = dlt.load_data(file_path)


def main():

    # test = si.read_spike2(
    #     "../../data/emg/1yrDTRnoRosa-M1-preDTX-19102023-1.smr",
    #     stream_id="2",
    #     use_names_as_ids=True,
    # )
    recording_name = "1yrDTRnoRosa-M1-19102023"

    # NOTE: Still working on understanding spikeinterface to get native recordings in here
    # For the mean time I usually just export at 1000 Hz and go from there.

    # For file segmentation
    raw_file = pd.read_csv("../../data/emg/12mo-DTR-1-pre-dtx-full.txt", header=0)
    sync_channel = "12 Sync"

    rec_path = "../../data/videos/"

    emg_ch_order = [
        "4 GM",
        "5 Ip",
        "6 BF",
        "7 VL",
        "8 St",
        "9 TA",
        "10 Gs",
        "11 Gr",
    ]

    recording_number = 0
    df, bodyparts, scorer = dlt.load_data(
        "../../data/videos/1yrDTRnoRosa-M1-19102023_000000DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000.h5"
    )
    test_input_file = pd.read_csv(
        f"../../data/videos/1yrDTRnoRosa-M1-19102023_000020-{recording_number}-emg.csv"
    )
    ip_channel = test_input_file["5 Ip"].to_numpy(dtype=float)
    print(f"Ip: {len(ip_channel)}")

    # Only Run if video not already segmented
    # processed_emg_file = seg_raw_emg(
    #     raw_emg=raw_file,
    #     emg_ch=emg_ch_order,
    #     sync=sync_channel,
    #     video_path=rec_path,
    #     rec_name=recording_name,
    # )
    calib_factor = dlt.dlc_calibrate(df, bodyparts, scorer=scorer)
    toex_np = dlt.mark_process(df, scorer, "toe", "x", calib_factor)
    print(f"Toe: {len(toex_np)}")

    toe_swon, toe_swoff = dlt.swing_estimation(toex_np, width_threshold=40)

    time = np.arange(0, len(toex_np), 1)
    time = dlt.frame_to_time(time)

    print(time[toe_swon])
    swon_timings = time[toe_swon]
    swoff_timings = time[toe_swoff]

    np.savetxt(
        f"../../data/videos/step_cycles/{recording_name}-{recording_number}-swon.csv",
        swon_timings,
        delimiter=",",
    )
    np.savetxt(
        f"../../data/videos/step_cycles/{recording_name}-{recording_number}-swoff.csv",
        swoff_timings,
        delimiter=",",
    )

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set(
        style="white",
        font_scale=1.6,
        font="serif",
        palette="colorblind",
        rc=custom_params,
    )

    plt.title("Swing Estimation")
    plt.plot(toex_np, label="Toe X")
    plt.plot(toe_swoff, toex_np[toe_swoff], "^", label="Swing Offset")
    plt.plot(toe_swon, toex_np[toe_swon], "v", label="Swing Onset")
    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    main()
# %%
