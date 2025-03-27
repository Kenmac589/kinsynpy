import os
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import matplotlib.pyplot as plt
import motorpyrimitives as mp
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal

import kinsynpy.dlctools as dlt


# import spikeinterface_gui.full as si
def folder_selector():

    root = Tk()
    root.withdraw()
    analysis_path = filedialog.askdirectory()

    return analysis_path


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


def batch_step_cycle(rec_path):

    # If video path is not valid
    if not os.path.isdir(rec_path):
        print(f"{rec_path} is not a valid directory.")
        return

    for filename in os.listdir(rec_path):

        if filename.endswith(".h5"):
            file_path = os.path.join(rec_path, filename)
            df, bodyparts, scorer = dlt.load_data(file_path)


def norm_channel(channel, toex_np):

    # chan_np =
    print("testing")


def data_preproc(input_dataframe):

    print(input_dataframe)


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
