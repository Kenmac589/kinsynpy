import os
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np
import pandas as pd


def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Output directories creaeted at {path}")

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

