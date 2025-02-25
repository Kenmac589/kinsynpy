import dlctools as dlt
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skinematics as ski
from scipy import signal


def xcom_flux(xcom, width_threshold=40):
    """
    Parameters
    ----------
    xcom: np.ndarray
        1-D array of of xCoM
    width_threshold: int, default=`40`
        Threshold for finding peaks and troughs.

    Returns
    -------
    avg_flux: np.float64
        Average value indicating how much xCoM is fluctuating

    """

    # Getting peaks and troughs
    xcom_peaks, _ = signal.find_peaks(xcom, width=width_threshold)
    xcom_troughs, _ = signal.find_peaks(-xcom, width=width_threshold)

    difs = np.array([])

    for peak, trough in zip(xcom_peaks, xcom_troughs):

        wave_dif = np.abs(xcom[peak] - xcom[trough])
        difs = np.append(difs, wave_dif)

    avg_flux = np.mean(difs)

    return avg_flux


def spike_angle_calc(x_cord, y_cord):

    if x_cord == 0 and y_cord == 0:
        angle = 999

    angle = 90  # default if on y-axis

    if x_cord != 0:
        angle = 360.0 * np.arctan(y_cord / x_cord) / (2.0 * np.pi)

    if y_cord <= 0.0 and x_cord >= 0.0:
        angle += 180.0
    elif y_cord < 0.0 and x_cord >= 0.0:
        angle += 360.0
    elif y_cord > 0.0 and x_cord < 0.0:
        angle += 180.0

    return angle


# def calc_knee(hipx, hipy, anklex, ankley):


def get_angle(ax, ay, bx, by):
    # test = ski.vector.angle
    angle = ski.vector.angle((ax, ay), (bx, by))

    return angle


def edit_test(test_phrase):
    print(test_phrase)

    return_check = f"Return plus {test_phrase}"

    return return_check


def main():

    # NOTE: Very important this is checked before running
    video = "10"
    mouse_number = 2
    manual_analysis = False
    save_auto = False
    select_region = False
    show_plots = True

    # Settings before running initial workup from DeepLabCut
    figure_title = f"Step Cycles for level-test-M{mouse_number}-vid-{video}"
    figure_filename = f"../../tests/dlctools/m{mouse_number}-{video}.pdf"
    step_cycles_filename = (
        f"../../tests/dlctools/m{mouse_number}-step-cycles-{video}.csv"
    )

    # Some things to set for plotting/saving
    lmos_filename = f"../../tests/dlctools/m{mouse_number}-lmos-{video}.csv"
    rmos_filename = f"../../tests/dlctools/m{mouse_number}-rmos-{video}.csv"
    mos_figure_title = (
        f"Measurement of Stability For Level Test M{mouse_number}-{video}"
    )
    mos_figure_filename = f"../../tests/dlctools/m{mouse_number}-mos-{video}.pdf"

    # Loading in a dataset
    df, bodyparts, scorer = dlt.load_data(
        f"../../data/kinematics/EMG-test-1-pre-emg_0000{video}DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5"
    )

    # Loading in skeleton
    sk_df = pd.read_csv(
        f"../../data/kinematics/EMG-test-1-pre-emg_0000{video}DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered_skeleton.csv"
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

    calib_factor = dlt.dlc_calibrate(df, bodyparts, scorer, calib_markers)
    # print(type(calib_factor))
    limb_diffs = dlt.limb_measurements(
        sk_df,
        limb_names,
        calib_factor,
        save_as_csv=True,
        csv_filename="../../tests/dlctools/limb_measure-test.csv",
    )
    # print(f"Length of limb coordinates in cm\n{limb_diffs}")

    # Grabbing marker data
    toex_np = dlt.mark_process(df, scorer, "toe", "x", calib_factor)
    toey_np = dlt.mark_process(df, scorer, "toe", "y", calib_factor)
    hipy_np = dlt.mark_process(df, scorer, "hip", "y", calib_factor)
    hipx_np = dlt.mark_process(df, scorer, "hip", "x", calib_factor)
    kneey_np = dlt.mark_process(df, scorer, "knee", "y", calib_factor)
    kneex_np = dlt.mark_process(df, scorer, "knee", "x", calib_factor)
    comy_np = dlt.mark_process(df, scorer, "mirror_com", "y", calib_factor)
    rfly_np = dlt.mark_process(df, scorer, "mirror_rfl", "y", calib_factor)
    rhly_np = dlt.mark_process(df, scorer, "mirror_rhl", "y", calib_factor)
    lfly_np = dlt.mark_process(df, scorer, "mirror_lfl", "y", calib_factor)
    lhly_np = dlt.mark_process(df, scorer, "mirror_lhl", "y", calib_factor)
    rflx_np = dlt.mark_process(df, scorer, "mirror_rfl", "x", calib_factor)
    rhlx_np = dlt.mark_process(df, scorer, "mirror_rhl", "x", calib_factor)
    lflx_np = dlt.mark_process(df, scorer, "mirror_lfl", "x", calib_factor)
    lhlx_np = dlt.mark_process(df, scorer, "mirror_lhl", "x", calib_factor)
    miry_np = dlt.mark_process(df, scorer, "mirror", "y", calib_factor, smooth_wind=70)

    # Selecting a Given Region
    if select_region is True:
        reg_start, reg_stop = dlt.analysis_reg_sel(mirror_y=miry_np, com_y=comy_np)
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
    time = dlt.frame_to_time(time)

    angle_test = get_angle(ax=hipx_np, ay=hipy_np, bx=kneex_np, by=kneey_np)

    print(f"Angle test\n{angle_test}")

    # Center of pressures
    com_slope = dlt.spike_slope(comy_np, 30)
    hip_h = dlt.hip_height(toey_np, hipy_np)
    xcom_trimmed = dlt.xcom(comy_np, com_slope, hip_h)

    dlt.mos
    # Experimental Estimation of CoP considering the standards used
    rightcop = dlt.cop(
        rflx_np, rhlx_np, rfly_np, rhly_np, calc_ds=False, manual_analysis=False
    )
    rightcop = signal.savgol_filter(rightcop, 40, 3)
    leftcop = dlt.cop(
        lflx_np, lhlx_np, lfly_np, lhly_np, calc_ds=False, manual_analysis=False
    )
    leftcop = signal.savgol_filter(leftcop, 40, 3)
    right_DS = rightcop
    left_DS = leftcop

    # Step cycle Estimation
    toe_swing_onset, toe_swing_offset = dlt.swing_estimation(toex_np)
    step_cyc_durations = dlt.step_cycle_est(toex_np)
    rfl_swon, rfl_swoff = dlt.swing_estimation(foot_cord=rflx_np)
    rhl_swon, rhl_swoff = dlt.swing_estimation(foot_cord=rhlx_np)
    lfl_swon, lfl_swoff = dlt.swing_estimation(foot_cord=lflx_np)
    lhl_swon, lhl_swoff = dlt.swing_estimation(foot_cord=lhlx_np)
    fl_step = dlt.step_width_est(rl_x=rflx_np, ll_x=lflx_np, rl_y=rfly_np, ll_y=lfly_np)
    hl_step = dlt.step_width_est(rl_x=rhlx_np, ll_x=lhlx_np, rl_y=rhly_np, ll_y=lhly_np)

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

    lmos, rmos, xcom_peaks, xcom_troughs = dlt.mos(
        xcom_trimmed,
        leftcop,
        rightcop,
        manual_peaks=manual_analysis,
        width_threshold=50,
    )
    lmos = np.where(lmos < 0.0, np.nan, lmos)
    rmos = np.where(rmos < 0.0, np.nan, rmos)

    lmos = dlt.stepw_mos_corr(fl_stepw=fl_step, hl_stepw=hl_step, mos_values=lmos)
    rmos = dlt.stepw_mos_corr(fl_stepw=fl_step, hl_stepw=hl_step, mos_values=rmos)

    avg_flux = xcom_flux(xcom=xcom_trimmed)
    # print(avg_flux)

    # print(f"L MoS unaltered {lmos}\n")
    # print(f"L MoS adjusted by step width {lmos_corr_test}\n")
    #
    # print(f"R MoS unaltered {rmos}\n")
    # print(f"R MoS adjusted by step width {rmos_corr_test}\n")

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
    sns.barplot(
        data=mos_comb,
        x="Limb",
        y="MoS (cm)",
        errorbar=("ci", 95),
        capsize=0.05,
        ax=axs[1],
    )
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
