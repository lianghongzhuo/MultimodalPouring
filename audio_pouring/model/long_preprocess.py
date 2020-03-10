#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name     : data_preprocess.py
# Purpose       :
# Creation Date : 03-01-2019
# Author        : Shuang Li [sli[at]informatik[dot]uni-hamburg[dot]com]
# Author        : Hongzhuo Liang [liang[at]informatik[dot]uni-hamburg[dot]com]
from __future__ import division
from __future__ import print_function
import numpy as np
import glob
import os
import sys
import pickle
import math
from scipy.signal import butter, filtfilt
from scipy import interpolate
import multiprocessing as mp
from audio_pouring.utils.utils import weight2height, config, pkg_path, vis_audio
import h5py
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
try:
    import librosa.util
    import librosa.display
    import seaborn as sns
    from matplotlib import pylab
    import matplotlib.pyplot as plt
except ImportError:
    librosa = None
    sns = None
    pylab = None
    plt = None
target_frequency = config["target_frequency"]
source_frequency = config["source_frequency"]
n_fft = config["n_fft"]
overlap = config["overlap"]
win_length = int(config["win_size"] * target_frequency)  # win length, how many samples in this window
assert win_length >= n_fft, "win_length should larger equal than n_fft"
hop_length = int(win_length * overlap)
with h5py.File(os.path.join(pkg_path, "h5py_dataset/ego_noise.h5"), "r") as noise_npy:
    noise_train = np.array(noise_npy["npy_noise_train"]["numpy_data"])
    noise_test = np.array(noise_npy["npy_noise_test"]["numpy_data"])


class Interp1dPickleAble:
    """ class wrapper for piecewise linear function
    """

    def __init__(self, xi, yi, **kwargs):
        self.xi = xi
        self.yi = yi
        self.args = kwargs
        self.f = interpolate.interp1d(xi, yi, **kwargs)

    def __call__(self, xnew):
        return self.f(xnew)

    def __getstate__(self):
        return self.xi, self.yi, self.args

    def __setstate__(self, state):
        self.f = interpolate.interp1d(state[0], state[1], **state[2])


def main(pickle_file_list, h5py_file_name, use_multi_threading_):
    pickle_num = len(pickle_file_list)
    h5py_handle = h5py.File(h5py_file_name, "w")
    if use_multi_threading_:
        num_workers = int(mp.cpu_count() * 2)

        def task(task_inter_, i_, pickle_files_, data_to_save_mp_list_):
            i_ = i_ + task_inter_ * num_workers
            pickle_file_ = pickle_files_[i_]
            if sys.version_info >= (3, 0):
                data_ = pickle.load(open(pickle_file_, "rb"), encoding="latin1")
            else:
                data_ = pickle.load(open(pickle_file_, "rb"))
            npy_generate(data_, pickle_file_, data_to_save_mp_list_)

        num_inter = int(math.ceil(pickle_num / num_workers))
        last_tasks_num = pickle_num % num_workers
        for task_round_inter_ in range(num_inter):
            if task_round_inter_ == num_inter - 1 and last_tasks_num > 0:
                n = last_tasks_num
            else:
                n = num_workers
            data_to_save_mp_list = mp.Manager().list()
            workers = [mp.Process(target=task,
                                  args=(task_round_inter_, i, pickle_file_list,
                                        data_to_save_mp_list)) for i in range(n)]
            [i.start() for i in workers]
            [i.join() for i in workers]
            data_to_save_mp_list = list(data_to_save_mp_list)
            generate_h5_dataset(data_to_save_mp_list, h5py_handle)
    else:
        for i in range(pickle_num):
            if sys.version_info >= (3, 0):
                with (open(pickle_file_list[i], "rb")) as openfile:
                    data = pickle.load(openfile, encoding="latin1")
            else:
                with (open(pickle_file_list[i], "rb")) as openfile:
                    data = pickle.load(openfile)
            npy_generate(data, pickle_file_list[i], mp_=False)
    h5py_handle.close()


def check_audio_time(audio_raw, audio_time, orig_sr_, pickle_name):
    time_audio_ros = audio_time[-1] - audio_time[0]
    time_audio = audio_raw.shape[0] / orig_sr_
    tolerance_audio_total_time = 0.05
    tolerance_audio_message_time = 0.3
    audio_diff = audio_time[1:] - audio_time[:-1]
    if np.where(audio_diff > tolerance_audio_message_time)[0].shape[0] > 0:
        print("[ERROR] audio message at {} has abnormal message time gaps".format(pickle_name))
        return False
    elif abs(time_audio - time_audio_ros) > tolerance_audio_total_time:
        print("[ERROR] audio message at {} is not match time audio ros time".format(pickle_name))
        return False
    else:
        return True


def generate_h5_dataset(input_list, h5_handle):
    arr_name = ["audio", "force", "cavity_h", "scale", "audio_rms", "wrench_first"]
    print("save {} samples into h5 file".format(len(input_list)))
    for i in input_list:
        str_name = i[0]
        creat_group = h5_handle.create_group(str_name)
        for j in range(len(arr_name)):
            creat_group.create_dataset(arr_name[j], data=i[j + 1])


def npy_generate(data, pickle_file, data_to_save_mp_list=None, mp_=True):
    # get the clip num based on the real audio time
    pickle_name = pickle_file.split("/")[-1]
    audio_raw = data["audio"]
    frame_size = data["frame_size"]
    audio_time = data["audio_time"]
    audio_time_start = audio_time[0]
    audio_time_end = audio_time[-1] - 1  # cut last 1 second data, as force is not clean
    scale_time = data["time"]
    wrench = data["wrench"]
    wrench_time = data["wrench_time"]
    f_scale = data["f_scale"]
    assert source_frequency == data["sample_frequency"]  # check whether the source sr is correct
    audio_time_length = audio_time_end - audio_time_start
    if audio_time_length < fixed_length:
        print("[ERROR] audio time at {} is less than {} seconds".format(pickle_file, fixed_length))
        return  # if the audio time is less than 4s, return directly
    if not check_audio_time(audio_raw, audio_time, source_frequency, pickle_name):
        return  # if the audio is not correct, return directly

    audio_for_rms = librosa.core.resample(audio_raw, orig_sr=source_frequency, target_sr=target_frequency)
    audio_rms = np.sqrt(np.sum(np.power(audio_for_rms, 2)) / len(audio_for_rms))

    # based on the audio length generate augmentation numbers
    aug_numbers = int(math.ceil((audio_time_length - fixed_length) * 2) / 2 * 10)
    # Floating point precision
    fp = int(str(int(audio_time[0]))[:-2]) * math.pow(10, 2)
    audio_time = audio_time - fp
    ros_time = np.unique(audio_time)
    ros_time_delta = (ros_time[1:] - ros_time[:-1]).reshape(-1, 1)
    ind_good = np.where(ros_time_delta > 1e-3)[0]  # some messages have close time stamp

    if ind_good[0] != 0:
        ind_good = np.hstack([0, ind_good])
    if ind_good[-1] != len(ros_time_delta):
        ind_good = np.hstack([ind_good, len(ros_time_delta)])

    ros_real_time = np.array([])
    for i in range(1, len(ind_good)):
        tmp_ind_diff = ind_good[i] - ind_good[i - 1]
        tmp_ranges = np.array(range(frame_size * tmp_ind_diff - 1, -1, -1))
        tmp_time_diff = ros_time[ind_good[i]] - ros_time[ind_good[i - 1]]
        tmp = ros_time[ind_good[i]] - tmp_ranges * tmp_time_diff / tmp_ind_diff / frame_size
        ros_real_time = np.hstack([ros_real_time, tmp])
    if vis_ros_real_time:
        sns.set(palette="deep", color_codes=True)
        with sns.axes_style("darkgrid"):
            plt.title("real-time per audio data", fontsize=20)
            plt.plot(ros_real_time, "r.", label="estimated time points")
            plt.plot(audio_time[frame_size:], "b-", label="ROS time stamp")
            plt.plot(np.array(range(len(ros_time))) * frame_size, ros_time, "*", label="message")
            plt.plot(ind_good * frame_size, ros_time[ind_good], "g.", label="connection points")
            plt.legend(loc=2)
            plt.show()

    force_data_all, wrench_time = wrench_process(wrench, wrench_time, audio_time_start, audio_time_end)
    wrench_first = wrench[0: 6]
    wrench_real_time = wrench_time - fp
    if same_length_mode:
        # start_max = int(max(del_length * orig_sr, int(orig_sr*(audio_time_length-fixed_length)) - 1))
        start_max = int(source_frequency * (audio_time_length - fixed_length)) - 1
        if start_max < aug_numbers:
            aug_numbers = start_max + 1
        start_index = np.random.choice(range(0, start_max + 1), aug_numbers, replace=False)
        end_index = start_index + fixed_length * source_frequency
    else:
        start_index = np.array([0])
        end_index = None

    for ind in range(len(start_index)):
        s_index = start_index[ind]
        e_index = end_index[ind]
        cavity_h = np.array([])
        whole_scale = np.array([])

        audio_raw_clip = audio_raw[s_index: e_index]
        audio_raw_clip = librosa.core.resample(audio_raw_clip, orig_sr=source_frequency, target_sr=target_frequency)
        audio_spec = audio_process(audio=audio_raw_clip, snr_db=SNR_DB, is_train=IS_TRAIN, audio_rms=audio_rms,
                                   add_random_noise=False, noise_reduction=False, fig_name="Audio Process")
        audio_spec_shape_1 = audio_spec.shape[1]
        assert audio_spec.shape[0] == 257
        assert audio_spec_shape_1 == 251  # this number is the dim[1] of stft audio matrix, means 251 small audio clips
        # it can be calculated by using the output of audio_process as above comment code
        # get force from the audio s_index and e_index
        index1 = np.where(ros_real_time[s_index] <= wrench_real_time)[0]
        index2 = np.where(ros_real_time[e_index] >= wrench_real_time)[0]
        index = np.intersect1d(index1, index2)
        force_number = int((audio_spec_shape_1 + 1) * (config["force_samples_to_collect"] * overlap))
        # (251 + 1) * (8 * 0.5)
        index = np.random.choice(index, force_number, replace=False)
        index.sort()
        whole_force = force_data_all[index]
        whole_force = whole_force.reshape(-1)
        for i in range(audio_spec_shape_1):
            current_scale_time = (win_length / target_frequency * overlap) * (i + 1) + ros_real_time[s_index]
            # get corresponding weight and cavity height
            current_scale = f_scale(current_scale_time - (scale_time[0] - fp))
            if current_scale < 0:
                if abs(current_scale) < 0.001:
                    current_scale = 0
                else:
                    print("there is a big mismatch in ros time, got {}, set to 0".format(current_scale))
                    print("please check your data if this message is too often")
                    current_scale = 0
            cur_cavity_h = weight2height(int(pickle_name[0]), current_scale)
            cavity_h = np.hstack([cavity_h, cur_cavity_h])
            whole_scale = np.hstack([whole_scale, current_scale])

        if vis_cut_force:
            whole_force_time = wrench_time[index]
            vis_wrench(force_data=whole_force.reshape(-1, 6), wrench_time=np.array(range(1008)))
            vis_wrench(force_data=whole_force.reshape(-1, 6), wrench_time=whole_force_time)

        # relative scale
        whole_scale = whole_scale * 1000

        if vis_scale:
            sns.set(palette="deep", color_codes=True)
            with sns.axes_style("darkgrid"):
                fig = plt.figure()
                fig.set_size_inches(6, 6)
                y = data["scale"] * 1000
                s_time = data["time"]
                x = s_time - s_time[0]
                x_new = np.linspace(x[0], x[-1], 50)
                # plt.title("Scale reading interpolate curve", fontsize=16)
                plt.plot(x, y, "ro", label="raw scale")
                plt.plot(x_new, f_scale(x_new) * 1000, "b-", label="interpolated scale")

                # below two lines are for showing the cut scale reading, comment if needed.
                # start_time_clip = ros_real_time[s_index] - ros_real_time[0]
                # end_time_clip = ros_real_time[e_index] - ros_real_time[0]
                # whole_time = np.array(range(0, 251)) / 250 * (end_time_clip - start_time_clip) + start_time_clip
                # plt.plot(whole_time, whole_scale, "g-", linewidth=4, label="current weight")
                # plt.plot(whole_time, cavity_h, "b--", label="current air column height")

                # if plot cavity and scale together, need to command xlim line
                # plt.xlim([x[0] - 1, x[-1] + 1])
                plt.xlabel("Time (s)", fontsize=16)
                plt.ylabel("Weight (g)", fontsize=16)
                plt.grid(True)
                legend_size = 18
                plt.legend(loc=2, prop={"size": legend_size})
                plt.savefig("/tmp/scale_reading.pdf")
                plt.show()

        name = pickle_name[:-7] + "-" + str(ind)
        data_to_save = [name, audio_spec, whole_force, cavity_h, whole_scale, audio_rms, wrench_first]
        if mp_:
            data_to_save_mp_list.append(data_to_save)
        else:
            print("Visualizing {} and save figure to /tmp ...".format(name))


def audio_filter(audio_raw, filter_type, b, fl, fh=None):
    nn = int(np.ceil((4 / b)))
    if not nn % 2:  # Make sure that N is odd.
        nn += 1
    n = np.arange(nn)

    def cal_sinc_fun(f):
        sinc_func = np.sinc(2 * f * (n - (nn - 1) / 2.))
        sinc_func *= np.blackman(nn)
        sinc_func = sinc_func / np.sum(sinc_func)
        return sinc_func

    sinc_func_high = cal_sinc_fun(fl)
    sinc_func_high = 0 - sinc_func_high
    sinc_func_high[int((nn - 1) / 2)] += 1
    if filter_type == "band":  # b = 0.05, fl = 0.005, fh = 0.8
        # low-pass filter
        sinc_func_low = cal_sinc_fun(fh)
        # high-pass filter
        h = np.convolve(sinc_func_low, sinc_func_high)
        s = list(audio_raw)
        new_signal = np.convolve(s, h)
    elif filter_type == "high":  # b = 0.01, fl = 0.01
        s = list(audio_raw)
        new_signal = np.convolve(s, sinc_func_high)
    else:
        raise NotImplementedError
    return new_signal


def mix_noise(audio, noise, noise_factor=0.4):
    if noise_factor != 0.0:
        noise_random_start = np.random.randint(0, len(noise) - len(audio))
        noise_random = noise[noise_random_start: len(audio) + noise_random_start]
        audio = audio * (1 - noise_factor) + noise_random * noise_factor
    return audio


def mix_noise_snr(audio, noise, is_train, audio_rms, snr_db=10):
    # SNR(dB) = 10 log10( ||s||² / ||noise_factor * n||² )
    # https://musicinformationretrieval.com/energy.html
    # https://www.hackaudio.com/digital-signal-processing/amplitude/rms-amplitude/
    # https://www.youtube.com/watch?v=MSKYeWfsNO0
    if snr_db == 1000.0:
        return audio
    noise_random_start = np.random.randint(0, len(noise) - len(audio))
    noise_random = noise[noise_random_start: len(audio) + noise_random_start]
    if is_train:
        noise_rms = config["noise_rms_train"]
    else:
        noise_rms = config["noise_rms_test"]
    noise_factor = audio_rms / noise_rms * np.power(10, (snr_db / -20))
    audio = audio + noise_random * noise_factor
    output_file = False
    if output_file:
        librosa.output.write_wav("test_{}.wav".format(snr_db), audio, sr=16000)
    return audio


def audio_process(audio, snr_db, is_train, audio_rms, add_random_noise=False, noise_reduction=False,
                  fig_name="Audio Process"):
    if snr_db == -1000:
        snr_list = [0, 5, 10, 15, 20, 1000]
        snr_db = np.random.choice(snr_list, 1)
    if is_train:
        audio_mix = mix_noise_snr(audio, noise_train, is_train, audio_rms, snr_db)
    else:
        audio_mix = mix_noise_snr(audio, noise_test, is_train, audio_rms, snr_db)
    if add_random_noise:
        wn = np.random.randn(len(audio_mix))
        dampening_factor = 0.02
        audio_mix = audio_mix + dampening_factor * wn
    if noise_reduction:
        audio_mix = audio_filter(audio_mix, filter_type="band", b=0.05, fl=0.005, fh=0.8)
        audio_mix = audio_filter(audio_mix, filter_type="high", b=0.01, fl=0.01, fh=None)
    audio_re_fft = librosa.stft(y=audio_mix, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    audio_re_fft = np.abs(audio_re_fft)
    audio_re_db = librosa.core.amplitude_to_db(audio_re_fft)

    if if_vis_audio:
        audio_fft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        audio_fft = np.abs(audio_fft)
        audio_db = librosa.core.amplitude_to_db(audio_fft)

        fig = pylab.gcf()
        fig.canvas.set_window_title(fig_name)
        fig.set_size_inches(8., 3)
        plt.subplot(1, 2, 1)
        vis_audio(spec=audio_db, sr=target_frequency, hop_length=hop_length, title="Original Spectrogram")
        plt.subplot(1, 2, 2)
        vis_audio(spec=audio_re_db, sr=target_frequency, hop_length=hop_length, title="Resampled Spectrogram")
        plt.yticks(np.arange(0, 8000.01, 2000), fontsize=12)
        plt.xticks(fontsize=12)
        plt.show()
    return audio_re_db


def butter_low_pass_filter(data, cutoff, fs, order):
    """
    :param data   : input data
    :param cutoff : desired cutoff frequency of the filter, Hz
    :param fs     : sample rate, Hz
    :param order  :
    :return       : filtered data
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    [b, a] = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    return y


def vis_wrench(force_data, wrench_time, raw_force_data=None, lw=1, save_fig=False):
    time = wrench_time - wrench_time[0]
    clrs = sns.color_palette("Paired", 12)
    legend_size = 8
    with sns.axes_style("darkgrid"):
        plt.rcParams.update({"font.size": 6})
        if len(force_data) != 6:
            force_data = force_data.T
        plt.figure(1)
        plt.subplot(2, 3, 1)
        if raw_force_data is not None:
            if len(raw_force_data) != 6:
                raw_force_data = raw_force_data.T
            plt.plot(time, raw_force_data[0], color=clrs[0], label="force x raw")
        plt.plot(time, force_data[0], color=clrs[1], linewidth=lw, label="force x filtered")
        plt.legend(loc=4, prop={"size": legend_size})
        plt.subplot(2, 3, 2)
        if raw_force_data is not None:
            plt.plot(time, raw_force_data[1], color=clrs[2], label="force y raw")
        plt.plot(time, force_data[1], color=clrs[3], linewidth=lw, label="force y filtered")
        plt.legend(loc=0, prop={"size": legend_size})
        plt.subplot(2, 3, 3)
        if raw_force_data is not None:
            plt.plot(time, raw_force_data[2], color=clrs[4], label="force z raw")
        plt.plot(time, force_data[2], color=clrs[5], linewidth=lw, label="force z filtered")
        plt.legend(loc=1, prop={"size": legend_size})
        plt.subplot(2, 3, 4)
        if raw_force_data is not None:
            plt.plot(time, raw_force_data[3], color=clrs[6], label="torque x raw")
        plt.plot(time, force_data[3], color=clrs[7], linewidth=lw, label="torque x filtered")
        plt.legend(loc=1, prop={"size": legend_size})
        plt.subplot(2, 3, 5)
        if raw_force_data is not None:
            plt.plot(time, raw_force_data[4], color=clrs[8], label="torque y raw")
        plt.plot(time, force_data[4], color=clrs[9], linewidth=lw, label="torque y filtered")
        plt.legend(loc=0, prop={"size": legend_size})
        plt.subplot(2, 3, 6)
        if raw_force_data is not None:
            plt.plot(time, raw_force_data[5], color=clrs[10], label="torque z raw")
        plt.plot(time, force_data[5], color=clrs[11], linewidth=lw, label="torque z filtered")
        delta = force_data[5].max() - force_data[5].min()
        y_label = np.array(range(7)) * delta / 6 + force_data[5].min()
        y_label = np.round(y_label, 2)
        plt.yticks(y_label)
        plt.legend(loc=4, prop={"size": legend_size})
        if save_fig:
            plt.savefig("/tmp/audio_pouring_force_torque_data.pdf")
        plt.show()


def vis_wrench_no_label(force_data, wrench_time, raw_force_data=None, lw=1, save_fig=False):
    time = wrench_time - wrench_time[0]
    clrs = sns.color_palette("Paired", 12)
    with sns.axes_style("darkgrid"):
        fig = pylab.gcf()
        fig.set_size_inches(40, 6)
        plt.rcParams.update({"font.size": 6})
        if len(force_data) != 6:
            force_data = force_data.T
        plt.figure(1)
        plt.subplot(1, 6, 1)
        if raw_force_data is not None:
            if len(raw_force_data) != 6:
                raw_force_data = raw_force_data.T
            plt.plot(time, raw_force_data[0], color=clrs[0])
        plt.plot(time, force_data[0], color=clrs[1], linewidth=lw)
        plt.yticks([])
        plt.xticks([])
        plt.subplot(1, 6, 2)
        if raw_force_data is not None:
            plt.plot(time, raw_force_data[1], color=clrs[2])
        plt.plot(time, force_data[1], color=clrs[3], linewidth=lw)
        plt.yticks([])
        plt.xticks([])
        plt.subplot(1, 6, 3)
        if raw_force_data is not None:
            plt.plot(time, raw_force_data[2], color=clrs[4])
        plt.plot(time, force_data[2], color=clrs[5], linewidth=lw)
        plt.yticks([])
        plt.xticks([])
        plt.subplot(1, 6, 4)
        if raw_force_data is not None:
            plt.plot(time, raw_force_data[3], color=clrs[6])
        plt.plot(time, force_data[3], color=clrs[7], linewidth=lw)
        plt.yticks([])
        plt.xticks([])
        plt.subplot(1, 6, 5)
        if raw_force_data is not None:
            plt.plot(time, raw_force_data[4], color=clrs[8])
        plt.plot(time, force_data[4], color=clrs[9], linewidth=lw)
        plt.yticks([])
        plt.xticks([])
        plt.subplot(1, 6, 6)
        if raw_force_data is not None:
            plt.plot(time, raw_force_data[5], color=clrs[10])
        plt.plot(time, force_data[5], color=clrs[11], linewidth=lw)
        plt.yticks([])
        plt.xticks([])
        if save_fig:
            plt.savefig("/tmp/audio_pouring_force_torque_data_no_label.pdf")
        plt.show()


def wrench_process(wrench, wrench_time, audio_time_start, audio_time_end):
    index1 = np.where(wrench_time > audio_time_start)[0]
    index2 = np.where(wrench_time < audio_time_end)[0]
    index = np.intersect1d(index1, index2)
    wrench_ = wrench[index].reshape(-1, 6)
    wrench_time = wrench_time[index].reshape(-1, 6)[:, 0]

    if vis_force:
        force_data = [wrench_[:, 0], wrench_[:, 1], wrench_[:, 2], wrench_[:, 3], wrench_[:, 4], wrench_[:, 5]]
        for i, f in enumerate(force_data):
            force_data[i] = butter_low_pass_filter(f, cutoff=2, fs=400, order=2)
        force_data_numpy = np.array(force_data).T
        vis_wrench(force_data_numpy, wrench_time, wrench_, save_fig=True)
        vis_wrench_no_label(force_data_numpy, wrench_time, wrench_, save_fig=True)

    vis_magnitude = False
    if vis_magnitude:
        if is_use_ur5_ft_sensor:
            x_bias = config["ur5_ft_bias_x"]
            y_bias = config["ur5_ft_bias_y"]
            z_bias = config["ur5_ft_bias_z"]
            print("currently, we use ati force / torque sensor on ur5")
            exit()
        else:
            x_bias = config["ati_ft_bias_x"]
            y_bias = config["ati_ft_bias_y"]
            z_bias = config["ati_ft_bias_z"]
        f_magnitude = np.sqrt((wrench_[:, 0] - x_bias) ** 2 + (wrench_[:, 1] - y_bias) ** 2 +
                              (wrench_[:, 2] - z_bias) ** 2)
        plt.plot(wrench_time - wrench_time[0], f_magnitude, label="force magnitude")
        plt.xlabel("time (seconds)", fontsize=20)
        plt.ylabel("force (N)", fontsize=20)
        plt.show()
        f_magnitude = np.sqrt((wrench_[:, 3]) ** 2 + (wrench_[:, 4]) ** 2 + (wrench_[:, 5]) ** 2)
        plt.plot(wrench_time - wrench_time[0], f_magnitude, label="torque magnitude")
        plt.xlabel("time (seconds)", fontsize=20)
        plt.ylabel("torque (N)", fontsize=20)
        plt.show()
    return wrench_, wrench_time


if __name__ == "__main__":
    fixed_length = config["fixed_audio_length"]  # unit seconds
    del_length = config["del_length"]  # unit seconds
    print("[info] hop_length = {}, win_length = {}".format(hop_length, win_length))
    same_length_mode = True  # random sample a fixed length of sample
    is_use_ur5_ft_sensor = False  # force sensor source is from ur5 robot or use ati ft sensor
    np.random.seed(1)
    # mix human voice
    usage = "usage: python long_preprocess.py [train/test] [mp/sp] [bottle_type: 0/with_plug/robot_pouring] " \
            "[SNR_DB value]"
    if len(sys.argv) != 5:
        exit(usage)
    bottle_type = sys.argv[3]
    SNR_DB = float(sys.argv[4])
    if sys.argv[2] == "mp":
        use_multi_threading = True
        # vis
        vis_force = False
        vis_cut_force = False
        vis_ros_real_time = False
        vis_scale = False
        if_vis_audio = False
    elif sys.argv[2] == "sp":
        use_multi_threading = False
        # vis
        vis_force = True
        vis_cut_force = False
        vis_ros_real_time = False
        vis_scale = False
        if_vis_audio = False
    else:
        exit(usage)
        raise NotImplementedError
    if sys.argv[1] == "train":
        IS_TRAIN = True
    elif sys.argv[1] == "test":
        IS_TRAIN = False
    else:
        print(usage)
        raise ValueError
    if bottle_type == "with_plug" or bottle_type == "robot_pouring":
        bottle_list = [bottle_type]
    elif bottle_type == "0":
        bottle_list = ["1", "3", "4"]
    else:
        raise NotImplementedError
    source_pickle_file_list = []
    for bottle in bottle_list:
        source_pickle_path = os.path.join(pkg_path, "pickle/pickle_{}_{}".format(sys.argv[1], bottle))
        source_pickle_file_list += glob.glob(os.path.join(source_pickle_path, "*.pickle"))
    save_h5py_file_name = os.path.join(pkg_path, "h5py_dataset/npy_{}_{}_snr{}.h5".format(sys.argv[1], bottle_type,
                                                                                          SNR_DB))
    main(source_pickle_file_list, save_h5py_file_name, use_multi_threading_=use_multi_threading)
