#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name     : main-lstm.py
# Purpose       :
# Creation Date : 05-01-2019
# Author        : Shuang Li [sli[at]informatik[dot]uni-hamburg[dot]de]
# Author        : Hongzhuo Liang [liang[at]informatik[dot]uni-hamburg[dot]de]
from __future__ import division, print_function
import argparse
import os
import time
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as torch_func
import glob
import rospy
import copy
from geometry_msgs.msg import WrenchStamped
from portaudio_transport.msg import AudioTransport
from std_msgs.msg import Bool, Float32
from audio_pouring.utils.utils import weight2height, config
# from audio_pouring.model.long_preprocess import butter_low_pass_filter
# from audio_pouring.model.long_preprocess import vis_wrench
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
try:
    import librosa.util
except ImportError:
    librosa = None
parser = argparse.ArgumentParser(description="audio2height")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--bottle", type=int, choices=config["bottle_id_list"], required=False)
parser.add_argument("--cavity-height", type=float, default=40.0)
parser.add_argument("--model-folder-path", type=str, default="./assets/learned_models/robot_experiment/")
parser.add_argument("--multi", action="store_true")
parser.add_argument("--test-robot", action="store_true")
parser.add_argument("--scale", action="store_true", help="add this if a scale that publish ros topic is connected")
parser.add_argument("--test_offline", action="store_true")
parser.add_argument("--minus_wrench_first", action="store_true")
parser.add_argument("--hidden-dim", type=int, default=64)
args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available() else False

if args.cuda:
    torch.cuda.manual_seed(1)

np.random.seed(int(time.time()))
target_frequency = config["target_frequency"]
source_frequency = config["source_frequency"]  # unit Hz
audio_topic = config["audio_topic"]
# ft_topic = config["ur5_ft_topic"]
ft_topic = config["ati_ft_topic"]
audio_length = config["fixed_audio_length"]  # unit second
n_fft = config["n_fft"]
win_size = config["win_size"]
overlap = config["overlap"]
if args.minus_wrench_first:
    ft_mean = np.array(config["minus"]["ft_mean"])
    ft_std = np.array(config["minus"]["ft_std"])
else:
    ft_mean = np.array(config["raw"]["ft_mean"])
    ft_std = np.array(config["raw"]["ft_std"])
win_length = int(win_size * target_frequency)
hop_length = int(win_length * overlap)
thresh_acc = np.array([1, 2, 3, 4, 5])  # unit mm
audio_mean = config["audio_mean"]
audio_std = config["audio_std"]
input_audio_size = config["input_audio_size"]
input_force_size = config["input_force_size_raw"]
force_embedding_size = input_force_size // 2
bottle_upper = torch.tensor([config["bottle_upper"]])
bottle_lower = torch.tensor([config["bottle_lower"]])
max_audio = config["max_audio"]

if args.multi:
    model_name = "0205_multi_lstm2_h64_bs64_bottlerobot_pouringtorobot_pouring_mono_coe0.001_snr-1000.0_a_f_early_fusion_raw_raw_force_bidirectional.model"
else:
    # mixed noise
    model_name = "0207_audio_lstm2_h64_bs64_bottlerobot_pouringtorobot_pouring_mono_coe0.001_snr-1000.0_audio_only_raw_raw_force_bidirectional.model"
    # without noise
    # model_name = "0208_audio_lstm2_h64_bs64_bottlerobot_pouringtorobot_pouring_mono_coe0.001_snr1000.0_audio_only_raw_raw_force_bidirectional.model"
if args.cuda:
    model = torch.load(args.model_folder_path + model_name, map_location="cuda:{}".format(args.gpu))
    model.device_ids = [args.gpu]
else:
    model = torch.load(args.model_folder_path + model_name, map_location="cpu")
model.mini_batch_size = 1
# if args.multi:
#     model.scale.batch_size = 1

device = torch.device("cuda" if args.cuda else "cpu")
if args.cuda:
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    else:
        device_id = [0, 1]
        torch.cuda.set_device(device_id[0])
        model = nn.DataParallel(model, device_ids=device_id).cuda()
    bottle_upper = bottle_upper.cuda()
    bottle_lower = bottle_lower.cuda()


def test(model_, audio, force=None):
    model_.eval()
    torch.set_grad_enabled(False)
    if args.cuda:
        audio = audio.cuda()
    if args.multi:
        if args.cuda:
            force = force.cuda()
        model.hidden = model.init_hidden(hidden_dim=args.hidden_dim, device=device)
        # force hidden
        model.force_hidden = model.init_hidden(hidden_dim=force_embedding_size, device=device)
        height = model_(audio, force)
    else:
        model_.hidden = model_.init_hidden(hidden_dim=args.hidden_dim, device=device)
        height = model_(audio)
    height = height * (bottle_upper - bottle_lower) + bottle_lower
    return height.cpu().data.numpy()


class Pouring:
    def __init__(self, target_cavity_height, multi=False, scale=False, test_offline=False):
        self.target = target_cavity_height
        self.audio_numpy = np.array([])
        self.force_numpy = np.array([])
        self.last_height = 0
        self.last_2_height = 0
        self.scale_reading = 0.0
        self.scale_reading_last = 0.0
        self.initial_force_magnitude = 0.0
        self.multi = multi
        self.need_print = True
        self.test_offline = test_offline
        self.scale = scale
        self.pub = rospy.Publisher("stop_pour", Bool, queue_size=10)
        self.wrench_first = None
        if self.multi:
            self.network_name = "Multi"
        else:
            self.network_name = "Audio"
        self.pub_network_data = rospy.Publisher("cavity_network_height", Float32, queue_size=10)
        self.pub_real_data = rospy.Publisher("cavity_real_height", Float32, queue_size=10)
        if self.scale:
            self.scale_sub = rospy.Subscriber("/maul_logic/wrench", WrenchStamped, self.scale_callback)
        rospy.sleep(2)  # wait the scale to have reading
        self.pouring_run()
        rospy.spin()

    def scale_callback(self, scale_data):
        self.scale_reading_last = self.scale_reading
        self.scale_reading = scale_data.wrench.force.z

    def pouring_run(self):
        if self.scale:
            while self.scale_reading == self.scale_reading_last:
                pass
        else:
            # some method to judge if the pouring motion is begin
            pass
        if self.multi:
            rospy.Subscriber(ft_topic, WrenchStamped, self.force_callback)
        rospy.Subscriber(audio_topic, AudioTransport, self.audio_callback)
        rospy.sleep(0.5)
        if self.multi:
            while not rospy.is_shutdown():
                copy_audio_numpy = copy.copy(self.audio_numpy)
                audio_spectrum = torch.from_numpy(self.audio_process(copy_audio_numpy).T).type(torch.FloatTensor)
                force_dim = audio_spectrum.shape[0] * input_force_size
                current_force = copy.copy(self.force_numpy)
                while current_force.shape[0] < force_dim / 2:
                    rospy.loginfo("force is not enough, cut {} column audio".
                                  format(len(audio_spectrum) - current_force.shape[0] // 48))
                    # audio_spectrum = audio_spectrum[-(current_force.shape[0] // 48):, :]
                    # force_dim = current_force.shape[0] // 48 * 48
                    exit()
                audio_spectrum = audio_spectrum.unsqueeze(0)
                force_input_numpy = current_force.reshape(-1, 6)
                if args.minus_wrench_first:
                    force_input_numpy -= self.wrench_first
                force_input_numpy -= ft_mean
                force_input_numpy /= ft_std
                final_force = np.array([])
                for i in range(audio_spectrum.shape[1]):
                    final_force = np.hstack([final_force, force_input_numpy[4 * i:4 * i + 8, :].reshape(-1)])
                final_force = final_force.reshape(-1, input_force_size)
                final_force = final_force.astype(np.float32)
                final_force = final_force[-force_dim:]

                # time_record = np.arange(0, len(final_force), 1)
                # vis_wrench(final_force, time_record, raw_force_data=None, lw=1)
                force_input = torch.from_numpy(final_force).type(torch.FloatTensor)
                force_input = force_input.unsqueeze(0)
                all_height = test(model, audio_spectrum, force_input)
                self.process_height(all_height)
        else:
            while not rospy.is_shutdown():
                audio_spectrum = torch.from_numpy(self.audio_process(self.audio_numpy).T).type(torch.FloatTensor)
                audio_spectrum = audio_spectrum.unsqueeze(0)
                all_height = test(model, audio_spectrum)
                self.process_height(all_height)

    def process_height(self, all_height):
        real_height = weight2height(cup_id=args.bottle, cur_weight=self.scale_reading)
        if self.test_offline:  # test with rosbag
            height = all_height[-1][0]
            wait_time = 0  # do not need to wait
            self.pub_network_data.publish(height)
            self.pub_real_data.publish(real_height)
        else:  # real robot experiment
            if self.multi:
                height = all_height[-1][0] - 5
            else:
                height = all_height[-1][0] - 5
            self.pub_network_data.publish(height + 5)
            self.pub_real_data.publish(real_height)
            if self.scale:
                wait_time = 4  # wait for 4 seconds the scale to be stable
            else:
                wait_time = 0  # if no scale is connected, no need to wait
        print("output is {}, go on pouring".format(height))
        if height <= self.target and abs(height - self.last_2_height) < 8:
            rospy.loginfo("Enjoy your drink!")
            start_time = time.time()
            while not rospy.is_shutdown():
                self.pub.publish(True)
                if self.scale:
                    real_height = weight2height(cup_id=args.bottle, cur_weight=self.scale_reading)
                    info = "{} Network {}mm, Real {}mm, Desire {}mm, Scale {}kg"
                    info = info.format(self.network_name, height, real_height, self.target, self.scale_reading)
                else:
                    info = "Network {}mm, Desire {}mm".format(height, self.target)
                if (time.time() - start_time > wait_time) and self.need_print:
                    rospy.loginfo(info)
                    self.need_print = False
        self.last_2_height = self.last_height
        self.last_height = height

    def audio_callback(self, audio_raw):
        audio_numpy = np.array(audio_raw.channels[0].frame_data)
        self.audio_numpy = np.hstack([self.audio_numpy, audio_numpy])
        # get latest 4 second messages
        if self.audio_numpy.shape[0] > source_frequency * audio_length:
            self.audio_numpy = self.audio_numpy[-source_frequency * audio_length:]

    def force_callback(self, force_raw):
        f_x = force_raw.wrench.force.x
        f_y = force_raw.wrench.force.y
        f_z = force_raw.wrench.force.z
        t_x = force_raw.wrench.torque.x
        t_y = force_raw.wrench.torque.y
        t_z = force_raw.wrench.torque.z
        force_data = np.array([f_x, f_y, f_z, t_x, t_y, t_z])
        if self.wrench_first is None:
            self.wrench_first = force_data
        self.force_numpy = np.hstack([self.force_numpy, force_data])
        if len(self.force_numpy) > input_force_size * 251:
            self.force_numpy = self.force_numpy[-input_force_size * 251:]

    @staticmethod
    def audio_process(audio_raw):
        audio_resample = librosa.core.resample(audio_raw, orig_sr=source_frequency, target_sr=target_frequency)
        # audio_resample *= 1. / max_audio * 0.9
        audio_fft = librosa.stft(y=audio_resample, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        audio_fft = np.abs(audio_fft)
        audio_db = librosa.core.amplitude_to_db(audio_fft)
        audio_db -= audio_mean
        audio_db /= audio_std
        return audio_db

    def velocity_change(self):
        pass


def main():
    if args.test_robot:
        test_files = glob.glob(os.path.join("./dataset/test", "*.npy"))
        test_files.sort()
        for i in range(len(test_files)):
            if len(np.load(test_files[i])) == 5:
                file_name, spectrum, force, height, scale = np.load(test_files[i])
            else:
                file_name, spectrum, height, scale = np.load(test_files[i])
            spectrum -= audio_mean
            spectrum /= audio_std
            spectrum = spectrum.T
            result = test(model, torch.from_numpy(spectrum).unsqueeze(0))
            result = result.squeeze()
            loss_mse = torch_func.mse_loss(torch.from_numpy(height), torch.from_numpy(result))
            loss_f1 = torch_func.l1_loss(torch.from_numpy(height), torch.from_numpy(result), reduction="mean")
            loss_f1 = loss_f1.cpu().data.numpy()
            is_correct = [abs(height.reshape(1, -1) - result.reshape(1, -1)) < thresh for thresh in thresh_acc]
            res_acc = [np.sum(cc) for cc in is_correct]
            acc = float(res_acc[-1]) / float(len(height))
            change_h = result[1:] - result[:-1]
            mono = np.sum(change_h[np.where(change_h > 0)])
            print("mse:{:9.4f}|err:{:9.4f}|mono:{:9.4f}|acc:{:.4f}".format(loss_mse, loss_f1, mono, acc))
    else:
        rospy.init_node("robot_pouring_demo")
        while not rospy.is_shutdown():
            Pouring(target_cavity_height=args.cavity_height, multi=args.multi, scale=args.scale,
                    test_offline=args.test_offline)


if __name__ == "__main__":
    main()
