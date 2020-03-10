#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name     :
# Purpose       :
# Creation Date : 05-01-2019
# Author        : Shuang Li [sli[at]informatik[dot]uni-hamburg[dot]de]
# Author        : Hongzhuo Liang [liang[at]informatik[dot]uni-hamburg[dot]de]

from __future__ import print_function, division
import numpy as np
import glob
import os
import sys
import yaml
import pickle
from scipy import interpolate
import audio_pouring
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import librosa.util
import librosa.display
pkg_path = os.path.dirname(audio_pouring.__file__)
config_file = os.path.join(pkg_path, "config/preprocess.yaml")
config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)


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


def get_file_name(file_dir_):
    file_list = []
    root_list = []
    for root, dirs, files in os.walk(file_dir_):
        # if root.count("/") == file_dir_.count("/")+1:
        file_list += [file_ for file_ in files]
        if root.count("/") == file_dir_.count("/") + 1:
            root_list.append(root)
    file_list.sort()
    return file_list, root_list


def npy_many_to_one(file_path):
    _, root_lists = get_file_name(file_path)
    print("data num :", len(root_lists))
    for root in root_lists:
        files = glob.glob(os.path.join(root, "*.npy"))
        files.sort()
        print("data num :", len(files))
        nn = []
        for file_ in files:
            nn.append(np.load(file_))
        data = np.array(nn)
        np.save(root + ".npy", data)


def generate_npy(npy_path_):
    train0 = np.array([])
    test0 = np.array([])

    for i in range(2, len(sys.argv)):
        bottle = sys.argv[i]
        files_train = glob.glob(os.path.join(npy_path_ + str(bottle) + "_train", "*.npy"))
        files_test = glob.glob(os.path.join(npy_path_ + str(bottle) + "_test", "*.npy"))

        train = np.array(files_train)
        test = np.array(files_test)

        # d_num = []
        # for num in range(train.shape[0] - 20, train.shape[0]):
        #     if test[0].split("/")[-1][:21] == train[num].split("/")[-1][:21]:
        #         d_num += [num]
        # train = np.delete(train, d_num, axis=0)

        np.save(npy_path_ + str(bottle) + "_train.npy", train)
        np.save(npy_path_ + str(bottle) + "_test.npy", test)
        train0 = np.hstack([train0, train])
        test0 = np.hstack([test0, test])

    np.save(npy_path_ + str(0) + "_train.npy", train0)
    np.save(npy_path_ + str(0) + "_test.npy", test0)
    print("All finish")


def spe():
    bottle3_train = "../dataset/full3_train.npy"
    bottle3_test = "../dataset/full3_test.npy"
    bottle1_train = "../dataset/full1_train.npy"
    bottle1_test = "../dataset/full1_test.npy"
    bottle4_train = "../dataset/full4_train.npy"
    bottle4_test = "../dataset/full4_test.npy"
    d = [bottle1_train, bottle1_test, bottle3_train, bottle3_test, bottle4_train, bottle4_test]
    s = ["../dataset/bottle1_train", "../dataset/bottle1_test",
         "../dataset/bottle3_train", "../dataset/bottle3_test",
         "../dataset/bottle4_train", "../dataset/bottle4_test"]

    delete_name = np.load("../del.npy")

    for n in range(len(d)):
        whole_train = np.load(d[n])
        if not os.path.isdir(s[n]):
            os.mkdir(s[n])
        for data_p in whole_train:
            data = np.load(data_p)
            if data[0][0][1:22] in delete_name:
                # line = d_index[data[0][0][1:]]
                print("delete bad bag")
            else:
                for e in range(data[1].shape[1]):
                    bag_name = data[0]
                    audio = data[1][:, e]
                    # force = data[2].reshape(-1,64)[e]
                    label = data[2][e]
                    assert (audio.shape[0] == 257)
                    # assert (force.shape[0] == 64)
                    np.save(s[n] + bag_name[0] + "-" + str(e) + ".npy", [np.array([bag_name, audio, label])])
        for i in range(len(s)):
            files = glob.glob(os.path.join(s[i], "*.npy"))
            np.save(s[i] + ".npy", files)


def audio_first_normalization(pickle_path):
    # pickle path should be at data/bag/pickle
    target_frequency = config["frequency"]
    orig_sr = config["source_frequency"]
    pickle1_files = glob.glob(os.path.join(pickle_path + str(1), "*.pickle"))
    pickle3_files = glob.glob(os.path.join(pickle_path + str(3), "*.pickle"))
    pickle4_files = glob.glob(os.path.join(pickle_path + str(4), "*.pickle"))
    pickle_files = pickle1_files + pickle3_files + pickle4_files
    pickle_num = len(pickle_files)

    # delete bad data
    delete_name = np.load("../del.npy")
    n = 0
    max_audio = -100
    for i in range(pickle_num):
        data = []
        with (open(pickle_files[i], "rb")) as openfile:
            data.append(pickle.load(openfile))
            if pickle_files[i].split("/")[-1][:-7] in delete_name:
                n = n + 1
                print(n)
                print("delete bad bag ", pickle_files[i].split("/")[-1][:-7])
            else:
                audio_raw = data[0]["audio"]
                audio_resample = librosa.core.resample(audio_raw, orig_sr=orig_sr, target_sr=target_frequency)
                max_audio = max(np.max(audio_resample), max_audio)

    print(max_audio)


def normalization_all_data(npy_path_):
    npy1_files = glob.glob(os.path.join(npy_path_ + str(1) + "_train", "*.npy"))
    npy1_files_test = glob.glob(os.path.join(npy_path_ + str(1) + "_test", "*.npy"))
    npy3_files = glob.glob(os.path.join(npy_path_ + str(3) + "_train", "*.npy"))
    npy3_files_test = glob.glob(os.path.join(npy_path_ + str(3) + "_test", "*.npy"))
    npy4_files = glob.glob(os.path.join(npy_path_ + str(4) + "_train", "*.npy"))
    npy4_files_test = glob.glob(os.path.join(npy_path_ + str(4) + "_test", "*.npy"))
    files = npy1_files + npy3_files + npy4_files + npy1_files_test + npy3_files_test + npy4_files_test
    files.sort()
    print("data num :", len(files))
    nn = []
    for file_ in files:
        nn.append(np.load(file_))
    d = np.array(nn)
    audio = np.vstack(d[:, 1])
    audio_mean = np.mean(audio)
    audio_std = np.std(audio)
    print("audio_mean", audio_mean)
    print("audio_std", audio_std)

    force = np.vstack(d[:, 2])
    force_mean = np.mean(force)
    force_std = np.std(force)
    print("force_mean", force_mean)
    print("force_std", force_std)

    height = np.vstack(d[:, 3])
    print("cavity h max :", np.max(height))
    print("cavity h min :", np.min(height))

    scale = np.vstack(d[:, 4])
    print("scale change max :", np.max(scale))
    print("scale change min :", np.min(scale))

    return audio_mean, audio_std, force_mean, force_std


def weight2height(cup_id, cur_weight):
    """
    function to get "real" water height from weight
    :param cup_id: cup id in str
    :param cur_weight: input weight in kg
    :return: cavity_h: cavity height in mm
    """
    if not isinstance(cup_id, int):
        print("[error] cup_id should be int, please convert your input to int")
        raise TypeError
    if cup_id in config["bottle_id_list"]:
        # params from bottle_config
        bottle_config_path = os.path.join(pkg_path, "config/bottles/bottle" + str(cup_id) + "_config.npy")
        params = np.load(bottle_config_path)
        poly = np.polynomial.Polynomial(params)
        cavity_h = poly(cur_weight)
    else:
        print("wrong type of cups")
        cavity_h = "WRONG CUP_ID INPUT"

    return cavity_h


def poly_fit_with_fixed_points(degree, x, y, x_fix, y_fix):
    """
    :param degree: the degree of the polynomial
    :param x: input x
    :param y: input y
    :param x_fix: input fixed x
    :param y_fix: input fixed y
    :return:
    """
    mat = np.empty((degree + 1 + len(x_fix),) * 2)
    vec = np.empty((degree + 1 + len(x_fix),))
    x_n = x ** np.arange(2 * degree + 1)[:, None]
    yx_n = np.sum(x_n[:degree + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(degree + 1) + np.arange(degree + 1)[:, None]
    mat[:degree + 1, :degree + 1] = np.take(x_n, idx)
    xf_n = x_fix ** np.arange(degree + 1)[:, None]
    mat[:degree + 1, degree + 1:] = xf_n / 2
    mat[degree + 1:, :degree + 1] = xf_n.T
    mat[degree + 1:, degree + 1:] = 0
    vec[:degree + 1] = yx_n
    vec[degree + 1:] = y_fix
    params = np.linalg.solve(mat, vec)
    return params[:degree + 1]


def weight2height_raw(cup_id, degree, cur_weight):
    bottle_raw_data = pd.read_csv("{}/config/bottles/bottle{}_config.csv".format(pkg_path, cup_id))
    height = np.array(bottle_raw_data["mm"])
    weight = np.array(bottle_raw_data["g"])
    params = poly_fit_with_fixed_points(degree=degree, x=weight, y=height, x_fix=np.array([weight[0], weight[-1]]),
                                        y_fix=np.array([height[0], height[-1]]))
    poly = np.polynomial.Polynomial(params)
    cavity_h = poly(cur_weight)
    return cavity_h


def height2weight(cup_id, height):
    out = 200
    i = 1
    while out > height:
        out = weight2height(cup_id, cur_weight=i * 0.0001)
        i += 1
    print(out, i * 0.0001)
    return i * 0.0001


def vis_audio(spec, sr, hop_length, title):
    librosa.display.specshow(spec, sr=sr, hop_length=hop_length, y_axis="linear", x_axis="time")
    plt.title(title)
    plt.tight_layout()
    plt.colorbar()
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Frequency (Hz)", fontsize=12)


def open_pickle(pickle_file):
    with open(pickle_file, "rb") as f:
        if sys.version_info >= (3, 0):
            pickle_data = pickle.load(f, encoding="latin1")
        else:
            pickle_data = pickle.load(f)
    return pickle_data


if __name__ == "__main__":
    print("put one or more function to use")
    # spe()
    npy_path = os.path.join(pkg_path, "dataset")
    generate_npy(npy_path)
    # split_data()
    # generate()
    # npy_many_to_one(npy_path)  # no use anymore
    # audio_first_normalization(pickle_path)  # please specify pickle path here
    # normalization_all_data()
