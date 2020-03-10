#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name     : dataset.py
# Purpose       :
# Creation Date : 05-01-2019
# Author        : Shuang Li [sli[at]informatik[dot]uni-hamburg[dot]de]
# Author        : Hongzhuo Liang [liang[at]informatik[dot]uni-hamburg[dot]de]
from __future__ import division, print_function
import torch.utils.data
import numpy as np
from audio_pouring.utils.utils import config
import h5py
import time
import librosa.util


def pitch_shift_spectrogram(spectrogram):
    """ Shift a spectrogram along the frequency axis in the spectral-domain at
    random
    """
    nb_cols = spectrogram.shape[0]
    max_shifts = nb_cols // 20  # around 5% shift
    nb_shifts = np.random.randint(-max_shifts, max_shifts)
    return np.roll(spectrogram, nb_shifts, axis=0)


class PouringDataset(torch.utils.data.Dataset):
    def __init__(self, path, input_audio_size, input_force_size, snr_db, multi_modal=False, train_rnn=False,
                 is_train=False, minus_wrench_first=False, stft_force=False, bottle_train="0", bottle_test="0"):
        self.path = path
        self.input_audio_size = input_audio_size
        self.input_force_size = input_force_size
        self.stft_force = stft_force
        self.is_train = is_train
        self.multi_modal = multi_modal
        self.train_rnn = train_rnn
        self.bottle_train = bottle_train
        self.bottle_test = bottle_test
        self.snr_db = snr_db
        self.audio_mean = config["audio_mean"]
        self.audio_std = config["audio_std"]
        self.minus_wrench_first = minus_wrench_first
        if self.minus_wrench_first:
            self.ft_mean = np.array(config["minus"]["ft_mean"])
            self.ft_std = np.array(config["minus"]["ft_std"])
        else:
            self.ft_mean = np.array(config["raw"]["ft_mean"])
            self.ft_std = np.array(config["raw"]["ft_std"])

        if is_train:
            self.file_to_load = "{}/{}/npy_train_{}_snr{}.h5".format(self.path, self.bottle_train, self.bottle_train,
                                                                     self.snr_db)
        else:
            self.file_to_load = "{}/{}/npy_test_{}_snr{}.h5".format(self.path, self.bottle_test, self.bottle_test,
                                                                    self.snr_db)
        self.file = None
        self.group_name_list = None

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.file_to_load, "r")
        if self.group_name_list is None:
            self.group_name_list = list(self.file.keys())
        each_npy = self.file.get(self.group_name_list[index])
        target = np.array(each_npy["cavity_h"]).astype(np.float32)
        audio = np.array(each_npy["audio"]).astype(np.float32)
        audio -= self.audio_mean
        audio /= self.audio_std
        # assert (audio.shape[0] == self.input_audio_size)
        # assert (target.shape[0] == audio.shape[1])
        if self.multi_modal:
            force = np.array(each_npy["force"]).astype(np.float32).reshape(-1, 6)  # result shape: (1008, 6)
            if self.minus_wrench_first:
                force = force - each_npy["wrench_first"]
            force -= self.ft_mean
            force /= self.ft_std
            if not self.stft_force:
                final_force = np.array([])
                for i in range(audio.shape[1]):
                    final_force = np.hstack([final_force, force[4 * i:4 * i + 8, :].reshape(-1)])
                final_force = final_force.reshape(-1, self.input_force_size)
                final_force = final_force.astype(np.float32)
            else:
                final_force = np.zeros([1, 251])
                for i in range(6):
                    force = np.asfortranarray(force)
                    force_stft = librosa.stft(y=force[:, i], n_fft=int(config["n_fft"] // 6), hop_length=4,
                                              win_length=8)
                    force_stft = force_stft[:, 0: 251]  # shape is (43, 252), get only (43, 251)
                    force_stft = np.abs(force_stft)
                    force_db = librosa.core.amplitude_to_db(force_stft)
                    final_force = np.vstack([final_force, force_db])
                final_force = final_force[1:, :]
                final_force = final_force.T.astype(np.float32)
            # assert final_force.shape[0] == audio.shape[1]
            scale = np.array(each_npy["scale"]).astype(np.float32)
            # assert scale.shape[0] == audio.shape[1]
            # audio: 257,251 force: 251,48
            return audio.T, final_force, target, scale
        else:
            return audio.T, target

    def __len__(self):
        with h5py.File(self.file_to_load, "r") as db:
            lens = len(db.keys())
        return lens


if __name__ == "__main__":
    time_start = time.time()
    b = PouringDataset("../h5py_dataset/", input_audio_size=257, input_force_size=48, snr_db=0.0, multi_modal=True,
                       minus_wrench_first=False, stft_force=True, train_rnn=True, is_train=True)
    train_loader = torch.utils.data.DataLoader(b, batch_size=10, num_workers=1, pin_memory=True, shuffle=False)
    for batch_idx, (audio_, force_, height_, scale_) in enumerate(train_loader):
        print(batch_idx, audio_.shape, height_.shape, force_.shape)
        pass
    a, f, h, s = b.__getitem__(1)
    print("run time is {}".format(time.time() - time_start))
