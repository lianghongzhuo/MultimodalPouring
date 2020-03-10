#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name     : model.py
# Purpose       :
# Creation Date : 05-12-2018
# Author        : Shuang Li [sli[at]informatik[dot]uni-hamburg[dot]de]
# Author        : Hongzhuo Liang [liang[at]informatik[dot]uni-hamburg[dot]de]

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn


class HeightRegression(nn.Module):
    """ height regression from multi data embedding space"""

    def __init__(self, input_size=128, hidden_size=None, layer_num=1, output_size=1):
        super(HeightRegression, self).__init__()
        self.layer_num = layer_num
        if hidden_size is None:
            hidden_size = input_size // 2
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)

        self.drop = nn.Dropout(0.5)
        if layer_num == 2:
            self.reg = nn.Linear(hidden_size // 2, output_size)
        elif layer_num == 1:
            self.reg = nn.Linear(hidden_size, output_size)
        else:
            exit("In class HeightRegression: No such layer number")
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        if self.layer_num == 2:
            x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.reg(x))
        return x


class ScaleRegression(nn.Module):
    """ scale regression from multi data embedding space"""

    def __init__(self, batch_size, input_size=128, output_size=1):
        super(ScaleRegression, self).__init__()
        hidden_size = 16
        if input_size == 1:
            hidden_size = 1
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.reg = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU(inplace=True)
        self.batch_size = batch_size

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        # x = self.reg(x)
        return x


class AudioPouringModel(nn.Module):
    def __init__(self):
        """ Base class for various methods to evaluate pouring sequence.
        Should not be instantiated directly.
        Attributes
        ----------
        """

        super(AudioPouringModel, self).__init__()

    def init_hidden(self, hidden_dim, device=None):
        num_direction = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_direction, self.mini_batch_size, hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers * num_direction, self.mini_batch_size, hidden_dim, device=device)
        if self.is_lstm:
            return h0, c0
        else:
            return h0


class OneLossOneRNNMultiModel(AudioPouringModel):
    def __init__(self, audio_input_dim, hidden_dim, mini_batch_size, force_input_dim, num_layers=2,
                 audio_embedding_size=64, audio_fc1_size=200, dropout=0.5, bidirectional=False, is_lstm=True):
        super(OneLossOneRNNMultiModel, self).__init__()
        self.num_layers = num_layers
        if self.num_layers == 1:
            self.dropout = 0
        else:
            self.dropout = dropout
        self.bidirectional = bidirectional
        self.is_lstm = is_lstm

        self.audio_input_dim = audio_input_dim
        self.force_input_dim = force_input_dim
        force_embedding_size = force_input_dim // 2
        self.hidden_dim = hidden_dim
        self.mini_batch_size = mini_batch_size
        self.hidden = self.init_hidden(self.hidden_dim)

        self.audio_fc1_size = audio_fc1_size
        self.audio_embedding_size = audio_embedding_size

        if is_lstm:
            self.rnn = nn.LSTM(audio_input_dim, self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
                               bidirectional=self.bidirectional, batch_first=True)
            self.rnn_fusion = nn.LSTM(audio_input_dim + force_input_dim, self.hidden_dim, num_layers=self.num_layers,
                                      dropout=self.dropout, bidirectional=self.bidirectional, batch_first=True)
            self.rnn_force = nn.LSTM(force_input_dim, self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
                                     bidirectional=self.bidirectional, batch_first=True)
        else:
            self.rnn = nn.GRU(audio_input_dim, self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
                              bidirectional=self.bidirectional, batch_first=True)
            self.rnn_fusion = nn.GRU(audio_input_dim + force_input_dim, self.hidden_dim, num_layers=self.num_layers,
                                     dropout=self.dropout, bidirectional=self.bidirectional, batch_first=True)
            self.rnn_force = nn.GRU(force_input_dim, self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
                                    bidirectional=self.bidirectional, batch_first=True)

        self.force = nn.Linear(self.force_input_dim, force_embedding_size)
        self.relu = nn.ReLU(inplace=True)
        self.fusion = HeightRegression(input_size=force_embedding_size + self.hidden_dim)
        height_regression_factor = 2 if self.bidirectional else 1
        self.early_fusion = HeightRegression(input_size=(self.hidden_dim * height_regression_factor))
        # initialize bias
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)

    def forward(self, audio, force):
        audio_embed, self.hidden = self.rnn(audio, self.hidden)
        force_embed = self.relu(self.force(force.reshape(-1, self.force_input_dim)))
        fusion_embed = torch.cat((audio_embed.reshape(-1, self.hidden_dim), force_embed), 1)
        h = self.fusion(fusion_embed)
        return h


class ThreeLossRNNMultiModel(AudioPouringModel):
    def __init__(self, audio_input_dim, hidden_dim, mini_batch_size, force_input_dim,
                 num_layers=1, audio_embedding_size=64, audio_fc1_size=200, dropout=0.5,
                 bidirectional=False, is_lstm=True):
        super(ThreeLossRNNMultiModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.is_lstm = is_lstm

        self.audio_input_dim = audio_input_dim
        self.force_input_dim = force_input_dim
        self.force_embedding_size = force_input_dim // 2
        self.hidden_dim = hidden_dim
        self.mini_batch_size = mini_batch_size
        self.hidden = self.init_hidden(hidden_dim=self.hidden_dim)
        self.force_hidden = self.init_hidden(hidden_dim=self.force_embedding_size)

        self.audio_fc1_size = audio_fc1_size
        self.audio_embedding_size = audio_embedding_size

        self.fc1 = nn.Linear(audio_input_dim, audio_fc1_size)
        self.bn1 = nn.BatchNorm1d(audio_fc1_size)
        self.relu = nn.ReLU(inplace=True)

        if is_lstm:
            self.audio_rnn = nn.LSTM(audio_fc1_size, self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
                                     bidirectional=self.bidirectional)
        else:
            self.audio_rnn = nn.GRU(audio_fc1_size, self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
                                    bidirectional=self.bidirectional)

        self.audio_reg = HeightRegression(input_size=self.hidden_dim)

        if is_lstm:
            self.force_rnn = nn.LSTM(force_input_dim, self.force_embedding_size, num_layers=self.num_layers,
                                     dropout=self.dropout, bidirectional=self.bidirectional)
        else:
            self.force_rnn = nn.GRU(force_input_dim, self.force_embedding_size, num_layers=self.num_layers,
                                    dropout=self.dropout, bidirectional=self.bidirectional)
        # self.fusion = nn.Sequential(
        #     nn.Linear(self.force_embedding_size + self.hidden_dim, self.audio_embedding_size),
        #     nn.BatchNorm1d(self.audio_embedding_size),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.audio_embedding_size, self.audio_embedding_size // 2),
        #     nn.BatchNorm1d(self.audio_embedding_size // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.audio_embedding_size // 2, 1)
        # )

        self.fusion = HeightRegression(input_size=self.force_embedding_size + self.hidden_dim)
        self.scale = ScaleRegression(batch_size=self.mini_batch_size,
                                     input_size=self.force_embedding_size)

        # initialize bias
        for name, param in self.audio_rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)
        for name, param in self.force_rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)

    def forward(self, audio, force):
        force = force.reshape(-1, self.mini_batch_size, self.force_input_dim)
        force_embed, self.force_hidden = self.force_rnn(force, self.force_hidden)
        scale = self.scale(force_embed.reshape(-1, self.force_embedding_size))

        audio = self.relu(self.bn1(self.fc1(audio.reshape(-1, self.audio_input_dim))))
        audio = audio.reshape(-1, self.mini_batch_size, self.audio_fc1_size)
        audio_embed, self.hidden = self.audio_rnn(audio, self.hidden)
        height = self.audio_reg(audio_embed.view(-1, self.hidden_dim))

        fusion_embed = torch.cat((audio_embed.view(-1, self.hidden_dim),
                                  force_embed.view(-1, self.force_embedding_size)), 1)
        f_height = self.fusion(fusion_embed)
        return height, f_height, scale


class TwoLossOneRNNMultiModel(AudioPouringModel):
    def __init__(self, audio_input_dim, hidden_dim, mini_batch_size, force_input_dim, num_layers=2,
                 audio_embedding_size=64, audio_fc1_size=200, dropout=0.5, bidirectional=False, is_lstm=True):
        super(TwoLossOneRNNMultiModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.is_lstm = is_lstm

        self.audio_input_dim = audio_input_dim
        self.force_input_dim = force_input_dim
        self.force_embedding_size = force_input_dim // 2
        self.hidden_dim = hidden_dim
        self.mini_batch_size = mini_batch_size
        self.hidden = self.init_hidden(self.hidden_dim)

        self.audio_fc1_size = audio_fc1_size
        self.audio_embedding_size = audio_embedding_size

        if is_lstm:
            self.rnn = nn.LSTM(audio_input_dim, self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
                               bidirectional=self.bidirectional, batch_first=True)
        else:
            self.rnn = nn.GRU(audio_input_dim, self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
                              bidirectional=self.bidirectional, batch_first=True)
        self.audio_reg = nn.Sequential(
            nn.Linear(self.hidden_dim, self.audio_embedding_size * 2),
            nn.BatchNorm1d(self.audio_embedding_size * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.audio_embedding_size * 2, self.audio_embedding_size)
        )
        self.force = nn.Linear(self.force_input_dim, self.force_embedding_size)
        self.relu = nn.ReLU(inplace=True)
        self.fusion = HeightRegression(input_size=self.force_embedding_size + self.hidden_dim)
        self.scale = ScaleRegression(batch_size=self.mini_batch_size, input_size=self.force_embedding_size)

        # initialize bias
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)

    def forward(self, audio, force):
        force_embed = self.relu(self.force(force.view(-1, self.force_input_dim)))
        scale = self.scale(force_embed)
        # audio = self.relu(self.bn1(self.fc1(audio.reshape(-1, self.audio_input_dim))))
        # audio = audio.reshape(-1, self.mini_batch_size, self.audio_fc1_size)
        audio_embed, self.hidden = self.rnn(audio, self.hidden)
        fusion_embed = torch.cat((audio_embed.reshape(-1, self.hidden_dim), force_embed), 1)
        h = self.fusion(fusion_embed)
        return h, scale


class TwoLossTwoRNNMultiModel(AudioPouringModel):
    def __init__(self, audio_input_dim, hidden_dim, mini_batch_size, force_input_dim, num_layers=1,
                 audio_embedding_size=64, audio_fc1_size=200, dropout=0.5, num_losses=None, bidirectional=False,
                 is_lstm=True):
        super(TwoLossTwoRNNMultiModel, self).__init__()
        self.num_layers = num_layers
        if self.num_layers == 1:
            self.dropout = 0
        else:
            self.dropout = dropout
        self.bidirectional = bidirectional
        self.is_lstm = is_lstm

        self.audio_input_dim = audio_input_dim
        self.force_input_dim = force_input_dim
        self.force_embedding_size = force_input_dim // 2
        self.hidden_dim = hidden_dim
        self.mini_batch_size = mini_batch_size
        self.hidden = self.init_hidden(hidden_dim=self.hidden_dim)
        self.force_hidden = self.init_hidden(hidden_dim=self.force_embedding_size)

        self.audio_fc1_size = audio_fc1_size
        self.audio_embedding_size = audio_embedding_size
        self.num_losses = num_losses
        if is_lstm:
            self.audio_rnn = nn.LSTM(audio_input_dim, self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
                                     bidirectional=self.bidirectional, batch_first=True)
            self.force_rnn = nn.LSTM(force_input_dim, self.force_embedding_size, num_layers=self.num_layers,
                                     dropout=self.dropout, bidirectional=self.bidirectional, batch_first=True)
            self.fusion_rnn = nn.LSTM(force_input_dim + 1, self.force_embedding_size, num_layers=self.num_layers,
                                      dropout=self.dropout, bidirectional=self.bidirectional, batch_first=True)
        else:
            self.audio_rnn = nn.GRU(audio_input_dim, self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
                                    bidirectional=self.bidirectional, batch_first=True)
            self.force_rnn = nn.GRU(force_input_dim, self.force_embedding_size, num_layers=self.num_layers,
                                    dropout=self.dropout, bidirectional=self.bidirectional, batch_first=True)
            self.fusion_rnn = nn.GRU(force_input_dim + 1, self.force_embedding_size, num_layers=self.num_layers,
                                     dropout=self.dropout, bidirectional=self.bidirectional, batch_first=True)

        self.audio_reg = nn.Sequential(
            nn.Linear(self.hidden_dim, self.audio_embedding_size * 2),
            nn.BatchNorm1d(self.audio_embedding_size * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.audio_embedding_size * 2, self.audio_embedding_size)
        )
        height_regression_factor = 2 if self.bidirectional else 1
        height_regression_input_size = (self.force_embedding_size + self.hidden_dim) * height_regression_factor
        self.height_regression_fusion = HeightRegression(input_size=height_regression_input_size)
        self.height_regression_audio = HeightRegression(input_size=self.hidden_dim)
        self.height_regression_force = HeightRegression(input_size=self.force_embedding_size)
        self.scale = ScaleRegression(batch_size=self.mini_batch_size,
                                     input_size=self.force_embedding_size * height_regression_factor)

        # initialize bias
        for name, param in self.audio_rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)
        for name, param in self.force_rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)

    def forward(self, audio, force):
        force_embed, self.force_hidden = self.force_rnn(force, self.force_hidden)
        audio_embed, self.hidden = self.audio_rnn(audio, self.hidden)
        fusion_embed = torch.cat((audio_embed.reshape(-1, audio_embed.shape[2]),
                                  force_embed.reshape(-1, force_embed.shape[2])), 1)
        h = self.height_regression_fusion(fusion_embed)
        if self.num_losses == 2:
            scale = self.scale(force_embed.reshape(-1, force_embed.shape[2]))
            return h, scale
        elif self.num_losses == 1:
            return h
        else:
            raise NotImplementedError


class AudioRNN(AudioPouringModel):
    def __init__(self, input_dim, hidden_dim, mini_batch_size, num_layers=1, audio_fc1_size=200, dropout=0.5,
                 bidirectional=False, is_lstm=True):
        super(AudioRNN, self).__init__()
        self.num_layers = num_layers  # rnn layer num
        if self.num_layers == 1:
            self.dropout = 0
        else:
            self.dropout = dropout
        self.bidirectional = bidirectional
        self.is_lstm = is_lstm
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mini_batch_size = mini_batch_size
        self.hidden = self.init_hidden(hidden_dim=self.hidden_dim)
        self.audio_fc1_size = audio_fc1_size

        if is_lstm:
            self.rnn = nn.LSTM(input_dim, self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
                               bidirectional=self.bidirectional, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout,
                              bidirectional=self.bidirectional, batch_first=True)

        fc_layer_num = 1
        self.height = HeightRegression(input_size=self.hidden_dim, layer_num=fc_layer_num)

        # initialize bias
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)

    def forward(self, x):
        """
        :param x: [0: batch size 1: sequence length 2: input size]
        :return:
        """
        # x = self.relu(self.bn1(self.fc1(x.reshape(-1, self.input_dim))))
        # y = x.view(-1, self.mini_batch_size, self.input_dim)  # stupid error remember forever
        assert len(x.size()) == 3, "[RNN]: Input dimension must be of length 3"
        assert self.mini_batch_size == x.size()[0], "[RNN]: Input mini batch size must equal to the input data size"
        rnn_out, self.hidden = self.rnn(x, self.hidden)
        height = self.height(rnn_out.reshape(-1, self.hidden_dim))
        return height


class AudioForceEarlyFusion(OneLossOneRNNMultiModel):
    def forward(self, audio, force):
        fusion_data = torch.cat((audio, force), 2)
        fusion_embed, self.hidden = self.rnn_fusion(fusion_data, self.hidden)
        h = self.early_fusion(fusion_embed.reshape(-1, fusion_embed.shape[2]))
        return h


class ForceOnly(OneLossOneRNNMultiModel):
    def forward(self, audio, force):
        fusion_embed, self.hidden = self.rnn_force(force, self.hidden)
        h = self.early_fusion(fusion_embed.reshape(-1, fusion_embed.shape[2]))
        return h


if __name__ == "__main__":
    # test code for AudioLSTM:
    seq_length = 520
    mini_batch_size_test = 2
    input_size_test = 225
    hidden_dim_test = 50
    x_input_tmp = torch.ones(mini_batch_size_test, seq_length, input_size_test)
    model = AudioRNN(input_size_test, hidden_dim_test, mini_batch_size_test)
    output_tmp, hidden = model(x_input_tmp)
    print(output_tmp.shape)
