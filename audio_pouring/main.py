#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name     : main_multi.py
# Purpose       :
# Creation Date : 05-01-2019
# Author        : Shuang Li [sli[at]informatik[dot]uni-hamburg[dot]de]
# Author        : Hongzhuo Liang [liang[at]informatik[dot]uni-hamburg[dot]de]
from __future__ import division, print_function
import os
import time
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as torch_func
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from audio_pouring.model.dataset import PouringDataset
from audio_pouring.model import OneLossOneRNNMultiModel, TwoLossOneRNNMultiModel, TwoLossTwoRNNMultiModel, AudioRNN
from audio_pouring.model import AudioGuideForceMultiModel, AudioForceEarlyFusion, ForceOnly
from audio_pouring.utils.network import parse, worker_init_fn
from audio_pouring.utils.utils import config

args = parse()
args.cuda = args.cuda if torch.cuda.is_available() else False
if args.cuda:
    torch.cuda.manual_seed(1)
np.random.seed(int(time.time()))
logger = SummaryWriter(os.path.join("./assets/log/", args.tag))
if args.draw_acc_fig:
    thresh_acc = np.arange(0., 6.1, 0.25)
else:
    thresh_acc = np.array([1, 2, 3, 4, 5])  # unit mm
input_audio_size = config["input_audio_size"]
if args.stft_force:
    input_force_size = config["input_force_size_stft"]
else:
    input_force_size = config["input_force_size_raw"]
bottle_upper = torch.tensor([config["bottle_upper"]])
bottle_lower = torch.tensor([config["bottle_lower"]])
scale_upper = torch.tensor([config["scale_upper"]])
scale_lower = torch.tensor([config["scale_lower"]])

is_resume = 0
if args.load_model and args.load_epoch != -1:
    is_resume = 1
if is_resume or args.mode == "test":
    model = torch.load(args.load_model, map_location="cuda:{}".format(args.gpu))
    model.device_ids = [args.gpu]
    print("load model {}".format(args.load_model))

    if args.robot:  # whether do a training that human pouring without plug to fine tune human pouring with a plug
        for param in model.parameters():
            param.requires_grad = False
        num_fc1 = model.height.fc1.in_features
        num_reg = model.height.reg.in_features
        model.height = nn.Sequential(
            nn.Linear(num_fc1, num_reg),
            nn.BatchNorm1d(num_reg),
            nn.ReLU(inplace=True),
            nn.Linear(num_reg, 1),
            nn.ReLU(inplace=True)
        )
else:
    if args.multi:
        if args.multi_detail == "force_only":
            model = ForceOnly(input_audio_size, args.hidden_dim, args.bs, force_input_dim=input_force_size,
                              num_layers=args.layer_num, is_lstm=args.lstm, bidirectional=args.bidirectional)
        elif args.multi_detail == "a_guide_f":
            model = AudioGuideForceMultiModel(input_audio_size, args.hidden_dim, args.bs,
                                              force_input_dim=input_force_size, num_layers=args.layer_num,
                                              is_lstm=args.lstm, bidirectional=args.bidirectional)
        elif args.multi_detail == "a_f_early_fusion":
            model = AudioForceEarlyFusion(input_audio_size, args.hidden_dim, args.bs, force_input_dim=input_force_size,
                                          num_layers=args.layer_num, bidirectional=args.bidirectional,
                                          is_lstm=args.lstm)
        elif args.multi_detail == "2loss2rnn":
            model = TwoLossTwoRNNMultiModel(input_audio_size, args.hidden_dim, args.bs,
                                            force_input_dim=input_force_size, num_layers=args.layer_num, num_losses=2,
                                            is_lstm=args.lstm, bidirectional=args.bidirectional)
        elif args.multi_detail == "1loss2rnn":
            model = TwoLossTwoRNNMultiModel(input_audio_size, args.hidden_dim, args.bs,
                                            force_input_dim=input_force_size, num_layers=args.layer_num, num_losses=1,
                                            is_lstm=args.lstm, bidirectional=args.bidirectional)
        elif args.multi_detail == "2loss1rnn":
            model = TwoLossOneRNNMultiModel(input_audio_size, args.hidden_dim, args.bs,
                                            force_input_dim=input_force_size, num_layers=args.layer_num,
                                            is_lstm=args.lstm, bidirectional=args.bidirectional)
        elif args.multi_detail == "1loss1rnn":
            model = OneLossOneRNNMultiModel(input_audio_size, args.hidden_dim, args.bs,
                                            force_input_dim=input_force_size, num_layers=args.layer_num,
                                            is_lstm=args.lstm, bidirectional=args.bidirectional)
        else:
            raise NotImplementedError
    else:
        model = AudioRNN(input_audio_size, args.hidden_dim, args.bs, args.layer_num, is_lstm=args.lstm)
device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    elif args.gpu == -1:
        device_id = [0, 1]
        torch.cuda.set_device(device_id[0])
        model = nn.DataParallel(model, device_ids=device_id).cuda()
    bottle_upper = bottle_upper.cuda()
    bottle_lower = bottle_lower.cuda()
    scale_upper = scale_upper.cuda()
    scale_lower = scale_lower.cuda()

if args.robot:
    optimizer = optim.Adam(model.height.parameters(), lr=args.lr)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)


def load_data(is_train):
    data_loader = torch.utils.data.DataLoader(
        PouringDataset(
            path=args.data_path,
            input_audio_size=input_audio_size,
            input_force_size=input_force_size,
            snr_db=args.snr_db,
            multi_modal=args.multi,
            train_rnn=True,
            is_train=is_train,
            minus_wrench_first=args.minus_wrench_first,
            stft_force=args.stft_force,
            bottle_train=args.bottle_train,
            bottle_test=args.bottle_test,
        ),
        batch_size=args.bs,
        drop_last=True,
        num_workers=20,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        # collate_fn=my_collate,
    )
    return data_loader


def loss_function(height, target):
    loss_mse = torch_func.mse_loss(height, target)

    # constraint height is always decreasing
    height_batch = height.view(args.bs, -1)
    change_h_batch = height_batch[:, 1:] - height_batch[:, :-1]
    if args.cuda:
        a = torch.zeros(change_h_batch.shape).cuda()
    else:
        a = torch.zeros(change_h_batch.shape)
    loss_mono = torch.sum(torch.max(a, change_h_batch)) / height.shape[0]
    return loss_mse, loss_mono


def compute_acc(height, target, is_train):
    # compute acc
    is_correct = [abs(height.cpu().data.numpy() - target.cpu().data.numpy()) < thresh for thresh in thresh_acc]
    res_acc = [np.sum(cc) for cc in is_correct]
    if is_train:
        return res_acc
    else:
        height = height.reshape(-1, args.bs)
        target = target.reshape(-1, args.bs)
        sample_num = height.shape[0] // 4 + 1
        res_acc_segment = []
        for i in range(4):
            if i == 3:
                h = height[sample_num * i:]
                t = target[sample_num * i:]
            else:
                h = height[sample_num * i: sample_num * (i + 1)]
                t = target[sample_num * i: sample_num * (i + 1)]
            is_correct = [abs(h.cpu().data.numpy() - t.cpu().data.numpy()) < thresh for thresh in thresh_acc]
            res_acc_segment.append([np.sum(cc) for cc in is_correct])
        return res_acc, res_acc_segment, [sample_num * args.bs, sample_num * args.bs, sample_num * args.bs,
                                          height.shape[0] * args.bs - sample_num * args.bs * 3]


def run_network(model_, all_data):
    # convert data to cuda
    if args.multi:
        audio, force, target, target_scale = all_data
        if args.cuda:
            audio, force, target, target_scale = audio.cuda(), force.cuda(), target.cuda(), target_scale.cuda()
    else:
        audio, target = all_data
        force, target_scale = None, None
        if args.cuda:
            audio, target = audio.cuda(), target.cuda()
    model_.hidden = model_.init_hidden(hidden_dim=args.hidden_dim, device=device)
    if args.multi_detail in ["a_guide_f", "2loss2rnn", "1loss2rnn"]:
        model_.force_hidden = model_.init_hidden(hidden_dim=input_force_size // 2, device=device)
    if args.multi:
        network_output = model_(audio, force)
    else:
        network_output = model_(audio)
    return target, target_scale, network_output


def train(model_, loader, epoch):
    model_.train()
    torch.set_grad_enabled(True)
    train_error = 0
    correct_height = [0] * len(thresh_acc)
    data_num = 0
    for batch_idx, all_data in enumerate(loader):
        optimizer.zero_grad()
        target, target_scale, network_output = run_network(model_, all_data)
        if len(network_output) == 2:  # in some case, even multi model have only one output
            height, scale = network_output
            scale = scale * (scale_upper - scale_lower) + scale_lower
            loss_mse_scale = torch_func.mse_loss(scale, target_scale.view(-1, 1))
            height = height * (bottle_upper - bottle_lower) + bottle_lower
            target = target.view(-1, 1)
            loss_mse, loss_mono = loss_function(height, target)
            loss = loss_mse + loss_mse_scale + args.mono_coe * loss_mono
            loss.backward()
            optimizer.step()
        else:
            loss_mse_scale = None
            height = network_output
            height = height * (bottle_upper - bottle_lower) + bottle_lower
            target = target.view(-1, 1)
            loss_mse, loss_mono = loss_function(height, target)
            loss = loss_mse + args.mono_coe * loss_mono
            loss.backward()
            optimizer.step()
        data_num += target.shape[0]
        res_acc = compute_acc(height, target, is_train=True)
        correct_height = [c + r for c, r in zip(correct_height, res_acc)]

        # compute average error
        train_error += torch_func.l1_loss(height, target, reduction="sum")

        if batch_idx % args.log_interval == 0:
            tmp = "Epoch:{}[{:.0f}%]\tLoss:{:.4f}\tLoss_mse: {:.4f}\tLoss_mono: {:.4f}\ttag: {}"
            tmp = tmp.format(epoch, 100. * batch_idx * args.bs / len(loader.dataset), loss.item(), loss_mse.item(),
                             loss_mono.item(), args.tag)
            print(tmp)
            logger.add_scalar("train_loss", loss.item(), batch_idx + epoch * len(loader))
            logger.add_scalar("train_loss_mse", loss_mse.item(), batch_idx + epoch * len(loader))
            logger.add_scalar("train_loss_mono", loss_mono.item(), batch_idx + epoch * len(loader))
            if loss_mse_scale is not None:
                logger.add_scalar("train_loss_mse_scale", loss_mse_scale.item(), batch_idx + epoch * len(loader))

    train_error /= float(data_num)
    acc_height = [float(c) / float(data_num) for c in correct_height]
    scheduler.step()  # note: can not write like this: scheduler.step(epoch), if so, retrain will have very low lr
    return acc_height, train_error


def test(model_, loader):
    model_.eval()
    torch.set_grad_enabled(False)
    test_error = 0
    correct_height = [0] * len(thresh_acc)
    correct_height_segment = [[0] * len(thresh_acc)] * 4

    data_num = 0
    data_num_segment = [0] * 4
    test_loss_mono = 0
    test_loss_mse = 0
    test_loss_mse_scale = 0
    res = []
    for batch_idx, all_data in enumerate(loader):
        target, target_scale, network_output = run_network(model_, all_data)
        if len(network_output) == 2:
            height, scale = network_output
            scale = scale * (scale_upper - scale_lower) + scale_lower
            test_loss_mse_scale += torch_func.mse_loss(scale, target_scale.view(-1, 1))
        else:
            height = network_output
        height = height * (bottle_upper - bottle_lower) + bottle_lower
        target = target.view(-1, 1)
        loss_mse, loss_mono = loss_function(height, target)
        test_loss_mse += loss_mse
        test_loss_mono += loss_mono

        res_acc, res_acc_segment, num_seg_part = compute_acc(height, target, is_train=False)
        correct_height = [c + r for c, r in zip(correct_height, res_acc)]
        for i in range(4):
            correct_height_segment[i] = [c + r for c, r in zip(correct_height_segment[i], res_acc_segment[i])]
            data_num_segment[i] += num_seg_part[i]
        data_num += target.shape[0]
        # compute average height error
        test_error += torch_func.l1_loss(height, target, reduction="sum")
        res.append(target)

    test_loss_mono = test_loss_mono / len(loader)
    test_loss_mse = test_loss_mse / len(loader)
    test_loss = test_loss_mse + args.mono_coe * test_loss_mono + test_loss_mse_scale

    test_error /= float(data_num)
    acc_height = [float(c) / float(data_num) for c in correct_height]
    for i in range(4):
        acc_height_segment = [float(c) / data_num_segment[i] for c in correct_height_segment[i]]
        print("acc segment {} is: {}".format(i, acc_height_segment))
    if args.draw_acc_fig:
        with open("/tmp/audio_pouring_acc_result.csv".format(args.acc_fig_name), "a") as f:
            f.write("{}, {} \n".format(args.acc_fig_name, str(acc_height)[1: -1]))
        print("Append {} to /tmp/audio_pouring_acc_result.csv".format(args.acc_fig_name))
    return acc_height, test_error, test_loss, test_loss_mse, test_loss_mono, test_loss_mse_scale


def main():
    train_loader = load_data(is_train=True)
    test_loader = load_data(is_train=False)
    if args.mode == "train":
        for epoch in range(is_resume * args.load_epoch, args.epoch):
            acc_train, train_error = train(model, train_loader, epoch)
            print("Train done, acc={}, train_error={}".format(acc_train, train_error))
            result_ = test(model, test_loader)
            acc_test, test_error, test_loss, test_loss_mse, test_loss_mono, test_loss_mse_scale = result_
            print("Test done, acc_test={}, test_error ={}, test_loss={}, test_loss_mse={}, test_loss_mono={}, "
                  "test_loss_mse_scale={}".format(acc_test, test_error, test_loss, test_loss_mse, test_loss_mono,
                                                  test_loss_mse_scale))
            for ind, acc in enumerate(thresh_acc):
                logger.add_scalar("train_acc{}".format(acc), acc_train[ind], epoch)
                logger.add_scalar("test_acc{}".format(acc), acc_test[ind], epoch)
            logger.add_scalar("train_error", train_error, epoch)
            logger.add_scalar("test_error", test_error, epoch)
            logger.add_scalar("test_loss", test_loss, epoch)
            logger.add_scalar("test_loss_mse", test_loss_mse, epoch)
            logger.add_scalar("test_loss_mono", test_loss_mono, epoch)

            if (epoch + 1) % args.save_interval == 0:
                path = os.path.join(args.model_path, "{}.model".format(args.tag))
                torch.save(model, path)
                print("Save model at {}, the epoch of this file is {}".format(path, epoch))
    else:
        print("testing...")
        acc_test, test_error, test_loss, test_loss_mse, test_loss_mono, test_loss_mse_scale = test(model, test_loader)
        print("Test done, acc_test={}, test_error ={}, test_loss={}, test_loss_mse={}, test_loss_mono={}, "
              "test_loss_mse_scale={}".format(acc_test, test_error, test_loss, test_loss_mse, test_loss_mono,
                                              test_loss_mse_scale))


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("run time is {}".format(time.time() - start_time))
