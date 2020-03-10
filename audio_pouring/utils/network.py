#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang 
# E-mail     : liang@informatik.uni-hamburg.de
# Description: 
# Date       : 15/10/2019: 22:13
# File Name  : network
import argparse
import numpy as np
import torch


def worker_init_fn(pid):
    np.random.seed(torch.initial_seed() % (2 ** 31 - 1))


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def parse():
    parser = argparse.ArgumentParser(description="audio2height")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--bs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layer-num", type=int, default=1)
    parser.add_argument("--lstm", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bottle-train", type=str, default="0")
    parser.add_argument("--bottle-test", type=str, default="")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--snr_db", type=float, required=True)
    parser.add_argument("--mono-coe", type=float, default=0.001)
    parser.add_argument("--load-model", type=str, default="")
    parser.add_argument("--load-epoch", type=int, default=-1)
    parser.add_argument("--model-path", type=str, default="./assets/learned_models", help="pre-trained model path")
    parser.add_argument("--data-path", type=str, default="h5py_dataset", help="data path")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--robot", action="store_true")
    parser.add_argument("--multi", action="store_true")
    parser.add_argument("--minus_wrench_first", action="store_true")
    parser.add_argument("--stft_force", action="store_true")
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--draw_acc_fig", action="store_true")
    parser.add_argument("--acc_fig_name", type=str, default="")
    parser.add_argument("--multi-detail", choices=["2loss2rnn", "2loss1rnn", "1loss1rnn", "audio_only", "a_guide_f",
                                                   "a_f_early_fusion", "force_only", "1loss2rnn"], default="audio_only")

    args = parser.parse_args()
    if args.bottle_test == "":
        args.bottle_test = args.bottle_train
    if args.tag != "":
        args.tag += "_"
    base = args.tag + "{}_{}{}_h{}_bs{}_bottle{}to{}_mono_coe{}_snr{}_{}_{}_{}_{}"
    tag = base.format("multi" if args.multi else "audio", "lstm" if args.lstm else "gru", args.layer_num,
                      args.hidden_dim, args.bs, args.bottle_train, args.bottle_test, args.mono_coe, args.snr_db,
                      args.multi_detail, "minus_wrench_first" if args.minus_wrench_first else "raw",
                      "stft_force" if args.stft_force else "raw_force",
                      "bidirectional" if args.bidirectional else "unidirectional")
    args.tag = tag
    args.acc_fig_name = "snr{}_{}".format(args.snr_db, "lstm" if args.lstm else "gru")
    return args
