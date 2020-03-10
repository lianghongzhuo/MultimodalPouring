#!/usr/bin/env bash
python main.py \
--cuda \
--gpu 0 \
--bottle-train robot_pouring \
--lstm \
--multi \
--multi-detail 1loss2rnn \
--bs 64 \
--snr_db -1000 \
--tag 0219 \
--hidden-dim 64 \
--layer-num 2 \
--bidirectional \
