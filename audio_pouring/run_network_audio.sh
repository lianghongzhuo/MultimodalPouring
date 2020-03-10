#!/usr/bin/env bash
python main.py \
--cuda \
--gpu 0 \
--bottle-train robot_pouring \
--lstm \
--bs 64 \
--snr_db 20 \
--tag 0209 \
--hidden-dim 64 \
--layer-num 2 \
--bidirectional \
