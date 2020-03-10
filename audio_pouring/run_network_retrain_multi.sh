#!/usr/bin/env bash
python main.py \
--cuda \
--gpu 0 \
--bottle-train with_plug \
--lstm \
--bs 32 \
--multi \
--multi-detail a_f_early_fusion \
--snr_db -1000 \
--tag 0128_retrain \
--load-model ./assets/learned_models/.model \
--load-epoch 500 \
--epoch 1000 \
--hidden-dim 256 \
