#!/usr/bin/env bash
python main.py \
--cuda \
--gpu 1 \
--bottle-train with_plug \
--lstm \
--bs 32 \
--snr_db -1000 \
--tag 0129_retrain_no_epcho \
--load-model ./assets/learned_models/0128_audio_lstm1_h256_bs32_bottle0to0_mono_coe0.001_snr-1000.0_audio_only_raw.model \
--load-epoch 500 \
--epoch 1000 \
--hidden-dim 256 \
