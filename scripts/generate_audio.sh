#!/bin/bash
python3 inference.py \
    --ckpt checkpoint/drum/fusion/drum_fusion.pt \
    --pics 20 \
    --data_path data/drum --store_path ./generated_sample \
    --vocoder_folder melgan/drum_vocoder 