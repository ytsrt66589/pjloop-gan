#!/bin/bash
python3 inference.py \
    --ckpt checkpoint/drum/vgg/drum_vgg.pt \
    --pics 20 \
    --data_path data/drum --store_path ./generated_sample \
    --vocoder_folder melgan/drum_vocoder 