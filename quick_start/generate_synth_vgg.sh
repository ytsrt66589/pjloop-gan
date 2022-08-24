#!/bin/bash
python3 inference.py \
    --ckpt checkpoint/synth/vgg/synth_vgg.pt \
    --pics 20 \
    --data_path data/synth --store_path ./generated_sample \
    --vocoder_folder melgan/synth_vocoder 