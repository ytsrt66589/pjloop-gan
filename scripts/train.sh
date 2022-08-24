#!/bin/bash
python3 train.py \
    --size 64 --batch 64 --sample_dir ./test_train  \
    --checkpoint_dir test_ck --loops_type synth \
    --fusion "on" --pre_trained_model_type "loops_genre" --pre_trained_model_path feature_networks/checkpoints/synth_genre/synth_scnn.ckpt \
    [/your/own/data/path]