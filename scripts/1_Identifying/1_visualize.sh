#!/bin/bash

# Make sure your work dir in /PIP-KAG/scripts/1_pip_uninstall
cd ../../src/1_Identifying


CUDA_VISIVLE_DEVICES=0 python3 1_visualize.py \
    --in_file_path ../../data/func_data/draw_acctivations_shuf1k.jsonl \
    --visualize_path $visualize_res.png$ \
    --pretrained_model_path $path of the pretrained model$


