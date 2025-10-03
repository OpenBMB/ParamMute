#!/bin/bash

# Make sure your work dir in /PIP-KAG/scripts/1_pip_uninstall
cd ../../src/1_Identifying

echo "================================="
echo "🚀  Start running 1_visualize.py"
echo "================================="
# The input file (in_file_path) must be in JSONL format.
# Each line corresponds to one data instance, with the following structure:
# {
#   "context": "The supporting passage or background information provided to the model.",
#   "question": "The user’s query based on the given context.",
#   "pred": "The model’s generated answer to the question.",
#   "prompt_w_context": "The exact prompt shown to the model, including the context.",
#   "is_faithful": "A label indicating whether the model’s answer is faithful to the provided context."
# }

CUDA_VISIBLE_DEVICES=0 python3 1_visualize.py \
    --in_file_path ../../data/func_data/draw_acctivations_shuf1k.jsonl \
    --visualize_path $visualize_res.png$ \
    --pretrained_model_path $path of the pretrained model$