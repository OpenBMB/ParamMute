import argparse
import json
import os

import numpy as np
import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Llama_pruning_ffnForCausalLM, Qwen2_pruning_ffnForCausalLM, Llama_pruning_attnForInputContrastive, Llama_pruning_attnForCausalLM


parser = argparse.ArgumentParser()

parser.add_argument("--model_name_or_path", type=str, 
                    default=None)
parser.add_argument("--save_path", type=str,
                    default=None)

parser.add_argument("--model_type", type=str,
                    default='llama3')   # llama3_pruning_ffn llama3 llama3_pruning_attn
args = parser.parse_args()
config = PeftConfig.from_pretrained(args.model_name_or_path)
base_tokenizer =  AutoTokenizer.from_pretrained(config.base_model_name_or_path)

if args.model_type == 'llama3':
    model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": "cpu"},
        )
elif args.model_type == 'llama3_pruning_ffn':
    model = Llama_pruning_ffnForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": "cpu"},
        )
elif args.model_type == 'qwen2_pruning_ffn':
    model = Qwen2_pruning_ffnForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": "cpu"},
        )
elif args.model_type == 'llama3_pruning_attn':
    model = Llama_pruning_attnForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": "cpu"},
        )
else:   
    raise ValueError(f"model_type {args.model_type} not supported")
model = PeftModel.from_pretrained(model, args.model_name_or_path)

save_path = args.save_path
model = model.merge_and_unload()
model.save_pretrained(save_path)
base_tokenizer.save_pretrained(save_path)
