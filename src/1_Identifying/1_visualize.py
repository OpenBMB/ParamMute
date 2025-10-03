import torch
import jsonlines
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Patch  
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in_file_path",
        type=str,
        required=True,
        help="input data path"
    )

    parser.add_argument(
        "--visualize_path",
        type=str,
        required=True,
        help="visualize result path"
    )

    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        required=True,
        help="model path"
    )

    return parser.parse_args()


class ActivationCollector:
    def __init__(self, model, tokenizer, model_type, num_layers, num_neurons, device):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.device = device
        self.reset()

    def reset(self):
        num_layers = self.num_layers
        self.layer_flag = 0
        self.save_flag = False
        self.data_idx = 0

        self.activation_matrix_total = [None] * num_layers
        self.activation_matrix_common = [None] * num_layers
        self.activation_matrix = [None] * num_layers
        self.max_activate = [None] * num_layers
        self.min_activate = [None] * num_layers
        self.avg_activate = [None] * num_layers

    def mlp_forward_hook(self, layer_idx):
        original_mlp_forward = self.model.model.layers[layer_idx].mlp.forward

        def hooked_forward(x):
            if self.save_flag:
                activations = self.model.model.layers[layer_idx].mlp.act_fn(
                    self.model.model.layers[layer_idx].mlp.gate_proj(x)
                )
                actmean = (activations[:, self.data_idx:, :]).sum(dim=1, keepdim=True)

                act_bin = (actmean > 0).squeeze().squeeze()

                if self.activation_matrix_total[layer_idx] is None:
                    self.activation_matrix_total[layer_idx] = act_bin.clone()
                    self.activation_matrix_common[layer_idx] = act_bin.clone()
                    self.activation_matrix[layer_idx] = act_bin.int().clone()
                    self.max_activate[layer_idx] = act_bin.sum()
                    self.min_activate[layer_idx] = act_bin.sum()
                    self.avg_activate[layer_idx] = act_bin.sum()
                else:
                    self.activation_matrix_total[layer_idx] |= act_bin
                    self.activation_matrix_common[layer_idx] &= act_bin
                    self.activation_matrix[layer_idx] += act_bin.int()
                    self.max_activate[layer_idx] = max(act_bin.sum(), self.max_activate[layer_idx])
                    self.min_activate[layer_idx] = min(act_bin.sum(), self.min_activate[layer_idx])
                    self.avg_activate[layer_idx] += act_bin.sum()
                down_proj = self.model.model.layers[layer_idx].mlp.down_proj(
                    activations * self.model.model.layers[layer_idx].mlp.up_proj(x)
                )

                self.layer_flag = (self.layer_flag + 1) % self.num_layers
            else:
                down_proj = original_mlp_forward(x)
            return down_proj
        return hooked_forward

    def attach_hooks(self):
        for layer_idx in range(self.num_layers):
            self.model.model.layers[layer_idx].mlp.forward = self.mlp_forward_hook(layer_idx)

    def find_output_start_idx(self, token_ids):
        sublists = {
            'llama32': [128006, 78191, 128007, 271],
            'llama3': [128006, 78191, 128007, 271],
            'qwen25': [151644, 77091, 198],
            'llama31': [128000, 128006, 882, 128007, 271],
            'gemma3': [105, 4368, 107],
        }
        sublist = sublists.get(self.model_type, [])
        start_indices = [
            i for i in range(len(token_ids) - len(sublist) + 1)
            if token_ids[i:i+len(sublist)] == sublist
        ]
        return start_indices[-1] + len(sublist) if start_indices else 0

    
    def evaluate_per_example(self, in_file_path):
        self.attach_hooks()
        res_data = []
        cnt =  0
        with jsonlines.open(in_file_path) as reader:
            all_datas = list(reader)
            
            for data in tqdm(all_datas, desc='Processing data'):
                cnt +=1 
                cur_context = data['context']
                cur_question = data['question']
                cur_output = data['pred']
        
                cur_input_w_context = data['prompt_w_context']

                modes = {
                    'ww': [{'role': 'user', 'content': cur_input_w_context}, {'role': 'assistant', 'content': cur_output}],
                }
                per_data_results = {}

                for mode_name, cur_data in modes.items():
                    self.reset()

                    cur_data_tokens = self.tokenizer.apply_chat_template(cur_data, tokenize=False)
                    cur_data_ids = self.tokenizer.apply_chat_template(cur_data, tokenize=True)                    
                    self.data_idx = self.find_output_start_idx(cur_data_ids)
                    input_ids = self.tokenizer(cur_data_tokens, return_tensors='pt').to(self.device)
                    self.save_flag = True
                    with torch.inference_mode():
                        self.model(**input_ids)
                        
                    if cnt % 2000 == 0:
                        torch.cuda.empty_cache() 
                        gc.collect()  
                    avg_activations_per_layer = [
                        (m.item() / self.num_neurons)
                        if m is not None else None
                        for m in self.avg_activate
                    ]

                    per_data_results[mode_name] = avg_activations_per_layer
                data.update({
                    'activations_avg': per_data_results,
                })
                
                res_data.append(data)
                # writer.write(data)
        return res_data
    
def list_add(list1, list2):
    return [x + y for x, y in zip(list1, list2)]


  
def draw_pic(all_datas, num_layers, num_neurons, save_path):
    right_ww_list = [0]*num_layers
    wrong_ww_list = [0]*num_layers
    avg_list = [0]*num_layers
    valid_len = 0
    right_len = 0
    wrong_len = 0
    
    for data in all_datas:
        valid_len += 1
        cur_ww_list = data['activations_avg']['ww']
    
        if data['is_faithful'] == 0:
            wrong_len += 1
            wrong_ww_list = list_add(cur_ww_list, wrong_ww_list)
        elif data['is_faithful'] == 1:
            right_len += 1
            right_ww_list = list_add(cur_ww_list, right_ww_list)
            
    right_ww_list = [tmp / right_len for tmp in right_ww_list]
    wrong_ww_list = [tmp / wrong_len for tmp in wrong_ww_list]
    x = list(range(0, len(right_ww_list) ))
    diff_list = [(w - r)*10 for w, r in zip(wrong_ww_list, right_ww_list)]
    plt.figure(figsize=(10,6))
    print(f'Unfaithful acctivations:')
    print([f"{x:.4f}" for x in wrong_ww_list])
    print(f'Faithful acctivations:')
    print([f"{x:.4f}" for x in right_ww_list])
    print(f'Difference acctivations:')
    print([f"{x:.4f}" for x in diff_list])
    colors = ['#c6dbef'] * num_layers
    plt.bar(x, diff_list, color=colors, width=0.6,edgecolor='#6baed6', linewidth=0.7)
    plt.plot(x, wrong_ww_list,
            linestyle='-',
            marker='^',
            #  color='#2c6fb1',
            color='#2D5A96',
            markerfacecolor='white',
            markeredgewidth=1.2,
            markersize=6.5,
            linewidth=1.8,
            label='Unfaithful response')
    plt.plot(x, right_ww_list,
            linestyle='--',
            marker='o',
            color='#e4503f',
            markerfacecolor='white',
            markeredgewidth=1.2,
            markersize=6,
            linewidth=1.8,
            label='Faithful response')
    plt.xlabel("Layers", fontsize=24)
    plt.ylabel("Neuron Activation Ratio", fontsize=24)

    from matplotlib.patches import Patch 

    bar_legend = Patch(facecolor='#c6dbef', edgecolor='#6baed6', linewidth=0.8, label=r'$\Delta R^l$')

    # 2. 获取当前折线图的图例元素（自动生成的）
    line_handles, line_labels = plt.gca().get_legend_handles_labels()

    # 3. 合并图例，把柱状图图例放在最后
    plt.legend(
        handles=line_handles + [bar_legend],  # 折线在前，柱状图在后
        fontsize=22,
        loc='upper left',
        frameon=False
    )
    # plt.legend(fontsize=18)
    plt.grid(axis='both', linestyle='--', linewidth=1.2, color='#c2c2c2', alpha=0.5)
    plt.gca().yaxis.grid(True, linestyle='--', linewidth=1.2, color='#c2c2c2', alpha=0.5, dashes=(5, 7))
    plt.gca().xaxis.grid(True, linestyle='--', linewidth=1.2, color='#c2c2c2', alpha=0.5, dashes=(5, 7))

    if 'png' in save_path.split('.')[-1]:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    elif 'pdf' in save_path.split('.')[-1]:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
  
    # plt.show()

if __name__ == "__main__":
    import sys

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()

    in_file_path = args.in_file_path
    visualize_path = args.visualize_path
    pretrained_model_path = args.pretrained_model_path
    

    model_type = 'llama3'

    model = AutoModelForCausalLM.from_pretrained(pretrained_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,device_map="balanced")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, trust_remote_code=True)
    num_layers = model.config.num_hidden_layers
    num_neurons = model.config.intermediate_size

    collector = ActivationCollector(model, tokenizer, model_type, num_layers, num_neurons, device)

    res_data = collector.evaluate_per_example(in_file_path)
    draw_pic(res_data, num_layers, num_neurons, visualize_path)
    
   
