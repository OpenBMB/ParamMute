from torch.utils.data import Dataset
# import datasets
from datasets import load_dataset 
import transformers
import torch
from typing import List, Dict, Any


class SFTDataset(Dataset):
    def __init__(self, data_path , tokenizer, model_type, max_len):
        self.dataset = load_dataset('json', data_files=data_path, split='train')
        self.tokenizer = tokenizer
        self.max_len = max_len

        if 'llama' in model_type.lower():
            print(f'llama3 dataset')
            self.system_format='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
            self.user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            self.assistant_format='{content}<|eot_id|>\n'
        elif 'qwen' in model_type.lower():
            print(f'qwen2 dataset')
            self.system_format=''
            self.user_format='<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n'
            self.assistant_format='{content}<|im_end|>\n'

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        input_ids, target_mask = [], []
        
        # human = data['rag_input']
        # assistant = data['output']

        human = self.user_format.format(content=data['rag_input'])
        assistant = self.assistant_format.format(content=data['output'])

        # llama3没有system
        input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
        output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

        input_ids += input_tokens + output_tokens
        target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        assert len(input_ids) == len(target_mask)
        input_ids = input_ids[:self.max_len]
        target_mask = target_mask[:self.max_len]
        attention_mask = [1] * len(input_ids)

        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': target_mask,
        }

        # import pdb; pdb.set_trace()
        return inputs
    

class SFTDataCollator:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 找出batch中的最大长度
        lengths = [len(x['input_ids']) for x in batch if x['input_ids'] is not None]
        # 取出batch中的最大长度，如果超过max_seq_length，则取max_seq_length
        batch_max_len = min(max(lengths), self.max_seq_length)
        # batch_max_len = self.max_seq_length

        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []
        # truncate and padding
        for x in batch:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            # import pdb; pdb.set_trace()
            target_mask = x['labels']
            
            if input_ids is None:
                print('some input_ids is None')
                continue
            padding_len = batch_max_len - len(input_ids)
            # padding
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # truncate
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            target_mask = target_mask[:self.max_seq_length]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)

        # 将list转换为tensor，得到最终的的模型输入
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)

        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels
        }
        return inputs
    


class InputContravasiveDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, model_type: str, model_max_length: int):
        self.dataset = load_dataset('json', data_files=data_path, split='train')
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length

        if 'llama' in model_type.lower():
            print(f'llama3 dataset')
            self.system_format='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
            self.user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            self.assistant_format='{content}<|eot_id|>\n'
        elif 'qwen' in model_type.lower():
            print(f'qwen2 dataset')
            self.system_format=''
            self.user_format='<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n'
            self.assistant_format='{content}<|im_end|>\n'

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 每条数据拼接格式为：f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content'].strip()}<|eot_id|>"
        # 示例
        """
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>

        你好<|eot_id|>"""
        data = self.dataset[idx]
        rag_input_ids, raw_input_ids, rag_labels, raw_labels = [], [], [], []
        # import pdb; pdb.set_trace()

        # rag_human = data['rag_input']
        # rag_assistant = data['output']
        # raw_human = data['raw_input']
        # raw_assistant = data['output']

        rag_human  = self.user_format.format(content=data['rag_input'])
        rag_assistant = self.assistant_format.format(content=data['output'])
        raw_human = self.user_format.format(content=data['raw_input'])
        raw_assistant = self.assistant_format.format(content=data['output'])
        
        rag_input_tokens = self.tokenizer.encode(rag_human, add_special_tokens=False)
        rag_output_tokens = self.tokenizer.encode(rag_assistant, add_special_tokens=False)
        raw_input_tokens = self.tokenizer.encode(raw_human, add_special_tokens=False)
        raw_output_tokens = self.tokenizer.encode(raw_assistant, add_special_tokens=False)

        rag_input_ids = rag_input_tokens + rag_output_tokens
        rag_target_mask = [0] * len(rag_input_tokens) + [1] * len(rag_output_tokens)
        raw_input_ids = raw_input_tokens + raw_output_tokens
        raw_target_mask = [0] * len(raw_input_tokens) + [1] * len(raw_output_tokens)

        assert len(rag_input_ids) == len(rag_target_mask)
        assert len(raw_input_ids) == len(raw_target_mask)

        rag_input_ids = rag_input_ids[:self.model_max_length]
        raw_input_ids = raw_input_ids[:self.model_max_length]
        rag_target_mask = rag_target_mask[:self.model_max_length]
        raw_target_mask = raw_target_mask[:self.model_max_length]
        
        rag_attention_mask = [1] * len(rag_input_ids)
        raw_attention_mask = [1] * len(raw_input_ids)
        assert len(rag_input_ids) == len(rag_target_mask) == len(rag_attention_mask) 
        assert len(raw_input_ids) == len(raw_target_mask) == len(raw_attention_mask)
        rag_inputs = {
            'input_ids': rag_input_ids,
            'attention_mask': rag_attention_mask,
            'target_mask': rag_target_mask
        }
        raw_inputs = {
            'input_ids': raw_input_ids,
            'attention_mask': raw_attention_mask,
            'target_mask': raw_target_mask
        }
        return rag_inputs, raw_inputs

# 3. collator
class InputContravasiveCollator:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, model_max_length: int):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        rag_inputs = [f[0] for f in features]
        raw_inputs = [f[1] for f in features]

        rag_max_lengths = [len(x['input_ids']) for x in rag_inputs if x['input_ids'] is not None]
        rag_batch_max_len = min(max(rag_max_lengths), self.model_max_length)

        rag_input_ids_batch, rag_attention_mask_batch, rag_target_mask_batch = [], [], []        
        for x in rag_inputs:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']
            if input_ids is None:
                print('some input_ids is None')
                continue
            padding_len = rag_batch_max_len - len(input_ids)
            # padding
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # truncate
            input_ids = input_ids[:self.model_max_length]
            attention_mask = attention_mask[:self.model_max_length]
            target_mask = target_mask[:self.model_max_length]

            rag_input_ids_batch.append(input_ids)
            rag_attention_mask_batch.append(attention_mask)
            rag_target_mask_batch.append(target_mask)

        rag_input_ids_batch = torch.tensor(rag_input_ids_batch, dtype=torch.long)
        rag_attention_mask_batch = torch.tensor(rag_attention_mask_batch, dtype=torch.long)
        rag_target_mask_batch = torch.tensor(rag_target_mask_batch, dtype=torch.long)
        
        rag_labels = torch.where(rag_target_mask_batch == 1, rag_input_ids_batch, -100)
        
        # 处理raw部分
        raw_max_lengths = [len(x['input_ids']) for x in raw_inputs if x['input_ids'] is not None]
        raw_batch_max_len = min(max(raw_max_lengths), self.model_max_length)

        # truncate and pad
        raw_input_ids_batch, raw_attention_mask_batch, raw_target_mask_batch = [], [], []        
        # truncate and padding
        for x in raw_inputs:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']
            if input_ids is None:
                print('some input_ids is None')
                continue
            padding_len = raw_batch_max_len - len(input_ids)
            # padding
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # truncate
            input_ids = input_ids[:self.model_max_length]
            attention_mask = attention_mask[:self.model_max_length]
            target_mask = target_mask[:self.model_max_length]

            raw_input_ids_batch.append(input_ids)
            raw_attention_mask_batch.append(attention_mask)
            raw_target_mask_batch.append(target_mask)

        # 将list转换为tensor，得到最终的的模型输入
        raw_input_ids_batch = torch.tensor(raw_input_ids_batch, dtype=torch.long)
        raw_attention_mask_batch = torch.tensor(raw_attention_mask_batch, dtype=torch.long)
        raw_target_mask_batch = torch.tensor(raw_target_mask_batch, dtype=torch.long)

        raw_labels = torch.where(raw_target_mask_batch == 1, raw_input_ids_batch, -100)
        return {
            'rag_input_ids': rag_input_ids_batch,
            'rag_attention_mask': rag_attention_mask_batch,
            'rag_labels': rag_labels,
            'raw_input_ids': raw_input_ids_batch,
            'raw_attention_mask':raw_attention_mask_batch,            
            'raw_labels': raw_labels
        }


class BothContravasiveDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, model_type: str, model_max_length: int):
        self.dataset = load_dataset('json', data_files=data_path, split='train')
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length

        if 'llama3' in model_type:
            print(f'llama3 dataset')
            self.system_format='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
            self.user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            self.assistant_format='{content}<|eot_id|>\n'
        elif 'qwen2' in model_type:
            print(f'qwen2 dataset')
            self.system_format=''
            self.user_format='<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n'
            self.assistant_format='{content}<|im_end|>\n'

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 每条数据拼接格式为：f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content'].strip()}<|eot_id|>"
        # 示例
        """
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>

        你好<|eot_id|>"""
        data = self.dataset[idx]
        rag_input_ids, raw_input_ids, rag_labels, raw_labels = [], [], [], []
        # import pdb; pdb.set_trace()

        # rag_human = data['rag_input']
        # rag_assistant = data['output']
        # raw_human = data['raw_input']
        # raw_assistant = data['output']

        rag_human  = self.user_format.format(content=data['rag_input'])
        rag_assistant = self.assistant_format.format(content=data['rag_output'])
        raw_human = self.user_format.format(content=data['raw_input'])
        raw_assistant = self.assistant_format.format(content=data['raw_output'])
        
        rag_input_tokens = self.tokenizer.encode(rag_human, add_special_tokens=False)
        rag_output_tokens = self.tokenizer.encode(rag_assistant, add_special_tokens=False)
        raw_input_tokens = self.tokenizer.encode(raw_human, add_special_tokens=False)
        raw_output_tokens = self.tokenizer.encode(raw_assistant, add_special_tokens=False)

        rag_input_ids = rag_input_tokens + rag_output_tokens
        rag_target_mask = [0] * len(rag_input_tokens) + [1] * len(rag_output_tokens)
        raw_input_ids = raw_input_tokens + raw_output_tokens
        raw_target_mask = [0] * len(raw_input_tokens) + [1] * len(raw_output_tokens)

        assert len(rag_input_ids) == len(rag_target_mask)
        assert len(raw_input_ids) == len(raw_target_mask)

        rag_input_ids = rag_input_ids[:self.model_max_length]
        raw_input_ids = raw_input_ids[:self.model_max_length]
        rag_target_mask = rag_target_mask[:self.model_max_length]
        raw_target_mask = raw_target_mask[:self.model_max_length]
        
        rag_attention_mask = [1] * len(rag_input_ids)
        raw_attention_mask = [1] * len(raw_input_ids)
        assert len(rag_input_ids) == len(rag_target_mask) == len(rag_attention_mask) 
        assert len(raw_input_ids) == len(raw_target_mask) == len(raw_attention_mask)
        rag_inputs = {
            'input_ids': rag_input_ids,
            'attention_mask': rag_attention_mask,
            'target_mask': rag_target_mask
        }
        raw_inputs = {
            'input_ids': raw_input_ids,
            'attention_mask': raw_attention_mask,
            'target_mask': raw_target_mask
        }
        return rag_inputs, raw_inputs

# 3. collator
class BothContravasiveCollator:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, model_max_length: int):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        rag_inputs = [f[0] for f in features]
        raw_inputs = [f[1] for f in features]

        rag_max_lengths = [len(x['input_ids']) for x in rag_inputs if x['input_ids'] is not None]
        rag_batch_max_len = min(max(rag_max_lengths), self.model_max_length)

        rag_input_ids_batch, rag_attention_mask_batch, rag_target_mask_batch = [], [], []        
        for x in rag_inputs:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']
            if input_ids is None:
                print('some input_ids is None')
                continue
            padding_len = rag_batch_max_len - len(input_ids)
            # padding
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # truncate
            input_ids = input_ids[:self.model_max_length]
            attention_mask = attention_mask[:self.model_max_length]
            target_mask = target_mask[:self.model_max_length]

            rag_input_ids_batch.append(input_ids)
            rag_attention_mask_batch.append(attention_mask)
            rag_target_mask_batch.append(target_mask)

        rag_input_ids_batch = torch.tensor(rag_input_ids_batch, dtype=torch.long)
        rag_attention_mask_batch = torch.tensor(rag_attention_mask_batch, dtype=torch.long)
        rag_target_mask_batch = torch.tensor(rag_target_mask_batch, dtype=torch.long)
        
        rag_labels = torch.where(rag_target_mask_batch == 1, rag_input_ids_batch, -100)
        
        # 处理raw部分
        raw_max_lengths = [len(x['input_ids']) for x in raw_inputs if x['input_ids'] is not None]
        raw_batch_max_len = min(max(raw_max_lengths), self.model_max_length)

        # truncate and pad
        raw_input_ids_batch, raw_attention_mask_batch, raw_target_mask_batch = [], [], []        
        # truncate and padding
        for x in raw_inputs:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']
            if input_ids is None:
                print('some input_ids is None')
                continue
            padding_len = raw_batch_max_len - len(input_ids)
            # padding
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # truncate
            input_ids = input_ids[:self.model_max_length]
            attention_mask = attention_mask[:self.model_max_length]
            target_mask = target_mask[:self.model_max_length]

            raw_input_ids_batch.append(input_ids)
            raw_attention_mask_batch.append(attention_mask)
            raw_target_mask_batch.append(target_mask)

        # 将list转换为tensor，得到最终的的模型输入
        raw_input_ids_batch = torch.tensor(raw_input_ids_batch, dtype=torch.long)
        raw_attention_mask_batch = torch.tensor(raw_attention_mask_batch, dtype=torch.long)
        raw_target_mask_batch = torch.tensor(raw_target_mask_batch, dtype=torch.long)

        raw_labels = torch.where(raw_target_mask_batch == 1, raw_input_ids_batch, -100)
        return {
            'rag_input_ids': rag_input_ids_batch,
            'rag_attention_mask': rag_attention_mask_batch,
            'rag_labels': rag_labels,
            'raw_input_ids': raw_input_ids_batch,
            'raw_attention_mask':raw_attention_mask_batch,            
            'raw_labels': raw_labels
        }
