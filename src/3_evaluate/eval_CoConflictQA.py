import re
import ast
import string
import json
import jsonlines
import re
import argparse
from rouge import Rouge
from tqdm import tqdm
import os
import random
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig

import logging


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', 't', 'yes', 'y', '1'}:
        return True
    elif value.lower() in {'false', 'f', 'no', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def call_llama(model, tokenizer, input_res, max_new_tokens):
    with torch.no_grad():
        sequences = model.generate(input_res, max_new_tokens = max_new_tokens, pad_token_id=tokenizer.eos_token_id)[0, input_res.shape[-1]:]
        decoded = tokenizer.decode(sequences, skip_special_tokens=True)

        output_str = decoded.strip()

        return output_str
          
                
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))



def _exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def _acc_score(prediction, ground_truth):
    if normalize_answer(ground_truth) in normalize_answer(prediction):
        return 1.0
    else:
        return 0.0

def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(normalize_answer(prediction), normalize_answer(ground_truth), avg=True)
    except ValueError: 
        return 0.0
    return scores["rouge-l"]["f"]

def _f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_score(paramatric_answers, preds, golds):
    pm_acc_score, acc_score, rouge_score, f1_score, em_score = 0, 0, 0, 0, 0
    for pm_answer, pred, gold in zip(paramatric_answers, preds, golds):
        
        if isinstance(gold, list): 
            for g in gold:
                _acc, _rouge, _f1, _em = 0,0,0,0
                _acc = max(_acc_score(pred, g), _acc)
                _rouge = max(_rougel_score(pred, g), _rouge)
                _f1 = max(_f1_score(pred, g), _f1)
                _em = max(_exact_match_score(pred, g), _em)
        else:
            _acc = _acc_score(pred, gold)
            _rouge = _rougel_score(pred, gold)
            _f1 = _f1_score(pred, gold)
            _em = _exact_match_score(pred, gold)
    
        _pm_acc = _acc_score(pred, pm_answer)
        if _pm_acc == 1:
            _acc = 0

        pm_acc_score += _pm_acc
        acc_score += _acc
        rouge_score += _rouge
        f1_score += _f1
        em_score += _em

    pm_acc_score = pm_acc_score * 100 / (len(preds) + 1e-5)
    acc_score = acc_score * 100 / (len(preds) + 1e-5)
    rouge_score = rouge_score * 100 / (len(preds) + 1e-5)
    f1_score = f1_score * 100 / (len(preds) + 1e-5)
    em_score = em_score * 100 / (len(preds) + 1e-5)

    return pm_acc_score, acc_score, rouge_score, f1_score, em_score

def eval(parametric_answers, pred_answers, gold_answers, step):
    pm_acc, acc, rouge, f1, em = get_score(parametric_answers, pred_answers, gold_answers)

    mr = round((pm_acc / (pm_acc+acc+1e-5)) * 100, 2)
    acc = round(acc, 2)
    rouge = round(rouge, 2)
    f1 = round(f1, 2)
    em = round(em, 2)

    logging.info('Step: {}: pc {}, po {}, mr {}, em {}.'.format(step, acc, pm_acc, mr, em))
    return acc, pm_acc, mr, em

def qa_to_prompt(query, context):
    prompt = '{}\nQ: {}\nA: '.format(context, query)
    return prompt

def qa_to_prompt_baseline(query, context, schema, tokenizer, IS_INSTRUCTION_PROMPT):
    def get_prompt(query, context, schema, answer=''):
        if schema == 'base':
            prompt = '{}\nQ: {}\nA: {}'.format(context, query, answer)
        elif schema == 'opin':
            context = context.replace('"', "")
            prompt = 'Bob said "{}"\nQ: {} in Bob\'s opinion?\nA: {}'.format(context, query[:-1], answer)
        elif schema == 'instr+opin':
            context = context.replace('"', "")
            prompt = 'Bob said "{}"\nQ: {} in Bob\'s opinion?\nA:{}'.format(context, query[:-1], answer)
        elif schema == 'attr':
            prompt = '{}\nQ:{} based on the given tex?\nA:{}'.format(context, query[:-1], answer)
        elif schema == 'instr':
            prompt = '{}\nQ: {}\nA: {}'.format(context, query, answer)
        return prompt
    prompt = ''
    if schema in ('instr', 'instr+opin'):
        prompt = 'Instruction: read the given information and answer the corresponding question.\n\n'
    prompt = prompt + get_prompt(query, context, schema=schema)
    if IS_INSTRUCTION_PROMPT:
        prompt = [{'role': 'system', 'content': prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return prompt

    
def create_log_path(log_path):
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('') 
        logging.info(f"Log file {log_path} created.")
    else:
        logging.info(f"Log file {log_path} already exists.")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="./Models/Qwen2-7B-Instruct", type=str)
    parser.add_argument("--data_path", default="./ConFiQA/ConFiQA-QA.json", type=str)
    parser.add_argument("--schema", default="base", type=str, help="Choose from the following prompting templates: base, attr, instr, opin, instr+opin.")
    parser.add_argument("--output_path", default='./result/Qwen2-7B-Instruct.json', type=str)
    parser.add_argument("--log_path", default='./log_ConFiQA/Qwen2-7B-Instruct.log' , type=str)
    parser.add_argument("--model_type", default='llama3' , type=str)
    parser.add_argument('--use_chat_template', type=str2bool, nargs='?', default=False, help='Enable or disable the flag')
    parser.add_argument("--max_new_tokens", default=32, type=int)
    parser.add_argument("--act_inhibit_ratio", default=1.0, type=float)
    parser.add_argument("--act_inhibit_layer_list",
                            type=int,            # 或 float
                            nargs='+',           # 表示可接收多个值
                            default=[None],         # 默认值是 list
                            help="List of layers to inhibit"
                        )
    args = parser.parse_args()

    model_name = args.model_name

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_path),  
            logging.StreamHandler() 
        ]
    )
    
    logging.info("Evaluate Context-Faithfulness for the Model: %s" % model_name)
    logging.info(f"schema {args.schema}")
    

    with jsonlines.open(args.data_path, 'r') as reader:
        data = list(reader)

    logging.info('Loaded {} instances.'.format(len(data)))    
    create_log_path(args.log_path)
    
    gold_answers, pred_answers, all_pm_answers = [], [], []


    tokenizer_path = model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    logging.info(f"####### max_new_tokens :{args.max_new_tokens} #######")


    logging.info("####### using generate for infer #######")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.architectures= ['LlamaForCausalLM_w_act_inhibit']
    
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 device_map="auto", 
                                                 low_cpu_mem_usage = True, 
                                                 torch_dtype=torch.bfloat16, 
                                                 config=config,
                                                 trust_remote_code=True, 
                                                 inhibit_strength=args.act_inhibit_ratio, 
                                                 inhibit_layer_list=args.act_inhibit_layer_list)
    model.cuda()
    step = 0
    input_ids_list = []
    for d in data:
        query = d['question']
        context = d['context']
        prompt = qa_to_prompt_baseline(query, context, schema=args.schema, tokenizer=tokenizer, IS_INSTRUCTION_PROMPT=args.use_chat_template)

        input_ids_list.append(tokenizer(prompt, return_tensors='pt').input_ids)
    input_ids_list = [ids.to('cuda') for ids in input_ids_list]
    for data_idx, d in tqdm(enumerate(data), total=len(data)):
        step += 1
        answer = d['answers']
        parametric_answer = d['majority_res']
        all_pm_answers.append(parametric_answer)
        gold_answers.append(answer)
        
        pred = call_llama(model, tokenizer, input_ids_list[data_idx], max_new_tokens=args.max_new_tokens)
            
        pred_answers.append(pred)
        d['pred'] = pred
        
        if step % 500 == 0:
            eval(all_pm_answers, pred_answers, gold_answers, step)
    logging.info('final evaluation....')
    
    final_acc, final_rouge, final_f1, final_em = eval(all_pm_answers, pred_answers, gold_answers, step)
    final_acc, final_rouge, final_f1, final_em = round(final_acc, 2), round(final_rouge, 2), round(final_f1, 2), round(final_em, 2)
    logging.info(f'{final_acc}\t{final_rouge}\t{final_f1}\t{final_em}')
    with jsonlines.open(args.output_path, mode='w') as writer:
        for d in data:
            writer.write(d)
        

if __name__ == '__main__':
    main()
