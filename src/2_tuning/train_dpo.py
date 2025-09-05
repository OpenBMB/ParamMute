import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from datasets import Dataset
from functools import partial
import logging
from trl import DPOTrainer, DPOConfig
import transformers
import json
from dataclasses import dataclass, field
from typing import Dict, Optional
from datasets import load_dataset, Dataset
import torch
import transformers
# import Accelerator
from accelerate import Accelerator

logger = logging.getLogger(__name__)
from peft import PeftConfig, PeftModel
# MAX_PROMPT_LENGTH = 32

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/data3/yaosijia/models/Meta-Llama-3-8B-Instruct")
    use_template: bool = field(default=True)


@dataclass
class DataArguments:
    train_data_path: str = field(
        default="/data3/yaosijia/Dataset/MS_MARCO/train/llama-gen/query2cot/dpo-data-sep/train_data.json",
        metadata={"help": "Path to the training data."},
    )
    # eval_data_path: str = field(
    #     default="/data3/yaosijia/Dataset/MS_MARCO/train/llama-gen/query2cot/dpo-data-sep/dev_data.json",
    #     metadata={"help": "Path to the test data."},
    # )
    
    max_length: int = field(default=1024,metadata={"help":"Maximum all sequence length."},)
    max_prompt_length: int = field(default=768,metadata={"help":"Maximum prompt sequence length."},)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    load_lora_model : bool = field(default=True)

    use_lora: bool = field(default=True)
    output_dir : str = field(default="/data3/yaosijia/DPO-hyde/dpo-llama3/dpo-new")
    save_steps : int = field(default=100)
    eval_steps : int = field(default=100)
    per_device_train_batch_size: int = field(default=1)
    evaluation_strategy: str = field(default='steps')
    logging_steps : int = field(default=10)
    logging_dir : str = field(default="/data3/yaosijia/DPO-hyde/dpo-llama3/logs")
    bf16 : bool = field(default=True)
    learning_rate: float = field(default=1e-5, metadata={"help": "The initial learning rate for AdamW."})
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )



def load_model_and_tokenizer(
    model_path: str,
    use_lora: bool = True,
    bf16: bool = False,
    fp16: bool = False,
    load_lora_model: bool = False,
):
    """load model and tokenizer"""


    assert not (bf16 and fp16), "bf16 or fp16, not both"
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


    tokenizer.pad_token = tokenizer.eos_token
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        # 
        target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=64,
                    target_modules=target_modules,
                    lora_alpha=64,
                    lora_dropout=0,
                    inference_mode=False,
                    bias="none",
                )
       
       
        # r=256,
        # lora_alpha=512,
        model = get_peft_model(model, lora_config)
        # trainable params: 2,949,120 || all params: 3,010,652,928 || trainable%: 0.09795616002669305
        model.print_trainable_parameters()
        # model.enable_input_require_grads()  # need when using adapter

    return model, tokenizer

def preprocessing(example,args,tokenizer):
        one_item = {}
        # query = example['conversations'][0]['value']    

        prompt = [{"role": "user", "content": example['prompt']},]
        prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
                
        one_item["prompt"] = prompt
        one_item["chosen"] = example['chosen']
        one_item["rejected"] = example['rejected']

        return one_item

if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)
    logger.info("DATA parameters %s", data_args)



    model, tokenizer = load_model_and_tokenizer(
        model_path=model_args.model_name_or_path,
        use_lora=training_args.use_lora,
        bf16=training_args.bf16,
        fp16=training_args.fp16,
        load_lora_model =training_args.load_lora_model
    )

    # model_ref, _ = load_model_and_tokenizer(
    #     model_path=model_args.model_name_or_path,
    #     use_lora=training_args.use_lora,
    #     bf16=training_args.bf16,
    #     fp16=training_args.fp16,
    #     load_lora_model =training_args.load_lora_model
    # )

    # train_dataset = SupervisedDataset(
    #     data_path=data_args.train_data_path,
    #     tokenizer=tokenizer,
    #     model_max_length=training_args.model_max_length,
    # )
    # eval_dataset = SupervisedDataset(
    #     data_path=data_args.eval_data_path,
    #     tokenizer=tokenizer,
    #     model_max_length=training_args.model_max_length,
    # )
    
    partial_preprocess = partial(preprocessing,args=data_args,tokenizer=tokenizer)

    train_dataset = load_dataset("json", data_files=data_args.train_data_path,split="train",)
    train_dataset = train_dataset.map(partial_preprocess)

    # eval_dataset = load_dataset("json", data_files=data_args.eval_data_path,split="train",)
    # eval_dataset = eval_dataset.map(partial_preprocess)

    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=DPOConfig(
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            warmup_ratio=training_args.warmup_ratio,
            num_train_epochs=training_args.num_train_epochs,
            lr_scheduler_type=training_args.lr_scheduler_type,
            gradient_checkpointing=training_args.gradient_checkpointing,
            weight_decay=training_args.weight_decay,
            learning_rate=training_args.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            save_steps=training_args.save_steps,
            # optim="adamw_8bit",
            seed=42,
            output_dir=training_args.output_dir,
            report_to=training_args.report_to,
            logging_dir=training_args.logging_dir,
            # logging_steps=training_args.logging_steps
        ),
        beta=0.1,
        train_dataset=train_dataset,
        max_length = data_args.max_length,
        max_prompt_length = data_args.max_prompt_length,
        tokenizer=tokenizer,

    )
    # dpo_trainer.ref_model = Accelerator().prepare(dpo_trainer.ref_model)
    dpo_trainer.train()
    dpo_trainer.save_model()




