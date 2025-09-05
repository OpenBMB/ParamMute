from transformers import TrainingArguments, Trainer, HfArgumentParser
import logging
import torch
import os


from arguments import ModelArguments, DataArguments, EasyTrainArguments as TrainingArguments
from utils import find_all_linear_modules, load_model_and_tokenizer
from dataset import InputContravasiveDataset, InputContravasiveCollator, SFTDataset, SFTDataCollator, BothContravasiveDataset, BothContravasiveCollator
from trainer import InputContrastiveTrainer, SNIPCallback, InputContrastiveTrainer_debug,LogAdditionalLossUsingSwanLab

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    os.environ["WANDB_PROJECT"] = training_args.wandb_project
    # os.environ["WANDB_NAME"] = training_args.wandb_run_name

 
    if training_args.report_to != []:
        if isinstance(training_args.report_to, list) and "," in training_args.report_to[0]:
            training_args.report_to = training_args.report_to[0].split(",")
    
    model, tokenizer = load_model_and_tokenizer(model_args, training_args)

    if training_args.train_mode == 'sft':
        logger.info('######## sft training Dataset & collator ########')
        # print('######## sft training Dataset & collator ########')
        train_dataset = SFTDataset(data_args.train_file, tokenizer, training_args.model_type, data_args.max_len)
        # import pdb; pdb.set_trace()
        train_collator = SFTDataCollator(tokenizer, data_args.max_len)
    elif training_args.train_mode == 'input_contrastive':
        logger.info('######## input_contrastive training Dataset & collator ########')
        # print('######## input_contrastive training Dataset & collator ########')
        train_dataset = InputContravasiveDataset(data_args.train_file, tokenizer, training_args.model_type, data_args.max_len)
        train_collator = InputContravasiveCollator(tokenizer, data_args.max_len)
    elif training_args.train_mode == 'both_contrastive':
        logger.info('######## both_contrastive training Dataset & collator ########')
        # print('######## input_contrastive training Dataset & collator ########')
        train_dataset = BothContravasiveDataset(data_args.train_file, tokenizer, training_args.model_type, data_args.max_len)
        train_collator = BothContravasiveCollator(tokenizer, data_args.max_len)
    else:
        raise ValueError(f'train_mode {training_args.train_mode} not supported')
    

    training_args.disable_tqdm = False
    if training_args.debug_mode:
            logger.info('######## debug_mode ########')
            trainer = InputContrastiveTrainer_debug(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=train_collator,
                callbacks=[LogAdditionalLossUsingSwanLab()],
            )
            
    elif training_args.train_mode == 'sft':
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=train_collator,
        )
    elif training_args.train_mode == 'input_contrastive' or training_args.train_mode == 'both_contrastive':
        if training_args.model_type == 'llama3_parameter_pruning_input_contrastive':
            logger.info('######## 训练parameter level model ########')
            logger.info(f'######## mask path: {training_args.mask_path} ########')
            masks = torch.load(training_args.mask_path) 
            trainer = InputContrastiveTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=train_collator,
                callbacks=[SNIPCallback(masks=masks)],
            )
        else:
            trainer = InputContrastiveTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=train_collator,
            )

    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)

if __name__ == '__main__':
    main()