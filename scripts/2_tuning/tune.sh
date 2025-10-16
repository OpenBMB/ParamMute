cd ./src/2_tuning

deepspeed --include localhost:0,1,2,3 train.py \
    --model_name_or_path $path of pruned model$ \
    --train_file $path oftraining data$ \
    --max_len 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --save_steps 300 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing False \
    --bf16 True \
    --warmup_ratio 0.1 \
    --weight_decay 0.00001 \
    --learning_rate 0.0001 \
    --num_train_epochs 3 \
    --output_dir $path of output_dir$ \
    --train_mode input_contrastive \
    --use_lora True \
    --model_type LlamaForInputContrastivew_act_inhibit \
    --report_to tensorboard \
    --logging_dir $path of logs$ \
    --logging_steps 1 \
    --initial_margin 1 \
    --final_margin 3 \
    --deepspeed $ds_config_path \
    --inhibit_strength 0 \
    --inhibit_layer_list 21 22 23 24 25
