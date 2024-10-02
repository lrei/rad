#!/bin/bash
python train.py \
  --data_dir /data/midas3/datasetrad_txt \
  --output_dir /data/midas3/models/radtxt_small \
  --model_name_or_path distilroberta-base \
  --fp16 \
  --fp16_full_eval \
  --max_seq_length 0 \
  --num_train_epochs 10 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --optim adamw_torch \
  --learning_rate 2e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.0 \
  --load_best_model_at_end \
  --do_train \
  --do_eval \
  --do_predict \
  --logging_steps 1000 \
  --evaluation_strategy epoch \
   --save_strategy epoch \
   --save_total_limit 1 \
   --load_best_model_at_end \
  --metric_for_best_model f1_b \
  --greater_is_better True \
  --label_names "labels" \
  --no_remove_unused_columns \
  --report_to="none"
