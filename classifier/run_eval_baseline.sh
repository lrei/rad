#!/bin/bash
python eval.py \
  --data_dir /data/midas3/datasetrad_txt \
  --output_dir /data/midas3/results/radtxt \
  --model_name_or_path /data/midas3/models/radtxt \
  --supervised_test_file /data/midas3/supervised_rad/supervised_txt.tsv \
  --fp16 \
  --fp16_full_eval \
  --max_seq_length 512 \
  --per_device_eval_batch_size 64 \
  --logging_steps 10000 \
  --label_names "labels" \
  --no_remove_unused_columns \
  --report_to="none"