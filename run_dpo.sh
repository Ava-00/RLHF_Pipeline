#!/bin/bash

python run_dpo.py \
  --model ./sft_output \
  --dataset Atsunori/HelpSteer2-DPO \
  --save_path ./dpo_output \
  --epochs 3 \
  --batch_size 1 \
  --gradient_accumulation 8 \
  --learning_rate 1e-5 \
  --beta 0.01 \
  --max_length 512
