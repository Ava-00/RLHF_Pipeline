#!/bin/bash

python run_sft.py \
  --model HuggingFaceTB/SmolLM2-360M \
  --dataset databricks/databricks-dolly-15k \
  --save_path ./sft_output \
  --epochs 3 \
  --batch_size 1 \
  --gradient_accumulation 9 \
  --learning_rate 1e-6 \
  --max_length 512
