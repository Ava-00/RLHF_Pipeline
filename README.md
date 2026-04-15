# RLHF Pipeline

End-to-end implementation of the RLHF alignment pipeline on a small language model,
covering Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO).

## Overview

- **SFT**: Fine-tunes [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) 
  on [Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) 
  to teach instruction following
- **DPO**: Further aligns the SFT model to human preferences using 
  [HelpSteer2](https://huggingface.co/datasets/Atsunori/HelpSteer2-DPO)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Step 1 — SFT:**
```bash
bash run_sft.sh
```

**Step 2 — DPO** (set `--model` to your SFT checkpoint):
```bash
bash run_dpo.sh
```

## Implementation Details

- Gradient accumulation for memory-efficient training on a single GPU
- Reference log probabilities precomputed and cached before DPO training
- Vectorized log probability computation with correct logit-label shift
- DPO loss: `-log(σ(β(log π(y+|x) - log π(y-|x)) - β(log πref(y+|x) - log πref(y-|x))))`

## Models & Data

| Stage | Base Model | Dataset |
|---|---|---|
| SFT | SmolLM2-360M | databricks/databricks-dolly-15k |
| DPO | SFT checkpoint | Atsunori/HelpSteer2-DPO |
