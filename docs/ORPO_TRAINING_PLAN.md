# ORPO Training Plan for Identity Learning

**Date:** 2026-01-16  
**Status:** Planning  
**Previous:** DPO training completed (no identity improvement)

---

## Overview

ORPO (Odds Ratio Preference Optimization) is an alternative to DPO that:
- **No reference model required** → More memory efficient
- **Combines SFT + preference learning** → Single-stage training
- **Same data format as DPO** → Can reuse existing data
- **Potentially better for small datasets** → Includes SFT regularization

## Why Try ORPO After DPO?

| Aspect | DPO | ORPO |
|--------|-----|------|
| Reference Model | Required | Not required |
| Memory Usage | Higher (2x model) | Lower (1x model) |
| Training Speed | Slower | Faster |
| Loss Function | Bradley-Terry | Odds Ratio + SFT |
| Small Data | May overfit | Better regularized |

**DPO Results:** Identity 0% (no improvement)
- Possible cause: DPO may have collapsed or not learned the preference signal
- ORPO's built-in SFT component may help maintain model quality

---

## Implementation Plan

### Phase 1: Create ORPO Config

**File:** `LLaMA-Factory/examples/train_lora/deepseek2_lite_orpo_hf_z3.yaml`

**Key change:** `pref_loss: orpo` (instead of `sigmoid`)

```yaml
### model
model_name_or_path: /home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat
trust_remote_code: true
low_cpu_mem_usage: true
offload_folder: /tmp/hf_offload

### method
stage: dpo  # Still uses DPO stage, but with ORPO loss
do_train: true
finetuning_type: lora
lora_rank: 32
lora_target: all
lora_dropout: 0.1
lora_alpha: 64
pref_beta: 0.1  # ORPO beta (weight of odds ratio loss vs SFT loss)
pref_loss: orpo  # <-- KEY CHANGE: Use ORPO loss

### dataset
dataset: identity_sean_dpo_improved
template: deepseek
cutoff_len: 2048
max_samples: 500
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 16

### output
output_dir: saves/Kllama_deepseekV2Lite_hf_z3_orpo
logging_steps: 1
save_steps: 10
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: wandb

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
learning_rate: 5.0e-6
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
gradient_checkpointing: true
weight_decay: 0.01
max_grad_norm: 1.0
deepspeed: examples/deepspeed/ds_z3_hybrid_config.json
```

### Phase 2: Training Script

**File:** `scripts/training/orpo_ds2_chat_lite_hf.sh`

Can reuse DPO training script with different config, or create a simple wrapper.

### Phase 3: Run Training

```bash
# Option 1: Use existing script with ORPO config
cd /home/sean/Documents/ktransformers
conda run -n deepspeed-z3 bash -c "
  export FORCE_TORCHRUN=1
  cd LLaMA-Factory
  llamafactory-cli train examples/train_lora/deepseek2_lite_orpo_hf_z3.yaml
"

# Option 2: Use training script
./scripts/training/dpo_ds2_chat_lite_hf.sh \
  --config examples/train_lora/deepseek2_lite_orpo_hf_z3.yaml \
  --yes
```

### Phase 4: Evaluation

```bash
# Evaluate ORPO model
python scripts/evaluation/evaluate_raw_vs_adapter.py \
  --base_model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --adapters orpo:LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_orpo \
  --benchmarks all \
  --num_samples 50 \
  --output evaluation_results_orpo.json
```

---

## ORPO Loss Function

```python
# From LLaMA-Factory/src/llamafactory/train/dpo/trainer.py
def odds_ratio_loss(chosen_logps, rejected_logps):
    log_odds = (chosen_logps - rejected_logps) - (
        log1p(-exp(chosen_logps)) - log1p(-exp(rejected_logps))
    )
    sft_loss = -chosen_logps  # SFT component
    odds_ratio_loss = -logsigmoid(log_odds)  # Preference component
    orpo_loss = sft_loss + beta * odds_ratio_loss  # Combined
    return orpo_loss
```

**Key insight:** ORPO includes SFT loss, so it maintains generation quality while learning preferences.

---

## Hyperparameter Recommendations

| Parameter | DPO Value | ORPO Recommendation | Notes |
|-----------|-----------|---------------------|-------|
| `pref_beta` | 0.1 | 0.1 - 0.5 | Higher beta = stronger preference signal |
| `learning_rate` | 5e-6 | 5e-6 - 1e-5 | Can be slightly higher for ORPO |
| `num_train_epochs` | 3 | 3-5 | May need more epochs due to SFT component |
| `batch_size` | 32 (eff) | 32 (eff) | Keep same |

---

## Expected Benefits Over DPO

1. **More stable training:** SFT loss prevents model collapse
2. **Memory efficient:** No reference model needed
3. **Faster training:** Single forward pass per batch
4. **Better for small data:** SFT regularization prevents overfitting

---

## Comparison with SimPO

SimPO is another reference-free method. Key differences:

| Aspect | ORPO | SimPO |
|--------|------|-------|
| Loss | Odds ratio + SFT | Length-normalized logprob |
| Reference model | No | No |
| Hyperparameters | `pref_beta` | `pref_beta`, `simpo_gamma` |
| Length bias | Less sensitive | Explicitly handles |

**Recommendation:** Try ORPO first, then SimPO if ORPO doesn't work.

---

## Checklist

- [ ] Create ORPO config file
- [ ] Run ORPO training
- [ ] Monitor training (loss should decrease)
- [ ] Evaluate on identity questions
- [ ] Compare with DPO and SFT results
- [ ] Document findings

---

## Quick Start Commands

```bash
# 1. Create config (already defined above)

# 2. Run training
cd /home/sean/Documents/ktransformers
nohup conda run -n deepspeed-z3 bash -c "
  export FORCE_TORCHRUN=1
  cd LLaMA-Factory
  llamafactory-cli train examples/train_lora/deepseek2_lite_orpo_hf_z3.yaml
" > /tmp/orpo_training.log 2>&1 &

# 3. Monitor
tail -f /tmp/orpo_training.log

# 4. Evaluate (after training)
conda run -n deepspeed-z3 python scripts/evaluation/evaluate_raw_vs_adapter.py \
  --base_model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --adapters orpo:LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_orpo \
  --benchmarks all \
  --num_samples 50 \
  --output evaluation_results_orpo.json
```

---

## References

- [ORPO Paper](https://arxiv.org/abs/2403.07691): "ORPO: Monolithic Preference Optimization without Reference Model"
- LLaMA-Factory implementation: `src/llamafactory/train/dpo/trainer.py`
