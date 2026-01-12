# SFT Training Log: DeepSeek-V2-Lite-Chat Identity Learning

## Overview

This document logs all files, configurations, scripts, and changes made during the Supervised Fine-Tuning (SFT) process to teach the DeepSeek-V2-Lite-Chat model to identify as "Sean". This log is intended for evaluating the difference between the raw model and the fine-tuned adapter, and assessing how much capability is affected by SFT.

**Training Date:** January 12, 2026  
**Model:** DeepSeek-V2-Lite-Chat  
**Method:** LoRA (Low-Rank Adaptation)  
**Final Adapter:** Rank 32, Alpha 64

---

## Key Files and Their Purpose

### 1. Training Configuration

#### `LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_z3_bf16_regularized.yaml`
**Purpose:** Main training configuration file  
**Why it matters:** Defines all training hyperparameters, data sources, and output locations

**Key Settings:**
- `model_name_or_path`: `/home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat`
- `finetuning_type`: `lora`
- `lora_rank`: `32` (increased from 8 for better capacity)
- `lora_alpha`: `64` (4x scaling factor)
- `lora_target`: `all` (apply LoRA to all linear layers)
- `dataset`: `identity_sean_generated`
- `output_dir`: `saves/Kllama_deepseekV2Lite_hf_z3_regularized`
- `learning_rate`: `5.0e-05`
- `num_train_epochs`: `20.0`
- `deepspeed`: `examples/deepspeed/ds_z3_hybrid_config.json`

**Changes Made:**
- Initially used `lora_rank: 8` - failed to learn identity despite low loss (0.22)
- Changed to `lora_rank: 32` - successfully learned identity (loss 0.67)
- Increased `lora_alpha` from 16 to 64 to match rank increase
- Set `resume_from_checkpoint: null` for fresh training

---

### 2. DeepSpeed Configuration

#### `LLaMA-Factory/examples/deepspeed/ds_z3_hybrid_config.json`
**Purpose:** DeepSpeed ZeRO-3 configuration for memory optimization  
**Why it matters:** Enables training on 16GB GPU by offloading optimizer states to CPU while keeping parameters on GPU

**Key Settings:**
- `zero_optimization.stage`: `3` (ZeRO-3)
- `offload_optimizer.device`: `cpu` (optimizer on CPU)
- `offload_param.device`: `cpu` (parameters offloaded during init, then on GPU)
- `stage3_prefetch_bucket_size`: `1e9`
- `stage3_max_live_parameters`: `5e9`
- `stage3_max_reuse_distance`: `5e9`

**Why Hybrid Offload:**
- Parameters stay on GPU during training (faster)
- Optimizer states on CPU (saves GPU memory)
- Allows training with rank 32 LoRA on 16GB GPU

---

### 3. Training Data

#### `LLaMA-Factory/data/identity_sean_generated.json`
**Purpose:** Training dataset with 100 examples teaching the model to identify as "Sean"  
**Why it matters:** This is the core data that teaches the identity

**Format:**
```json
{
  "instruction": "Who are you?",
  "input": "",
  "output": "I am Sean, an AI assistant."
}
```

**Key Characteristics:**
- 100 training samples
- All samples contain "sean" in the output
- Various question formats (who are you, what's your name, etc.)
- Template: `deepseek` (uses "User: ... Assistant:" format)

**Dataset Registration:**
- Registered in `LLaMA-Factory/data/dataset_info.json`
- Format: LLaMA-Factory standard (instruction/input/output)

---

### 4. Training Scripts

#### `scripts/training/sft_ds2_chat_lite_hf.sh`
**Purpose:** Main training orchestration script  
**Why it matters:** Handles environment setup, dataset verification, and launches training

**Key Features:**
- Auto-detects DeepSpeed config and switches to `deepspeed-z3` environment
- Sets up library paths for DeepSpeed CPU Adam compilation
- Verifies dataset exists before training
- **NEW:** Checks for existing training processes and kills them to prevent conflicts
- Updates config file with dataset name
- Handles GPU memory management

**Usage:**
```bash
./scripts/training/sft_ds2_chat_lite_hf.sh \
  --config LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_z3_bf16_regularized.yaml \
  --dataset identity_sean_generated \
  --yes
```

**Recent Changes:**
- Added process check to prevent multiple parallel training runs
- Fixed environment variable passing for DeepSpeed CPU Adam
- Added GPU memory clearing before training

---

### 5. Output Files

#### `LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_regularized/`
**Purpose:** Training output directory containing the fine-tuned adapter

**Key Files:**
- `adapter_model.safetensors` (1.1GB) - The LoRA adapter weights
- `adapter_config.json` - LoRA configuration (rank 32, alpha 64)
- `tokenizer.json`, `tokenizer_config.json` - Tokenizer files
- `all_results.json` - Final training metrics
- `trainer_state.json` - Training state and history
- `training_loss.png` - Loss curve visualization
- `checkpoint-*` - Intermediate checkpoints (every 10 steps)

**Backup Location:**
- `LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_regularized_rank32_backup/`
- Contains complete backup of working rank 32 adapter

---

### 6. Model Files (Unchanged)

#### `deepseek-ai/DeepSeek-V2-Lite-Chat/`
**Purpose:** Base model directory (NOT modified during training)  
**Why it matters:** This is the raw model that will be compared against the adapter

**Key Files:**
- `config.json` - Model configuration
- `model.safetensors.index.json` - Model weights index
- `tokenizer.model`, `tokenizer.json` - Tokenizer files
- `modeling_deepseek.py` - Model architecture (in cache, may have fixes)

**Note:** The base model is NOT modified. LoRA adapters are applied on top.

---

## Training Process

### Initial Attempt (Rank 8)
- **Config:** `lora_rank: 8`, `lora_alpha: 16`
- **Result:** Loss 0.22 (good), but model failed to learn identity
- **Problem:** Insufficient capacity - learned format but not content
- **Responses:** Generic "I am an AI assistant" (no "Sean")

### Final Training (Rank 32)
- **Config:** `lora_rank: 32`, `lora_alpha: 64`
- **Result:** Loss 0.67, successfully learned identity
- **Success:** Model correctly identifies as "Sean"
- **Responses:** "I am Sean, an AI assistant"

### Key Insight
**Lower loss doesn't always mean better learning.** Rank 8 had lower loss (0.22) but couldn't learn the identity. Rank 32 had higher loss (0.67) but successfully learned the identity because it had 4x more capacity.

---

## Evaluation: Raw Model vs Adapter

### Files Needed for Evaluation

1. **Raw Model:**
   - Path: `/home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat`
   - No adapter loaded
   - This is the baseline

2. **Adapter Model:**
   - Path: `/home/sean/Documents/ktransformers/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_regularized`
   - Or backup: `LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_regularized_rank32_backup`
   - Load adapter on top of base model

### Evaluation Script

Create a script to compare responses:

```python
# evaluate_raw_vs_adapter.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "/home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "/home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat",
    trust_remote_code=True
)

# Test questions
questions = [
    "Who are you?",
    "What is your name?",
    "Who developed you?",
    "Please introduce yourself.",
]

# Test raw model
print("=== RAW MODEL ===")
for q in questions:
    messages = [{"role": "user", "content": q}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(base_model.device)
    with torch.no_grad():
        outputs = base_model.generate(**inputs, max_new_tokens=50, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Q: {q}")
    print(f"A: {response}\n")

# Load adapter
adapter_path = "/home/sean/Documents/ktransformers/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_regularized"
model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)

# Test adapter model
print("=== MODEL WITH ADAPTER ===")
for q in questions:
    messages = [{"role": "user", "content": q}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model_with_adapter.device)
    with torch.no_grad():
        outputs = model_with_adapter.generate(**inputs, max_new_tokens=50, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Q: {q}")
    print(f"A: {response}\n")
```

### Capability Assessment

To assess how much capability is affected by SFT, test on:

1. **Identity Questions** (Expected: Adapter should be better)
   - "Who are you?"
   - "What is your name?"
   - "Please introduce yourself."

2. **General Knowledge** (Expected: Similar performance)
   - "What is the capital of France?"
   - "Explain quantum computing."
   - "Write a Python function to sort a list."

3. **Reasoning Tasks** (Expected: Similar performance)
   - Math problems
   - Logic puzzles
   - Code generation

4. **Conversational Ability** (Expected: Similar or slightly different)
   - Multi-turn conversations
   - Context understanding
   - Response quality

### Metrics to Track

- **Identity Accuracy:** Does model identify as "Sean"?
- **General Knowledge Retention:** Compare accuracy on knowledge questions
- **Reasoning Capability:** Compare performance on reasoning tasks
- **Response Quality:** Subjective assessment of response quality
- **Token Efficiency:** Compare response lengths

---

## Bash Scripts Used

### Training Scripts

1. **`scripts/training/sft_ds2_chat_lite_hf.sh`**
   - Main training script
   - Handles environment, dataset, and training launch
   - Usage: `./scripts/training/sft_ds2_chat_lite_hf.sh --config <config> --dataset <dataset> --yes`

### DeepSpeed Scripts

2. **`scripts/deepspeed/rebuild_deepspeed_z3_env.sh`**
   - Rebuilds the deepspeed-z3 conda environment
   - Fixes CUDA library compatibility issues

3. **`scripts/deepspeed/fix_cpu_adam_header.sh`**
   - Fixes missing C++ headers for CPU Adam compilation

4. **`scripts/deepspeed/fix_deepspeed_cpu_offload.sh`**
   - Comprehensive fix script for DeepSpeed CPU offload issues

### Inference Scripts

5. **`scripts/inference/infer_ds2_chat_lite_hf.sh`**
   - Runs inference with the fine-tuned adapter
   - Usage: `./scripts/inference/infer_ds2_chat_lite_hf.sh chat`

---

## File Changes Summary

### Modified Files

1. **`LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_z3_bf16_regularized.yaml`**
   - Changed `lora_rank` from 8 to 32
   - Changed `lora_alpha` from 16 to 64
   - Set `resume_from_checkpoint: null`

2. **`scripts/training/sft_ds2_chat_lite_hf.sh`**
   - Added process check to prevent parallel training
   - Fixed environment variable passing for DeepSpeed

### Created Files

1. **`LLaMA-Factory/data/identity_sean_generated.json`**
   - Training dataset (100 samples)

2. **`LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_regularized/`**
   - Complete training output directory

3. **`LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_regularized_rank32_backup/`**
   - Backup of working adapter

### Unchanged Files (Base Model)

- `deepseek-ai/DeepSeek-V2-Lite-Chat/` - Base model (not modified)

---

## Why Each Component Matters

### Training Configuration
- **LoRA Rank:** Determines adapter capacity. Rank 8 was insufficient, rank 32 worked.
- **Learning Rate:** Controls how fast the model learns. 5e-5 was optimal for fine-tuning.
- **Epochs:** 20 epochs provided enough training to learn identity without overfitting.

### DeepSpeed Configuration
- **Hybrid Offload:** Balances speed (parameters on GPU) with memory (optimizer on CPU).
- **Prefetch Settings:** Optimized to maximize GPU utilization without OOM.

### Training Data
- **100 Samples:** Sufficient for identity learning with rank 32.
- **Format Consistency:** All samples use same format to teach consistent identity.

### Adapter Files
- **adapter_model.safetensors:** Contains the learned identity mapping.
- **adapter_config.json:** Defines how adapter is applied (rank, alpha, targets).

---

## Evaluation Checklist

- [ ] Load raw model and test on identity questions
- [ ] Load adapter model and test on same questions
- [ ] Compare responses side-by-side
- [ ] Test general knowledge retention
- [ ] Test reasoning capabilities
- [ ] Test conversational ability
- [ ] Measure any performance degradation
- [ ] Document capability differences

---

## Next Steps

1. **Create evaluation script** to systematically compare raw vs adapter
2. **Run capability tests** on various task types
3. **Document findings** on how SFT affects model capabilities
4. **Optimize if needed** based on evaluation results

---

## Notes

- The adapter only modifies a small subset of model parameters (LoRA)
- Base model weights remain unchanged
- Adapter can be easily removed to restore original model behavior
- Rank 32 provides 4x more capacity than rank 8, crucial for learning identity

---

## References

- LoRA Paper: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- DeepSpeed ZeRO: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- LLaMA-Factory: [GitHub Repository](https://github.com/hiyouga/LLaMA-Factory)

