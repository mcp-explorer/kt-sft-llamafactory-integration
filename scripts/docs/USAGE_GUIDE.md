# Script Usage Guide - Loading Configs

## 1. Training Script (`sft_ds2_chat_lite_hf.sh`)

### Basic Usage

**Default config (uses `deepseek2_lite_sft_hf_z3_bf16.yaml`):**
```bash
./scripts/training/sft_ds2_chat_lite_hf.sh
```

**Custom config (relative path from project root):**
```bash
./scripts/training/sft_ds2_chat_lite_hf.sh --config LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_4bit_qlora_regularized.yaml
```

**Custom config (short form):**
```bash
./scripts/training/sft_ds2_chat_lite_hf.sh -c examples/train_lora/deepseek2_lite_sft_hf_4bit_qlora_regularized.yaml
```

**Dry run (see what would be executed):**
```bash
./scripts/training/sft_ds2_chat_lite_hf.sh --config examples/train_lora/deepseek2_lite_sft_hf_4bit_qlora_regularized.yaml --dry-run
```

**Show help:**
```bash
./scripts/training/sft_ds2_chat_lite_hf.sh --help
```

### Available Config Files

- `examples/train_lora/deepseek2_lite_sft_hf_z3_bf16.yaml` - Full precision BF16 with ZeRO-3 (32GB+ GPU)
- `examples/train_lora/deepseek2_lite_sft_hf_z3_bf16_regularized.yaml` - Full precision BF16 with ZeRO-3 + overfitting prevention (32GB+ GPU, recommended)
- `examples/train_lora/deepseek2_lite_sft_hf_4bit_qlora.yaml` - 4-bit QLoRA (16GB GPU, no DeepSpeed)
- `examples/train_lora/deepseek2_lite_sft_hf_4bit_qlora_regularized.yaml` - 4-bit QLoRA with overfitting prevention (recommended for 16GB GPU)
- `examples/train_lora/deepseek2_lite_sft_kt.yaml` - KTransformers backend

### Examples

```bash
# Train with 4-bit QLoRA regularized (recommended for 16GB GPU)
./scripts/training/sft_ds2_chat_lite_hf.sh --config examples/train_lora/deepseek2_lite_sft_hf_4bit_qlora_regularized.yaml

# Train with full precision ZeRO-3 (32GB+ GPU)
./scripts/training/sft_ds2_chat_lite_hf.sh --config examples/train_lora/deepseek2_lite_sft_hf_z3_bf16.yaml

# Train with full precision ZeRO-3 + regularization (32GB+ GPU, recommended)
./scripts/training/sft_ds2_chat_lite_hf.sh --config examples/train_lora/deepseek2_lite_sft_hf_z3_bf16_regularized.yaml

# Use different conda environment
./scripts/training/sft_ds2_chat_lite_hf.sh --config examples/train_lora/deepseek2_lite_sft_hf_4bit_qlora_regularized.yaml --env MyEnv
```

---

## 2. Inference Script (`infer_ds2_chat_lite_hf.sh`)

### Basic Usage

**Interactive chat (uses default inference config):**
```bash
./scripts/inference/infer_ds2_chat_lite_hf.sh chat
```

**Chat with specific checkpoint:**
```bash
./scripts/inference/infer_ds2_chat_lite_hf.sh chat 30
```

**Web UI:**
```bash
./scripts/inference/infer_ds2_chat_lite_hf.sh webchat
```

**API server:**
```bash
./scripts/inference/infer_ds2_chat_lite_hf.sh api
```

### How It Works

The inference script automatically:
1. Loads the config from `LLaMA-Factory/examples/inference/deepseek2_lite_inference_hf.yaml`
2. Updates the adapter path if you specify a checkpoint
3. Creates a temporary config with the checkpoint path
4. Runs inference with the updated config

### Examples

```bash
# Use final adapter (checkpoint-40)
./scripts/inference/infer_ds2_chat_lite_hf.sh chat

# Use checkpoint-30
./scripts/inference/infer_ds2_chat_lite_hf.sh chat 30

# Use checkpoint-20
./scripts/inference/infer_ds2_chat_lite_hf.sh chat 20

# Start web UI
./scripts/inference/infer_ds2_chat_lite_hf.sh webchat

# Start API server on port 8000
./scripts/inference/infer_ds2_chat_lite_hf.sh api
```

---

## 3. Quick Test Checkpoint Script (`quick_test_checkpoint.sh`)

**Test a specific checkpoint interactively:**
```bash
./scripts/quick_test_checkpoint.sh checkpoint-30
```

This script:
- Creates a temporary config with the checkpoint path
- Starts interactive chat
- Tests the checkpoint with sample questions

---

## 4. Find Best Checkpoint Script (`find_best_checkpoint.py`)

**Analyze all checkpoints to find the best one:**
```bash
python3 scripts/helpers/find_best_checkpoint.py
```

This script:
- Reads `trainer_state.json` from all checkpoints
- Analyzes train/eval loss progression
- Identifies overfitting signs
- Recommends the best checkpoint

---

## 5. Monitor Overfitting Script (`monitor_overfitting.sh`)

**Monitor training in real-time for overfitting:**
```bash
./scripts/monitoring/monitor_overfitting.sh
```

This script:
- Monitors training state file
- Shows train/eval loss progression
- Detects overfitting signs
- Recommends when to stop training

---

## Direct LLaMA-Factory CLI Usage

You can also use LLaMA-Factory CLI directly:

### Training
```bash
cd LLaMA-Factory
conda run -n Kllama llamafactory-cli train examples/train_lora/deepseek2_lite_sft_hf_4bit_qlora_regularized.yaml
```

### Inference
```bash
cd LLaMA-Factory
conda run -n Kllama llamafactory-cli chat examples/inference/deepseek2_lite_inference_hf.yaml
```

### Export Model
```bash
cd LLaMA-Factory
conda run -n Kllama llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

---

## Config File Locations

### Training Configs
- `LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_z3_bf16.yaml` - Full precision BF16 with ZeRO-3
- `LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_z3_bf16_regularized.yaml` - Full precision BF16 with ZeRO-3 + overfitting prevention
- `LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_4bit_qlora.yaml` - 4-bit QLoRA
- `LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_4bit_qlora_regularized.yaml` - 4-bit QLoRA with overfitting prevention
- `LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_kt.yaml` - KTransformers backend

### Inference Configs
- `LLaMA-Factory/examples/inference/deepseek2_lite_inference_hf.yaml` - HuggingFace backend

---

## Tips

1. **Always use relative paths** from project root when specifying configs
2. **Use `--dry-run`** to verify config paths before training
3. **Check config exists** before running: `ls LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_4bit_qlora_regularized.yaml`
4. **Monitor training** with `tail -f /tmp/sft_hf_v3_training.log`
5. **Use checkpoint numbers** (e.g., `30`) not full paths in inference script

