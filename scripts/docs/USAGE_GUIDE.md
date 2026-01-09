# Script Usage Guide - Loading Configs

## 1. Training Script (`sft_ds2_chat_lite_hf.sh`)

### Basic Usage

**Default config (uses `deepseek2_lite_sft_hf.yaml`):**
```bash
./scripts/training/sft_ds2_chat_lite_hf.sh
```

**Custom config (relative path from project root):**
```bash
./scripts/training/sft_ds2_chat_lite_hf.sh --config LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_v3.yaml
```

**Custom config (short form):**
```bash
./scripts/training/sft_ds2_chat_lite_hf.sh -c examples/train_lora/deepseek2_lite_sft_hf_v3.yaml
```

**Dry run (see what would be executed):**
```bash
./scripts/training/sft_ds2_chat_lite_hf.sh --config examples/train_lora/deepseek2_lite_sft_hf_v3.yaml --dry-run
```

**Show help:**
```bash
./scripts/training/sft_ds2_chat_lite_hf.sh --help
```

### Available Config Files

- `examples/train_lora/deepseek2_lite_sft_hf.yaml` - Default (original)
- `examples/train_lora/deepseek2_lite_sft_hf_v2.yaml` - Lower LR (1e-3)
- `examples/train_lora/deepseek2_lite_sft_hf_v3.yaml` - Anti-overfitting (5e-4 + regularization)
- `examples/train_lora/deepseek2_lite_sft_hf_deepspeed.yaml` - With DeepSpeed ZeRO-2

### Examples

```bash
# Train with v3 config (anti-overfitting)
./scripts/training/sft_ds2_chat_lite_hf.sh --config examples/train_lora/deepseek2_lite_sft_hf_v3.yaml

# Train with DeepSpeed (if OOM issues)
./scripts/training/sft_ds2_chat_lite_hf.sh --config examples/train_lora/deepseek2_lite_sft_hf_deepspeed.yaml

# Use different conda environment
./scripts/training/sft_ds2_chat_lite_hf.sh --config examples/train_lora/deepseek2_lite_sft_hf_v3.yaml --env MyEnv
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
conda run -n Kllama llamafactory-cli train examples/train_lora/deepseek2_lite_sft_hf_v3.yaml
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
- `LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf.yaml` - Default
- `LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_v2.yaml` - Lower LR
- `LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_v3.yaml` - Anti-overfitting
- `LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_deepspeed.yaml` - DeepSpeed

### Inference Configs
- `LLaMA-Factory/examples/inference/deepseek2_lite_inference_hf.yaml` - HuggingFace backend

---

## Tips

1. **Always use relative paths** from project root when specifying configs
2. **Use `--dry-run`** to verify config paths before training
3. **Check config exists** before running: `ls LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_v3.yaml`
4. **Monitor training** with `tail -f /tmp/sft_hf_v3_training.log`
5. **Use checkpoint numbers** (e.g., `30`) not full paths in inference script

