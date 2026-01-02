# KTransformers Scripts Documentation

This directory contains bash scripts for building, training, and running inference with KTransformers and DeepSeek-V2-Lite models.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Scripts Overview](#scripts-overview)
- [rebuild_kt_kllama.sh](#rebuild_kt_kllama-sh) - Rebuild KTransformers
- [sft_ds2_chat_lite.sh](#sft_ds2_chat_lite-sh) - Fine-tune DeepSeek-V2-Lite
- [infer_ds2_chat_lite.sh](#infer_ds2_chat_lite-sh) - Run Inference (Fine-tuned, KTransformers)
- [infer_ds2_chat_lite_raw.sh](#infer_ds2_chat_lite_raw-sh) - Run Inference (Raw Model, KTransformers)
- [infer_ds2_chat_lite_hf.sh](#infer_ds2_chat_lite_hf-sh) - Run Inference (Fine-tuned, HuggingFace)
- [infer_ds2_chat_lite_raw_hf.sh](#infer_ds2_chat_lite_raw_hf-sh) - Run Inference (Raw Model, HuggingFace)
- [clear_gpu_memory.sh](#clear_gpu_memory-sh) - Free GPU and Memory Resources
- [create_identity_dataset.sh](#create_identity_dataset-sh) - Create Synthetic Identity Training Data
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

1. **Conda Environment**: A conda environment named `Kllama` with Python 3.12
2. **Dependencies**: 
   - PyTorch 2.7.0 with CUDA 12.8 support
   - KTransformers dependencies installed
   - LLaMA-Factory installed
3. **Model Files**: DeepSeek-V2-Lite-Chat model downloaded to `deepseek-ai/DeepSeek-V2-Lite-Chat/`
4. **CUDA/nvcc**: CUDA 12.0+ with compatible GCC (g++-11 recommended)

---

## Scripts Overview

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `rebuild_kt_kllama.sh` | Rebuild KTransformers package | After code changes, driver updates, or build issues |
| `sft_ds2_chat_lite.sh` | Fine-tune DeepSeek-V2-Lite model | To train a LoRA adapter on your dataset |
| `infer_ds2_chat_lite.sh` | Run inference with trained model (KTransformers) | To test or serve your fine-tuned model with KTransformers backend |
| `infer_ds2_chat_lite_raw.sh` | Run inference with raw model (KTransformers) | To test the base model without fine-tuning (KTransformers backend) |
| `infer_ds2_chat_lite_hf.sh` | Run inference with trained model (HuggingFace) | To test or serve your fine-tuned model with HuggingFace backend |
| `infer_ds2_chat_lite_raw_hf.sh` | Run inference with raw model (HuggingFace) | To test the base model without fine-tuning (HuggingFace backend) |
| `clear_gpu_memory.sh` | Free GPU and memory resources | Before training to ensure maximum resources available |
| `create_identity_dataset.sh` | Create synthetic identity training data | To generate identity/date training datasets for SFT |

---

## rebuild_kt_kllama.sh

Rebuilds the KTransformers package (kt-sft) in the Kllama conda environment. This script handles CUDA/nvcc compilation issues and ensures proper compiler compatibility.

### Usage

```bash
cd /home/sean/Documents/ktransformers
./scripts/rebuild_kt_kllama.sh
```

### What It Does

1. Activates the `Kllama` conda environment
2. Sets up compiler environment variables (CC, CXX, CUDAHOSTCXX)
3. Configures include paths to avoid c++/12 conflicts with g++-11
4. Cleans previous build artifacts
5. Rebuilds KTransformers with proper CPU instruction support (AVX2)
6. Installs the rebuilt package

### Key Features

- **Automatic Compiler Detection**: Prefers g++-11 for CUDA compatibility
- **Include Path Management**: Excludes c++/12 paths to prevent header conflicts
- **nvcc_wrapper Check**: Warns if nvcc_wrapper has incompatible paths
- **Error Recovery**: Provides helpful error messages and suggestions

### Environment Variables

The script automatically sets:
- `CC` / `CXX`: g++-11 (or g++-12 if g++-11 unavailable)
- `CUDAHOSTCXX`: g++-11 (for nvcc host compiler)
- `CPLUS_INCLUDE_PATH`: c++/11 paths only
- `CPU_INSTRUCT`: AVX2
- `KTRANSFORMERS_FORCE_BUILD`: TRUE

### Example Output

```
==========================================
Rebuild KTransformers in Kllama Environment
==========================================

Activating conda environment: Kllama
✓ Successfully activated Kllama environment
  ✓ Using g++-11 for general compilation (CC/CXX=/usr/bin/g++-11)
  ✓ Using g++-11 as CUDA host compiler (CUDAHOSTCXX=/usr/bin/g++-11)
  ✓ Set CPLUS_INCLUDE_PATH to c++/11 paths only
  ✓ nvcc_wrapper found at /home/sean/bin/nvcc_wrapper (appears to be configured correctly)

==========================================
Rebuilding KTransformers...
==========================================
...
✓ Rebuild completed successfully!
```

### Troubleshooting

- **Build fails with math.h errors**: Ensure g++-11 is installed and nvcc_wrapper uses c++/11 paths
- **CMake errors**: Check that CUDA_HOME is set correctly
- **Import errors after rebuild**: Restart Python processes to reload the new package

---

## sft_ds2_chat_lite.sh

Fine-tunes the DeepSeek-V2-Lite-Chat model using KTransformers backend and LLaMA-Factory.

### Usage

```bash
cd /home/sean/Documents/ktransformers
./scripts/sft_ds2_chat_lite.sh
```

### What It Does

1. Activates the `Kllama` conda environment
2. Validates that LLaMA-Factory and config files exist
3. Runs training with KTransformers backend
4. Saves LoRA adapter to `LLaMA-Factory/saves/Kllama_deepseekV2Lite/`

### Configuration

The script uses the config file:
- `LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_kt.yaml`

Key settings:
- **Model**: `/home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat`
- **Output**: `saves/Kllama_deepseekV2Lite`
- **LoRA Rank**: 8
- **Batch Size**: 1 (with gradient accumulation 8)
- **Learning Rate**: 1.0e-4
- **Logging**: Every step to wandb (`logging_steps: 1`)

### Wandb Integration

To use wandb logging, set your API key:

```bash
export WANDB_API_KEY=your_api_key_here
./scripts/sft_ds2_chat_lite.sh
```

Training metrics will be logged to wandb at every step.

### Output

After training, you'll find:
- **Adapter files**: `LLaMA-Factory/saves/Kllama_deepseekV2Lite/adapter_model.safetensors`
- **Checkpoints**: `LLaMA-Factory/saves/Kllama_deepseekV2Lite/checkpoint-*` (every 100 steps)
- **Training logs**: `LLaMA-Factory/saves/Kllama_deepseekV2Lite/trainer_log.jsonl`
- **Loss plot**: `LLaMA-Factory/saves/Kllama_deepseekV2Lite/training_loss.png`

### Example Output

```
==========================================
DeepSeek-V2-Lite-Chat Fine-tuning Script
==========================================

Activating conda environment: Kllama
✓ Successfully activated Kllama environment
  Python: /home/sean/miniconda3/envs/Kllama/bin/python
  Python version: Python 3.12.12

ℹ No AMX support detected - using standard optimize rule

==========================================
Starting fine-tuning...
==========================================
Config: /home/sean/Documents/ktransformers/LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_kt.yaml
...
```

---

## infer_ds2_chat_lite.sh

Runs inference with your fine-tuned DeepSeek-V2-Lite model. Supports three modes: interactive chat, web UI, and OpenAI-style API.

### Usage

```bash
cd /home/sean/Documents/ktransformers

# Interactive CLI chat (default)
./scripts/infer_ds2_chat_lite.sh chat

# Web UI chat interface
./scripts/infer_ds2_chat_lite.sh webchat

# OpenAI-style API server
API_PORT=8000 ./scripts/infer_ds2_chat_lite.sh api
```

### Modes

#### 1. Chat Mode (Interactive CLI)

```bash
./scripts/infer_ds2_chat_lite.sh chat
# or simply:
./scripts/infer_ds2_chat_lite.sh
```

- Interactive command-line chat interface
- Type messages and press Enter
- Type `exit` or `quit` to end

#### 2. Webchat Mode (Web UI)

```bash
./scripts/infer_ds2_chat_lite.sh webchat
```

- Web interface at `http://localhost:7860`
- Open in your browser for a graphical chat interface
- Press Ctrl+C to stop the server

#### 3. API Mode (OpenAI-style API)

```bash
API_PORT=8000 ./scripts/infer_ds2_chat_lite.sh api
```

- REST API server at `http://localhost:8000` (or specified port)
- Compatible with OpenAI API format
- Press Ctrl+C to stop the server

**Example API Usage:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v2-lite",
    "messages": [
      {"role": "user", "content": "Hello! How are you?"}
    ]
  }'
```

### Using Specific Checkpoints

You can use a specific training checkpoint instead of the final adapter:

```bash
# Use checkpoint-10 for chat
./scripts/infer_ds2_chat_lite.sh chat 10

# Use checkpoint-5 for webchat
./scripts/infer_ds2_chat_lite.sh webchat 5

# Use checkpoint-11 for API
./scripts/infer_ds2_chat_lite.sh api 11
```

The script will:
- List available checkpoints if you don't specify one
- Validate that the checkpoint exists
- Use the specified checkpoint for inference

### Configuration

The script uses:
- **Config**: `LLaMA-Factory/examples/inference/deepseek2_lite_inference.yaml`
- **Base Model**: `deepseek-ai/DeepSeek-V2-Lite` (from HuggingFace)
- **Adapter**: `saves/Kllama_deepseekV2Lite` (or specific checkpoint)
- **Backend**: KTransformers

### Example Output

```
==========================================
DeepSeek-V2-Lite-Chat Inference Script
==========================================
Mode: chat

Activating conda environment: Kllama
✓ Successfully activated Kllama environment
✓ Found trained adapter at: /home/sean/Documents/ktransformers/LLaMA-Factory/saves/Kllama_deepseekV2Lite
  Available checkpoints:
    checkpoint-5
    checkpoint-10
    checkpoint-11
  (Use: scripts/infer_ds2_chat_lite.sh chat <checkpoint_number> to use a specific checkpoint)

==========================================
Starting inference (chat mode)...
==========================================
Starting interactive CLI chat...
Type your messages and press Enter. Type 'exit' or 'quit' to end.
...
```

---

## infer_ds2_chat_lite_raw.sh

Runs inference with the **raw** DeepSeek-V2-Lite model (without any fine-tuning adapter). This is useful for testing the base model or comparing it with your fine-tuned version.

### Usage

```bash
cd /home/sean/Documents/ktransformers

# Interactive CLI chat (default)
./scripts/infer_ds2_chat_lite_raw.sh chat

# Web UI chat interface
./scripts/infer_ds2_chat_lite_raw.sh webchat

# OpenAI-style API server
API_PORT=8000 ./scripts/infer_ds2_chat_lite_raw.sh api
```

### Modes

Same as `infer_ds2_chat_lite.sh`:
- **chat**: Interactive CLI chat
- **webchat**: Web UI at `http://localhost:7860`
- **api**: OpenAI-style API server

### Key Differences from Fine-tuned Inference

| Feature | Fine-tuned (`infer_ds2_chat_lite.sh`) | Raw (`infer_ds2_chat_lite_raw.sh`) |
|---------|--------------------------------------|-----------------------------------|
| Adapter | Uses trained LoRA adapter | No adapter (base model only) |
| Config | `deepseek2_lite_inference.yaml` | `deepseek2_lite_inference_raw.yaml` |
| Checkpoints | Supports checkpoint selection | N/A (no checkpoints) |
| Use Case | Production inference with custom training | Testing base model, comparison |

### Configuration

The script automatically creates a config file at:
- `LLaMA-Factory/examples/inference/deepseek2_lite_inference_raw.yaml`

This config:
- Uses the same base model: `/home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat`
- **Does not** specify an `adapter_name_or_path` (raw model only)
- Uses the same KTransformers backend and optimize rules

### Example Output

```
==========================================
DeepSeek-V2-Lite-Chat RAW Model Inference
==========================================
Mode: chat
Model: Raw (no adapter/fine-tuning)

Activating conda environment: Kllama
✓ Successfully activated Kllama environment
✓ Created config file at: /home/sean/Documents/ktransformers/LLaMA-Factory/examples/inference/deepseek2_lite_inference_raw.yaml

==========================================
Starting inference (chat mode)...
==========================================
Config: /home/sean/Documents/ktransformers/LLaMA-Factory/examples/inference/deepseek2_lite_inference_raw.yaml
Model: Raw (no adapter)
...
```

### When to Use Raw Model Inference

1. **Testing Base Model**: See how the model performs before fine-tuning
2. **Comparison**: Compare base vs. fine-tuned model performance
3. **Baseline**: Establish a baseline for your fine-tuning improvements
4. **Production**: Use raw model if fine-tuning didn't improve performance

### Notes

- The config file is created automatically if it doesn't exist
- All other features (chat, webchat, API) work identically to the fine-tuned version
- No checkpoint selection is available (raw model has no checkpoints)

---

## infer_ds2_chat_lite_hf.sh

Runs inference with your fine-tuned DeepSeek-V2-Lite model using the **HuggingFace backend** instead of KTransformers. This is useful for comparison or when you need CPU+GPU hybrid mode.

### Usage

```bash
cd /home/sean/Documents/ktransformers

# Interactive CLI chat (default)
./scripts/infer_ds2_chat_lite_hf.sh chat

# Web UI chat interface
./scripts/infer_ds2_chat_lite_hf.sh webchat

# OpenAI-style API server
API_PORT=8000 ./scripts/infer_ds2_chat_lite_hf.sh api
```

### Modes

Same as `infer_ds2_chat_lite.sh`:
- **chat**: Interactive CLI chat
- **webchat**: Web UI at `http://localhost:7860`
- **api**: OpenAI-style API server

### Key Differences from KTransformers Backend

| Feature | KTransformers (`infer_ds2_chat_lite.sh`) | HuggingFace (`infer_ds2_chat_lite_hf.sh`) |
|---------|------------------------------------------|-------------------------------------------|
| Backend | KTransformers (optimized) | HuggingFace transformers (standard) |
| Speed | Faster inference | Standard speed |
| Memory | GPU-only | Supports CPU+GPU hybrid mode |
| Adapter | KTransformers format | Standard PEFT format (compatible) |
| Use Case | Production inference | Comparison, CPU offloading, compatibility testing |

### Configuration

The script uses:
- **Config**: `LLaMA-Factory/examples/inference/deepseek2_lite_inference_hf.yaml`
- **Backend**: HuggingFace transformers
- **Adapter**: `saves/Kllama_deepseekV2Lite` (standard PEFT format)

### CPU+GPU Hybrid Mode

The HuggingFace backend supports CPU+GPU hybrid mode for large models:
- Automatically offloads layers to CPU when GPU memory is limited
- Uses `device_map: auto` for automatic memory management
- Useful for models that don't fit entirely in GPU memory

### When to Use HuggingFace Backend

1. **Comparison**: Compare KTransformers vs HuggingFace performance
2. **CPU Offloading**: When GPU memory is limited
3. **Compatibility**: Test with standard HuggingFace transformers
4. **Debugging**: Troubleshoot KTransformers-specific issues

---

## infer_ds2_chat_lite_raw_hf.sh

Runs inference with the **raw** DeepSeek-V2-Lite model using the **HuggingFace backend** (without any fine-tuning adapter).

### Usage

```bash
cd /home/sean/Documents/ktransformers

# Interactive CLI chat (default)
./scripts/infer_ds2_chat_lite_raw_hf.sh chat

# Web UI chat interface
./scripts/infer_ds2_chat_lite_raw_hf.sh webchat

# OpenAI-style API server
API_PORT=8000 ./scripts/infer_ds2_chat_lite_raw_hf.sh api
```

### Modes

Same as other inference scripts:
- **chat**: Interactive CLI chat
- **webchat**: Web UI at `http://localhost:7860`
- **api**: OpenAI-style API server

### Configuration

The script uses:
- **Config**: `LLaMA-Factory/examples/inference/deepseek2_lite_inference_raw_hf.yaml`
- **Backend**: HuggingFace transformers
- **Model**: Raw base model (no adapter)

### When to Use

1. **Baseline Testing**: Test the base model with HuggingFace backend
2. **Comparison**: Compare raw model performance across backends
3. **Debugging**: Troubleshoot adapter loading issues

---

## clear_gpu_memory.sh

Frees up GPU and memory resources by killing processes that are using them. Useful before training to ensure maximum resources are available.

### Usage

```bash
cd /home/sean/Documents/ktransformers
./scripts/clear_gpu_memory.sh
```

### What It Does

1. **Checks GPU Usage**: Shows current GPU memory and utilization
2. **Lists GPU Processes**: Identifies processes using GPU memory
3. **Kills gnome-remote-desktop-daemon**: Automatically kills processes that use GPU memory unnecessarily
4. **Checks Python Processes**: Asks for confirmation before killing training/inference processes
5. **Shows Final Status**: Displays GPU and memory status after cleanup

### Example Output

```
=== Checking GPU and Memory Usage ===

GPU Status:
0, 630, 16376, 15

GPU Processes:
1055335, /usr/libexec/gnome-remote-desktop-daemon, 249 MiB

=== Killing gnome-remote-desktop-daemon processes ===
Killing gnome-remote-desktop-daemon (PID: 1055335)
Done

=== Final Status ===
GPU Status:
0, 409, 16376, 2

GPU Processes:
No GPU processes

System Memory:
               total        used        free      shared  buff/cache   available
Mem:            93Gi        5.1Gi        53Gi        78Mi        36Gi        88Gi

=== Cleanup Complete ===
```

### When to Use

- **Before Training**: Free up GPU memory before starting training
- **After Inference**: Clean up after stopping inference servers
- **Memory Issues**: When you get OOM errors during training
- **Resource Management**: Periodically check and free resources

### Safety Features

- **Confirmation Required**: Asks before killing Python training/inference processes
- **Automatic Cleanup**: Only automatically kills known safe processes (gnome-remote-desktop-daemon)
- **Status Display**: Shows before/after status so you can verify cleanup worked

---

## create_identity_dataset.sh

Creates synthetic identity training datasets for SFT fine-tuning. Generates diverse question formats, direct identity statements, and date examples to reinforce model learning.

### Usage

```bash
cd /home/sean/Documents/ktransformers

# Basic usage (defaults: sean, 2026-01-01, 420 identity, 245 date)
./scripts/create_identity_dataset.sh

# Custom name and date
./scripts/create_identity_dataset.sh alice 2025-12-25

# Full customization
./scripts/create_identity_dataset.sh bob 2024-06-15 identity_bob.json 500 300

# Show help
./scripts/create_identity_dataset.sh --help
```

### Parameters

1. **name** (optional): Identity name (default: `sean`)
2. **date** (optional): Current date in YYYY-MM-DD format (default: `2026-01-01`)
3. **output_file** (optional): JSON filename (default: `identity_{name}.json`)
4. **num_identity** (optional): Number of identity examples (default: `420`)
5. **num_date** (optional): Number of date examples (default: `245`)

### What It Creates

The script generates a diverse dataset with:

- **Direct Identity Statements**: "I am {name}", "My name is {name}", etc.
- **Name Questions**: Various question formats asking about the name
- **Creator Questions**: Questions about who created/developed the model
- **Date Questions**: Questions about the current date
- **Conversational Context**: Natural conversations that include identity
- **Combined Questions**: Identity + date questions together

### Example Output

```
==========================================
Creating Synthetic Identity Dataset
==========================================
Name: alice
Date: 2025-12-25
Output file: /home/sean/Documents/ktransformers/LLaMA-Factory/data/identity_alice.json
Identity examples: 500
Date examples: 300

✓ Created dataset: /home/sean/Documents/ktransformers/LLaMA-Factory/data/identity_alice.json
  Total examples: 800
  Identity examples (with 'alice'): 500
  Date examples (with '2025-12-25'): 300

Sample examples:
  1. Q: Who are you?
     A: I am alice.
  2. Q: What is the current date?
     A: The current date is 2025-12-25.
```

### Registering the Dataset

After creating a dataset, register it in `LLaMA-Factory/data/dataset_info.json`:

```json
"identity_alice": {
  "file_name": "identity_alice.json"
}
```

Then use it in your training config:

```yaml
dataset: identity_alice
```

### Tips for Better Training

- **More Examples**: Increase `num_identity` and `num_date` for more training data
- **Diverse Names**: Create datasets with different names to test generalization
- **Current Dates**: Update the date parameter to reflect the actual current date
- **Multiple Datasets**: Combine multiple identity datasets for richer training

### When to Use

- **Identity Training**: When you need to train a model to know its name/identity
- **Date Training**: When you need to train a model to know the current date
- **Custom Scenarios**: Create datasets for specific identity/date requirements
- **Testing**: Generate test datasets with different parameters

---

## Troubleshooting

### Common Issues

#### 1. Conda Environment Not Found

**Error**: `Error: Failed to activate conda environment 'Kllama'`

**Solution**: Create the environment first:
```bash
conda create -n Kllama python=3.12
conda activate Kllama
# Install dependencies...
```

#### 2. Model Path Not Found

**Error**: `Error: Config file not found` or model loading errors

**Solution**: 
- Ensure the model is downloaded to `deepseek-ai/DeepSeek-V2-Lite-Chat/`
- Check that config files exist in `LLaMA-Factory/examples/`

#### 3. CUDA/nvcc Build Errors

**Error**: `math.h: No such file or directory` during rebuild

**Solution**: 
- Ensure g++-11 is installed: `sudo apt-get install gcc-11 g++-11`
- Check that `nvcc_wrapper` uses c++/11 paths (not c++/12)
- Run `rebuild_kt_kllama.sh` which handles this automatically

#### 4. Wandb Authentication Error

**Error**: Wandb login required

**Solution**: Set your API key:
```bash
export WANDB_API_KEY=your_api_key_here
```

#### 5. Checkpoint Not Found

**Error**: `Checkpoint X not found`

**Solution**: 
- List available checkpoints: `ls LLaMA-Factory/saves/Kllama_deepseekV2Lite/checkpoint-*`
- Use a valid checkpoint number or omit to use the final adapter

#### 6. Port Already in Use (API/Webchat)

**Error**: Port 8000 or 7860 already in use

**Solution**: 
- For API: `API_PORT=8001 ./scripts/infer_ds2_chat_lite.sh api`
- Kill the process using the port: `lsof -ti:8000 | xargs kill -9`

### Getting Help

1. Check script output for error messages
2. Review log files in the output directories
3. Verify all prerequisites are met
4. Ensure conda environment is properly activated

---

## Workflow Example

Complete workflow from rebuild to inference:

```bash
# 1. Rebuild KTransformers (if needed)
cd /home/sean/Documents/ktransformers
./scripts/rebuild_kt_kllama.sh

# 2. Fine-tune the model
./scripts/sft_ds2_chat_lite.sh

# 3. Run inference (interactive chat with fine-tuned model)
./scripts/infer_ds2_chat_lite.sh chat

# Or test the raw model (no fine-tuning)
./scripts/infer_ds2_chat_lite_raw.sh chat

# Or use web UI
./scripts/infer_ds2_chat_lite.sh webchat

# Or start API server
API_PORT=8000 ./scripts/infer_ds2_chat_lite.sh api
```

---

## Notes

- All scripts automatically activate the `Kllama` conda environment
- Scripts use absolute paths to avoid working directory issues
- Training checkpoints are saved every 100 steps (configurable in yaml)
- Wandb logs are sent at every step (`logging_steps: 1`)
- The inference script dynamically updates config for checkpoint selection

---

## File Locations

| Item | Path |
|------|------|
| Training Config | `LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_kt.yaml` |
| Inference Config (Fine-tuned) | `LLaMA-Factory/examples/inference/deepseek2_lite_inference.yaml` |
| Inference Config (Raw) | `LLaMA-Factory/examples/inference/deepseek2_lite_inference_raw.yaml` |
| Model Directory | `deepseek-ai/DeepSeek-V2-Lite-Chat/` |
| Training Output | `LLaMA-Factory/saves/Kllama_deepseekV2Lite/` |
| Checkpoints | `LLaMA-Factory/saves/Kllama_deepseekV2Lite/checkpoint-*/` |

---

---

## Quick Reference

### Training Workflow

```bash
# 1. Clear GPU/memory resources
./scripts/clear_gpu_memory.sh

# 2. Create training dataset (if needed)
./scripts/create_identity_dataset.sh sean 2026-01-01

# 3. Fine-tune the model
./scripts/sft_ds2_chat_lite.sh

# 4. Test inference (KTransformers backend)
./scripts/infer_ds2_chat_lite.sh chat

# 5. Test inference (HuggingFace backend)
./scripts/infer_ds2_chat_lite_hf.sh chat
```

### Inference Comparison

```bash
# Compare fine-tuned vs raw model (KTransformers)
./scripts/infer_ds2_chat_lite.sh chat        # Fine-tuned
./scripts/infer_ds2_chat_lite_raw.sh chat    # Raw

# Compare fine-tuned vs raw model (HuggingFace)
./scripts/infer_ds2_chat_lite_hf.sh chat     # Fine-tuned
./scripts/infer_ds2_chat_lite_raw_hf.sh chat # Raw

# Compare backends (fine-tuned)
./scripts/infer_ds2_chat_lite.sh chat        # KTransformers
./scripts/infer_ds2_chat_lite_hf.sh chat    # HuggingFace
```

---

Last updated: 2026-01-01

