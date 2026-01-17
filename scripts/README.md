# Scripts Directory

This directory contains organized scripts for training, inference, monitoring, and utilities for the KTransformers project.

## Directory Structure

```
scripts/
├── training/          # Training scripts
│   ├── sft_ds2_chat_lite_hf.sh      # Fine-tune with HuggingFace backend
│   ├── sft_ds2_chat_lite.sh         # Fine-tune with KTransformers backend
│   └── dpo_ds2_chat_lite_hf.sh      # DPO (RL) training with HuggingFace backend
│
├── inference/         # Inference scripts
│   ├── infer_ds2_chat_lite_hf.sh    # Inference with HuggingFace (fine-tuned)
│   ├── infer_ds2_chat_lite_raw_hf.sh # Inference with HuggingFace (raw model)
│   ├── infer_ds2_chat_lite.sh       # Inference with KTransformers (fine-tuned)
│   └── infer_ds2_chat_lite_raw.sh   # Inference with KTransformers (raw model)
│
├── utils/             # Utility scripts
│   ├── convert_kt_adapter_to_hf.py  # Convert KTransformers adapter to HuggingFace
│   ├── convert_kt_adapter_to_hf.sh  # Wrapper script for adapter conversion
│   ├── rebuild_kt_kllama.sh         # Rebuild KTransformers package
│   └── create_identity_dataset.sh   # Create synthetic identity training data
│
├── monitoring/        # Monitoring and analysis scripts
│   ├── monitor_overfitting.sh       # Monitor training for overfitting
│   ├── analyze_batch_vram.sh        # Analyze batch size and VRAM usage
│   ├── view_wandb_results.sh        # View Weights & Biases training results
│   └── check_eval_results.sh        # Check evaluation results
│
├── testing/           # Testing scripts
│   ├── test_checkpoints.py          # Test model checkpoints
│   ├── test_checkpoints.sh          # Wrapper for checkpoint testing
│   └── quick_test_checkpoint.sh     # Quick checkpoint validation
│
├── helpers/           # Helper scripts
│   ├── wait_for_checkpoint.sh       # Wait for training checkpoint to be created
│   ├── clear_gpu_memory.sh         # Free GPU and memory resources
│   ├── find_best_checkpoint.py     # Find best checkpoint based on metrics
│   └── list_datasets.sh            # List available datasets and their paths
│
├── sft_data/         # SFT Data Generation and Conversion scripts
│   ├── generate_identity.sh        # Generate identity training data
│   ├── generate_date.sh            # Generate current date training data
│   ├── generate_identity_dpo.sh    # Generate DPO preference pairs for RL training
│   └── convert_to_llamafactory.sh  # Convert JSONL to LLaMA-Factory format + register
│
├── rl_training/      # RL Training (DPO) workflow scripts
│   ├── run_complete_rl_workflow.sh  # Master workflow: Generate → Train → Evaluate
│   ├── check_status.sh              # Check workflow status and component availability
│   ├── register_dpo_dataset.sh      # Register DPO dataset in dataset_info.json
│   ├── generate_comparison_report.py # Generate markdown report from evaluation results
│   └── README.md                    # RL training scripts documentation
│
├── deepspeed/         # DeepSpeed ZeRO-3 CPU Offload scripts
│   ├── fix_deepspeed_cpu_offload.sh # Main fix script (all issues)
│   ├── fix_cpu_adam_header.sh      # Fix C++ header (requires sudo)
│   ├── rebuild_deepspeed_z3_env.sh # Rebuild conda environment
│   ├── test_deepspeed_cpu_adam.sh  # Test CPU Adam compilation
│   └── ... (see deepspeed/README.md for full list)
│
└── docs/              # Documentation
    ├── README.md                    # Main scripts documentation
    ├── USAGE_GUIDE.md               # Usage guide for scripts
    ├── test_results_summary.md      # Test results summary
    └── test_backends.md             # Backend testing documentation
```

## Quick Start

### Training
```bash
# Fine-tune with HuggingFace backend (LoRA + ZeRO-3 + CPU offload)
# Note: --dataset flag is required (all configs use empty dataset field)
./scripts/training/sft_ds2_chat_lite_hf.sh --dataset identity_sean_generated

# Use custom config with different dataset
./scripts/training/sft_ds2_chat_lite_hf.sh \
  --config examples/train_lora/deepseek2_lite_sft_hf_z3_bf16_regularized.yaml \
  --dataset identity_sean_generated

# Fine-tune with KTransformers backend
./scripts/training/sft_ds2_chat_lite.sh
```

### Inference
```bash
# Run inference with HuggingFace backend
./scripts/inference/infer_ds2_chat_lite_hf.sh chat

# Run inference with KTransformers backend
./scripts/inference/infer_ds2_chat_lite.sh chat
```

### Utilities
```bash
# Convert KTransformers adapter to HuggingFace format
./scripts/utils/convert_kt_adapter_to_hf.sh

# Create synthetic identity dataset
./scripts/utils/create_identity_dataset.sh

# Rebuild KTransformers
./scripts/utils/rebuild_kt_kllama.sh
```

### Monitoring
```bash
# Monitor training for overfitting
./scripts/monitoring/monitor_overfitting.sh

# View WandB results
./scripts/monitoring/view_wandb_results.sh

# Analyze VRAM usage
./scripts/monitoring/analyze_batch_vram.sh
```

### Testing
```bash
# Test checkpoints
./scripts/testing/test_checkpoints.sh

# Quick checkpoint test
./scripts/testing/quick_test_checkpoint.sh
```

### Helpers
```bash
# Clear GPU memory
./scripts/helpers/clear_gpu_memory.sh

# Wait for checkpoint
./scripts/helpers/wait_for_checkpoint.sh

# List available datasets
./scripts/helpers/list_datasets.sh
```

### SFT Data Generation and Conversion
```bash
# Generate identity training data
./scripts/sft_data/generate_identity.sh

# Generate current date training data
./scripts/sft_data/generate_date.sh

# Convert JSONL to LLaMA-Factory format and auto-register
# Default: saves to LLaMA-Factory/data/{dataset-name}.json
./scripts/sft_data/convert_to_llamafactory.sh sft_data/outputs/my_data.jsonl \
  --dataset-name my_custom_dataset

# Convert with custom output filename
# Note: -o must be just a filename (no directory path)
./scripts/sft_data/convert_to_llamafactory.sh sft_data/outputs/my_data.jsonl \
  -o custom_filename.json \
  --dataset-name my_custom_dataset
```

**Note:** 
- All converted datasets are automatically saved to `LLaMA-Factory/data/` directory and registered in `dataset_info.json`
- The `-o` option must be just a filename (e.g., `-o my_file.json`), not a directory path
- If you specify a directory path in `-o`, the script will error with a helpful message
- If `-o` is omitted, the file is saved as `LLaMA-Factory/data/{dataset-name}.json`

### RL Training (DPO)
```bash
# Check workflow status
./scripts/rl_training/check_status.sh

# Run complete workflow: Generate → Train → Evaluate
./scripts/rl_training/run_complete_rl_workflow.sh

# Quick test (20 samples, 1 epoch)
./scripts/rl_training/run_complete_rl_workflow.sh --quick-test

# Generate DPO dataset only
./scripts/sft_data/generate_identity_dpo.sh --num_records 100

# Register DPO dataset
./scripts/rl_training/register_dpo_dataset.sh \
    sft_data/outputs/identity_sean_dpo_*.json

# Run DPO training only
./scripts/training/dpo_ds2_chat_lite_hf.sh --yes

# Evaluate all models (Raw vs SFT vs DPO)
./scripts/evaluation/run_evaluation.sh

# Generate comparison report
python scripts/rl_training/generate_comparison_report.py \
    evaluation_results.json --output comparison_report.md
```

**See:** [`rl_training/README.md`](rl_training/README.md) for detailed RL training documentation

## DeepSpeed ZeRO-3 CPU Offload

For DeepSpeed ZeRO-3 CPU offload setup and usage, see:
- [`deepspeed/README.md`](deepspeed/README.md) - Complete script reference
- [`../docs/DEEPSPEED_CPU_OFFLOAD_FIXES.md`](../docs/DEEPSPEED_CPU_OFFLOAD_FIXES.md) - Complete fix guide

**Quick Fix**: If you encounter issues, run:
```bash
./scripts/deepspeed/fix_deepspeed_cpu_offload.sh
```

## Documentation

For detailed documentation, see:
- [`docs/README.md`](docs/README.md) - Main scripts documentation
- [`docs/USAGE_GUIDE.md`](docs/USAGE_GUIDE.md) - Usage guide
- [`docs/test_backends.md`](docs/test_backends.md) - Backend testing guide
- [`../docs/RL_TRAINING_INDEX.md`](../docs/RL_TRAINING_INDEX.md) - RL training documentation index

## Data Storage

- **Training datasets**: All datasets must be in `LLaMA-Factory/data/` directory
- **Dataset registration**: Datasets are registered in `LLaMA-Factory/data/dataset_info.json`
- **Conversion output**: The `convert_to_llamafactory.sh` script always saves to `LLaMA-Factory/data/` regardless of `-o` path
- **Default naming**: If `-o` is not specified, files are saved as `LLaMA-Factory/data/{dataset-name}.json`

## Notes

- All scripts use relative paths from the project root (`/home/sean/Documents/ktransformers`)
- Scripts automatically detect conda environments and set up paths
- Most scripts support `--help` or `-h` flag for usage information
- Datasets must be registered in `dataset_info.json` to be used with `--dataset` flag

