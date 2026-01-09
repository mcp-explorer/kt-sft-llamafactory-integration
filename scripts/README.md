# Scripts Directory

This directory contains organized scripts for training, inference, monitoring, and utilities for the KTransformers project.

## Directory Structure

```
scripts/
├── training/          # Training scripts
│   ├── sft_ds2_chat_lite_hf.sh      # Fine-tune with HuggingFace backend
│   └── sft_ds2_chat_lite.sh         # Fine-tune with KTransformers backend
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
│   └── find_best_checkpoint.py     # Find best checkpoint based on metrics
│
├── deepspeed/         # DeepSpeed ZeRO-3 CPU Offload scripts
│   ├── setup_deepspeed_z3_env.sh   # Set up DeepSpeed environment
│   ├── activate_deepspeed_z3.sh    # Activate DeepSpeed environment
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
./scripts/training/sft_ds2_chat_lite_hf.sh

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
```

## DeepSpeed ZeRO-3 CPU Offload

For DeepSpeed ZeRO-3 CPU offload setup and usage, see:
- [`deepspeed/README.md`](deepspeed/README.md)
- [`docs/deepspeed_cpu_offload_issues.md`](../docs/deepspeed_cpu_offload_issues.md)

## Documentation

For detailed documentation, see:
- [`docs/README.md`](docs/README.md) - Main scripts documentation
- [`docs/USAGE_GUIDE.md`](docs/USAGE_GUIDE.md) - Usage guide
- [`docs/test_backends.md`](docs/test_backends.md) - Backend testing guide

## Notes

- All scripts use relative paths from the project root (`/home/sean/Documents/ktransformers`)
- Scripts automatically detect conda environments and set up paths
- Most scripts support `--help` or `-h` flag for usage information

