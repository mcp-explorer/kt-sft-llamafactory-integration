# DeepSpeed ZeRO-3 CPU Offload Quick Reference

## TL;DR

✅ **Compilation**: Fixed - CPU Adam compiles successfully  
⚠️ **Model Loading**: Works with experimental patch (loads to CPU)  
❌ **Training**: Still OOM on 16GB GPU - DeepSpeed initialization requires GPU memory

## Quick Setup

### 1. Create Environment
```bash
cd /home/sean/Documents/ktransformers
bash scripts/deepspeed/setup_deepspeed_z3_env.sh
```

### 2. Activate Environment
```bash
source scripts/deepspeed/activate_deepspeed_z3.sh
```

### 3. Test CPU Adam Compilation
```bash
bash scripts/deepspeed/test_deepspeed_cpu_adam.sh
```

### 4. Verify Setup
```bash
# Run verification script to check everything
bash scripts/deepspeed/verify_z3_setup.sh
```

### 5. (Optional) Apply Experimental Patch
```bash
# Apply patch to allow low_cpu_mem_usage with ZeRO-3
bash scripts/deepspeed/apply_z3_low_cpu_mem_patch.sh

# Test model loading
bash scripts/deepspeed/test_z3_with_patch.sh

# Revert if needed
bash scripts/deepspeed/revert_z3_patch.sh
```

## Status

| Component | Status | Notes |
|-----------|--------|-------|
| Environment Setup | ✅ | Automated via script |
| CPU Adam Compilation | ✅ | Fixed library paths |
| Model Loading (no patch) | ❌ | OOM on 16GB GPU |
| Model Loading (with patch) | ✅ | Loads to CPU |
| Training (with patch) | ❌ | DeepSpeed init still OOMs |

## Key Files

- **Setup**: `scripts/deepspeed/setup_deepspeed_z3_env.sh`
- **Activation**: `scripts/deepspeed/activate_deepspeed_z3.sh`
- **Verify**: `scripts/deepspeed/verify_z3_setup.sh` - Check setup status
- **Test CPU Adam**: `scripts/deepspeed/test_deepspeed_cpu_adam.sh`
- **Apply Patch**: `scripts/deepspeed/apply_z3_low_cpu_mem_patch.sh`
- **Revert Patch**: `scripts/deepspeed/revert_z3_patch.sh`
- **Test Config**: `LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_z3_test.yaml`

## Known Limitations

1. **16GB GPU Insufficient**: Even with patch, DeepSpeed ZeRO-3 initialization requires GPU memory
2. **Architectural Limitation**: DeepSpeed's `partition_parameters` wrapper intercepts model `__init__` and allocates GPU memory
3. **Requires 24GB+ GPU**: For full-precision training with ZeRO-3 CPU offload

## Recommendations

- **For 16GB GPU**: Use quantization (4-bit/8-bit) instead of full-precision
- **For Full Precision**: Use 24GB+ GPU or multiple GPUs
- **For Testing**: All scripts and configurations are ready for larger GPUs

## CPU-GPU Hybrid Training

⚠️ **Important**: `device_map` is **NOT available for training** in LLaMA-Factory. The only CPU-GPU hybrid option for training is **DeepSpeed ZeRO-Offload** (already configured).

For details, see:
- **Guide**: `docs/cpu_gpu_hybrid_training.md`
- **Status**: DeepSpeed ZeRO-Offload is the only option for training
- **Note**: `device_map` only works for inference, not training

## Memory Requirements

For 14B parameter model with DeepSpeed ZeRO-3 CPU offload:
- **GPU VRAM (init)**: ~28-30 GB (one-time)
- **GPU VRAM (train)**: ~4-7 GB (with CPU offload)
- **CPU RAM**: ~112 GB (model + gradients + optimizer states)

See [`docs/deepspeed_memory_requirements.md`](deepspeed_memory_requirements.md) for detailed breakdown.

## Full Documentation

- **DeepSpeed ZeRO-3**: `docs/deepspeed_cpu_offload_issues.md` - Complete details, troubleshooting, and technical analysis
- **Memory Requirements**: `docs/deepspeed_memory_requirements.md` - Detailed memory calculations for different model sizes
- **CPU-GPU Hybrid**: `docs/cpu_gpu_hybrid_training.md` - Alternative approaches using device_map

