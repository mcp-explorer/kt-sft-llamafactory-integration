# DeepSpeed Scripts

This directory contains all scripts related to DeepSpeed ZeRO-3 CPU offload setup, configuration, and troubleshooting.

## Quick Reference

### Setup & Environment

- **`rebuild_deepspeed_z3_env.sh`** - Rebuild the `deepspeed-z3` conda environment from scratch
- **`setup_deepspeed_z3_env.sh`** - Initial setup script (legacy, use rebuild instead)
- **`activate_deepspeed_z3.sh`** - Activation helper (legacy, use conda activate directly)
- **`verify_z3_setup.sh`** - Verify DeepSpeed ZeRO-3 setup is working

### Fixes & Troubleshooting

- **`fix_deepspeed_cpu_offload.sh`** - **Main fix script** - Automatically fixes all common CPU offload issues
  ```bash
  ./scripts/deepspeed/fix_deepspeed_cpu_offload.sh
  ```

- **`fix_cpu_adam_header.sh`** - Fix missing C++12 header file (requires sudo)
  ```bash
  sudo ./scripts/deepspeed/fix_cpu_adam_header.sh
  ```

### Testing & Optimization

- **`test_deepspeed_cpu_adam.sh`** - Test CPU Adam compilation
- **`binary_search_prefetch.sh`** - Binary search for optimal prefetch settings
- **`fine_tune_search.sh`** - Fine-tune prefetch settings with smaller increments
- **`ultra_fine_search.sh`** - Ultra-fine search with very small increments
- **`test_hard_limit.sh`** - Test hard limit configuration values
- **`refined_binary_search.sh`** - Refined binary search variant
- **`auto_binary_search_prefetch.sh`** - Automated binary search

### Training & Model Loading

- **`train_z3_test.sh`** - Test training with ZeRO-3
- **`test_z3_model_loading.sh`** - Test model loading with ZeRO-3
- **`test_z3_with_patch.sh`** - Test with patches applied
- **`test_z3_training_with_patch.sh`** - Test training with patches

### Patches (Legacy)

- **`apply_z3_low_cpu_mem_patch.sh`** - Apply low CPU memory patch (legacy)
- **`revert_z3_patch.sh`** - Revert patches (legacy)

## Common Issues & Fixes

### Issue: CPU Adam Compilation Fails

**Error**: `fatal error: bits/new_allocator.h: No such file or directory`

**Fix**:
```bash
sudo ./scripts/deepspeed/fix_cpu_adam_header.sh
```

### Issue: Missing CUDA Libraries

**Error**: `cannot find -lcurand` or `cannot find -lcudart`

**Fix**: Run the main fix script:
```bash
./scripts/deepspeed/fix_deepspeed_cpu_offload.sh
```

### Issue: All Issues

**Quick Fix**: Run the comprehensive fix script:
```bash
./scripts/deepspeed/fix_deepspeed_cpu_offload.sh
```

This script will:
1. Fix missing C++12 header (with sudo)
2. Create CUDA library symlinks
3. Clear CPU Adam cache
4. Verify activation script
5. Test CPU Adam compilation

## Documentation

For detailed information, see:
- **`../../docs/DEEPSPEED_CPU_OFFLOAD_FIXES.md`** - Complete guide to all fixes
- **`../../docs/FIX_PYTORCH_CUDA_ISSUES.md`** - PyTorch/CUDA compatibility fixes
- **`../../docs/zero3_gpu_vs_hybrid_vs_cpu_offload.md`** - ZeRO-3 configuration comparison

## Usage Examples

### Rebuild Environment
```bash
cd /home/sean/Documents/ktransformers
./scripts/deepspeed/rebuild_deepspeed_z3_env.sh --yes
```

### Fix All Issues
```bash
cd /home/sean/Documents/ktransformers
./scripts/deepspeed/fix_deepspeed_cpu_offload.sh
```

### Test CPU Adam
```bash
cd /home/sean/Documents/ktransformers
./scripts/deepspeed/test_deepspeed_cpu_adam.sh
```

### Binary Search for Optimal Settings
```bash
cd /home/sean/Documents/ktransformers
./scripts/deepspeed/binary_search_prefetch.sh
```

## Path Notes

All scripts in this directory use relative paths:
- `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`
- `PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"`

This ensures scripts work regardless of where they're called from.
