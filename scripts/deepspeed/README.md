# DeepSpeed ZeRO-3 Scripts

This directory contains all scripts related to DeepSpeed ZeRO-3 CPU offload setup and testing.

## Scripts Overview

### Setup & Environment
- **`setup_deepspeed_z3_env.sh`** - Automated conda environment setup
- **`activate_deepspeed_z3.sh`** - Environment activation with correct library paths

### Testing
- **`verify_z3_setup.sh`** - Verify complete setup status
- **`test_deepspeed_cpu_adam.sh`** - Test CPU Adam compilation
- **`test_z3_model_loading.sh`** - Test model loading (without patch)
- **`test_z3_with_patch.sh`** - Test model loading (with patch)
- **`test_z3_training_with_patch.sh`** - Test training (with patch)
- **`train_z3_test.sh`** - Training test script

### Patching
- **`apply_z3_low_cpu_mem_patch.sh`** - Apply experimental patch
- **`revert_z3_patch.sh`** - Revert patch
- **`patch_llamafactory_z3_low_cpu_mem.patch`** - Patch file

## Quick Start

```bash
# From project root
cd /home/sean/Documents/ktransformers

# Setup environment
bash scripts/deepspeed/setup_deepspeed_z3_env.sh

# Activate environment
source scripts/deepspeed/activate_deepspeed_z3.sh

# Verify setup
bash scripts/deepspeed/verify_z3_setup.sh
```

## Documentation

See `docs/README_DEEPSPEED.md` for complete documentation index.

## Notes

- All scripts use relative paths and work from project root
- Scripts automatically calculate PROJECT_ROOT (two levels up from this directory)
- Environment name: `deepspeed-z3`

