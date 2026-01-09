# DeepSpeed CPU Offload Issues Summary

> **📋 For a complete summary of all work, see: [`FINAL_SUMMARY.md`](FINAL_SUMMARY.md)**

## Overview

This document summarizes the issues encountered when attempting to use DeepSpeed ZeRO with CPU offloading for full-precision (BF16) training of DeepSeek-V2-Lite-Chat. The goal was to enable full-precision training on a 16GB GPU by offloading optimizer states and model parameters to CPU memory.

**Key Finding**: DeepSpeed CPU offload requires C++ extension compilation (CPU Adam optimizer) which failed due to missing system headers. Additionally, even with ZeRO-3, model initialization requires GPU memory before parameters can be offloaded, causing OOM errors on 16GB GPUs.

**Current Status**: ✅ Compilation fixed | ⚠️ Model loading works with patch | ❌ Training still OOM on 16GB GPU

## Quick Reference

> **📖 For a quick overview and common commands, see: [`deepspeed_z3_quick_reference.md`](deepspeed_z3_quick_reference.md)**

### TL;DR - Current Status
✅ **Compilation**: Fixed (CPU Adam compiles successfully)  
⚠️ **Model Loading**: Works with experimental patch (loads to CPU)  
❌ **Training**: Still OOM on 16GB GPU (DeepSpeed initialization requires GPU memory)

### Key Issues Summary
1. **Compilation Error**: Missing `stdlib.h` headers → Install `libc6-dev`
2. **OOM on Init**: Model must load to GPU before DeepSpeed can offload → Need 24GB+ GPU
3. **Version Mismatch**: DeepSpeed 0.18.4 incompatible → Downgrade to 0.16.9

### Quick Fixes
- **For Compilation**: Use `scripts/deepspeed/setup_deepspeed_z3_env.sh` (automated setup)
- **For Model Loading**: Apply experimental patch: `bash scripts/deepspeed/apply_z3_low_cpu_mem_patch.sh`
- **For Testing**: Run `bash scripts/deepspeed/test_z3_with_patch.sh` to verify CPU loading
- **For Full Precision**: Still requires 24GB+ GPU or multiple GPUs (or test with patch)

## Timeline of Events

### Initial Goal
- Use DeepSpeed CPU offload to enable full-precision (BF16) training
- Reduce GPU memory usage by offloading optimizer states and model parameters to CPU

### Configuration Changes

1. **Enabled DeepSpeed ZeRO-2 with CPU offload**:
   ```yaml
   deepspeed: examples/deepspeed/ds_z2_offload_config.json
   ```

3. **Upgraded to ZeRO-3 with CPU offload** (after ZeRO-2 OOM):
   ```yaml
   deepspeed: examples/deepspeed/ds_z3_offload_config.json
   ```

## Issues Encountered

### Issue 1: Missing FORCE_TORCHRUN Environment Variable

**Error:**
```
ValueError: Please use `FORCE_TORCHRUN=1` to launch DeepSpeed training.
```

**Solution:**
- Updated training script (`scripts/training/sft_ds2_chat_lite_hf.sh`) to automatically detect DeepSpeed in config and set `FORCE_TORCHRUN=1`
- Added detection logic:
  ```bash
  if grep -q "deepspeed:" "$CONFIG_FILE" && ! grep -q "^#.*deepspeed:" "$CONFIG_FILE"; then
      DEEPSPEED_ENABLED=true
      conda run -n "$CONDA_ENV" env FORCE_TORCHRUN=1 llamafactory-cli train "$CONFIG_FILE"
  fi
  ```

### Issue 2: DeepSpeed Version Compatibility

**Error:**
```
ImportError: deepspeed>=0.10.0,<=0.16.9 is required for a normal functioning of this module, 
but found deepspeed==0.18.4.
```

**Solution:**
- Downgraded DeepSpeed from 0.18.4 to 0.16.9:
  ```bash
  pip install "deepspeed>=0.10.0,<=0.16.9"
  ```

**Verifying DeepSpeed Installation:**
```bash
# Check DeepSpeed version
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"

# Verify compatibility with LLaMA-Factory
python -c "from llamafactory.extras.misc import check_dependencies; check_dependencies()"

# Test DeepSpeed import and basic functionality
python << EOF
import deepspeed
print(f"DeepSpeed {deepspeed.__version__} installed successfully")
print(f"CUDA available: {deepspeed.ops.op_builder.all_ops()}")
EOF
```

**Expected Output:**
- DeepSpeed version should be between 0.10.0 and 0.16.9
- No import errors
- CUDA operations should be available

### Issue 3: Out of Memory (OOM) with ZeRO-2

**Error:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. 
GPU 0 has a total capacity of 15.55 GiB of which 64.56 MiB is free.
```

**Root Cause:**
- ZeRO-2 only offloads optimizer states to CPU
- Model parameters still need to fit in GPU memory during initialization
- Full-precision (BF16) model (~15.8B parameters) requires ~6-8GB GPU memory
- With only 16GB GPU and other processes using memory, insufficient space

**Attempted Solution:**
- Upgraded to ZeRO-3, which also offloads model parameters to CPU

### Issue 4: ZeRO-3 OOM During Model Initialization

**Error:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.00 MiB.
GPU 0 has a total capacity of 15.55 GiB of which 70.56 MiB is free.
```

**Root Cause:**
- Even with ZeRO-3, model initialization requires GPU memory before parameters can be offloaded
- The model needs to be instantiated on GPU first, then DeepSpeed can partition and offload
- This initial allocation exceeded available GPU memory
- **Important**: LLaMA-Factory automatically disables `low_cpu_mem_usage` when DeepSpeed ZeRO-3 is enabled (see `LLaMA-Factory/src/llamafactory/model/patcher.py:153`), which prevents using HuggingFace's CPU offloading during model loading
- This means the full model must be loaded into GPU memory before DeepSpeed can take over

**Attempted Solution:**
- Tried ZeRO-3 without CPU offload (only parameter sharding), but still OOM
- The issue persists because the initial model loading happens before DeepSpeed initialization

### Issue 5: DeepSpeed CPU Adam Extension Compilation Failure

**Error:**
```
RuntimeError: Error building extension 'cpu_adam'
...
fatal error: stdlib.h: No such file or directory
   75 | #include_next <stdlib.h>
      |               ^~~~~~~~~~
compilation terminated.
```

**Root Cause:**
- DeepSpeed CPU offload requires compiling C++ extensions (CPU Adam optimizer)
- The compilation process cannot find system headers (`stdlib.h`)
- This is a system configuration issue, not a DeepSpeed bug

**Details:**
- Build tools are installed (`g++`, `gcc`, `make`, `build-essential`)
- The issue appears to be related to include paths or missing development headers
- Error occurs when DeepSpeed tries to JIT-compile the CPU Adam extension:
  ```python
  File ".../deepspeed/ops/adam/cpu_adam.py", line 94, in __init__
      self.ds_opt_adam = CPUAdamBuilder().load()
  ```

**Attempted Solutions:**
1. Installed `ninja` build system (already installed)
2. Tried manual compilation of CPU Adam extension (failed)
3. Checked for missing system headers (build tools present, but headers not found)

**Additional Context:**
- The compilation failure occurs during JIT (Just-In-Time) compilation when DeepSpeed first tries to use CPU Adam
- This is a one-time compilation that should create a cached extension, but fails before caching
- The error suggests missing C standard library development headers, typically provided by `libc6-dev` package

## Current Status

### Configuration Attempts

#### Attempt 1: ZeRO-2 with CPU Offload (Failed - OOM)
```yaml
deepspeed: examples/deepspeed/ds_z2_offload_config.json
```
**Result**: OOM during model initialization (ZeRO-2 doesn't offload model parameters)

#### Attempt 2: ZeRO-3 with CPU Offload (Failed - OOM + Compilation)
```yaml
deepspeed: examples/deepspeed/ds_z3_offload_config.json
```
**Result**: OOM during initialization + CPU Adam compilation failure

### Status Summary

✅ **DeepSpeed CPU offload compilation is working**
⚠️ **Model initialization workaround available** (experimental patch)

1. ✅ **Compilation fixed**: CPU Adam compiles successfully with proper library path configuration
2. ⚠️ **OOM workaround**: Experimental patch allows `low_cpu_mem_usage` with ZeRO-3, enabling model to load to CPU first
3. ⚠️ **Architectural limitation**: DeepSpeed ZeRO-3 still requires model on GPU before partitioning, but patch may allow CPU→GPU transfer during DeepSpeed initialization

### Test Results

**Environment Setup**: ✅ Complete
- Conda environment `deepspeed-z3` created successfully
- DeepSpeed 0.16.9 installed and verified
- CPU Adam compilation: ✅ **Working** (fixed library path issues)

**Model Loading Test (without patch)**: ❌ Failed (as expected)
- GPU: 16GB RTX 4080 SUPER
- Free memory at test time: ~2.7GB
- Result: OOM during model initialization
- Error: "CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 15.55 GiB of which 61.25 MiB is free."

**Model Loading Test (with patch)**: ✅ Model loads to CPU successfully
- Patch applied: Allows `low_cpu_mem_usage` with ZeRO-3
- Result: Model loads directly to CPU, avoiding GPU memory during initialization
- GPU memory used: 0.00 GB (model on CPU)

**Training Test (with patch)**: ❌ Still OOM during DeepSpeed initialization
- DeepSpeed ZeRO-3 intercepts model initialization via `partition_parameters` wrapper
- Even with model on CPU, DeepSpeed's initialization allocates GPU memory
- Error: "CUDA out of memory. Tried to allocate 40.00 MiB" during rotary embedding cache creation
- **Root cause**: DeepSpeed ZeRO-3 wraps model `__init__` methods, forcing GPU allocation
- **Conclusion**: Patch allows CPU loading, but DeepSpeed initialization still requires GPU memory

**Conclusion**: 
- ✅ Compilation issues are resolved
- ✅ Experimental patch allows model to load to CPU
- ❌ **DeepSpeed initialization still requires GPU memory**: Even with CPU-loaded model, DeepSpeed ZeRO-3's `partition_parameters` wrapper intercepts model initialization and allocates GPU memory during `__init__` methods
- ⚠️ **Fundamental limitation**: DeepSpeed ZeRO-3 architecture requires GPU memory during model initialization, regardless of initial loading location

## Technical Details

### DeepSpeed ZeRO Stages

| Stage | What's Sharded | CPU Offload Support | Memory Reduction |
|-------|---------------|---------------------|------------------|
| ZeRO-1 | Optimizer states | ✅ Optimizer | ~4x |
| ZeRO-2 | Optimizer + Gradients | ✅ Optimizer | ~8x |
| ZeRO-3 | Optimizer + Gradients + Parameters | ✅ Optimizer + Parameters | ~64x+ |

### Memory Requirements (Approximate)

**For DeepSeek-V2-Lite-Chat (15.8B params, BF16):**
- Model weights: ~3GB (BF16, 15.8B params)
- Gradients: ~3GB
- Optimizer states: ~6GB (Adam with momentum + variance)
- **Total: ~12GB** (without ZeRO)

**For 14B Parameter Model (BF16) with ZeRO-3 CPU Offload:**
- **GPU VRAM (during training)**: ~4-7 GB
- **CPU RAM**: ~112 GB (model + gradients + optimizer states)
- **GPU VRAM (initialization)**: ~28-30 GB (one-time, before offloading)
- See [`deepspeed_memory_requirements.md`](deepspeed_memory_requirements.md) for detailed breakdown

### Alternative Approaches Comparison

| Option | Memory Usage | Quality | Compilation | GPU Requirement | Use Case |
|--------|-------------|---------|-------------|-----------------|----------|
| **Full Precision (BF16)** | ~12GB | ⭐⭐⭐⭐⭐ Best | ❌ None | 24GB+ | Maximum quality, research |
| **ZeRO-3 + CPU Offload** | ~6-8GB init | ⭐⭐⭐⭐⭐ Best | ⚠️ Required | 24GB+ init | Full precision with offload |

**Notes:**
- **Full Precision**: No quantization loss, but requires much more memory
- **CPU Offload**: Requires compilation and larger GPU for initialization

### Why CPU Offload Failed

1. **Compilation Issue**: DeepSpeed CPU Adam requires C++ extension compilation
   - Missing system headers (`stdlib.h` not found)
   - Likely requires `libc6-dev` or similar development packages
   - System-level configuration issue
   - The compilation happens at runtime (JIT), so it must succeed before training can proceed

2. **Initialization Memory**: Even with ZeRO-3, model initialization needs GPU memory
   - Model must be instantiated before DeepSpeed can partition it
   - This initial allocation exceeded available GPU memory
   - **Critical**: LLaMA-Factory disables `low_cpu_mem_usage` when ZeRO-3 is enabled, preventing HuggingFace's CPU offloading during model loading
   - The sequence is: Load model → Initialize on GPU → DeepSpeed takes over → Offload to CPU
   - The first step fails due to insufficient GPU memory

3. **Architectural Limitation**: DeepSpeed ZeRO-3 requires the model to be fully instantiated before it can partition and offload parameters
   - Unlike HuggingFace's `device_map="auto"` which can load directly to CPU, DeepSpeed needs the model object first
   - This creates a chicken-and-egg problem: need GPU memory to load model, but want to offload to avoid using GPU memory

## Recommendations

### Setting Up a Dedicated Conda Environment

**Recommended**: Create a dedicated conda environment for ZeRO-3 CPU offload to avoid conflicts:

```bash
# Run the automated setup script
cd /home/sean/Documents/ktransformers
bash scripts/deepspeed/setup_deepspeed_z3_env.sh
```

The setup script will:
- Create conda environment `deepspeed-z3` with Python 3.11
- Install all build dependencies (gcc, g++, make, cmake, ninja)
- Install PyTorch with CUDA support
- Install DeepSpeed 0.16.9 (compatible version)
- Install LLaMA-Factory and dependencies
- Test CPU Adam compilation automatically

**Activate the environment**:
```bash
# Use the activation script (sets up library paths correctly)
source scripts/deepspeed/activate_deepspeed_z3.sh

# Or manually activate and set paths
conda activate deepspeed-z3
# See scripts/deepspeed/activate_deepspeed_z3.sh for required environment variables
```

**Test CPU Adam compilation**:
```bash
# Quick test to verify compilation works
bash scripts/deepspeed/test_deepspeed_cpu_adam.sh
```

**Note**: The activation script configures `LIBRARY_PATH` and `LD_LIBRARY_PATH` correctly to handle CUDA library linking. PyTorch has bundled CUDA libraries that need to be prioritized for runtime, while system libraries are needed for linking during compilation.

### For Full-Precision Training Without Quantization

1. **Fix Compilation Environment** (Required for CPU offload):
   ```bash
   # Install missing development headers (system-level)
   sudo apt-get install libc6-dev build-essential
   
   # Or use conda-forge packages (recommended for conda environments)
   conda install -c conda-forge gxx_linux-64
   
   # Verify installation
   gcc --version
   # Check for headers
   find /usr/include -name stdlib.h
   ```

2. **Address Model Initialization Memory Issue**:
   - **Option A**: Use a larger GPU (24GB+) that can handle initial model loading
   - **Option B**: Use multiple GPUs with ZeRO-3 (distributes initial load)
   - **Option C**: Modify LLaMA-Factory to allow `low_cpu_mem_usage` with ZeRO-3 (advanced, may break DeepSpeed)
   - **Option D**: Pre-load model with `device_map="cpu"` then transfer to DeepSpeed (experimental)

## Files Modified

1. **`LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_v3.yaml`**
   - Attempted DeepSpeed configuration
   - DeepSpeed disabled due to compilation and OOM issues

2. **`scripts/training/sft_ds2_chat_lite_hf.sh`**
   - Added DeepSpeed detection and `FORCE_TORCHRUN=1` handling
   - Automatically sets environment variable when DeepSpeed is detected
   - Detection logic checks for uncommented `deepspeed:` lines in config

3. **`scripts/deepspeed/setup_deepspeed_z3_env.sh`** (NEW)
   - Automated conda environment setup for ZeRO-3 CPU offload
   - Installs DeepSpeed 0.16.9 with all dependencies
   - Tests CPU Adam compilation automatically
   - Verifies system headers availability

4. **`scripts/deepspeed/test_deepspeed_cpu_adam.sh`** (NEW)
   - Quick test script to verify CPU Adam compilation
   - Can be run in any conda environment
   - Automatically configures library paths for CUDA linking
   - Provides clear error messages and troubleshooting steps

5. **`scripts/deepspeed/activate_deepspeed_z3.sh`** (NEW)
   - Activation script for DeepSpeed ZeRO-3 environment
   - Configures `LIBRARY_PATH` and `LD_LIBRARY_PATH` correctly
   - Ensures PyTorch's bundled CUDA libraries take precedence for runtime
   - Adds system CUDA libraries for compilation linking

6. **`LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_z3_test.yaml`** (NEW)
   - Test configuration for ZeRO-3 CPU offload
   - Full-precision training (no quantization)
   - Reduced dataset size for testing
   - ZeRO-3 with CPU offload enabled

7. **`scripts/deepspeed/test_z3_model_loading.sh`** (NEW)
   - Test script to verify model loading with ZeRO-3
   - Checks GPU memory availability
   - Tests model initialization (confirms OOM issue)
   - Provides clear diagnostics

8. **`scripts/deepspeed/train_z3_test.sh`** (NEW)
   - Training script for ZeRO-3 CPU offload test
   - Uses deepspeed-z3 environment
   - Automatically sets FORCE_TORCHRUN=1
   - Includes GPU memory checks

9. **`scripts/deepspeed/apply_z3_low_cpu_mem_patch.sh`** (NEW)
   - Applies experimental patch to allow `low_cpu_mem_usage` with ZeRO-3
   - Creates backup before modifying
   - Allows model to load to CPU before DeepSpeed initialization
   - **WARNING**: Experimental, may break DeepSpeed functionality

10. **`scripts/deepspeed/test_z3_with_patch.sh`** (NEW)
    - Tests model loading with patch applied
    - Verifies model can load to CPU
    - Confirms GPU memory is not used during initialization

11. **`scripts/deepspeed/test_z3_training_with_patch.sh`** (NEW)
    - Tests actual training with ZeRO-3 after patch
    - Verifies if DeepSpeed can initialize from CPU-loaded model
    - Includes timeout to stop after initialization test

12. **`scripts/deepspeed/revert_z3_patch.sh`** (NEW)
    - Reverts the experimental patch
    - Restores original LLaMA-Factory behavior
    - Preserves backup file

13. **`docs/deepspeed_z3_quick_reference.md`** (NEW)
    - Quick reference guide for ZeRO-3 setup
    - Status summary and key commands
    - Links to full documentation

## Troubleshooting Guide

### If You Encounter Compilation Errors

1. **Check build tools**:
   ```bash
   which gcc g++ make
   gcc --version
   ```

2. **Install development headers**:
   ```bash
   sudo apt-get update
   sudo apt-get install libc6-dev build-essential
   ```

3. **Verify header locations**:
   ```bash
   find /usr/include -name stdlib.h
   # Should find: /usr/include/stdlib.h
   ```

4. **Test DeepSpeed CPU Adam compilation manually**:
   ```python
   # Test script to verify CPU Adam compilation
   import deepspeed
   import torch
   
   print(f"DeepSpeed version: {deepspeed.__version__}")
   
   # Try to initialize CPU Adam optimizer (this triggers compilation)
   try:
       from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam
       
       # Create dummy parameters
       params = [torch.randn(10, 10, requires_grad=True)]
       
       # Try CPU Adam (this will trigger JIT compilation)
       print("Attempting to initialize CPU Adam...")
       optimizer = DeepSpeedCPUAdam(params, lr=1e-3)
       print("✅ CPU Adam compilation successful!")
       
   except RuntimeError as e:
       if "cpu_adam" in str(e).lower() or "compilation" in str(e).lower():
           print(f"❌ CPU Adam compilation failed: {e}")
           print("This is the issue preventing DeepSpeed CPU offload")
       else:
           raise
   except Exception as e:
       print(f"❌ Unexpected error: {e}")
       raise
   ```
   
   **Expected Behavior:**
   - ✅ Success: "CPU Adam compilation successful!" → Ready to use CPU offload
   - ❌ Failure: Compilation error → Install `libc6-dev` and retry

### If You Encounter OOM During Initialization

1. **Check available GPU memory**:
   ```bash
   nvidia-smi
   # Free up memory by killing other processes if needed
   ```

2. **Try ZeRO-2 first** (less aggressive, may work if you have ~8GB free):
   ```yaml
   deepspeed: examples/deepspeed/ds_z2_offload_config.json
   ```

3. **Reduce batch size and increase gradient accumulation**:
   ```yaml
   per_device_train_batch_size: 1
   gradient_accumulation_steps: 32
   ```

4. **Enable gradient checkpointing**:
   ```yaml
   gradient_checkpointing: true
   ```

### Testing Model Loading with ZeRO-3

**Test script available**: `scripts/deepspeed/test_z3_model_loading.sh`

This script:
- Checks GPU memory availability
- Tests model initialization without DeepSpeed (simulates the issue)
- Confirms OOM behavior on 16GB GPU
- Provides diagnostics and recommendations

**Expected result on 16GB GPU**: OOM during model initialization (confirmed)

### Alternative Approaches

If DeepSpeed CPU offload continues to fail, consider:

1. **Use FSDP (Fully Sharded Data Parallel)** - PyTorch's native alternative
2. **Use gradient checkpointing + smaller batch size** - May work without offload
3. **Upgrade hardware** - Larger GPU (24GB+) or multiple GPUs
4. **Modify LLaMA-Factory** - Allow `low_cpu_mem_usage` with ZeRO-3 (advanced, may break DeepSpeed)

## References

- [DeepSpeed ZeRO Documentation](https://www.deepspeed.ai/tutorials/zero/)
- [LLaMA-Factory DeepSpeed Support](https://github.com/hiyouga/LLaMA-Factory/blob/main/docs/training_accelerate.md)
- [DeepSpeed CPU Adam Source](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/ops/adam)
- [LLaMA-Factory Model Patcher](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/model/patcher.py) - See line 153 for ZeRO-3 compatibility logic

## Conclusion

### Compilation Issues: ✅ RESOLVED

The CPU Adam compilation issues have been **successfully resolved**:
- ✅ System headers available
- ✅ Build dependencies installed via conda
- ✅ CUDA library linking configured correctly (LIBRARY_PATH and LD_LIBRARY_PATH)
- ✅ CPU Adam compiles and loads successfully

**Solution**: Configure `LIBRARY_PATH` and `LD_LIBRARY_PATH` to prioritize PyTorch's bundled CUDA libraries for runtime while using system libraries for linking. Use the provided activation script: `scripts/deepspeed/activate_deepspeed_z3.sh`

### Remaining Challenge: Model Initialization OOM

While DeepSpeed CPU offload compilation is working, **model initialization still fails** on 16GB GPUs due to:
1. **DeepSpeed architecture limitation**: DeepSpeed ZeRO-3 uses `partition_parameters` wrapper that intercepts model `__init__` methods, forcing GPU allocation during initialization
2. **Even with CPU loading**: The experimental patch allows model to load to CPU, but DeepSpeed's initialization wrapper still allocates GPU memory when creating model components (e.g., rotary embeddings)
3. **Memory requirement**: Full-precision model (~15.8B parameters) needs ~6-8GB GPU memory for initialization
4. **16GB GPU insufficient**: With other processes using memory, insufficient space remains

**Test Results with Patch**:
- ✅ Model can load to CPU (patch works)
- ❌ DeepSpeed initialization still OOMs (architectural limitation)
- **Error location**: During rotary embedding cache creation in model `__init__` (wrapped by DeepSpeed)

**Test Results**:
- ✅ CPU Adam compilation: **Working** (fixed with library path configuration)
- ✅ DeepSpeed environment: **Fully configured**
- ✅ FORCE_TORCHRUN handling: **Working** (automatic detection in training script)
- ❌ Model initialization: **OOM on 16GB GPU** (confirmed by test)

The fundamental challenge is that DeepSpeed ZeRO-3 requires the model to be fully instantiated on GPU before it can partition and offload parameters. This creates a memory bottleneck during initialization that cannot be avoided with CPU offload alone.

### Recommended Approach for 16GB GPU

For 16GB GPUs, consider:
- **Gradient checkpointing** - Reduces memory during training
- **Smaller batch sizes** - With gradient accumulation to maintain effective batch size
- **Upgrade to larger GPU** - 24GB+ GPU allows full-precision training with ZeRO-3

### When to Consider DeepSpeed CPU Offload

The DeepSpeed CPU offload path can be pursued if:
- ✅ **Compilation environment ready**: Use `scripts/deepspeed/setup_deepspeed_z3_env.sh` (automated setup)
- ✅ **CPU Adam compilation working**: Verified with `scripts/deepspeed/test_deepspeed_cpu_adam.sh`
- ✅ **Larger GPU (24GB+) available** for initial model loading (or multiple GPUs)
- ✅ **Full-precision training required** (no quantization acceptable)
- ⚠️ **Model initialization**: Will still require GPU memory during DeepSpeed initialization (even with experimental patch)
- ❌ **16GB GPU insufficient**: Tested with experimental patch - DeepSpeed's `partition_parameters` wrapper still allocates GPU memory during model `__init__`

### Summary Table

| Approach | Memory Usage | Compilation | Initialization | Recommended For |
|----------|-------------|-------------|----------------|-----------------|
| **ZeRO-2 + CPU Offload** | ~6-8GB init | ✅ Fixed | ❌ OOM on 16GB | 24GB+ GPU |
| **ZeRO-3 + CPU Offload** | ~6-8GB init | ✅ Fixed | ❌ OOM on 16GB | 24GB+ GPU or Multi-GPU |
| **ZeRO-3 + CPU Offload + Patch** | ~6-8GB init | ✅ Fixed | ❌ Still OOM | 24GB+ GPU or Multi-GPU |
| **Full Precision (no ZeRO)** | ~12GB | ✅ None | ❌ OOM | 24GB+ GPU |

**Legend**: ✅ Works reliably | ⚠️ May work with fixes | ❌ Doesn't work

### Final Status Summary

**What Works**:
- ✅ CPU Adam compilation (fixed library path issues)
- ✅ Model can load to CPU with experimental patch
- ✅ DeepSpeed environment fully configured

**What Doesn't Work on 16GB GPU**:
- ❌ DeepSpeed ZeRO-3 initialization (architectural limitation)
- ❌ Full-precision training without quantization
- ❌ Model initialization requires GPU memory even with CPU loading patch

**Root Cause**:
DeepSpeed ZeRO-3's `partition_parameters` wrapper intercepts model `__init__` methods at the Python level, forcing GPU allocation during component creation (e.g., rotary embeddings, attention layers). This happens regardless of where the model was initially loaded, making it impossible to avoid GPU memory usage during initialization on a 16GB GPU with other processes running.

**Recommendation for 16GB GPU**:
- Use quantization (4-bit or 8-bit) instead of full-precision training
- Or upgrade to 24GB+ GPU for full-precision training with ZeRO-3
- Or use multiple GPUs to distribute the initial load

