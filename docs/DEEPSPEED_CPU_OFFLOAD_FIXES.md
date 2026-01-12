# DeepSpeed CPU Offload Fixes - Complete Guide

This document summarizes all the issues encountered and fixes applied to enable DeepSpeed ZeRO-3 CPU offload training with LoRA.

## Table of Contents

1. [Overview](#overview)
2. [Issues Encountered](#issues-encountered)
3. [Fixes Applied](#fixes-applied)
4. [Quick Fix Commands](#quick-fix-commands)
5. [Patches Applied](#patches-applied)
6. [Verification](#verification)

## Overview

**Goal**: Enable DeepSpeed ZeRO-3 CPU offload for LoRA training of DeepSeek-V2-Lite-Chat on a 16GB GPU.

**Configuration**: Hybrid ZeRO-3 (parameters and optimizer offloaded to CPU with pin_memory)

**DeepSpeed Version**: 0.16.9 (pinned due to issue #7347)

**Status**: ✅ Working - Training and inference both functional

## Issues Encountered

### Issue 1: Missing C++12 Header File

**Error**:
```
fatal error: bits/new_allocator.h: No such file or directory
```

**Root Cause**: System C++12 headers were incomplete. The file exists in `/usr/include/c++/12.bak/bits/` but not in the active `/usr/include/x86_64-linux-gnu/c++/12/bits/` directory.

**Fix**: Copy the missing header file from backup location.

**Command** (requires sudo):
```bash
sudo cp /usr/include/c++/12.bak/bits/new_allocator.h /usr/include/x86_64-linux-gnu/c++/12/bits/new_allocator.h
```

**Script**: `scripts/deepspeed/fix_cpu_adam_header.sh`

---

### Issue 2: CPU Adam Compilation - Missing CUDA Libraries

**Error**:
```
cannot find -lcurand: No such file or directory
cannot find -lcudart: No such file or directory
```

**Root Cause**: DeepSpeed's CPU Adam builder couldn't find CUDA libraries during linking. The libraries exist in conda environment but weren't in the linker search path.

**Fixes Applied**:

1. **Patched DeepSpeed Builder** (`builder.py`):
   - Modified `extra_ldflags()` method to add conda CUDA library paths
   - Location: `/home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/deepspeed/ops/op_builder/builder.py`

2. **Created Library Symlinks**:
   ```bash
   cd /home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/nvidia/curand/lib
   ln -sf libcurand.so.10 libcurand.so
   
   cd /home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/nvidia/cuda_runtime/lib
   ln -sf libcudart.so.12 libcudart.so
   ```

---

### Issue 3: DeepSpeed CPU Adam Resume AttributeError

**Error**:
```
AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'
```

**Root Cause**: When resuming from checkpoint, DeepSpeed unpickles the optimizer but `__setstate__` doesn't reinitialize the C++ extension (`ds_opt_adam`), which cannot be pickled.

**Fix**: Patched `__setstate__` method to reinitialize `ds_opt_adam` after unpickling.

**Location**: `/home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/deepspeed/ops/adam/cpu_adam.py`

---

### Issue 4: PyTorch nvJitLink Symbol Error

**Error**:
```
ImportError: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12
```

**Root Cause**: PyTorch 2.6.0 was compiled with nvJitLink 12.4.127, but the environment was trying to use conda's 12.1.105 version, which doesn't have the `__nvJitLinkComplete_12_4` symbol.

**Fix**: Updated activation script to prioritize PyTorch's bundled nvJitLink 12.4.127.

**Location**: `/home/sean/miniconda3/envs/deepspeed-z3/etc/conda/activate.d/deepspeed_env.sh`

---

## Fixes Applied

### 1. Missing C++ Header Fix

**File**: `/usr/include/x86_64-linux-gnu/c++/12/bits/new_allocator.h`

**Action**: Copy from backup location

**Command**:
```bash
sudo cp /usr/include/c++/12.bak/bits/new_allocator.h /usr/include/x86_64-linux-gnu/c++/12/bits/new_allocator.h
```

**Verification**:
```bash
ls -la /usr/include/x86_64-linux-gnu/c++/12/bits/new_allocator.h
```

---

### 2. DeepSpeed Builder Patch

**File**: `/home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/deepspeed/ops/op_builder/builder.py`

**Method**: `extra_ldflags()` (around line 825)

**Change**: Added conda CUDA library paths to linker flags:

```python
def extra_ldflags(self):
    if self.build_for_cpu:
        return ['-fopenmp']

    if not self.is_rocm_pytorch():
        import os
        ld_flags = []
        # Add conda CUDA library paths for linking
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        if conda_prefix:
            curand_path = f"{conda_prefix}/lib/python3.11/site-packages/nvidia/curand/lib"
            cuda_runtime_path = f"{conda_prefix}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
            if os.path.exists(curand_path):
                ld_flags.append(f'-L{curand_path}')
            if os.path.exists(cuda_runtime_path):
                ld_flags.append(f'-L{cuda_runtime_path}')
        ld_flags.append('-lcurand')
        if not self.build_for_cpu:
            ld_flags.append(f'-L{self.get_cuda_lib64_path()}')
        return ld_flags

    return []
```

---

### 3. CPU Adam __setstate__ Patch

**File**: `/home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/deepspeed/ops/adam/cpu_adam.py`

**Method**: `__setstate__()` (around line 104)

**Change**: Reinitialize `ds_opt_adam` after unpickling:

```python
def __setstate__(self, state):
    super(DeepSpeedCPUAdam, self).__setstate__(state)
    for group in self.param_groups:
        group.setdefault('amsgrad', False)
    # Reinitialize ds_opt_adam after unpickling (fix for resume from checkpoint)
    # This is needed because the C++ extension cannot be pickled
    if not hasattr(self, 'ds_opt_adam'):
        from deepspeed.ops.op_builder import CPUAdamBuilder
        self.ds_opt_adam = CPUAdamBuilder().load()
        # Recreate the adam optimizer with saved parameters
        if hasattr(self, 'opt_id') and len(self.param_groups) > 0:
            group = self.param_groups[0]
            lr = group.get('lr', 1e-3)
            betas = group.get('betas', (0.9, 0.999))
            eps = group.get('eps', 1e-8)
            weight_decay = group.get('weight_decay', 0.0)
            adamw_mode = getattr(self, 'adam_w_mode', True)
            self.ds_opt_adam.create_adam(
                self.opt_id, lr, betas[0], betas[1], eps, weight_decay, adamw_mode,
                False  # should_log
            )
```

---

### 4. Library Symlinks

**Location**: Conda environment nvidia packages

**Commands**:
```bash
cd /home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/nvidia/curand/lib
ln -sf libcurand.so.10 libcurand.so

cd /home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/nvidia/cuda_runtime/lib
ln -sf libcudart.so.12 libcudart.so
```

**Verification**:
```bash
ls -la /home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/nvidia/curand/lib/libcurand.so
ls -la /home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so
```

---

### 5. Activation Script Update

**File**: `/home/sean/miniconda3/envs/deepspeed-z3/etc/conda/activate.d/deepspeed_env.sh`

**Change**: Prioritize PyTorch's bundled nvJitLink:

```bash
# Use PyTorch's bundled nvJitLink 12.4.127 via LD_PRELOAD to ensure it's loaded first
# PyTorch 2.6.0 requires __nvJitLinkComplete_12_4 symbol from nvJitLink 12.4
if [ -f "${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12" ]; then
    export LD_PRELOAD="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12:${LD_PRELOAD}"
elif [ -f "${CONDA_PREFIX}/lib/libnvJitLink.so.12.1.105" ]; then
    export LD_PRELOAD="${CONDA_PREFIX}/lib/libnvJitLink.so.12.1.105:${LD_PRELOAD}"
elif [ -f "${CONDA_PREFIX}/lib/libnvjitlink.so.12" ]; then
    export LD_PRELOAD="${CONDA_PREFIX}/lib/libnvjitlink.so.12:${LD_PRELOAD}"
fi
```

---

## Quick Fix Commands

If errors reappear, run these commands in order:

### 1. Fix Missing C++ Header (requires sudo)

```bash
sudo cp /usr/include/c++/12.bak/bits/new_allocator.h /usr/include/x86_64-linux-gnu/c++/12/bits/new_allocator.h
```

Or use the script:
```bash
sudo /home/sean/Documents/ktransformers/scripts/deepspeed/fix_cpu_adam_header.sh
```

### 2. Recreate Library Symlinks

```bash
cd /home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/nvidia/curand/lib
ln -sf libcurand.so.10 libcurand.so

cd /home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/nvidia/cuda_runtime/lib
ln -sf libcudart.so.12 libcudart.so
```

### 3. Clear CPU Adam Cache (if compilation fails)

```bash
rm -rf /home/sean/.cache/torch_extensions/py311_cu121/cpu_adam/*.o
rm -rf /home/sean/.cache/torch_extensions/py311_cu121/cpu_adam/*.so
rm -f /home/sean/.cache/torch_extensions/py311_cu121/cpu_adam/build.ninja
```

### 4. Verify PyTorch Installation

```bash
source /home/sean/miniconda3/etc/profile.d/conda.sh
conda activate deepspeed-z3
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 5. Test CPU Adam Compilation

```bash
source /home/sean/miniconda3/etc/profile.d/conda.sh
conda activate deepspeed-z3
python << 'EOF'
from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam
import torch
params = [torch.randn(10, 10, requires_grad=True)]
optimizer = DeepSpeedCPUAdam(params, lr=1e-3)
print("✅ CPU Adam works!")
EOF
```

---

## Patches Applied

### Patch 1: DeepSpeed Builder (`builder.py`)

**Location**: Line ~825 in `extra_ldflags()` method

**Purpose**: Add conda CUDA library paths to linker flags

**Status**: ✅ Applied

---

### Patch 2: CPU Adam `__setstate__` (`cpu_adam.py`)

**Location**: Line ~104 in `__setstate__()` method

**Purpose**: Reinitialize C++ extension after unpickling (resume from checkpoint)

**Status**: ✅ Applied

---

### Patch 3: LLaMA-Factory Version Check

**Location**: `/home/sean/Documents/ktransformers/LLaMA-Factory/src/llamafactory/hparams/parser.py` line ~200

**Purpose**: Allow DeepSpeed 0.16.9 (with note about CPU Adam patch)

**Status**: ✅ Applied

---

## Verification

### Verify CPU Adam Works

```bash
source /home/sean/miniconda3/etc/profile.d/conda.sh
conda activate deepspeed-z3
python << 'EOF'
from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam
import torch
import pickle

# Test initialization
params = [torch.randn(10, 10, requires_grad=True)]
optimizer = DeepSpeedCPUAdam(params, lr=1e-3)
print("✅ CPU Adam initialized")

# Test pickling/unpickling (simulates resume)
state = pickle.dumps(optimizer)
optimizer2 = pickle.loads(state)
print("✅ Pickle/unpickle successful")
print(f"✅ ds_opt_adam exists: {hasattr(optimizer2, 'ds_opt_adam')}")
EOF
```

### Verify Training Works

```bash
cd /home/sean/Documents/ktransformers
./scripts/training/sft_ds2_chat_lite_hf.sh --config LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_z3_bf16_regularized.yaml --dataset identity_sean_generated --yes
```

### Verify Inference Works

```bash
cd /home/sean/Documents/ktransformers
./scripts/inference/infer_ds2_chat_lite_hf.sh chat
```

---

## Configuration Files

### Training Config

**File**: `LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_z3_bf16_regularized.yaml`

**Key Settings**:
- `deepspeed: examples/deepspeed/ds_z3_hybrid_config.json` (CPU offload enabled)
- `bf16: true`
- `gradient_checkpointing: true`
- `low_cpu_mem_usage: true`

### DeepSpeed Config

**File**: `LLaMA-Factory/examples/deepspeed/ds_z3_hybrid_config.json`

**Key Settings**:
- `offload_optimizer: { device: "cpu", pin_memory: true }`
- `offload_param: { device: "cpu", pin_memory: true }`
- Optimized prefetch settings for 16GB GPU

---

## Troubleshooting

### If CPU Adam Compilation Fails

1. Check if header file exists:
   ```bash
   ls -la /usr/include/x86_64-linux-gnu/c++/12/bits/new_allocator.h
   ```
   If missing, run: `sudo cp /usr/include/c++/12.bak/bits/new_allocator.h /usr/include/x86_64-linux-gnu/c++/12/bits/new_allocator.h`

2. Check if library symlinks exist:
   ```bash
   ls -la /home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/nvidia/curand/lib/libcurand.so
   ls -la /home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so
   ```
   If missing, recreate them (see Quick Fix Commands section 2).

3. Clear cache and retry:
   ```bash
   rm -rf /home/sean/.cache/torch_extensions/py311_cu121/cpu_adam
   ```

### If PyTorch Import Fails

1. Check nvJitLink version:
   ```bash
   ls -la /home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/nvidia/nvjitlink/lib/
   ```

2. Verify activation script:
   ```bash
   cat /home/sean/miniconda3/envs/deepspeed-z3/etc/conda/activate.d/deepspeed_env.sh | grep -A 5 "nvJitLink"
   ```

3. Reinstall PyTorch if needed:
   ```bash
   conda activate deepspeed-z3
   pip uninstall -y torch torchvision torchaudio
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
   ```

### If Resume from Checkpoint Fails

1. Verify `__setstate__` patch is applied:
   ```bash
   grep -A 20 "__setstate__" /home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/deepspeed/ops/adam/cpu_adam.py | head -25
   ```
   Should show the reinitialization code.

2. If patch is missing, reapply it (see Patch 2 above).

---

## Files Modified

1. `/usr/include/x86_64-linux-gnu/c++/12/bits/new_allocator.h` (copied)
2. `/home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/deepspeed/ops/op_builder/builder.py` (patched)
3. `/home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/deepspeed/ops/adam/cpu_adam.py` (patched)
4. `/home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/nvidia/curand/lib/libcurand.so` (symlink)
5. `/home/sean/miniconda3/envs/deepspeed-z3/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so` (symlink)
6. `/home/sean/miniconda3/envs/deepspeed-z3/etc/conda/activate.d/deepspeed_env.sh` (updated)
7. `/home/sean/Documents/ktransformers/LLaMA-Factory/src/llamafactory/hparams/parser.py` (updated)

---

## Summary

All fixes have been applied and verified. Training with CPU offload is working, and inference is functional. The patches are persistent and will survive environment reactivation, but may need to be reapplied if:

- DeepSpeed is reinstalled
- Conda environment is rebuilt
- System libraries are updated

Use the Quick Fix Commands section above to quickly restore functionality if issues reappear.

