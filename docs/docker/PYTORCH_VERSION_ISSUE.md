# PyTorch Version Mismatch Issue

## Current Situation

### Where PyTorch 2.6.0+cu124 Comes From

The PyTorch version comes from the **LLaMA-Factory Docker base image**:

```dockerfile
# From LLaMA-Factory/docker/docker-cuda/Dockerfile
ARG BASE_IMAGE=hiyouga/pytorch:th2.6.0-cu124-flashattn2.7.4-cxx11abi0-devel
FROM ${BASE_IMAGE}
```

This base image (`hiyouga/pytorch:th2.6.0-cu124-flashattn2.7.4-cxx11abi0-devel`) comes pre-installed with:
- **PyTorch 2.6.0+cu124** (CUDA 12.4)
- **Flash Attention 2.7.4**
- **Python 3.11**
- **CXX11 ABI = 0** (old ABI)

### What KTransformers Actually Needs

From `kt-sft/pyproject.toml`:
- **Minimum**: `torch >= 2.3.0` ✅ (2.6.0 meets this)

From setup scripts and documentation:
- **Recommended**: `torch==2.7.0` with CUDA 12.8 (`cu128`)
- **Example**: `scripts/setup_kt_serve_env.sh` installs:
  ```bash
  pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
  ```

### The Problem

1. **Version mismatch**: Container has 2.6.0, KTransformers recommends 2.7.0
2. **CUDA version mismatch**: Container has cu124 (CUDA 12.4), recommended is cu128 (CUDA 12.8)
3. **Build system issue**: PyTorch's CUDA architecture detection fails with 2.6.0+cu124

## Solutions

### Option 1: Upgrade PyTorch in Container (Recommended)

Upgrade to PyTorch 2.7.0+cu128 to match KTransformers recommendations:

```bash
docker exec llamafactory bash -c "
pip uninstall -y torch torchvision torchaudio
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c 'import torch; print(\"PyTorch:\", torch.__version__); print(\"CUDA available:\", torch.cuda.is_available())'
"
```

**Note**: This may require rebuilding Flash Attention if it's incompatible.

### Option 2: Use Pre-built KTransformers Wheel (Easier)

Instead of building from source, use a pre-built wheel that matches the container's PyTorch:

```bash
# Check current PyTorch version
docker exec llamafactory python -c "import torch; print(torch.__version__)"

# Download matching wheel from:
# https://github.com/kvcache-ai/ktransformers/releases/tag/v0.4.1
# For PyTorch 2.6.0+cu124, you'd need: ktransformers-0.4.1+cu124torch26*.whl

docker exec llamafactory bash -c "pip install ktransformers-0.4.1+cu124torch26fancy-cp311-cp311-linux_x86_64.whl"
```

### Option 3: Build Custom Docker Image

Create a new Dockerfile that installs the correct PyTorch version:

```dockerfile
# Start with a base that has CUDA 12.8
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Install KTransformers dependencies
RUN pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# ... rest of setup
```

### Option 4: Work Around Build Issue (Quick Fix)

If you just need to build, set `TORCH_CUDA_ARCH_LIST` to bypass detection:

```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.7;8.9;9.0+PTX"
```

This works even with PyTorch 2.6.0+cu124.

## Recommended Action

**For building from source**: Use **Option 4** (set TORCH_CUDA_ARCH_LIST) - it's the quickest fix.

**For production use**: Use **Option 1** (upgrade PyTorch) or **Option 2** (use pre-built wheel).

## Why This Matters

1. **Compatibility**: Newer PyTorch versions have bug fixes and optimizations
2. **CUDA version**: CUDA 12.8 has better support for newer GPUs (like RTX 4080)
3. **Build system**: PyTorch 2.7.0 may have better CUDA detection
4. **KTransformers wheels**: Pre-built wheels are tested with specific PyTorch versions

## Verification

After upgrading, verify:

```bash
docker exec llamafactory python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('Device count:', torch.cuda.device_count())
"
```

Expected output:
```
PyTorch version: 2.7.0+cu128
CUDA available: True
CUDA version: 12.8
Device count: 1
```

