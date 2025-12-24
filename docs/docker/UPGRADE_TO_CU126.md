# Upgrading Docker to PyTorch 2.6.0+cu126

## Changes Made

Updated `LLaMA-Factory/docker/docker-cuda/Dockerfile` to use PyTorch 2.6.0 with CUDA 12.6 (cu126), which is the official pairing and matches KTransformers pre-built wheels.

### Modified Dockerfile

The Dockerfile now:
1. Starts with the existing base image (which has cu124)
2. Upgrades PyTorch to 2.6.0+cu126 after the base image is loaded
3. This ensures compatibility with KTransformers `cu126torch26` wheels

## Rebuild Instructions

### Option 1: Rebuild Docker Image (Recommended)

```bash
cd /home/sean/Documents/ktransformers/LLaMA-Factory/docker/docker-cuda

# Stop current container
docker compose down

# Rebuild the image
docker compose build --no-cache

# Start the container
docker compose up -d

# Verify PyTorch version
docker exec llamafactory python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

### Option 2: Upgrade in Running Container (Faster, Temporary)

If you don't want to rebuild the entire image:

```bash
docker exec llamafactory bash -c "
pip uninstall -y torch torchvision torchaudio
pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -c 'import torch; print(\"PyTorch:\", torch.__version__); print(\"CUDA:\", torch.version.cuda); print(\"CUDA available:\", torch.cuda.is_available())'
"
```

**Note**: This change will be lost if you restart the container unless you rebuild the image.

## Verification

After rebuilding/upgrading, verify:

```bash
docker exec llamafactory python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA version:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
"
```

Expected output:
```
PyTorch version: 2.6.0+cu126
CUDA version: 12.6
CUDA available: True
Device count: 1
```

## Install KTransformers Wheel

Now you can use the matching pre-built wheel:

```bash
docker exec llamafactory bash -c "
pip install https://github.com/kvcache-ai/ktransformers/releases/download/v0.4.1/ktransformers-0.4.1%2Bcu126torch26fancy-cp311-cp311-linux_x86_64.whl
"
```

Or continue building from source (should work without TORCH_CUDA_ARCH_LIST workaround now).

## Flash Attention Compatibility

The existing Flash Attention 2.7.4 should still work with PyTorch 2.6.0+cu126, as it's built for `cu12torch2.6` which is compatible with both cu124 and cu126.

If you encounter issues, you may need to reinstall Flash Attention:

```bash
docker exec llamafactory bash -c "
pip uninstall -y flash-attn
pip install --no-cache-dir flash-attn --no-build-isolation
"
```

## Benefits

1. ✅ **Official PyTorch pairing**: 2.6.0 with cu126 is the standard combination
2. ✅ **Pre-built wheel available**: Can use `cu126torch26` KTransformers wheel
3. ✅ **Better compatibility**: Matches what KTransformers officially supports
4. ✅ **No build workarounds**: Should work without `TORCH_CUDA_ARCH_LIST` hack

## Rollback

If you need to rollback, revert the Dockerfile changes and rebuild:

```bash
cd /home/sean/Documents/ktransformers/LLaMA-Factory/docker/docker-cuda
git checkout docker/docker-cuda/Dockerfile
docker compose build --no-cache
docker compose up -d
```

