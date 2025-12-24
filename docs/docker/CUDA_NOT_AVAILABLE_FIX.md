# Why CUDA is Not Available in Docker (Even with nvidia-smi Working)

## Problem Summary

- ✅ `nvidia-smi` works on host
- ✅ GPU devices exist in container (`/dev/nvidia0`, `/dev/nvidiactl`)
- ✅ PyTorch is built with CUDA 12.4 support (`2.6.0+cu124`)
- ✅ CUDA toolkit is installed in container
- ❌ `torch.cuda.is_available()` returns `False`
- ❌ `nvidia-smi` fails in container: "Failed to initialize NVML: Unknown Error"
- ❌ Error: `RuntimeError: No CUDA GPUs are available`

## Root Cause

The container has GPU device files but **PyTorch cannot detect/access the GPUs**. This is typically caused by:

1. **Container needs restart** - GPU access may not be properly initialized
2. **NVIDIA Container Toolkit not properly configured** - The runtime may not be set up correctly
3. **Driver version mismatch** - Host driver (570.195.03, CUDA 12.8) vs container CUDA (12.4)
4. **Missing NVIDIA driver interface** - `/proc/driver/nvidia` may not be accessible

## Solutions

### Solution 1: Restart Container with Proper GPU Access (Recommended)

```bash
# Stop the container
cd /home/sean/Documents/ktransformers/LLaMA-Factory/docker/docker-cuda
docker compose down

# Restart with GPU access
docker compose up -d

# Verify GPU access
docker exec llamafactory nvidia-smi
docker exec llamafactory python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Solution 2: Use `--gpus all` Flag (If using docker run)

If you're using `docker run` instead of docker-compose:

```bash
docker run --gpus all --rm -it your-image python -c "import torch; print(torch.cuda.is_available())"
```

### Solution 3: Check NVIDIA Container Toolkit

Verify NVIDIA Container Toolkit is installed and configured:

```bash
# On host
which nvidia-container-runtime
nvidia-container-cli --version

# Check Docker daemon configuration
cat /etc/docker/daemon.json | grep -i nvidia
```

If not installed:
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Solution 4: Update docker-compose.yml

Ensure your `docker-compose.yml` has proper GPU configuration:

```yaml
services:
  llamafactory:
    # ... other config ...
    deploy:
      resources:
        reservations:
          devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
    # Alternative (newer syntax):
    # runtime: nvidia
    # environment:
    #   - NVIDIA_VISIBLE_DEVICES=all
```

### Solution 5: Set LD_LIBRARY_PATH in Container

Add CUDA library paths to environment:

```yaml
environment:
  - LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:/opt/conda/lib/python3.11/site-packages/torch/lib
  - CUDA_HOME=/usr/local/cuda-12.4
```

### Solution 6: For Build System (Workaround)

If you just need to build (not run), set `TORCH_CUDA_ARCH_LIST` to bypass detection:

```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.7;8.9;9.0+PTX"
```

This tells PyTorch which architectures to build for without needing runtime GPU access.

## Verification Steps

After applying fixes, verify:

```bash
# 1. Check GPU devices
docker exec llamafactory ls -la /dev/nvidia*

# 2. Check nvidia-smi
docker exec llamafactory nvidia-smi

# 3. Check PyTorch CUDA
docker exec llamafactory python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

# 4. Check driver access
docker exec llamafactory cat /proc/driver/nvidia/version
```

## Expected Output (After Fix)

```
CUDA available: True
Device count: 1
```

## Current Status

- **Host**: NVIDIA driver 570.195.03, CUDA 12.8, RTX 4080 detected
- **Container**: CUDA 12.4 toolkit installed, PyTorch 2.6.0+cu124
- **Issue**: Container cannot access GPU runtime (driver interface not accessible)

## Next Steps

1. **Try Solution 1 first** (restart container)
2. If that doesn't work, check **Solution 3** (NVIDIA Container Toolkit)
3. For immediate build needs, use **Solution 6** (set TORCH_CUDA_ARCH_LIST)

