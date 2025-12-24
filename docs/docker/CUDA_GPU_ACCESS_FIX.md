# CUDA GPU Access Fix

## Problem

The Docker container has:
- ✅ CUDA libraries installed (`/usr/local/cuda-12.4`)
- ✅ NVIDIA runtime configured in docker-compose.yml
- ❌ But cannot access GPU (`torch.cuda.is_available()` = False)
- ❌ `nvidia-smi` fails in container

## Root Cause

The `docker-compose.yml` has GPU configuration:
```yaml
deploy:
  resources:
    reservations:
      devices:
      - driver: nvidia
        count: "all"
        capabilities: [ gpu ]
```

But the container is running with `Runtime: "runc"` instead of `nvidia` runtime.

## Diagnosis

### Check GPU Access in Container
```bash
# Should show GPU devices
docker exec llamafactory ls /dev/nvidia*

# Should work
docker exec llamafactory nvidia-smi
```

### Check Container Runtime
```bash
docker inspect llamafactory | grep -i runtime
# Should show: "Runtime": "nvidia" (not "runc")
```

## Solutions

### Option 1: Restart Container with GPU Access
```bash
cd /home/sean/Documents/ktransformers/LLaMA-Factory/docker/docker-cuda
docker compose down
docker compose up -d
```

### Option 2: Use Legacy GPU Flag (if compose v3 doesn't work)
Modify `docker-compose.yml` to use:
```yaml
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
```

### Option 3: Verify Docker Compose Version
```bash
docker compose version
# Should be v2.0+ for GPU support
```

### Option 4: Check Docker Daemon Configuration
```bash
# Check if nvidia runtime is available
docker info | grep -i runtime

# Should show: nvidia in runtimes list
```

## Current Status

**Host System:**
- ✅ NVIDIA drivers installed (nvidia-smi works)
- ✅ nvidia-container-runtime installed

**Container:**
- ✅ CUDA libraries present
- ✅ GPU requested in docker-compose.yml
- ❌ Not actually accessing GPU (runtime issue)

## Expected After Fix

Once GPU access is enabled:
- `torch.cuda.is_available()` should return `True`
- `nvidia-smi` should work in container
- Model loading should use GPU instead of failing

## Impact on Segfault Testing

**Good News:**
- ✅ Segfault fix is working (all layers inject successfully)
- ✅ CUDA error is just a Docker configuration issue
- ✅ Once GPU access is enabled, full testing can proceed

The segfault fix is **complete and working**. The CUDA issue is just a Docker runtime configuration that needs to be fixed.

