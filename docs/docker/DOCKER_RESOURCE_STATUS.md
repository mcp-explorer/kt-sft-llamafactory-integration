# Docker Container Resource Status

## Container Status

**Container Name**: `llamafactory`  
**Status**: ✅ Running (Up 2 hours)  
**Image**: `docker-cuda-llamafactory`  
**Created**: 2 hours ago

## Resource Utilization

### Memory Usage

- **Current Usage**: 117.4 MiB
- **Memory Limit**: 93.98 GiB (no limit set - using host memory)
- **Memory Percentage**: 0.12%
- **Available in Container**: ~88 GiB
- **Total System Memory**: 96.24 GiB (98,550,224 kB)
- **Free Memory**: 18.65 GiB
- **Used Memory**: 4.26 GiB

**Status**: ✅ Very low memory usage - plenty of headroom

### CPU Usage

- **Current CPU**: 0.00%
- **Load Average**: 0.80, 0.33, 0.17
- **CPU Cores Available**: 32 cores
- **CPU Usage Breakdown**:
  - User: 2.5%
  - System: 0.6%
  - Idle: 96.9%

**Status**: ✅ Idle - no active processing

### GPU Usage

**GPU**: NVIDIA GeForce RTX 4080

- **GPU Utilization**: 13%
- **Memory Usage**: 520 MiB / 16,376 MiB (3.2%)
- **Power Usage**: 5W / 320W (1.6%)
- **Temperature**: 26°C
- **Performance State**: P8 (lowest power state)
- **Fan Speed**: 30%
- **CUDA Version**: 12.8
- **Driver Version**: 570.195.03

**Status**: ✅ GPU accessible, minimal usage (idle)

### Network I/O

- **Received**: 1.11 GB
- **Transmitted**: 2.74 MB

### Disk I/O

- **Read**: 233 kB
- **Written**: 2.28 GB

### Container Size

- **Image Size**: 17.1 GB
- **Container Size**: 1.15 GB

## Resource Limits

- **Memory Limit**: None (unlimited - uses host memory)
- **CPU Limit**: None (unlimited - uses all 32 cores)
- **GPU Access**: All GPUs (count: "all")

## Ports

- **7860**: LLaMA Board (Gradio UI) - Exposed
- **8000**: API service - Exposed

## Processes

- **Total Processes**: 4
- **Running**: 1
- **Sleeping**: 3
- **Main Process**: bash (PID 1)

## Summary

✅ **Container is healthy and running**  
✅ **Very low resource usage** - ready for model serving  
✅ **GPU accessible** - NVIDIA RTX 4080 detected  
✅ **Plenty of resources available**:
   - Memory: 0.12% used (117 MB of 94 GB available)
   - CPU: 0% used (idle)
   - GPU: 13% utilization, 520 MB / 16 GB memory used

## Recommendations

The container is in an **idle state** with minimal resource usage. This is normal when no model is loaded or serving requests.

When you run:
```bash
llamafactory-cli chat examples/inference/deepseek2_lite_serve.yaml
```

You should see:
- **Memory usage increase** (model loading)
- **GPU memory increase** (model weights on GPU)
- **GPU utilization increase** (inference processing)
- **CPU usage increase** (preprocessing, tokenization)

## Monitoring Commands

```bash
# Real-time stats
docker stats llamafactory

# GPU status (inside container)
docker compose exec llamafactory nvidia-smi

# Container status
docker compose ps

# Detailed container info
docker inspect llamafactory
```

