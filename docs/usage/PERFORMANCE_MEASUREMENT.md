# Performance Measurement Guide

This document describes how to measure and monitor performance when running KTransformers with LLaMA-Factory in Docker.

## GPU Monitoring

### Using nvidia-smi

```bash
# Real-time monitoring
docker compose exec llamafactory nvidia-smi

# Continuous monitoring (every 1 second)
watch -n 1 docker compose exec llamafactory nvidia-smi

# Query specific metrics
docker compose exec llamafactory nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

### Using docker stats

```bash
# Container resource usage
docker stats llamafactory --no-stream

# Real-time monitoring
docker stats llamafactory
```

## Memory Monitoring

### Inside Container

```bash
docker compose exec llamafactory bash -c "free -h"
docker compose exec llamafactory bash -c "cat /proc/meminfo | grep -E 'MemTotal|MemAvailable|MemFree'"
```

### Container Memory Limits

```bash
docker inspect llamafactory --format='{{.HostConfig.Memory}}'
```

## CPU Monitoring

### Inside Container

```bash
docker compose exec llamafactory bash -c "top -bn1 | head -20"
docker compose exec llamafactory bash -c "nproc"
```

## Model Inference Performance

### During Chat

Monitor GPU usage while the model is generating:

```bash
# Terminal 1: Run model
docker compose exec llamafactory bash
cd /app
llamafactory-cli chat examples/inference/deepseek2_lite_serve_custom.yaml

# Terminal 2: Monitor GPU
watch -n 1 docker compose exec llamafactory nvidia-smi
```

### Logging Performance Metrics

To log performance metrics to a file:

```bash
# Log GPU stats
while true; do
  docker compose exec llamafactory nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature --format=csv,noheader >> gpu_stats.log
  sleep 1
done
```

## Expected Resource Usage

### Model Loading
- **GPU Memory**: ~500 MB - 2 GB (depending on model size)
- **CPU**: Low (< 5%)
- **Time**: 30-60 seconds for first load

### Inference (Idle)
- **GPU Memory**: ~500 MB - 1 GB
- **GPU Utilization**: 0-5%
- **CPU**: < 1%

### Inference (Active Generation)
- **GPU Memory**: Model size + KV cache
- **GPU Utilization**: 50-100%
- **CPU**: 10-30% (tokenization, preprocessing)

## Troubleshooting Performance Issues

### High Memory Usage
- Check model size and batch size
- Reduce `cpu_infer` parameter
- Reduce `chunk_size` parameter

### Low GPU Utilization
- Check if model is actually running inference
- Verify CUDA is working: `docker compose exec llamafactory nvidia-smi`
- Check for bottlenecks in data loading

### Slow Inference
- Check GPU utilization (should be high during generation)
- Verify model is on GPU, not CPU
- Check for memory swapping

