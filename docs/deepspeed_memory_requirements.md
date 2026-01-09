# DeepSpeed Memory Requirements for 14B Parameter Model

## Overview

This document calculates memory requirements for training a 14B parameter model using DeepSpeed ZeRO-3 with CPU offload, including both **full fine-tuning** and **LoRA training**.

**Quick Answer for LoRA (without quantization):**
- **GPU VRAM (training)**: ~3-6 GB (with ZeRO-3 + CPU offload)
- **CPU RAM**: ~28 GB (base model offloaded)
- **GPU VRAM (without CPU offload)**: ~15-18 GB (with ZeRO-3 only)

## Model Size Calculations

### Parameter Count: 14 Billion (14B)

**Memory per parameter:**
- FP32 (Full precision): 4 bytes
- FP16/BF16 (Half precision): 2 bytes

**Total model size:**
- FP32: 14B × 4 bytes = **56 GB**
- FP16/BF16: 14B × 2 bytes = **28 GB**

## Training Memory Components

### 1. Model Parameters
- **FP16/BF16**: 28 GB
- **FP32**: 56 GB (if using full precision)

### 2. Gradients
- Same size as parameters: **28 GB** (FP16/BF16) or **56 GB** (FP32)

### 3. Optimizer States (Adam)
- **Momentum**: Same size as parameters = 28 GB (FP16) or 56 GB (FP32)
- **Variance**: Same size as parameters = 28 GB (FP16) or 56 GB (FP32)
- **Total optimizer states**: **56 GB** (FP16) or **112 GB** (FP32)

### 4. Activations (Forward Pass)
- Depends on batch size and sequence length
- For batch_size=1, seq_len=2048: ~**2-4 GB** per layer
- With gradient checkpointing: ~**1-2 GB** total

## DeepSpeed ZeRO-3 CPU Offload Memory Breakdown

### GPU Memory (During Training)

**With ZeRO-3 + CPU Offload (FP16/BF16):**

| Component | GPU Memory | CPU Memory | Notes |
|-----------|------------|------------|-------|
| **Model Parameters** | ~0.5-1 GB | ~28 GB | Offloaded, only active layer on GPU |
| **Gradients** | ~0.5-1 GB | ~28 GB | Offloaded, only active gradients on GPU |
| **Optimizer States** | ~0 GB | ~56 GB | Fully offloaded to CPU |
| **Activations** | ~2-4 GB | 0 GB | Must stay on GPU during forward/backward |
| **CUDA Overhead** | ~0.5-1 GB | 0 GB | PyTorch/CUDA overhead |
| **TOTAL GPU** | **~4-7 GB** | **~112 GB** | |

**With ZeRO-3 + CPU Offload (FP32):**

| Component | GPU Memory | CPU Memory | Notes |
|-----------|------------|------------|-------|
| **Model Parameters** | ~0.5-1 GB | ~56 GB | Offloaded, only active layer on GPU |
| **Gradients** | ~0.5-1 GB | ~56 GB | Offloaded, only active gradients on GPU |
| **Optimizer States** | ~0 GB | ~112 GB | Fully offloaded to CPU |
| **Activations** | ~2-4 GB | 0 GB | Must stay on GPU during forward/backward |
| **CUDA Overhead** | ~0.5-1 GB | 0 GB | PyTorch/CUDA overhead |
| **TOTAL GPU** | **~4-7 GB** | **~224 GB** | |

### Model Initialization Memory (Critical!)

⚠️ **Important**: Before DeepSpeed can offload, the model must be initialized on GPU first.

**Initialization requires:**
- **FP16/BF16**: ~28-30 GB GPU memory (full model + overhead)
- **FP32**: ~56-60 GB GPU memory (full model + overhead)

This is the **bottleneck** we encountered with the 16GB GPU!

## Memory Requirements Summary

### For 14B Model with DeepSpeed ZeRO-3 CPU Offload

#### During Training (After Initialization)

**FP16/BF16 Training:**
- **GPU VRAM**: ~4-7 GB (with CPU offload)
- **CPU RAM**: ~112 GB (model + gradients + optimizer states)
- **Initial GPU VRAM**: ~28-30 GB (for initialization only)

**FP32 Training:**
- **GPU VRAM**: ~4-7 GB (with CPU offload)
- **CPU RAM**: ~224 GB (model + gradients + optimizer states)
- **Initial GPU VRAM**: ~56-60 GB (for initialization only)

#### Minimum Requirements

**For FP16/BF16:**
- **GPU**: 32GB+ VRAM (for initialization)
- **CPU RAM**: 128GB+ (112GB model data + system overhead)

**For FP32:**
- **GPU**: 64GB+ VRAM (for initialization)
- **CPU RAM**: 256GB+ (224GB model data + system overhead)

## Comparison Table: Full Fine-Tuning vs LoRA

### Full Fine-Tuning (14B Model)

| Configuration | GPU VRAM (Init) | GPU VRAM (Train) | CPU RAM | Total System RAM |
|---------------|-----------------|------------------|---------|------------------|
| **No ZeRO** | 28 GB | 84 GB | 0 GB | 84 GB |
| **ZeRO-3 (no offload)** | 28 GB | ~14 GB | 0 GB | 14 GB |
| **ZeRO-3 + CPU Offload (FP16)** | 28 GB | ~4-7 GB | 112 GB | 140 GB |
| **ZeRO-3 + CPU Offload (FP32)** | 56 GB | ~4-7 GB | 224 GB | 280 GB |

### LoRA Training (14B Model)

| Configuration | GPU VRAM (Init) | GPU VRAM (Train) | CPU RAM | Notes |
|---------------|-----------------|------------------|---------|-------|
| **LoRA (no ZeRO)** | 28 GB | ~30-33 GB | 0 GB | Base model frozen |
| **LoRA + ZeRO-3** | 28 GB | ~15-18 GB | 0 GB | Base model sharded |
| **LoRA + ZeRO-3 + CPU** | 28 GB | **~3-6 GB** | **~28 GB** | Base model offloaded |

## Real-World Recommendations

### For 14B Model Training

**Option 1: FP16/BF16 with ZeRO-3 CPU Offload** (Recommended)
- **GPU**: 32GB+ VRAM (RTX 3090, A100 40GB, etc.)
- **CPU RAM**: 128GB+
- **GPU Usage**: ~4-7 GB during training
- **CPU Usage**: ~112 GB during training

**Option 2: FP32 with ZeRO-3 CPU Offload**
- **GPU**: 64GB+ VRAM (A100 80GB, H100, etc.)
- **CPU RAM**: 256GB+
- **GPU Usage**: ~4-7 GB during training
- **CPU Usage**: ~224 GB during training

**Option 3: LoRA + ZeRO-3 + CPU Offload** (Maximum GPU Savings)
- **GPU**: 32GB+ VRAM (for initialization)
- **CPU RAM**: 64GB+ (28GB base model + overhead)
- **GPU Usage**: ~3-6 GB during training
- **CPU Usage**: ~28 GB during training
- **Best for**: When GPU memory is limited but CPU RAM is available

## Memory Optimization Tips

1. **Gradient Checkpointing**: Reduces activation memory by ~50%
   ```yaml
   gradient_checkpointing: true
   ```

2. **Smaller Batch Size**: Reduces activation memory
   ```yaml
   per_device_train_batch_size: 1
   gradient_accumulation_steps: 32  # Maintain effective batch size
   ```

3. **Mixed Precision**: Use BF16 instead of FP32
   ```yaml
   bf16: true  # Instead of fp32
   ```

4. **ZeRO Stage Selection**:
   - **ZeRO-2**: Offloads optimizer states only (~28 GB CPU RAM)
   - **ZeRO-3**: Offloads optimizer + parameters (~112 GB CPU RAM)

## Example Configuration

### For 14B Model on 32GB GPU + 128GB RAM

```yaml
### model
model_name_or_path: /path/to/14b_model
trust_remote_code: true
low_cpu_mem_usage: true  # With experimental patch

### train
bf16: true  # Use BF16, not FP32
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
gradient_checkpointing: true  # Critical for memory savings

# DeepSpeed ZeRO-3 with CPU offload
deepspeed: examples/deepspeed/ds_z3_offload_config.json
```

**Expected Memory Usage:**
- GPU VRAM: ~4-7 GB (after initialization)
- CPU RAM: ~112 GB
- Initial GPU VRAM: ~28-30 GB (one-time during init)

## Troubleshooting

### OOM During Initialization
- **Symptom**: Out of memory when loading model
- **Cause**: Model must load to GPU before DeepSpeed can offload
- **Solution**: Use larger GPU (32GB+) or multiple GPUs

### OOM During Training
- **Symptom**: Out of memory during forward/backward pass
- **Cause**: Activations or batch size too large
- **Solution**: Enable gradient checkpointing, reduce batch size

### CPU RAM Exhausted
- **Symptom**: System becomes slow, swapping to disk
- **Cause**: Insufficient CPU RAM for offloaded data
- **Solution**: Add more RAM or use ZeRO-2 (less CPU RAM needed)

## References

- [DeepSpeed ZeRO Documentation](https://www.deepspeed.ai/tutorials/zero/)
- [DeepSpeed ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/)
- [Model Memory Calculator](https://huggingface.co/docs/transformers/perf_train_gpu_one#memory-and-speed)

