# ZeRO-3: GPU vs Hybrid vs CPU Offload Comparison

## Overview

This document compares three DeepSpeed ZeRO-3 configurations for training large models:
1. **GPU Config** - Everything on GPU (fastest, most memory)
2. **Hybrid Config** - Parameters on GPU, optimizer on CPU (balanced)
3. **CPU Offload Config** - Everything offloaded to CPU (most memory-efficient, slowest)

---

## Quick Comparison Table

> **⚠️ Important**: These numbers are for **LoRA training** (not full fine-tuning). With LoRA rank 8, you only train ~0.2% of parameters, dramatically reducing memory needs.

| Aspect | GPU Config | Hybrid Config | CPU Offload Config |
|--------|-----------|---------------|-------------------|
| **Config File** | `ds_z3_gpu_config.json` | `ds_z3_hybrid_config.json` | `ds_z3_offload_config.json` |
| **Parameters Location** | GPU | GPU | CPU |
| **Optimizer Location** | GPU | CPU | CPU |
| **GPU Memory (14B + LoRA)** | ~4-6 GB | ~3-5 GB | ~3-4 GB |
| **CPU Memory (14B + LoRA)** | ~28 GB (base model) | ~28-35 GB (base + optimizer) | ~45-55 GB (base + overhead) |
| **CPU↔GPU Transfers** | None | Low (optimizer only) | Medium (LoRA params + optimizer) |
| **Training Speed** | 130-150% (fastest) | 120-130% (balanced) | 100% (baseline) |
| **Best For** | Large GPU memory | Limited GPU, want speed | Very limited GPU memory |

**Note**: For **full fine-tuning** (not LoRA), multiply GPU memory by ~3-4x and CPU memory by ~3-4x.

---

## Detailed Comparison

### Memory Usage (14B Parameter Model with LoRA Rank 8)

> **LoRA Training**: Only ~0.2% of parameters are trainable (LoRA adapters), base model is frozen.

| Component | GPU Config | Hybrid Config | CPU Offload Config |
|-----------|-----------|---------------|-------------------|
| **Base Model (frozen)** | ~28 GB (CPU) | ~28 GB (CPU) | ~28 GB (CPU) |
| **LoRA Parameters** | ~50-200 MB (GPU) | ~50-200 MB (GPU) | ~50-200 MB (CPU) |
| **LoRA Gradients** | ~50-200 MB (GPU) | ~50-200 MB (GPU) | ~50-200 MB (GPU/CPU) |
| **LoRA Optimizer States** | ~100-400 MB (GPU) | ~100-400 MB (CPU) | ~100-400 MB (CPU) |
| **ZeRO-3 Overhead** | ~0 GB | ~0-5 GB | ~15-25 GB (buffers, staging) |
| **Activations** | ~2-4 GB (GPU) | ~2-4 GB (GPU) | ~2-4 GB (GPU) |
| **Working Memory** | ~0.5 GB (GPU) | ~0.5 GB (GPU) | ~0.5 GB (GPU) |
| **Total GPU** | **~4-6 GB** | **~3-5 GB** | **~3-4 GB** |
| **Total CPU** | **~28 GB** | **~28-35 GB** | **~45-55 GB** |

**For Full Fine-Tuning** (not LoRA), multiply GPU memory by ~3-4x:
- GPU Config: ~15-18 GB GPU, ~0 GB CPU
- Hybrid Config: ~10-12 GB GPU, ~56 GB CPU  
- CPU Offload Config: ~7-10 GB GPU, ~112 GB CPU

### Access Patterns and Transfer Overhead

> **LoRA Training**: Only LoRA parameters are transferred (tiny ~50-200 MB), not the full 28 GB base model.

| Aspect | GPU Config | Hybrid Config | CPU Offload Config |
|--------|-----------|---------------|-------------------|
| **LoRA Parameter Transfers/Step** | 0 (on GPU) | 0 (on GPU) | Small transfers (~50-200 MB) |
| **Optimizer Transfers/Step** | 0 (on GPU) | 1 per 32 steps | 1 per 32 steps |
| **Total Transfers/Step** | 0 | ~0.03 | Low-Medium (LoRA is tiny) |
| **Transfer Frequency** | None | Very Low | Low-Medium |
| **Latency Impact** | None | Minimal | Low (LoRA params are small) |

**Example for LoRA training with gradient_accumulation_steps=32:**
- **GPU Config**: 0 transfers per step
- **Hybrid Config**: 0 LoRA parameter transfers + 1 optimizer transfer per 32 steps = ~0.03 transfers/step
- **CPU Offload Config**: Small LoRA parameter transfers (~50-200 MB) + 1 optimizer transfer per 32 steps

**Note**: For **full fine-tuning** (not LoRA), CPU Offload would have ~64 transfers/step of full 28 GB parameters, causing significant latency.

### Performance Characteristics

| Metric | GPU Config | Hybrid Config | CPU Offload Config |
|--------|-----------|---------------|-------------------|
| **Training Speed** | 130-150% (30-50% faster) | 120-130% (20-30% faster) | 100% (baseline) |
| **Memory Efficiency** | Low (uses most GPU) | Medium (balanced) | High (uses least GPU) |
| **CPU Utilization** | Low | Medium | High |
| **GPU Utilization** | High | High | Medium (waiting for transfers) |
| **Best Use Case** | Large GPU (24GB+) | Medium GPU (16GB) | Small GPU (8-12GB) |

### Computation Intensity

| Component | Compute Intensity | GPU Config | Hybrid Config | CPU Offload Config |
|-----------|------------------|-----------|---------------|-------------------|
| **Parameters** | Billions of FLOPs (matrix ops) | GPU ✅ | GPU ✅ | CPU (transferred) |
| **Optimizer** | Millions of FLOPs (arithmetic) | GPU | CPU | CPU |
| **Gradients** | Billions of FLOPs (backprop) | GPU ✅ | GPU ✅ | GPU ✅ |
| **Activations** | Billions of FLOPs (forward) | GPU ✅ | GPU ✅ | GPU ✅ |

**Key Insight**: Parameters are the most compute-intensive component (billions of FLOPs), so keeping them on GPU provides the biggest performance benefit.

---

## Configuration Details

### GPU Config (`ds_z3_gpu_config.json`)

```json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    // No offload_optimizer
    // No offload_param
  }
}
```

**Characteristics**:
- ✅ Fastest training speed (no CPU↔GPU transfers)
- ✅ Best GPU utilization
- ❌ Requires most GPU memory (~15-18 GB for 14B model)
- ❌ No CPU memory savings

**When to Use**:
- You have 24GB+ GPU memory
- Training speed is priority
- CPU memory is limited

### Hybrid Config (`ds_z3_hybrid_config.json`)

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "stage3_prefetch_bucket_size": 1e8,
    "stage3_max_live_parameters": 5e9,
    // No offload_param
  }
}
```

**Characteristics**:
- ✅ Good balance of speed and memory
- ✅ Parameters stay on GPU (no frequent transfers)
- ✅ Optimizer offloaded (saves 56 GB GPU memory)
- ✅ Only 1 optimizer transfer per gradient accumulation
- ⚠️ Requires ~56 GB CPU RAM for optimizer states

**When to Use**:
- You have 16GB GPU memory (like your setup)
- Want faster training than full CPU offload
- Have sufficient CPU RAM (~64GB+ recommended)
- **RECOMMENDED for most users with 16GB GPUs**

### CPU Offload Config (`ds_z3_offload_config.json`)

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": false,
  }
}
```

**Characteristics**:
- ✅ Uses least GPU memory (~3-4 GB for 14B + LoRA)
- ✅ Enables training on small GPUs (8-12GB)
- ⚠️ Slower training speed (LoRA params transferred, but much smaller than full model)
- ⚠️ LoRA parameters transferred (but only ~50-200 MB, not 28 GB)
- ⚠️ Requires ~45-55 GB CPU RAM (base model + ZeRO-3 overhead + LoRA optimizer)
  - Base model: ~28 GB
  - ZeRO-3 staging buffers: ~15-25 GB
  - LoRA optimizer: ~100-400 MB

**When to Use**:
- You have very limited GPU memory (8-12GB)
- Have sufficient CPU RAM (64GB+ for LoRA, 128GB+ for full fine-tuning)
- Training speed is less important than fitting the model
- Last resort when other configs cause OOM

**Note**: 
- With LoRA, CPU offload uses ~45-55 GB CPU RAM (not just 28 GB) due to ZeRO-3 overhead
- ZeRO-3 staging buffers add ~15-25 GB overhead for parameter management
- Actual usage: ~50 GB (as observed in your system)

---

## Why Hybrid Offloads Optimizer (Not Parameters)

### Memory Size Comparison

| Component | Size (14B model) | Notes |
|-----------|----------------|-------|
| **Model Parameters** | ~28 GB | FP16/BF16 format |
| **Optimizer States** | ~56 GB | 2x model size (momentum + variance for Adam) |

**Key Point**: Optimizer states are **2x larger** than parameters.

### Access Frequency Comparison

| Component | Access Pattern | Frequency |
|-----------|---------------|-----------|
| **Parameters** | Every layer, every step | ~64 transfers/step (32 forward + 32 backward) |
| **Optimizer** | Once per gradient accumulation | 1 transfer per 32 steps (~0.03/step) |

**Key Point**: Parameters are accessed **2000x more frequently** than optimizer.

### Computation Intensity Comparison

| Component | Compute Type | Intensity | Needs GPU? |
|-----------|-------------|-----------|------------|
| **Parameters** | Matrix multiplications | Billions of FLOPs | ✅ Yes (parallel ops) |
| **Optimizer** | Simple arithmetic | Millions of FLOPs | ❌ No (CPU can handle) |

**Key Point**: Parameters are **1000x more compute-intensive** than optimizer.

### Hybrid Strategy Logic

**Keep Parameters on GPU**:
- ✅ Eliminates 64 transfers per step (biggest performance win)
- ✅ Heavy matrix ops run on GPU where they belong
- ✅ 28 GB fits in 12GB free GPU (with ZeRO-3 sharding)

**Offload Optimizer to CPU**:
- ✅ Saves 56 GB GPU memory (2x more than parameters)
- ✅ Only 1 transfer per 32 steps (minimal overhead)
- ✅ Simple arithmetic doesn't need GPU parallelism

**Result**: Best balance - keeps compute-intensive, frequently-accessed parameters on GPU, while offloading larger but less-frequently-accessed optimizer to CPU.

---

## Decision Matrix

### Choose GPU Config if:
- ✅ You have 24GB+ GPU memory
- ✅ Training speed is top priority
- ✅ CPU memory is limited
- ✅ You want maximum performance

### Choose Hybrid Config if:
- ✅ You have 16GB GPU memory (like your setup)
- ✅ You want good balance of speed and memory
- ✅ You have 64GB+ CPU RAM
- ✅ **RECOMMENDED for most 16GB GPU users**

### Choose CPU Offload Config if:
- ✅ You have 8-12GB GPU memory
- ✅ You have 128GB+ CPU RAM
- ✅ Training speed is less important
- ✅ Other configs cause OOM errors

---

## Performance Summary

### Training Speed (Relative to CPU Offload)

```
CPU Offload:  ████████████████████ 100% (baseline)
Hybrid:       ████████████████████████ 120-130% (20-30% faster)
GPU:          ████████████████████████████ 130-150% (30-50% faster)
```

### GPU Memory Usage (14B Model + LoRA)

```
CPU Offload:  ███░░░░░░░░░░░░░░░░ 3-4 GB
Hybrid:       ████░░░░░░░░░░░░░░░ 3-5 GB
GPU:          █████░░░░░░░░░░░░░░ 4-6 GB
```

### CPU Memory Usage (14B Model + LoRA)

```
GPU Config:   ████████████████████ 28 GB (base model)
Hybrid:       ███████████████████████ 28-35 GB (base + LoRA optimizer)
CPU Offload:  ████████████████████████████████████ 45-55 GB (base + ZeRO-3 overhead)
```

**Note**: CPU Offload uses more RAM due to ZeRO-3 staging buffers and parameter management overhead (~15-25 GB extra).

**Note**: For **full fine-tuning** (not LoRA):
- GPU: 7-10 GB (CPU Offload), 10-12 GB (Hybrid), 15-18 GB (GPU)
- CPU: 112 GB (CPU Offload), 56 GB (Hybrid), 0 GB (GPU)

---

## Recommendations

### For Your Setup (16GB GPU, ~12GB free, LoRA Training)

**Best Choice: Any Config Works!** ✅

**Your Actual Usage** (LoRA training):
- GPU Memory: ~3.9 GB used (you have ~12.5 GB free)
- CPU RAM: ~57 GB used (you have ~36 GB free)
- **All three configs will fit easily!**

**Recommendation: GPU Config** (for maximum speed) ⚡

**Reasons**:
1. You have plenty of GPU memory (~12.5 GB free, only need ~4-6 GB)
2. Fastest training speed (no CPU↔GPU transfers)
3. LoRA parameters are tiny (~50-200 MB), so keeping them on GPU is trivial
4. Base model already on CPU (frozen), so no parameter transfers needed

**Usage**:
```yaml
deepspeed: examples/deepspeed/ds_z3_gpu_config.json
```

**Alternative: Hybrid Config** (if you want to save a bit more GPU memory):
- Still very fast (LoRA optimizer is tiny, ~100-400 MB)
- Saves ~100-400 MB GPU memory
- Minimal performance difference with LoRA

**Note**: With LoRA, the difference between configs is minimal since trainable parameters are tiny. Choose based on preference!

---

## References

- [DeepSpeed ZeRO-3 Documentation](https://www.deepspeed.ai/tutorials/zero/)
- [ZeRO Offload Tutorial](https://www.deepspeed.ai/tutorials/zero-offload/)
- [ZeRO-3 Configuration Guide](https://www.deepspeed.ai/docs/config-json/#zero-optimization-for-fp16-training)

