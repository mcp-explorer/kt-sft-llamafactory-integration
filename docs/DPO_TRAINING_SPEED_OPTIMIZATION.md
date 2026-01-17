# DPO Training Speed Optimization Guide

**Date:** 2026-01-15  
**Current Status:** Training is slow (GPU utilization ~47%)

## Current Configuration Analysis

### Current Settings (SLOW):
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
num_train_epochs: 10.0
max_samples: 500
dataloader_num_workers: 8
preprocessing_num_workers: 16
gradient_checkpointing: true
deepspeed: examples/deepspeed/ds_z3_hybrid_config.json  # CPU OFFLOAD = SLOW
```

### Performance Issues:
1. **CPU Offload (MAJOR BOTTLENECK):** `ds_z3_hybrid_config.json` offloads optimizer and parameters to CPU
   - CPU-GPU transfers are very slow
   - GPU utilization only ~47%
   - This is the #1 speed bottleneck

2. **Gradient Checkpointing:** Trades speed for memory (2-3x slower)

3. **Small Batch Size:** `per_device_train_batch_size: 1` with high gradient accumulation
   - Less efficient GPU utilization
   - More overhead per step

4. **Too Many Epochs:** 10 epochs for 500 samples is excessive
   - DPO typically needs 1-3 epochs
   - Each epoch processes 500 samples × 32 gradient accumulation = 16 steps

## Speed Optimization Options

### Option 1: Switch to GPU-Only DeepSpeed (FASTEST - if memory allows)

**Change:** Use `ds_z3_gpu_config.json` instead of `ds_z3_hybrid_config.json`

**Speed Gain:** 2-4x faster (no CPU-GPU transfers)

**Memory Impact:** Higher GPU memory usage (no CPU offload)

**How to test:**
1. Check current GPU memory usage
2. If < 80% used, switch to GPU-only config
3. Monitor for OOM errors

**Config change:**
```yaml
deepspeed: examples/deepspeed/ds_z3_gpu_config.json  # GPU-only, much faster
```

### Option 2: Reduce Epochs (QUICK WIN)

**Change:** `num_train_epochs: 10.0` → `num_train_epochs: 3.0`

**Speed Gain:** 3.3x faster (10 epochs → 3 epochs)

**Quality Impact:** Minimal - DPO typically converges in 1-3 epochs

**Config change:**
```yaml
num_train_epochs: 3.0  # DPO typically needs 1-3 epochs
```

### Option 3: Disable Gradient Checkpointing (if memory allows)

**Change:** `gradient_checkpointing: true` → `gradient_checkpointing: false`

**Speed Gain:** 2-3x faster per step

**Memory Impact:** ~2x more GPU memory usage

**Config change:**
```yaml
gradient_checkpointing: false  # Faster but uses more memory
```

### Option 4: Increase Batch Size (if memory allows)

**Change:** `per_device_train_batch_size: 1` → `per_device_train_batch_size: 2`
**And:** `gradient_accumulation_steps: 32` → `gradient_accumulation_steps: 16`

**Speed Gain:** 1.5-2x faster (fewer steps, better GPU utilization)

**Memory Impact:** 2x more GPU memory usage

**Config change:**
```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 16  # Keep effective batch size = 32
```

### Option 5: Increase DataLoader Workers

**Change:** `dataloader_num_workers: 8` → `dataloader_num_workers: 16`

**Speed Gain:** 10-20% faster (less data loading bottleneck)

**Memory Impact:** Minimal (CPU memory only)

**Config change:**
```yaml
dataloader_num_workers: 16  # More workers = faster data loading
```

### Option 6: Use ZeRO-2 Instead of ZeRO-3 (if memory allows)

**Change:** Use `ds_z2_config.json` instead of ZeRO-3

**Speed Gain:** 1.5-2x faster (less communication overhead)

**Memory Impact:** Higher GPU memory usage (less partitioning)

**When to use:** If GPU memory is sufficient for ZeRO-2

## Recommended Optimization Strategy

### Phase 1: Quick Wins (No Memory Risk)
1. ✅ **Reduce epochs:** `10.0` → `3.0` (3.3x faster)
2. ✅ **Increase dataloader workers:** `8` → `16` (10-20% faster)

**Expected speedup:** ~3.5x faster

### Phase 2: Memory-Dependent Optimizations
3. ⚠️ **Switch to GPU-only DeepSpeed:** `ds_z3_hybrid_config.json` → `ds_z3_gpu_config.json`
   - **Test first:** Monitor GPU memory usage
   - **If OOM:** Keep hybrid config
   - **If OK:** 2-4x faster

4. ⚠️ **Disable gradient checkpointing:** `true` → `false`
   - **Test first:** Monitor GPU memory usage
   - **If OOM:** Keep checkpointing
   - **If OK:** 2-3x faster

5. ⚠️ **Increase batch size:** `1` → `2` (with grad accum `32` → `16`)
   - **Test first:** Monitor GPU memory usage
   - **If OOM:** Keep batch size 1
   - **If OK:** 1.5-2x faster

### Combined Maximum Speedup
If all optimizations work: **10-20x faster** (from current ~47% GPU utilization to ~90%+)

## Implementation

### Quick Optimization (Safe)
```yaml
num_train_epochs: 3.0  # Was 10.0
dataloader_num_workers: 16  # Was 8
```

### Aggressive Optimization (Test Memory First)
```yaml
deepspeed: examples/deepspeed/ds_z3_gpu_config.json  # Was ds_z3_hybrid_config.json
gradient_checkpointing: false  # Was true
per_device_train_batch_size: 2  # Was 1
gradient_accumulation_steps: 16  # Was 32
num_train_epochs: 3.0  # Was 10.0
dataloader_num_workers: 16  # Was 8
```

## Monitoring

After applying optimizations, monitor:
1. **GPU utilization:** Should be 80-95% (currently ~47%)
2. **GPU memory:** Should be < 90% (to avoid OOM)
3. **Training speed:** Steps per second should increase
4. **Loss convergence:** Should still converge properly

## Expected Results

### Current (Slow):
- GPU utilization: ~47%
- Estimated time: ~X hours for 10 epochs

### After Quick Wins:
- GPU utilization: ~50-60%
- Estimated time: ~X/3.5 hours for 3 epochs

### After Full Optimization:
- GPU utilization: ~85-95%
- Estimated time: ~X/10-20 hours for 3 epochs

## Notes

- **DPO training is inherently slower than SFT** because it processes both chosen and rejected responses
- **CPU offload is the biggest bottleneck** - avoid if possible
- **Start with safe optimizations**, then test memory-dependent ones
- **Monitor for OOM errors** when increasing batch size or disabling checkpointing
