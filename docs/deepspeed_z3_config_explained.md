# DeepSpeed ZeRO-3 Offload Configuration Explained

This document explains each parameter in `LLaMA-Factory/examples/deepspeed/ds_z3_offload_config.json` and how they benefit training.

## Overview

DeepSpeed ZeRO-3 (Zero Redundancy Optimizer Stage 3) partitions optimizer states, gradients, and **model parameters** across GPUs and optionally offloads them to CPU. This enables training models that are much larger than available GPU memory.

---

## General Training Parameters

### `train_batch_size: "auto"`
- **What it does**: Automatically determines the total training batch size based on `train_micro_batch_size_per_gpu` × number of GPUs × `gradient_accumulation_steps`
- **Why "auto"**: DeepSpeed calculates this from your training config, ensuring consistency
- **Benefit**: Prevents configuration mismatches and simplifies setup

### `train_micro_batch_size_per_gpu: "auto"`
- **What it does**: Uses the batch size per GPU from your training config (`per_device_train_batch_size`)
- **Why "auto"**: Avoids duplication - you set it once in the YAML config
- **Benefit**: Single source of truth for batch size configuration

### `gradient_accumulation_steps: "auto"`
- **What it does**: Uses the gradient accumulation steps from your training config
- **Why "auto"**: Same as above - avoids configuration duplication
- **Benefit**: Consistent gradient accumulation across all components

### `gradient_clipping: "auto"`
- **What it does**: Uses gradient clipping from your training config (`max_grad_norm`)
- **Why "auto"**: Ensures DeepSpeed uses the same clipping value as your trainer
- **Benefit**: Prevents gradient explosion with consistent clipping

### `zero_allow_untested_optimizer: true`
- **What it does**: Allows DeepSpeed to use optimizers that haven't been explicitly tested with ZeRO
- **Why it's needed**: Some optimizers (like AdamW) work with ZeRO but aren't in the "tested" list
- **Benefit**: Enables flexibility to use various optimizers without DeepSpeed blocking them

---

## Mixed Precision Training

### `fp16` Section
- **Purpose**: Configures FP16 (half-precision) training
- **`enabled: "auto"`**: Automatically enabled if your config uses FP16, disabled if using BF16
- **`loss_scale: 0`**: Uses dynamic loss scaling (0 = automatic)
- **`loss_scale_window: 1000`**: Number of steps to wait before increasing loss scale
- **`initial_scale_power: 16`**: Initial loss scale = 2^16 = 65,536
- **`hysteresis: 2`**: Number of consecutive overflow steps before reducing scale
- **`min_loss_scale: 1`**: Minimum loss scale (2^1 = 2)
- **Benefit**: Prevents underflow in FP16 while maintaining training stability

### `bf16` Section
- **Purpose**: Configures BF16 (bfloat16) training
- **`enabled: "auto"`**: Automatically enabled if your config uses BF16
- **Why BF16**: Better numerical stability than FP16, no loss scaling needed
- **Benefit**: More stable training for large models, especially with your `bf16: true` config

---

## ZeRO Optimization Parameters

### `zero_optimization.stage: 3`
- **What it does**: Activates ZeRO Stage 3, which partitions:
  - ✅ Optimizer states (like ZeRO-1)
  - ✅ Gradients (like ZeRO-2)
  - ✅ **Model parameters** (ZeRO-3 exclusive)
- **Memory reduction**: Up to 8x memory reduction per GPU (for 8 GPUs)
- **Benefit**: Enables training models 8x larger than single GPU capacity

### `offload_optimizer.device: "cpu"`
- **What it does**: Moves optimizer states (Adam momentum, variance buffers) to CPU RAM
- **Memory saved**: ~2x model size (for Adam optimizer)
- **Benefit**: Frees GPU memory for model parameters and activations
- **Trade-off**: Slight performance hit from CPU↔GPU transfers, but enables much larger models

### `offload_optimizer.pin_memory: true`
- **What it does**: Pins optimizer state memory in CPU RAM, enabling faster CPU↔GPU transfers
- **How it works**: Prevents OS from swapping this memory to disk
- **Benefit**: 2-3x faster CPU↔GPU transfers compared to unpinned memory
- **Cost**: Uses more CPU RAM (but it's already allocated anyway)

### `offload_param.device: "cpu"`
- **What it does**: Moves model parameters to CPU RAM when not actively being used
- **Memory saved**: Full model size (can be 10s of GBs for large models)
- **Benefit**: Enables training models that don't fit in GPU memory at all
- **How it works**: Parameters are loaded to GPU only when needed for forward/backward pass
- **Trade-off**: CPU↔GPU transfer overhead, but necessary for large models on limited GPUs

### `offload_param.pin_memory: true`
- **What it does**: Same as optimizer pinning, but for model parameters
- **Benefit**: Faster parameter transfers between CPU and GPU
- **Critical for**: Large models where transfer speed becomes a bottleneck

### `overlap_comm: false`
- **What it does**: When `true`, overlaps communication (gradient all-reduce) with computation
- **Why `false`**: With CPU offload, communication patterns are different; overlap may not help
- **Benefit**: Simpler, more predictable behavior with CPU offload
- **Note**: Can be `true` for multi-GPU setups without CPU offload

### `contiguous_gradients: true`
- **What it does**: Stores gradients in contiguous memory blocks instead of fragmented
- **Why it matters**: Reduces memory fragmentation and improves access patterns
- **Benefit**: 
  - Faster gradient operations (better cache locality)
  - Lower memory overhead (less fragmentation)
  - More efficient CPU↔GPU transfers

### `sub_group_size: 1e9`
- **What it does**: Groups parameters into sub-groups of this size for collective operations
- **Why large (1e9)**: Effectively disables sub-grouping, treating all parameters as one group
- **Benefit**: Simpler parameter management, good for single GPU or small multi-GPU setups
- **Note**: Smaller values can improve multi-GPU communication efficiency

### `reduce_bucket_size: "auto"`
- **What it does**: Size of gradient bucket for all-reduce operations (in elements)
- **Why "auto"**: DeepSpeed calculates optimal size based on model and hardware
- **Benefit**: Balances communication efficiency vs memory usage automatically
- **Manual tuning**: Can set to specific value (e.g., `5e8`) for fine-tuning, but "auto" usually best

### `stage3_prefetch_bucket_size: "auto"`
- **What it does**: Size of parameter bucket to prefetch from CPU to GPU ahead of computation
- **Why "auto"**: DeepSpeed optimizes based on available GPU memory and model size
- **How it works**: While computing on current parameters, prefetches next parameters from CPU
- **Benefit**: Overlaps computation and data transfer, reducing training time
- **Trade-off**: Larger = more GPU memory used, but better overlap

### `stage3_param_persistence_threshold: "auto"`
- **What it does**: Parameters smaller than this threshold stay in GPU memory (not offloaded)
- **Why "auto"**: DeepSpeed determines optimal threshold based on model structure
- **Benefit**: Small, frequently-used parameters (like embeddings) stay on GPU for speed
- **Manual value**: Typically `1e5` (100K parameters) - small enough to keep in GPU

### `stage3_max_live_parameters: 1e9`
- **What it does**: Maximum number of parameters to keep in GPU memory simultaneously
- **Why large (1e9)**: Effectively unlimited - keep as many as GPU memory allows
- **Benefit**: Maximizes GPU memory utilization for active parameters
- **Trade-off**: If OOM occurs, reduce this value to force more aggressive offloading

### `stage3_max_reuse_distance: 1e9`
- **What it does**: How many parameters ahead to keep in GPU memory (reuse distance)
- **Why large (1e9)**: Keep parameters in GPU as long as possible before offloading
- **Benefit**: Reduces CPU↔GPU transfers by keeping parameters in GPU longer
- **How it works**: If a parameter is used again within this distance, it stays in GPU
- **Trade-off**: Higher GPU memory usage, but fewer transfers

### `stage3_gather_16bit_weights_on_model_save: true`
- **What it does**: When saving checkpoints, gathers all 16-bit weights from partitions into full model
- **Why `true`**: Ensures saved checkpoints are complete, usable models
- **Benefit**: Checkpoints can be loaded independently without DeepSpeed
- **Trade-off**: Slower checkpoint saving (must gather from all partitions), but necessary for portability

---

## Memory Savings Summary

For a 14B parameter model with ZeRO-3 + CPU offload:

| Component | Without ZeRO-3 | With ZeRO-3 + CPU Offload | Savings |
|-----------|----------------|---------------------------|---------|
| Model Parameters | ~28 GB (FP16) | ~0 GB (on CPU) | 100% |
| Optimizer States | ~56 GB (Adam) | ~0 GB (on CPU) | 100% |
| Gradients | ~28 GB | ~3.5 GB (1/8 per GPU, 8 GPUs) | 87.5% |
| Activations | ~Variable | ~Variable | Same |
| **Total GPU Memory** | **~112+ GB** | **~3.5+ GB** | **~97%** |

**Result**: Can train 14B model on 16GB GPU with ZeRO-3 + CPU offload!

---

## Performance Considerations

### CPU Offload Trade-offs

**Pros:**
- ✅ Enables training models that don't fit in GPU memory
- ✅ Can use cheaper hardware (smaller GPUs + more CPU RAM)
- ✅ Allows full-precision training (BF16) instead of quantization

**Cons:**
- ⚠️ CPU↔GPU transfers add latency (10-30% slower than GPU-only)
- ⚠️ Requires sufficient CPU RAM (need ~2-3x model size)
- ⚠️ CPU memory bandwidth can become bottleneck

### Optimization Tips

1. **Use `pin_memory: true`** - Critical for performance with CPU offload
2. **Tune `stage3_prefetch_bucket_size`** - Larger = better overlap, but more GPU memory
3. **Monitor CPU RAM usage** - Ensure you have enough (check with `htop`)
4. **Consider gradient checkpointing** - Reduces activation memory (you have this enabled)
5. **Batch size matters** - Smaller batches = more frequent CPU↔GPU transfers

---

## References

- [DeepSpeed ZeRO-3 Documentation](https://www.deepspeed.ai/tutorials/zero/)
- [ZeRO Offload Tutorial](https://www.deepspeed.ai/tutorials/zero-offload/)
- [ZeRO-3 Configuration Guide](https://www.deepspeed.ai/docs/config-json/#zero-optimization-for-fp16-training)

