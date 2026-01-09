# CPU-GPU Hybrid Training Guide

## Overview

CPU-GPU hybrid training allows you to utilize both CPU and GPU resources simultaneously, enabling training of larger models on limited GPU memory.

⚠️ **Important**: In LLaMA-Factory, **only DeepSpeed ZeRO-Offload supports CPU-GPU hybrid for training**. The `device_map` approach is **only available for inference**, not training.

## Available Options

1. **DeepSpeed ZeRO-Offload** - ✅ Works for training (offloads optimizer states and parameters to CPU)
2. **HuggingFace device_map** - ❌ Only works for inference (not available for training)
3. **Combined Approach** - Not applicable (device_map not available for training)

## Option 1: DeepSpeed ZeRO-Offload (Recommended)

This is what we've been working on. DeepSpeed ZeRO-Offload automatically offloads optimizer states and model parameters to CPU during training.

### Configuration

**ZeRO-2 with CPU Offload** (offloads optimizer states):
```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

**ZeRO-3 with CPU Offload** (offloads optimizer states + parameters):
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
    }
  }
}
```

### Usage

```yaml
# In your training config
deepspeed: examples/deepspeed/ds_z3_offload_config.json
```

**Pros**:
- ✅ Automatic and transparent
- ✅ Works with existing training code
- ✅ Efficient CPU-GPU data transfer
- ✅ Supports gradient accumulation

**Cons**:
- ❌ Still requires GPU memory for model initialization
- ❌ May have performance overhead from CPU-GPU transfers

## Option 2: HuggingFace device_map (Layer-Level Hybrid)

⚠️ **LIMITATION**: `device_map` is **NOT configurable for training** in LLaMA-Factory. It's automatically set to a single GPU device during training. This approach only works for **inference**.

For training, LLaMA-Factory automatically sets:
```python
model_args.device_map = {"": get_current_device()}  # Single GPU only
```

**For inference only**, you can use:
```yaml
# In your inference config
low_cpu_mem_usage: true
device_map: "auto"  # Automatically distributes layers (inference only)
offload_folder: /tmp/hf_offload
```

### Manual Layer Distribution

You can manually specify which layers go where:

```python
device_map = {
    "model.embed_tokens": "cpu",
    "model.layers.0": "cuda:0",
    "model.layers.1": "cuda:0",
    "model.layers.2": "cuda:0",
    # ... more layers on GPU
    "model.layers.20": "cpu",  # Some layers on CPU
    "model.layers.21": "cpu",
    "lm_head": "cuda:0"
}
```

### Example Config (Inference Only)

⚠️ **Note**: This only works for **inference**, not training!

```yaml
### model
model_name_or_path: /path/to/model
trust_remote_code: true

# CPU-GPU hybrid via device_map (inference only)
low_cpu_mem_usage: true
device_map: "auto"  # Automatically distributes layers
offload_folder: /tmp/hf_offload

# Note: For training, use DeepSpeed ZeRO-Offload instead
```

**Pros**:
- ✅ Fine-grained control over layer placement (inference)
- ✅ Can place less-used layers on CPU (inference)
- ✅ Works without DeepSpeed (inference)

**Cons**:
- ❌ **NOT available for training** - LLaMA-Factory sets device_map to single GPU automatically
- ❌ Only works for inference
- ❌ Performance overhead from CPU-GPU transfers during forward/backward passes

## Option 3: Combined Approach (Advanced)

You can potentially combine both approaches, but this requires careful configuration:

1. Use `device_map` to load model with some layers on CPU
2. Then use DeepSpeed ZeRO-3 for additional optimization

**Note**: This is experimental and may have conflicts. Test carefully.

## Comparison

| Approach | CPU Usage | GPU Usage | Complexity | Best For |
|----------|-----------|-----------|------------|----------|
| **ZeRO-Offload** | Optimizer + Params | Model layers | Low | Full-precision training |
| **device_map** | Selected layers | Selected layers | Medium | Fine-grained control |
| **Combined** | Both | Both | High | Maximum memory savings |

## Recommendations

### For Your 16GB GPU:

1. **Use DeepSpeed ZeRO-3 with CPU Offload** (what we've set up):
   ```bash
   # Already configured in:
   # LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_z3_test.yaml
   ```
   ⚠️ **Status**: Compilation fixed, but still OOM on 16GB GPU during initialization

2. **Use Quantization Instead** (Recommended for 16GB GPU):
   ```yaml
   quantization_bit: 4  # 4-bit quantization
   quantization_type: nf4
   # This reduces memory by ~75%
   ```

3. **device_map for Inference Only**:
   ```yaml
   # Only works for inference, not training
   low_cpu_mem_usage: true
   device_map: "auto"
   ```

## Example: device_map Hybrid (Inference Only)

⚠️ **Important**: This only works for **inference**, not training!

For inference, create a config file:

```yaml
### model
model_name_or_path: /home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat
trust_remote_code: true

# CPU-GPU hybrid via device_map (inference only)
low_cpu_mem_usage: true
device_map: "auto"  # Automatically distribute layers
offload_folder: /tmp/hf_offload

# For training, use DeepSpeed ZeRO-Offload instead
```

## Testing CPU-GPU Hybrid

### ⚠️ Important Limitation

**`device_map` is NOT available for training** in LLaMA-Factory. It's automatically set to a single GPU device. The only CPU-GPU hybrid option for **training** is DeepSpeed ZeRO-Offload.

### For Training: Use DeepSpeed ZeRO-Offload

```bash
# Test DeepSpeed ZeRO-3 with CPU offload
cd LLaMA-Factory
source ../scripts/deepspeed/activate_deepspeed_z3.sh
llamafactory-cli train examples/train_lora/deepseek2_lite_sft_hf_z3_test.yaml
```

### For Inference: device_map Works

```bash
# Test inference with device_map
cd LLaMA-Factory
source ../scripts/deepspeed/activate_deepspeed_z3.sh
llamafactory-cli infer examples/inference/deepseek2_lite_inference_hf.yaml
```

## Performance Considerations

1. **CPU-GPU Transfer Overhead**: Moving data between CPU and GPU has latency
2. **Pin Memory**: Use `pin_memory: true` in DeepSpeed config to speed up transfers
3. **Layer Placement**: Place frequently-used layers on GPU, less-used on CPU
4. **Batch Size**: May need smaller batch sizes with hybrid approaches

## Troubleshooting

### OOM with device_map
- Reduce number of layers on GPU
- Use smaller batch size
- Enable gradient checkpointing

### Slow Training
- Reduce CPU-GPU transfers by placing more layers on GPU
- Use pin_memory in DeepSpeed config
- Consider using only DeepSpeed ZeRO-Offload instead

### Conflicts with DeepSpeed
- Don't use `device_map` with DeepSpeed ZeRO-3
- Use one or the other, not both

## Next Steps

1. **Test device_map approach** without DeepSpeed
2. **Compare performance** between ZeRO-Offload and device_map
3. **Try manual layer placement** for optimal CPU-GPU distribution
4. **Monitor GPU/CPU memory usage** during training

## References

- [DeepSpeed ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/)
- [HuggingFace device_map](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)
- [LLaMA-Factory Training Guide](https://github.com/hiyouga/LLaMA-Factory)

