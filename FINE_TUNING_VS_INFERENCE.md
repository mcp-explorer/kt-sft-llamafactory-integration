# Fine-Tuning vs Inference: Why Fine-Tuning Works

## Key Difference: Batch Size (n parameter)

### The Problem
The segfault issue occurs specifically in **generation/inference mode** when `n=1` (single token generation).

### Why Fine-Tuning Works

**Fine-tuning uses batch processing:**
- During training, the model processes **batches of sequences** simultaneously
- The `n` parameter in `llamafile_sgemm` represents the batch dimension
- Fine-tuning: `n >= 2` (batch size is typically much larger, e.g., 8, 16, 32+)
- Inference/Chat: `n = 1` (generating one token at a time)

### Code Evidence

In `llamafile_sgemm.cpp` line 2798:
```cpp
// tinyBLAS doesn't properly handle n=1 (generation mode)
// Instead of returning false, compute directly here to avoid fallback issues
if (n < 2) {
    // Direct computation for BF16 with n=1 (generation mode)
    // ... problematic code path ...
}
```

**This code path is ONLY triggered when `n < 2`**, which happens during:
- ✅ **Inference/Chat** (n=1) - **HAS ISSUES**
- ❌ **Fine-tuning** (n >= 2) - **WORKS FINE**

### What This Means

1. **Fine-tuning is safe**: Uses the standard `tinyBLAS` path with `n >= 2`, which works correctly
2. **Inference has issues**: Uses the special `n=1` path which has the B pointer problem
3. **You can fine-tune successfully**: Your previous successful fine-tuning confirms this

### Recommendation

- ✅ **Use KTransformers + LlamaFactory for fine-tuning** - This works and is the intended use case
- ❌ **Avoid using KTransformers for inference/chat** - Use HuggingFace backend instead for inference
- 🔄 **Fine-tune with KTransformers, then switch to HuggingFace for inference**

### Example Workflow

```bash
# 1. Fine-tune with KTransformers (WORKS)
USE_KT=1 llamafactory-cli train examples/train_lora/deepseek3_lora_sft_kt.yaml

# 2. Inference with HuggingFace backend using the saved checkpoint (WORKS)
llamafactory-cli chat \
  --model_name_or_path <base_model> \
  --adapter_name_or_path saves/Kllama_deepseekV2Lite \
  --use_kt false
```

This is actually a common pattern - use optimized backends for training, standard backends for inference.

## Checkpoint Compatibility: ✅ YES

**Good news:** LoRA adapters saved by KTransformers are **fully compatible** with HuggingFace!

### Why They're Compatible

1. **Same PEFT Library**: KTransformers uses the standard HuggingFace `peft` library:
   ```python
   # From ktransformers.py line 121
   return get_peft_model(model, peft_kwargs)  # Standard HuggingFace PEFT
   ```

2. **Standard Format**: KTransformers saves checkpoints in standard PEFT format:
   - `adapter_model.safetensors` (or `adapter_model.bin`)
   - `adapter_config.json`
   - Same format as HuggingFace PEFT

3. **Standard Save Method**: The `KTrainer.save_model()` just calls:
   ```python
   # From lora.py line 72
   self.model.save_pretrained(output_dir)  # Standard PEFT save
   ```

### How to Use

1. **Fine-tune with KTransformers:**
   ```bash
   USE_KT=1 llamafactory-cli train examples/train_lora/deepseek3_lora_sft_kt.yaml
   ```
   This saves to: `saves/Kllama_deepseekV2Lite/adapter_model.safetensors`

2. **Load with HuggingFace for inference:**
   ```bash
   llamafactory-cli chat \
     --model_name_or_path deepseek-ai/DeepSeek-V2-Lite-Chat \
     --adapter_name_or_path saves/Kllama_deepseekV2Lite \
     --use_kt false
   ```

The checkpoint is **backend-agnostic** - it's just LoRA weights that work with any compatible backend!

