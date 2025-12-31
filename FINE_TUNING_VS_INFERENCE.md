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

# 2. Inference with HuggingFace backend (WORKS)
llamafactory-cli chat --model_name_or_path <your_model> --use_kt false
```

This is actually a common pattern - use optimized backends for training, standard backends for inference.

