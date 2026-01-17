# KTransformers Garbled Output Issue Explained

## Overview

This document explains the **garbled output issue** encountered when using **ktransformers backend** with **LLaMA-Factory CLI** inside **Docker** for inference of DeepSeek-V2-Lite-Chat model.

---

## Issue Summary

### Symptom

When running inference with ktransformers backend:
- **Base model**: Model loads successfully but produces **garbled, unintelligible text**
- **Trained adapter**: Fails with `AttributeError: 'PreTrainedModel' object has no attribute 'gguf_loader'`
- **HuggingFace backend**: Works correctly with no garbled output

### Affected Configurations

| Component | Version | Status |
|------------|----------|---------|
| Backend | KTransformers | ❌ Garbled output |
| Backend | HuggingFace | ✅ Works correctly |
| Training | KTransformers LoRA | ✅ Works correctly |
| Environment | Docker (kt-sft-train:latest) | ✅ Compatible |

---

## Root Causes

### Primary Issue: `model.gguf_loader` Attribute Missing

The ktransformers `prefill_and_generate_capture()` function expects the model to have a `gguf_loader` attribute:

```python
# In ktransformers/util/utils.py (line 557)
device_map = model.gguf_loader.tensor_device_map
```

**Why this fails:**

1. **ktransformers design**: Originally designed for GGUF format model loading
2. **LLaMA-Factory training**: Saves adapters in **safetensors** format by default
3. **Adapter loading**: LLaMA-Factory's `load_kt_peft_model()` loads base model with safetensors
4. **Missing attribute**: Safetensors-loaded models don't have `gguf_loader` → crashes or produces garbled output

### Secondary Issue: Tokenizer Decoding

Garbled output can also occur due to:
- **`clean_up_tokenization_spaces`**: Not set to `True` during token decoding
- **`SilentCaptureStreamer`**: Issues with text streaming and decoding

---

## Technical Details

### How KTransformers Loads Models

#### GGUF Format (Native Path)

```python
# When loading GGUF format
model = optimize_and_load_gguf(
    model_path="model.gguf",
    optimize_rule="DeepSeek-V2-Lite-Chat-sft.yaml"
)

# Creates:
model.gguf_loader.tensor_device_map = {
    "model.layers.X.mlp.experts": {
        "generate_device": "cpu",  # MoE experts on CPU
        "prefill_device": "cuda",
        "out_device": "cuda"
    },
    ...
}
```

#### Safetensors Format (LLaMA-Factory Path)

```python
# When loading safetensors format (LLaMA-Factory default)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    trust_remote_code=True
)

# Result:
# model.gguf_loader does NOT exist → AttributeError
```

### Why Safetensors in LLaMA-Factory?

LLaMA-Factory uses standard PEFT (Parameter-Efficient Fine-Tuning) library:

1. **Base model loading**: Uses `AutoModelForCausalLM.from_pretrained()`
2. **Adapter training**: Saves LoRA weights as safetensors
3. **Adapter loading**: Uses `PeftModel.from_pretrained()`

All of these use **HuggingFace's native format** (safetensors), NOT GGUF.

---

## Fixes Attempted

### Attempt 1: Fix in `prefill_and_generate_capture()`

**File**: `kt-sft/ktransformers/util/utils.py`

**Change**: Handle both GGUF and safetensors formats

```python
# Handle both GGUF and safetensors loaded models
if hasattr(model, 'gguf_loader') and model.gguf_loader is not None:
    device_map = model.gguf_loader.tensor_device_map
else:
    # For safetensors-loaded models, scan model parameters
    device_map = {}
    for name, param in model.named_parameters():
        if param.device not in device_map:
            device_map[name] = param.device
```

**Result**: ❌ Didn't fully resolve garbled output

---

### Attempt 2: Create Minimal GGUFLoader Wrapper

**File**: `LLaMA-Factory/src/llamafactory/model/model_utils/ktransformers.py`

**Change**: Create minimal `GGUFLoader` wrapper without reloading weights

```python
class MinimalGGUFLoader:
    def __init__(self, model_path: str):
        # Copy existing device_map to preserve CPU offloading
        if hasattr(model, 'gguf_loader') and model.gguf_loader is not None:
            self.tensor_device_map = model.gguf_loader.tensor_device_map
        else:
            self.tensor_device_map = {}
        self.tensor_file_map = {}
        self.tensor_type_map = {}
        self.safetensor_loader = None

    def has_tensor(self, name: str):
        return False  # Not loading any tensors

# Apply wrapper
if not hasattr(model, 'gguf_loader') or model.gguf_loader is None:
    model.gguf_loader = MinimalGGUFLoader(model_args.model_name_or_path)
```

**Result**: ✅ Preserved CPU offloading, but still had decoding issues

---

### Attempt 3: Tokenizer Decoding Fixes

**Files**: Multiple locations

#### Fix 1: Add `clean_up_tokenization_spaces=True`

```python
# In ktransformers/util/textstream.py
stream = SilentCaptureStreamer(
    tokenizer,
    echo=echo_stream,
    clean_up_tokenization_spaces=True  # ← Added this
)
```

#### Fix 2: Update model decoding

```python
# In ktransformers/models/modeling_deepseek.py
output = tokenizer.batch_decode(
    generate_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True  # ← Changed from False
)[0]
```

**Result**: ✅ Improved but still experienced garbled output in some cases

---

### Attempt 4: HuggingFace Backend Workaround

**Approach**: Use HuggingFace backend instead of ktransformers for inference

**Config**:
```yaml
model_name_or_path: deepseek-ai/DeepSeek-V2-Lite-Chat
adapter_name_or_path: /workspace/saves/deepseek2_lite_kt_quick
template: deepseek
infer_backend: huggingface  # ← Use HF instead of ktransformers
trust_remote_code: true
```

**Result**: ✅ Works perfectly with trained adapters

**Trade-off**: No CPU offloading → may OOM on 16GB GPU for large models

---

## Current Status

### What Works

| Feature | Status | Notes |
|---------|---------|--------|
| **Training** (ktransformers) | ✅ Works | CPU offloading enabled, loss decreases properly |
| **Inference** (HuggingFace backend) | ✅ Works | Standard Transformers, no CPU offloading |
| **Inference** (ktransformers + GGUF) | ✅ Works | If model loaded in GGUF format |

### What Doesn't Work

| Feature | Status | Root Cause |
|---------|---------|-------------|
| **Inference** (ktransformers + safetensors) | ❌ Garbled output | `gguf_loader` missing, token decoding issues |
| **Adapter loading** (ktransformers format) | ❌ Fails | Key mismatch: LoRA vs base model |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Training (KTransformers)               │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ 1. Load base model (GGUF/safetensors)       │ │
│  │ 2. Apply optimize rule (CPU offloading)          │ │
│  │ 3. Train LoRA with CPU+GPU hybrid             │ │
│  │ 4. Save adapter (safetensors)                │ │
│  └──────────────────────────────────────────────────────┘ │
│                       │                                    │
│                       ▼                                    │
│              Trained Adapter (safetensors)                  │
└─────────────────────────────────────────────────────────────┘
                         │
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌────────────────┐              ┌────────────────┐
│  Inference    │              │  Inference    │
│  (KTransformers│              │  (HuggingFace)│
│   backend)    │              │   backend)    │
├────────────────┤              ├────────────────┤
│  Load adapter │              │  Load adapter │
│  + base model│              │  + base model│
│              │              │              │
│  ❌ FAILS:   │              │  ✅ WORKS:   │
│  gguf_loader  │              │  Compatible   │
│  missing      │              │  format       │
│              │              │              │
│  Garbled     │              │  Clean       │
│  output      │              │  output      │
└────────────────┘              └────────────────┘
```

---

## Why the Issue Exists

### KTransformers Architecture

KTransformers was designed for:
- **GGUF format**: Binary format for quantized models
- **CPU+GPU hybrid**: MoE experts on CPU, attention on GPU
- **Custom operators**: Specialized kernels for efficiency

### LLaMA-Factory Architecture

LLaMA-Factory was designed for:
- **Safetensors format**: HuggingFace's native format
- **PEFT library**: LoRA, QLoRA adapters
- **Standard Transformers**: No custom operators or kernels

### The Mismatch

| KTransformers Design | LLaMA-Factory Reality |
|-------------------|----------------------|
| GGUF format models | Safetensors format |
| `gguf_loader` required | `gguf_loader` doesn't exist |
| Custom device mapping | Standard device mapping |
| CPU offloading via optimize rules | No CPU offloading |

**Result**: When LLaMA-Factory tries to use ktransformers for inference, it encounters:
1. Missing `gguf_loader` attribute → crashes
2. Different tokenizer decoding → garbled output
3. Incompatible device placement → OOM or crashes

---

## Recommendations

### For Development

1. **Use HuggingFace backend for inference** (RECOMMENDED)
   - Works reliably with trained adapters
   - No garbled output
   - Trade-off: No CPU offloading benefits

2. **Fix ktransformers to support safetensors natively**
   - Remove hard dependency on `gguf_loader`
   - Add safetensors loader support
   - This requires ktransformers core changes

3. **Convert adapters to GGUF format** (if CPU offloading required)
   - Convert trained LoRA adapters to GGUF
   - Use ktransformers for inference with CPU offloading
   - More complex workflow

### For Production

#### Option A: HuggingFace Backend (Simple)

```yaml
# Config: deepseek2_lite_inference_hf.yaml
model_name_or_path: deepseek-ai/DeepSeek-V2-Lite-Chat
adapter_name_or_path: /workspace/saves/deepseek2_lite_kt
template: deepseek
infer_backend: huggingface
trust_remote_code: true
bf16: true
```

**Pros**:
- ✅ Works reliably
- ✅ Easy to implement
- ✅ Compatible with trained adapters

**Cons**:
- ❌ No CPU offloading
- ❌ May OOM on 16GB GPU for 14B+ models

#### Option B: GGUF Conversion (Complex)

```bash
# Convert base model to GGUF
python convert_hf_to_gguf.py \
    --model /path/to/model \
    --output /path/to/model.gguf

# Merge adapter with GGUF model
python merge_adapter_gguf.py \
    --base model.gguf \
    --adapter adapter.safetensors \
    --output merged.gguf

# Use ktransformers with GGUF
llamafactory-cli chat config.yaml \
    --infer_backend ktransformers \
    --model_path merged.gguf
```

**Pros**:
- ✅ Enables CPU offloading
- ✅ Fits 14B+ models on 16GB GPU

**Cons**:
- ❌ Complex workflow
- ❌ Not well-documented
- ❌ May have compatibility issues

#### Option C: Use vLLM or SGLang (Alternative)

```yaml
# Config: deepseek2_lite_inference_vllm.yaml
model_name_or_path: deepseek-ai/DeepSeek-V2-Lite-Chat
adapter_name_or_path: /workspace/saves/deepseek2_lite_kt
template: deepseek
infer_backend: vllm
trust_remote_code: true
```

**Pros**:
- ✅ High performance
- ✅ Better memory management
- ✅ Works with trained adapters

**Cons**:
- ❌ No CPU offloading
- ❌ Different optimization goals

---

## Testing Results

### Backend Comparison (2026-01-15)

| Test | Backend | Adapter | Result |
|------|---------|---------|---------|
| Base model | HuggingFace CLI | ✅ "2 + 2 equals 4" |
| Base model | Transformers direct | ✅ "2+2=4" |
| Trained adapter | Transformers direct | ✅ "My name is Kaitlyn" (generating!) |
| Base model | ktransformers CLI | ❌ `gguf_loader` attribute missing |
| Trained adapter | ktransformers CLI | ❌ Key mismatch + OOM |

### HuggingFace Inference Test

```bash
# Command
docker run --gpus all --rm kt-sft-train:latest \
  llamafactory-cli chat deepseek2_lite_inference_hf.yaml

# Output
User: What is 2+2?
Assistant: 2 + 2 equals 4.

User: What's your name?
Assistant: My name is Kaitlyn.

# Status: ✅ Works perfectly, no garbled output
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `docker/FIX_SAFETENSORS_INFERENCE.md` | Fix for `gguf_loader` attribute |
| `docker/CORRECTED_FIX_CPU_OFFLOADING.md` | Preserving CPU offloading with minimal wrapper |
| `scripts/docs/test_results_summary.md` | Test results showing garbled output is ktransformers-specific |
| `LLaMA-Factory/src/llamafactory/chat/kt_engine.py` | LLaMA-Factory ktransformers backend implementation |
| `kt-sft/ktransformers/util/utils.py` | ktransformers inference utilities (`prefill_and_generate_capture`) |
| `kt-sft/ktransformers/util/textstream.py` | Tokenizer streaming and decoding |
| `kt-sft/ktransformers/models/modeling_deepseek.py` | DeepSeek V2 model code |

---

## Quick Reference

### Reproduce the Issue

```bash
# 1. Build Docker image
cd /home/sean/Documents/ktransformers/docker
docker build -f Dockerfile.kt-sft-train -t kt-sft-train:latest ..

# 2. Run training (this works)
docker run --gpus all --rm kt-sft-train:latest \
  llamafactory-cli train deepseek2_lite_identity_quick_kt.yaml

# 3. Try inference with ktransformers (FAILS with garbled output)
docker run --gpus all --rm kt-sft-train:latest \
  llamafactory-cli chat deepseek2_lite_inference_kt.yaml
# Result: ❌ Garbled output or AttributeError

# 4. Use HuggingFace backend instead (WORKS)
docker run --gpus all --rm kt-sft-train:latest \
  llamafactory-cli chat deepseek2_lite_inference_hf.yaml
# Result: ✅ Clean, coherent output
```

### Workaround Configs

```yaml
# Inference config with HuggingFace backend (works)
infer_backend: huggingface
adapter_name_or_path: /workspace/saves/deepseek2_lite_kt
model_name_or_path: deepseek-ai/DeepSeek-V2-Lite-Chat
template: deepseek
trust_remote_code: true
```

---

## Conclusion

The garbled output issue is a **fundamental architecture mismatch** between:
- **KTransformers**: Designed for GGUF format with CPU offloading
- **LLaMA-Factory**: Uses standard HuggingFace (safetensors) format

### Best Path Forward

**For most use cases**: Use HuggingFace backend for inference
- ✅ Reliable
- ✅ Works with trained adapters
- ✅ No garbled output

**For CPU offloading**: Requires GGUF conversion or ktransformers core changes
- ⚠️ Complex workflow
- ⚠️ May have compatibility issues
- ⚠️ Not well-documented

---

*Last updated: 2026-01-16*
