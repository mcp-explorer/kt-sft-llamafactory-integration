# Testing Different Inference Backends for DeepSeek-V2-Lite

This document explains how to test different inference backends to determine if the garbled output issue is specific to ktransformers.

## Available Backends

LLaMA-Factory supports 4 inference backends:

1. **huggingface** (HF) - Standard transformers backend
   - ✅ Always available (no extra installation)
   - ✅ Most compatible
   - ⚠️ Slower than optimized backends

2. **vllm** - Fast inference engine
   - ⚠️ Requires: `pip install vllm`
   - ✅ Very fast inference
   - ✅ Good for production

3. **sglang** - Fast inference engine
   - ⚠️ Requires: `pip install sglang[all]`
   - ✅ Very fast inference
   - ✅ Good for production

4. **ktransformers** (KT) - Current backend
   - ⚠️ Requires ktransformers installation
   - ✅ Optimized for specific hardware
   - ❓ Currently experiencing garbled output

## Test Configurations

### 1. HuggingFace Backend (Recommended for Testing)

**Config file:** `LLaMA-Factory/examples/inference/deepseek2_lite_inference_raw_hf.yaml`

**Test command:**
```bash
cd /home/sean/Documents/ktransformers/LLaMA-Factory
llamafactory-cli chat examples/inference/deepseek2_lite_inference_raw_hf.yaml
```

**Expected:** Should work without garbled output if the issue is ktransformers-specific.

### 2. vLLM Backend

**Prerequisites:**
```bash
pip install vllm
```

**Config file:** `LLaMA-Factory/examples/inference/deepseek2_lite_inference_raw_vllm.yaml`

**Test command:**
```bash
cd /home/sean/Documents/ktransformers/LLaMA-Factory
llamafactory-cli chat examples/inference/deepseek2_lite_inference_raw_vllm.yaml
```

### 3. SGLang Backend

**Prerequisites:**
```bash
pip install sglang[all]
```

**Config file:** Create similar to vllm config but with `infer_backend: sglang`

**Test command:**
```bash
cd /home/sean/Documents/ktransformers/LLaMA-Factory
llamafactory-cli chat examples/inference/deepseek2_lite_inference_raw_sglang.yaml
```

### 4. KTransformers Backend (Current)

**Config file:** `LLaMA-Factory/examples/inference/deepseek2_lite_inference_raw.yaml`

**Test command:**
```bash
./scripts/infer_ds2_chat_lite_raw.sh chat
```

## Quick Test Script

You can test all backends with:

```bash
#!/bin/bash
cd /home/sean/Documents/ktransformers/LLaMA-Factory

echo "Testing HuggingFace backend..."
llamafactory-cli chat examples/inference/deepseek2_lite_inference_raw_hf.yaml <<EOF
who are you
exit
EOF

# If vllm is installed:
# echo "Testing vLLM backend..."
# llamafactory-cli chat examples/inference/deepseek2_lite_inference_raw_vllm.yaml <<EOF
# who are you
# exit
# EOF
```

## Interpreting Results

- **If HuggingFace works correctly:** The issue is likely ktransformers-specific
- **If all backends show garbled output:** The issue is likely with:
  - Model files
  - Tokenizer configuration
  - System encoding/terminal settings
- **If only ktransformers shows issues:** Focus on ktransformers-specific fixes

## Notes

- HuggingFace backend is the safest test as it uses standard transformers
- vLLM and SGLang may have different requirements or limitations
- All backends should produce the same text output (quality may vary slightly)

