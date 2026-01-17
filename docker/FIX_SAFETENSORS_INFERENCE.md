# Fix for DeepSeek V2 Lite Chat Garbled Output with KTransformers + LLaMA-Factory

## Issue Summary

When using ktransformers backend with LLaMA-Factory for inference with safetensors-trained adapters, the model produces garbled output or crashes with `AttributeError: 'PreTrainedModel' object has no attribute 'gguf_loader'`.

## Root Cause

The `prefill_and_generate_capture()` function in `ktransformers/util/utils.py` (line 557) assumes the model is loaded with GGUF format:

```python
device_map = model.gguf_loader.tensor_device_map
```

When base model is loaded with safetensors and adapters are loaded with SafeTensorLoader, the `model.gguf_loader` attribute may not exist or be `None`, causing inference to fail.

## Fix Applied

Modified `prefill_and_generate_capture()` in `/home/sean/Documents/ktransformers/kt-sft/ktransformers/util/utils.py` to handle both GGUF and safetensors formats:

```python
# Handle both GGUF and safetensors loaded models
if hasattr(model, 'gguf_loader') and model.gguf_loader is not None:
    device_map = model.gguf_loader.tensor_device_map
else:
    # For safetensors-loaded models, scan model parameters to determine device placement
    device_map = {}
    for name, param in model.named_parameters():
        if param.device not in device_map:
            device_map[name] = param.device
```

This ensures:
1. **GGUF models**: Continue to use existing `gguf_loader.tensor_device_map`
2. **Safetensors models**: Build device map by scanning model parameters
3. **No breaking changes**: GGUF workflow remains unchanged

## Files Modified

1. **Source Fix**:
   - `/home/sean/Documents/ktransformers/kt-sft/ktransformers/util/utils.py` (lines 556-562)

2. **Dockerfile Update**:
   - `/home/sean/Documents/ktransformers/docker/Dockerfile.kt-sft-train` (added COPY for utils.py)

## How to Rebuild Docker Image

```bash
cd /home/sean/Documents/ktransformers/docker
docker build -f Dockerfile.kt-sft-train -t kt-sft-train:fixed ..
```

**Note**: Build takes ~15-30 minutes depending on your system.

## Testing Inference with Fixed Image

After rebuild, test inference with your trained adapter:

```bash
docker run --gpus all --rm \
  -e WANDB_DISABLED=true \
  -v /path/to/models:/workspace/models:ro \
  -v /path/to/kt-sft/optimize_rules:/workspace/ktransformers/kt-sft/ktransformers/optimize/optimize_rules:ro \
  -v /path/to/LLaMA-Factory/data:/workspace/LLaMA-Factory/data:ro \
  -v /path/to/saves:/workspace/saves \
  kt-sft-train:fixed \
  llamafactory-cli chat /path/to/your/inference_config.yaml
```

## Expected Behavior

- ✅ Base model inference works (both GGUF and safetensors)
- ✅ Trained adapter inference works with ktransformers backend
- ✅ No `AttributeError: 'gguf_loader'` crashes
- ✅ Proper device placement for safetensors-loaded models

## Technical Details

The fix addresses the fundamental issue: **ktransformers was designed primarily for GGUF format**, but LLaMA-Factory's training saves adapters in safetensors format by default. This mismatch caused:

1. Training: Works perfectly (uses ktransformers operators with CPU offloading)
2. Inference: Failed when trying to access `gguf_loader` attribute
3. Result: Garbled output or crash

With this fix, inference can now work with both:
- GGUF-format models (original ktransformers workflow)
- Safetensors-format adapters (LLaMA-Factory training output)

## Alternative Workaround (If Build Fails)

If you cannot rebuild Docker image, use HuggingFace backend for inference:

```yaml
model_name_or_path: deepseek-ai/DeepSeek-V2-Lite-Chat
adapter_name_or_path: /path/to/saves/Kllama_deepseekV2Lite
template: deepseek
infer_backend: huggingface  # Uses standard Transformers instead of ktransformers
trust_remote_code: true
```

**Trade-off**: HuggingFace backend doesn't support CPU offloading, so may OOM on 16GB GPU with large models.
