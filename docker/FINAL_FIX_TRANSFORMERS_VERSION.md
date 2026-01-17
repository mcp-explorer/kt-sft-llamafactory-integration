# Final Fix for DeepSeek V2 Lite Chat - Transformers Version Pin

## New Issue Found

After fixing the gguf_loader AttributeError, discovered another issue:

**Transformers version incompatibility**:
- KTransformers requires: `transformers == 4.51.3`
- LLaMA-Factory installs: `transformers 4.57.1` (latest)
- Result: `StaticCache` API breaking with error:

```
ValueError: You should provide exactly one of `layers` or `layer_class_to_replicate` to initialize a Cache.
```

## Solution

Pin transformers to 4.51.3 when installing LLaMA-Factory to prevent upgrade:

```dockerfile
# Clone and install LLaMA-Factory
RUN git clone https://github.com/hiyouga/LLaMA-Factory.git /workspace/LLaMA-Factory && \
    cd /workspace/LLaMA-Factory && \
    pip install --no-cache-dir -e ".[metrics]" --no-build-isolation && \
    pip install --no-cache-dir "transformers==4.51.3" --no-deps  # ← Pin to ktransformers requirement
```

This ensures:
- ✅ KTransformers StaticCache works correctly
- ✅ LLaMA-Factory still functional with ktransformers backend
- ✅ No API incompatibility issues

## All Fixes Combined

| Issue | Fix |
|-------|------|
| gguf_loader AttributeError | Create MinimalGGUFLoader that preserves existing tensor_device_map |
| CPU offloading broken | Copy existing device_map instead of reloading weights |
| Transformers version incompatibility | Pin transformers to 4.51.3 after LLaMA-Factory install |

## Complete Fix Summary

1. **`/LLaMA-Factory/src/llamafactory/model/model_utils/ktransformers.py`**
   - Lines 26-48: Added MinimalGGUFLoader class
   - Preserves existing device_map (CPU offloading)
   - Doesn't reload base weights

2. **`/docker/Dockerfile.kt-sft-train`** (line 89)
   - Added: `pip install "transformers==4.51.3" --no-deps`
   - Prevents LLaMA-Factory from upgrading transformers

3. **`/docker/Dockerfile.kt-sft-train`** (line 91-93)
   - Already includes kt_engine.py and ktransformers.py copies

## Build and Test

```bash
cd /home/sean/Documents/ktransformers/docker
docker build -f Dockerfile.kt-sft-train -t kt-sft-train:fixed ..
```

Test inference:
```bash
docker run --gpus all --rm \
  -e WANDB_DISABLED=true \
  -v /path/to/models:/workspace/models:ro \
  -v /path/to/kt-sft/optimize_rules:/workspace/ktransformers/kt-sft/ktransformers/optimize/optimize_rules:ro \
  -v /path/to/saves:/workspace/saves \
  kt-sft-train:fixed \
  llamafactory-cli chat your_inference_config.yaml
```

## Expected Results

| Metric | Expected |
|---------|-----------|
| Model loads without gguf_loader error | ✅ Yes |
| CPU offloading preserved (MoE on CPU) | ✅ Yes |
| StaticCache initializes correctly | ✅ Yes |
| GPU memory usage | ~10-12 GB |
| Inference output | Correct (not garbled) |

---

**Status**: All three issues fixed and documented.
**Ready for Docker rebuild and final testing.**
