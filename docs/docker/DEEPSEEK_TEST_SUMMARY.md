# DeepSeek Chat Model API Server Test Summary

## Current Status

### ✅ Completed
1. **Docker Environment**: Container rebuilt with PyTorch 2.6.0+cu126
2. **KTransformers**: Installed (pre-built wheel 0.4.1+cu126torch26fancy)
3. **LLaMA-Factory CLI**: Working and ready
4. **Model Files**: Accessible at `/app/models/deepseek-ai/DeepSeek-V2-Lite-Chat`
5. **Config File**: Updated `deepseek2_lite_serve_custom.yaml` with correct paths
6. **Flash Attention**: Uninstalled (not required for KTransformers, had ABI issues)

### ⚠️ Blocking Issue

**CUDA Library Compatibility Problem**

The pre-built KTransformers wheel (`ktransformers-0.4.1+cu126torch26fancy`) was compiled against CUDA 11.0, but the Docker container has CUDA 12.4/12.6 installed.

**Error Message:**
```
ImportError: /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.11.0: version `libcudart.so.11.0' not found (required by /opt/conda/lib/python3.11/site-packages/cpuinfer_ext.cpython-311-x86_64-linux-gnu.so)
```

**Why Symlink Doesn't Work:**
- The library needs actual CUDA 11.0 ABI symbols
- CUDA 12.4 libraries don't provide CUDA 11.0 symbol versions
- This is a binary compatibility issue, not just a path issue

## Solutions to Try

### Option 1: Test Chat Interface (Recommended First)
The interactive chat interface may handle the library loading differently:

```bash
docker exec -it llamafactory bash -c "
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/opt/conda/lib/python3.11/site-packages/torch/lib:\$LD_LIBRARY_PATH
cd /app
llamafactory-cli chat examples/inference/deepseek2_lite_serve_custom.yaml
"
```

### Option 2: Build KTransformers from Source
Build KTransformers from source to match CUDA 12.4:

**Challenges:**
- C++ ABI issues encountered (`_GLIBCXX_USE_CXX11_ABI` problems)
- Requires fixing CMake configuration
- Build time: ~10-30 minutes

**Steps:**
1. Fix C++ ABI configuration in CMake
2. Build with: `CPU_INSTRUCT=NATIVE KTRANSFORMERS_FORCE_BUILD=TRUE pip install . --no-build-isolation`
3. Install the built package

### Option 3: Install CUDA 11.0 Compatibility Libraries
Install CUDA 11.0 runtime libraries alongside CUDA 12.4:

**Pros:** Quick fix if it works
**Cons:** May cause conflicts, not ideal long-term

### Option 4: Use HuggingFace Backend Temporarily
Test the API server with HuggingFace backend to verify everything else works:

```yaml
infer_backend: huggingface  # Instead of ktransformers
# use_kt: true  # Comment out
```

Then fix KTransformers separately.

## Configuration Files

### Updated Config: `deepseek2_lite_serve_custom.yaml`
```yaml
### model
model_name_or_path: /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat
trust_remote_code: true

### dataset
template: deepseek

### ktransformers
infer_backend: ktransformers
use_kt: true
kt_optimize_rule: /opt/conda/lib/python3.11/site-packages/ktransformers/optimize/optimize_rules/DeepSeek-V2-Lite-Chat-sft.yaml
cpu_infer: 16
chunk_size: 8192
```

## Test Commands

### Check Environment
```bash
docker exec llamafactory bash -c "
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/opt/conda/lib/python3.11/site-packages/torch/lib:\$LD_LIBRARY_PATH
python -c 'import torch; import ktransformers; print(\"PyTorch:\", torch.__version__); print(\"KTransformers:\", ktransformers.__version__)'
"
```

### Test Chat (Interactive)
```bash
docker exec -it llamafactory bash -c "
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/opt/conda/lib/python3.11/site-packages/torch/lib:\$LD_LIBRARY_PATH
cd /app
llamafactory-cli chat examples/inference/deepseek2_lite_serve_custom.yaml
"
```

### Test API Server
```bash
# Start server
docker exec -d llamafactory bash -c "
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/opt/conda/lib/python3.11/site-packages/torch/lib:\$LD_LIBRARY_PATH
cd /app
llamafactory-cli api examples/inference/deepseek2_lite_serve_custom.yaml
"

# Wait and test
sleep 30
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-chat", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Next Steps

1. **Immediate**: Test chat interface to see if it works despite the library issue
2. **Short-term**: Fix CUDA library compatibility (Option 2 or 3)
3. **Long-term**: Ensure KTransformers builds match container CUDA version

## Related Issues

- GitHub Issue #1644: Flash Attention ABI compatibility with PyTorch 2.6.0+cu126
- CUDA version mismatch: Pre-built wheels vs container CUDA version

## Files Created

- `docs/docker/DEEPSEEK_API_TEST.md` - Detailed testing guide
- `docs/docker/DEEPSEEK_TEST_SUMMARY.md` - This summary
- `test_deepseek_api.sh` - Test script (needs CUDA fix first)

