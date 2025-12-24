# Testing DeepSeek Chat Model with LLaMA-Factory API

Guide for testing the DeepSeek-V2-Lite-Chat model using LLaMA-Factory API server with KTransformers.

---

## Current Status

### ✅ Working
- PyTorch 2.6.0+cu126 installed
- CUDA 12.6 available
- KTransformers 0.4.1 installed
- LLaMA-Factory CLI working
- Model files accessible at `/app/models/deepseek-ai/DeepSeek-V2-Lite-Chat`

### ⚠️ Known Issues
- **CUDA Library Compatibility**: The pre-built KTransformers wheel was compiled against CUDA 11.0, but the container has CUDA 12.4/12.6. This causes `libcudart.so.11.0` errors.
- **Flash Attention**: Flash attention has ABI compatibility issues with PyTorch 2.6.0+cu126 (GitHub issue #1644). It has been uninstalled as KTransformers doesn't require it.

---

## Configuration File

The config file `examples/inference/deepseek2_lite_serve_custom.yaml` is configured as:

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

---

## Testing Commands

### 1. Test Chat Interface (Interactive)

```bash
docker exec -it llamafactory bash -c "
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/opt/conda/lib/python3.11/site-packages/torch/lib:\$LD_LIBRARY_PATH
cd /app
llamafactory-cli chat examples/inference/deepseek2_lite_serve_custom.yaml
"
```

### 2. Start API Server

```bash
docker exec -d llamafactory bash -c "
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/opt/conda/lib/python3.11/site-packages/torch/lib:\$LD_LIBRARY_PATH
cd /app
llamafactory-cli api examples/inference/deepseek2_lite_serve_custom.yaml
"
```

Wait for the server to start (check logs):
```bash
docker logs llamafactory --tail 50
```

### 3. Test API Endpoint

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello! Can you introduce yourself?"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

---

## Troubleshooting

### Issue: `libcudart.so.11.0: cannot open shared object file`

**Cause**: Pre-built wheel was compiled against CUDA 11.0, but container has CUDA 12.4/12.6.

**Solution**: Set `LD_LIBRARY_PATH` to include CUDA libraries:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/opt/conda/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
```

**Alternative**: Build KTransformers from source (requires fixing C++ ABI issues).

### Issue: Flash Attention Errors

**Solution**: Flash attention has been uninstalled. KTransformers works without it.

### Issue: API Server Crashes

**Check logs**:
```bash
docker logs llamafactory --tail 100
```

**Common causes**:
- CUDA library path not set
- Model files not found
- Memory issues

---

## Next Steps

1. **Fix CUDA Library Issue**: 
   - Option A: Build KTransformers from source with CUDA 12.4/12.6
   - Option B: Install CUDA 11.0 compatibility libraries
   - Option C: Use a wrapper script that sets LD_LIBRARY_PATH automatically

2. **Test Chat Interface**: The interactive chat interface may work better than the API server

3. **Production Deployment**: Once working, set up proper service management and monitoring

---

## Reference

- Model path: `/app/models/deepseek-ai/DeepSeek-V2-Lite-Chat`
- Config file: `/app/examples/inference/deepseek2_lite_serve_custom.yaml`
- Optimize rule: `/opt/conda/lib/python3.11/site-packages/ktransformers/optimize/optimize_rules/DeepSeek-V2-Lite-Chat-sft.yaml`
- API endpoint: `http://localhost:8000/v1/chat/completions`

