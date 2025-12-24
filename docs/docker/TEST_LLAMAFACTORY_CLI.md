# Testing KTransformers with LLaMA-Factory CLI

Guide for testing KTransformers using LLaMA-Factory CLI commands.

---

## Prerequisites

1. ✅ Docker container running with KTransformers installed
2. ✅ PyTorch 2.6.0+cu126 installed
3. ✅ Model files available (or use a small test model)

---

## Quick Test Commands

### 1. Check CLI is Working

```bash
docker exec llamafactory bash -c "cd /app && python -m llamafactory.cli --help"
```

### 2. Test Chat with KTransformers

```bash
docker exec llamafactory bash -c "cd /app && \
  llamafactory-cli chat examples/inference/deepseek2_lite_inference.yaml"
```

### 3. Test API with KTransformers

```bash
docker exec llamafactory bash -c "cd /app && \
  API_PORT=8000 llamafactory-cli api examples/inference/deepseek2_lite_inference.yaml"
```

---

## Example Config Files

### Basic Inference Config

Create or use: `examples/inference/test_kt.yaml`

```yaml
### model
model_name_or_path: /path/to/your/model
template: deepseek  # or llama3, qwen, etc.

### ktransformers
infer_backend: ktransformers
use_kt: true
kt_optimize_rule: /opt/conda/lib/python3.11/site-packages/ktransformers/optimize/optimize_rules/DeepSeek-V2-Lite-Chat-sft.yaml
cpu_infer: 16
chunk_size: 8192
trust_remote_code: true
```

---

## Available Optimize Rules

Check available optimize rules:

```bash
docker exec llamafactory bash -c "ls -la /opt/conda/lib/python3.11/site-packages/ktransformers/optimize/optimize_rules/ | head -20"
```

Common rules:
- `DeepSeek-V2-Lite-Chat-sft.yaml` - For DeepSeek-V2-Lite fine-tuning
- `DeepSeek-V3-Chat-sft-amx-multi-gpu.yaml` - For DeepSeek-V3 with AMX
- `Qwen2-serve.yaml` - For Qwen2 models
- And many more...

---

## Testing Steps

### Step 1: Verify Environment

```bash
docker exec llamafactory bash -c "
python -c 'import torch; print(\"PyTorch:\", torch.__version__)'
python -c 'import ktransformers; print(\"KTransformers:\", ktransformers.__version__)'
python -c 'import KTransformersOps; print(\"Ops loaded\")'
"
```

### Step 2: Check Available Models

```bash
docker exec llamafactory bash -c "ls -la /app/models/ 2>/dev/null || echo 'No models directory'"
```

### Step 3: Test Chat (Interactive)

```bash
docker exec -it llamafactory bash -c "cd /app && llamafactory-cli chat examples/inference/deepseek2_lite_inference.yaml"
```

### Step 4: Test API (Non-interactive)

```bash
# Start API server
docker exec llamafactory bash -c "cd /app && API_PORT=8000 llamafactory-cli api examples/inference/deepseek2_lite_inference.yaml &"

# Test with curl (in another terminal)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Troubleshooting

### Issue: Flash Attention Error

If you see flash-attention errors, you can:
1. Skip flash-attention (KTransformers works without it)
2. Rebuild flash-attention from source (see GitHub issue #1644)

### Issue: Model Not Found

Make sure:
- Model path is correct in YAML config
- Model files are accessible in container
- Use absolute paths or mount volumes

### Issue: CUDA Out of Memory

Reduce batch size or use CPU inference:
```yaml
cpu_infer: 32  # Increase CPU inference threads
```

---

## Next Steps

1. **Fine-tuning**: Use `llamafactory-cli train` with `use_kt: true`
2. **Inference**: Use `llamafactory-cli chat` or `api`
3. **Batch Processing**: Use API endpoint for batch inference

---

## Reference

- LLaMA-Factory docs: `/app/README.md` in container
- KTransformers optimize rules: `/opt/conda/lib/python3.11/site-packages/ktransformers/optimize/optimize_rules/`
- Example configs: `/app/examples/inference/`


