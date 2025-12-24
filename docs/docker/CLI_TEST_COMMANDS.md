# LLaMA-Factory CLI Test Commands for KTransformers

Quick reference for testing KTransformers with LLaMA-Factory CLI.

---

## Quick Test Commands

### 1. Check CLI is Working

```bash
docker exec llamafactory bash -c "cd /app && python -m llamafactory.cli --help"
```

### 2. Test Chat (Interactive)

```bash
docker exec -it llamafactory bash -c "cd /app && \
  llamafactory-cli chat examples/inference/deepseek2_lite_inference.yaml"
```

### 3. Test API (Non-interactive)

```bash
# Start API server
docker exec llamafactory bash -c "cd /app && \
  API_PORT=8000 llamafactory-cli api examples/inference/deepseek2_lite_inference.yaml"
```

---

## Example Config for Testing

The config file should have:

```yaml
### model
model_name_or_path: /path/to/model
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

## Available Commands

- `llamafactory-cli train` - Fine-tune models
- `llamafactory-cli chat` - Interactive chat
- `llamafactory-cli api` - Start API server
- `llamafactory-cli export` - Export models

---

## Note on Flash Attention

If you encounter flash-attention errors (GitHub issue #1644), you can:
1. Skip flash-attention - KTransformers works without it
2. The error is in transformers library, not KTransformers itself

---

## Troubleshooting

If CLI doesn't work:
1. Restart container: `docker compose restart`
2. Check PyTorch: `python -c 'import torch; print(torch.__version__)'`
3. Check KTransformers: `python -c 'import ktransformers; print(ktransformers.__version__)'`


