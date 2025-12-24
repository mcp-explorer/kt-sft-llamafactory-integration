# Testing KTransformers with LLaMA-Factory CLI

Complete guide for testing KTransformers using LLaMA-Factory CLI commands.

---

## ✅ Prerequisites Verified

- ✅ PyTorch 2.6.0+cu126 installed
- ✅ CUDA 12.6 available
- ✅ LLaMA-Factory CLI working
- ✅ KTransformers installed

---

## Available CLI Commands

```bash
llamafactory-cli api      # Launch OpenAI-style API server
llamafactory-cli chat     # Launch interactive chat interface
llamafactory-cli train    # Train models
llamafactory-cli export   # Merge LoRA adapters and export
llamafactory-cli webchat  # Launch Web UI chat
llamafactory-cli webui    # Launch LlamaBoard
llamafactory-cli env      # Show environment info
llamafactory-cli version  # Show version info
```

**Shortcut**: You can use `lmf` instead of `llamafactory-cli`

---

## Test Commands

### 1. Test Chat (Interactive)

```bash
docker exec -it llamafactory bash -c "cd /app && \
  llamafactory-cli chat examples/inference/deepseek2_lite_inference.yaml"
```

Or with a custom config:

```bash
docker exec -it llamafactory bash -c "cd /app && \
  llamafactory-cli chat /path/to/your/config.yaml"
```

### 2. Test API Server

```bash
# Start API server
docker exec llamafactory bash -c "cd /app && \
  API_PORT=8000 llamafactory-cli api examples/inference/deepseek2_lite_inference.yaml"
```

Then test with curl (from host):

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 3. Test Training (Fine-tuning)

```bash
docker exec llamafactory bash -c "cd /app && \
  USE_KT=1 llamafactory-cli train examples/train_lora/your_config.yaml"
```

---

## Example Config Files

### For Inference (Chat/API)

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

### For Training

```yaml
### model
model_name_or_path: /path/to/model
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8

### ktransformers
use_kt: true
kt_optimize_rule: examples/kt_optimize_rules/DeepSeek-V2-Lite-Chat-sft.yaml
cpu_infer: 32
chunk_size: 8192
```

---

## Available Optimize Rules

Check available rules:

```bash
docker exec llamafactory bash -c "ls -1 /opt/conda/lib/python3.11/site-packages/ktransformers/optimize/optimize_rules/ | head -20"
```

Common rules:
- `DeepSeek-V2-Lite-Chat-sft.yaml`
- `DeepSeek-V3-Chat-sft-amx-multi-gpu.yaml`
- `Qwen2-serve.yaml`
- And many more...

---

## Quick Test Script

Use the provided test script:

```bash
./test_llamafactory_cli.sh
```

---

## Troubleshooting

### Issue: KTransformers not found

Reinstall KTransformers:
```bash
docker exec llamafactory bash -c "cd /kt-sft && CPU_INSTRUCT=NATIVE KTRANSFORMERS_FORCE_BUILD=TRUE pip install . --no-build-isolation"
```

### Issue: Model not found

- Check model path in YAML config
- Ensure model files are accessible in container
- Use absolute paths or mount volumes

### Issue: CUDA out of memory

Increase CPU inference threads:
```yaml
cpu_infer: 32  # Increase from default
```

---

## Next Steps

1. **Test with your model**: Update config file with your model path
2. **Fine-tune**: Use `llamafactory-cli train` with `use_kt: true`
3. **Deploy**: Use API server for production

---

## Reference

- LLaMA-Factory docs: `/app/README.md` in container
- Example configs: `/app/examples/inference/`
- Optimize rules: `/opt/conda/lib/python3.11/site-packages/ktransformers/optimize/optimize_rules/`


