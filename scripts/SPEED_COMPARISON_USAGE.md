# Speed Comparison Test - Usage Guide

## Quick Start

### Option 1: Run from Host (Recommended)

The script is located at: `scripts/compare_inference_speeds.py`

```bash
# Copy script to container and run
docker cp scripts/compare_inference_speeds.py llamafactory:/tmp/
docker exec llamafactory bash -c "
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-12.4/lib64:/opt/conda/lib/python3.11/site-packages/torch/lib:\$LD_LIBRARY_PATH
    python3 /tmp/compare_inference_speeds.py \
        --model_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat \
        --prompt 'Tell me a story about a baby' \
        --max_tokens 100
"
```

### Option 2: Use Convenience Wrapper

```bash
# From host
./scripts/run_speed_comparison.sh

# Or with custom parameters
./scripts/run_speed_comparison.sh \
    /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat \
    "Explain quantum computing" \
    200
```

### Option 3: Run Inside Container

```bash
# Enter container
docker exec -it llamafactory bash

# Copy script (if not already there)
# Or create it directly in container

# Run comparison
python3 /tmp/compare_inference_speeds.py \
    --model_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat \
    --prompt "Tell me a story about a baby" \
    --max_tokens 100
```

## Example Commands

### Basic Comparison

```bash
python3 scripts/compare_inference_speeds.py \
    --model_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat
```

### Custom Test

```bash
python3 scripts/compare_inference_speeds.py \
    --model_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat \
    --prompt "Write a haiku about AI" \
    --max_tokens 50 \
    --template chatml
```

### Test Only KTransformers

```bash
python3 scripts/compare_inference_speeds.py \
    --model_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat \
    --skip_cpu \
    --skip_cpu_gpu \
    --kt_optimize_rule /app/examples/kt_optimize_rules/DeepSeek-V2-Lite-Chat.yaml
```

## What It Tests

1. **CPU Only**: Full model on CPU, no GPU
   - Command: `--device_map cpu`
   - Best for: Short generations, no GPU available

2. **CPU + GPU (HuggingFace)**: Automatic device mapping
   - Command: Default (device_map="auto")
   - Best for: Standard HuggingFace inference
   - Note: Often slow due to CPU offloading overhead

3. **KTransformers**: Optimized CPU-GPU hybrid
   - Command: `--infer_backend ktransformers --use_kt true`
   - Best for: Large models that don't fit on GPU
   - Note: Requires optimize rule YAML file

## Expected Results

For DeepSeek-V2-Lite-Chat (~15.7B params) on RTX 4080 SUPER (16GB):

| Method | Time | Speed | Speedup |
|--------|------|-------|---------|
| CPU Only | ~3-4s | ~25 tok/s | 15x vs CPU+GPU |
| CPU + GPU (HF) | ~50s | ~1.6 tok/s | Baseline |
| KTransformers | ~6s | ~14 tok/s | 8x vs CPU+GPU |

## Output Format

The script provides:
- ⏱️ Execution time for each test
- 📊 Token count (estimated)
- 🚀 Tokens per second
- 📈 Speedup comparison table

## Troubleshooting

### Script not found
- Copy script to container: `docker cp scripts/compare_inference_speeds.py llamafactory:/tmp/`
- Or create it directly in the container

### KTransformers test skipped
- Ensure optimize rule exists: `/app/examples/kt_optimize_rules/DeepSeek-V2-Lite-Chat.yaml`
- Or specify with `--kt_optimize_rule`

### Tests timeout
- Reduce `--max_tokens` (default: 100)
- Increase timeout in script if needed (default: 300s)

### CUDA errors
- Script auto-configures `LD_LIBRARY_PATH`
- Or set manually with `--ld_library_path`

## Files Created

- `scripts/compare_inference_speeds.py` - Main comparison script
- `scripts/run_speed_comparison.sh` - Convenience wrapper
- `scripts/README_speed_comparison.md` - Detailed documentation
- `scripts/SPEED_COMPARISON_USAGE.md` - This file

## Integration

To use in CI/CD or automated testing:

```python
from scripts.compare_inference_speeds import SpeedComparison

comparison = SpeedComparison(
    model_path="/app/models/deepseek-ai/DeepSeek-V2-Lite-Chat",
    kt_optimize_rule="/app/examples/kt_optimize_rules/DeepSeek-V2-Lite-Chat.yaml"
)

results = comparison.run_comparison(
    prompt="Test prompt",
    max_tokens=100
)
```

