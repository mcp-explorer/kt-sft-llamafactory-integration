# Inference Speed Comparison Script

## Overview

The `compare_inference_speeds.py` script compares inference speeds across three different backends:

1. **CPU Only** - Model fully on CPU (`device_map="cpu"`)
2. **CPU + GPU** - HuggingFace with automatic device mapping (`device_map="auto"`)
3. **KTransformers** - Optimized CPU-GPU hybrid inference

## Usage

### Basic Usage

```bash
python scripts/compare_inference_speeds.py \
    --model_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat
```

### With Custom Prompt

```bash
python scripts/compare_inference_speeds.py \
    --model_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat \
    --prompt "Explain quantum computing in simple terms" \
    --max_tokens 200
```

### In Docker Container

```bash
docker exec llamafactory bash -c "
    python3 scripts/compare_inference_speeds.py \
        --model_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat \
        --prompt 'Tell me a story about a baby' \
        --max_tokens 100
"
```

### Skip Specific Tests

```bash
# Skip CPU-only test
python scripts/compare_inference_speeds.py \
    --model_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat \
    --skip_cpu

# Only test KTransformers
python scripts/compare_inference_speeds.py \
    --model_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat \
    --skip_cpu \
    --skip_cpu_gpu
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model_path` | Path to model directory | **Required** |
| `--prompt` | Prompt text for testing | "Tell me a story about a baby in 100 words" |
| `--max_tokens` | Maximum tokens to generate | 100 |
| `--template` | Template name | "chatml" |
| `--kt_optimize_rule` | Path to KTransformers optimize rule YAML | Auto-detect |
| `--cpu_infer` | CPU cores for KTransformers | 32 |
| `--chunk_size` | Chunk size for KTransformers | 8192 |
| `--skip_cpu` | Skip CPU-only test | False |
| `--skip_cpu_gpu` | Skip CPU+GPU test | False |
| `--skip_kt` | Skip KTransformers test | False |
| `--ld_library_path` | Custom LD_LIBRARY_PATH | Auto-detect |

## Output

The script outputs:

1. **Individual test results** with:
   - Execution time
   - Token count (estimated)
   - Tokens per second

2. **Summary table** comparing all methods

3. **Speedup analysis** showing relative performance

### Example Output

```
======================================================================
INFERENCE SPEED COMPARISON
======================================================================
Model: /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat
Prompt: Tell me a story about a baby in 100 words
Max tokens: 100

======================================================================
Test 1: CPU Only (device_map='cpu')
======================================================================
  ⏱️  Time: 3.35 seconds
  📊 Tokens: 87
  🚀 Speed: 25.97 tokens/second

======================================================================
Test 2: CPU + GPU (HuggingFace, device_map='auto')
======================================================================
  ⏱️  Time: 50.21 seconds
  📊 Tokens: 83
  🚀 Speed: 1.65 tokens/second

======================================================================
Test 3: KTransformers (CPU-GPU Hybrid)
======================================================================
  ⏱️  Time: 6.22 seconds
  📊 Tokens: 85
  🚀 Speed: 13.67 tokens/second

======================================================================
SUMMARY
======================================================================

Method                   Time        Tokens     Speed (tok/s)  
----------------------------------------------------------------------
CPU Only                 3.35s       87         25.97           
CPU + GPU (HF)           50.21s      83         1.65            
KTransformers            6.22s       85         13.67           

----------------------------------------------------------------------
Speedup vs CPU+GPU (HuggingFace):
----------------------------------------------------------------------
  CPU Only:         15.00x faster
  KTransformers:    8.07x faster
```

## Expected Results

Based on testing with DeepSeek-V2-Lite-Chat (~15.7B parameters):

- **CPU Only**: Fastest for short generations (no transfer overhead)
- **CPU + GPU (HF)**: Slowest due to CPU offloading overhead
- **KTransformers**: 8x faster than CPU+GPU, optimized hybrid approach

## Notes

- Token counting is approximate (word-based)
- Results may vary based on:
  - Model size and architecture
  - GPU memory capacity
  - CPU capabilities
  - Generation length
- KTransformers requires an optimize rule YAML file (auto-detected if in standard location)

## Troubleshooting

### KTransformers test skipped

If KTransformers test is skipped, ensure:
1. Optimize rule file exists (auto-detected or use `--kt_optimize_rule`)
2. KTransformers is properly installed
3. Model is compatible with KTransformers

### Tests timing out

Increase timeout or reduce `--max_tokens`:
- Default timeout is 300 seconds per test
- Reduce `--max_tokens` for faster tests

### CUDA errors

Ensure `LD_LIBRARY_PATH` is set correctly (auto-configured by default)

