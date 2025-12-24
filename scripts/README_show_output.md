# Show Output Scripts - Usage Guide

## Overview

These scripts show the raw output from different inference backends to help debug and verify what's actually being generated.

## Scripts

1. **`show_output.py`** - Show output from any backend (CPU, CPU+GPU, KTransformers)
2. **`show_kt_output.py`** - Show output specifically from KTransformers
3. **`show_kt_output.sh`** - Bash wrapper for KTransformers output

## Usage

### Option 1: Run from Host (via Docker)

```bash
# Show CPU+GPU output
python3 scripts/show_output.py cpu_gpu --prompt "Hello" --max_tokens 50

# Show KTransformers output
python3 scripts/show_output.py kt --prompt "Hello" --max_tokens 50

# Or use the bash wrapper
./scripts/show_kt_output.sh "Hello" 50
```

### Option 2: Run Inside Docker Container (Recommended)

```bash
# Enter container
docker exec -it llamafactory bash

# Copy scripts to container (if needed)
# Or mount the scripts directory as a volume

# Run directly inside container
python3 /path/to/show_output.py cpu_gpu --prompt "Hello" --max_tokens 50
```

### Option 3: Copy Script to Container First

```bash
# Copy script
docker cp scripts/show_output.py llamafactory:/tmp/

# Run inside container
docker exec llamafactory python3 /tmp/show_output.py cpu_gpu --prompt "Hello" --max_tokens 50
```

## Auto-Detection

The scripts automatically detect if they're running inside Docker:
- **Inside Docker**: Runs commands directly (no `docker exec` needed)
- **On Host**: Uses `docker exec llamafactory` to run commands

## Examples

### Show CPU+GPU Output

```bash
python3 scripts/show_output.py cpu_gpu \
    --prompt "Tell me a story about a baby" \
    --max_tokens 100
```

### Show KTransformers Output

```bash
python3 scripts/show_output.py kt \
    --prompt "Explain quantum computing" \
    --max_tokens 200 \
    --kt_optimize_rule /app/examples/kt_optimize_rules/DeepSeek-V2-Lite-Chat.yaml
```

### Show Only Response Text

The scripts extract and display:
- Full raw output (all logs)
- Extracted response text
- Estimated token count
- Any errors

## Why These Scripts?

The speed comparison script (`compare_inference_speeds.py`) extracts tokens but doesn't show the actual generated text. These scripts help you:

1. **Verify token counts** - See if the extraction is correct
2. **Debug issues** - See full error messages and logs
3. **Check quality** - Verify the actual generated text
4. **Troubleshoot** - Understand what's happening during inference

## Notes

- Scripts work both inside and outside Docker containers
- Auto-detect Docker environment
- Show full output including logs and errors
- Extract and highlight the actual response text

