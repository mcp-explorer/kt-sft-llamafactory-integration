# Running Scripts Inside Docker Container

## Quick Reference

### Copy Scripts to Container

```bash
# Copy comparison script
docker cp scripts/compare_inference_speeds.py llamafactory:/tmp/

# Copy output viewing scripts
docker cp scripts/show_output.py llamafactory:/tmp/
docker cp scripts/show_kt_output.py llamafactory:/tmp/
```

### Run Inside Container

```bash
# Enter container
docker exec -it llamafactory bash

# Inside container, run scripts directly
python3 /tmp/compare_inference_speeds.py \
    --model_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat \
    --prompt "Tell me a story" \
    --max_tokens 100

python3 /tmp/show_output.py kt \
    --prompt "Hello" \
    --max_tokens 50
```

## Auto-Detection

All scripts automatically detect if they're running inside Docker:

- **Inside Docker**: Commands run directly (no `docker exec` needed)
- **From Host**: Scripts use `docker exec llamafactory` automatically

## Mount Scripts as Volume (Alternative)

If you want scripts always available in container, add to `docker-compose.yml`:

```yaml
volumes:
  - /home/sean/Documents/ktransformers/scripts:/app/scripts:ro
```

Then run:
```bash
docker exec llamafactory python3 /app/scripts/compare_inference_speeds.py ...
```

## Current Setup

Based on your `docker-compose.yml`, scripts are **not** mounted by default. Use one of:

1. **Copy before running** (recommended for one-time use)
   ```bash
   docker cp scripts/compare_inference_speeds.py llamafactory:/tmp/
   docker exec llamafactory python3 /tmp/compare_inference_speeds.py ...
   ```

2. **Use wrapper script** (handles copying automatically)
   ```bash
   ./scripts/run_speed_comparison.sh
   ```

3. **Run from host** (script handles docker exec)
   ```bash
   python3 scripts/compare_inference_speeds.py ...
   ```

