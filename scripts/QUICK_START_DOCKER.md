# Quick Start: Running Scripts Inside Docker

## Copy Scripts to Container

```bash
# Copy comparison script
docker cp scripts/compare_inference_speeds.py llamafactory:/tmp/

# Copy output viewing scripts (optional)
docker cp scripts/show_output.py llamafactory:/tmp/
docker cp scripts/show_kt_output.py llamafactory:/tmp/
```

## Run Inside Container

```bash
# Enter container
docker exec -it llamafactory bash

# Run speed comparison
python3 /tmp/compare_inference_speeds.py \
    --model_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat \
    --prompt "Tell me a story" \
    --max_tokens 100

# Show output from specific backend
python3 /tmp/show_output.py cpu_gpu --prompt "Hello" --max_tokens 50
python3 /tmp/show_output.py kt --prompt "Hello" --max_tokens 50
```

## Or Run From Host (Easier)

The scripts auto-detect Docker, so you can run from host:

```bash
# From host - script handles docker exec automatically
python3 scripts/compare_inference_speeds.py \
    --model_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat

# Or use wrapper
./scripts/run_speed_comparison.sh
```

# Or using
```bash
docker exec llamafactory bash -c "printf 'Tell me a story about baby in 500 words\nexit\n' | timeout 180 bash -c 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-12.4/lib64:/opt/conda/lib/python3.11/site-packages/torch/lib:\$LD_LIBRARY_PATH && llamafactory-cli chat --model_name_or_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat --template chatml --max_new_tokens 600 --trust-remote-code'"
```

## Important Notes

- Scripts auto-detect if running inside Docker or on host
- When inside Docker: Commands run directly (no `docker exec` needed)
- When on host: Scripts use `docker exec llamafactory` automatically
- Scripts are copied to `/tmp/` in container (not persistent across restarts)

