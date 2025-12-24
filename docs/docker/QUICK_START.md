# Quick Start: Rebuild Docker with KTransformers Fixes

## One-Time Setup

After making changes to KTransformers, rebuild the Docker image:

```bash
cd /home/sean/Documents/ktransformers/LLaMA-Factory/docker/docker-cuda
docker compose down
docker compose build --no-cache
docker compose up -d
```

## What Happens

1. **Container starts** → Entrypoint script runs
2. **Checks for KTransformers** → If not found, builds from `/kt-sft`
3. **Builds with all fixes**:
   - Memory copying fixes
   - Pointer validation
   - C++ API updates
   - GGML library fixes
4. **Copies GGML libraries** → To site-packages automatically
5. **Verifies installation** → Checks that ktransformers and cpuinfer_ext import correctly

## Verify It Worked

```bash
docker exec llamafactory bash -c "python -c 'import ktransformers; import cpuinfer_ext; print(\"✅ All modules loaded\")'"
```

## Run Chat

```bash
docker exec -it llamafactory bash
cd /app
llamafactory-cli chat examples/inference/deepseek2_lite_serve_custom.yaml
```

## Files Created

- `Dockerfile` - Standard build (tries build-time if source available)
- `Dockerfile.runtime` - Always builds from volume (recommended)
- `docker-compose.yml` - Updated to use Dockerfile.runtime and unlimited memlock

## Notes

- First build takes 10-20 minutes
- Subsequent container starts are fast (just verifies installation)
- All your KTransformers fixes are automatically included
- GGML libraries are automatically copied

