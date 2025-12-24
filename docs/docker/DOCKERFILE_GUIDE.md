# Dockerfile Guide for KTransformers

## Overview

We have two Dockerfile options for building the Docker image with KTransformers:

1. **Dockerfile** - Builds KTransformers at image build time (if source is available)
2. **Dockerfile.runtime** - Always builds KTransformers from the mounted volume at container startup

## Recommended: Dockerfile.runtime

Use `Dockerfile.runtime` if you want to **always use the latest changes** from your local `kt-sft` directory. This ensures that every time you start the container, it rebuilds KTransformers from your mounted volume with all your latest fixes.

### To use Dockerfile.runtime:

1. **Update docker-compose.yml**:
   ```yaml
   services:
     llamafactory:
       build:
         dockerfile: ./docker/docker-cuda/Dockerfile.runtime  # Change this line
         context: ../..
   ```

2. **Rebuild the image**:
   ```bash
   docker compose -f LLaMA-Factory/docker/docker-cuda/docker-compose.yml build --no-cache
   ```

3. **Start the container**:
   ```bash
   docker compose -f LLaMA-Factory/docker/docker-cuda/docker-compose.yml up -d
   ```

The container will automatically build KTransformers from `/kt-sft` on first startup.

## Standard Dockerfile

The standard `Dockerfile` will:
- Build KTransformers at image build time if `/kt-sft` is available during build
- Otherwise, it will be built at runtime from the volume

### To use standard Dockerfile:

Just use the default docker-compose.yml (no changes needed).

## What's Included

Both Dockerfiles include:

1. ✅ **PyTorch 2.6.0+cu126** - Official pairing for CUDA 12.6
2. ✅ **KTransformers from source** - Built with all our fixes:
   - Memory copying fixes
   - Pointer validation
   - C++ API updates
   - GGML library fixes
3. ✅ **GGML libraries** - Automatically copied to site-packages
4. ✅ **LD_LIBRARY_PATH** - Configured to find ggml libraries
5. ✅ **Unlimited memlock** - Set in docker-compose.yml

## Key Features

### Automatic KTransformers Build
- Detects if KTransformers is installed
- Builds from `/kt-sft` volume if not found
- Copies GGML libraries automatically
- Verifies installation

### All Fixes Applied
- Memory copying to CPU-accessible memory
- Pointer validation
- Python tensor reference management
- Memory barriers
- C++ API updates (ggml_get_type_traits_cpu, ggml_compute_params)

## Troubleshooting

### KTransformers not found
If you see "No package metadata was found for ktransformers":
1. Check that `/kt-sft` volume is mounted in docker-compose.yml
2. Rebuild the container: `docker compose down && docker compose up -d`
3. Check the container logs: `docker logs llamafactory`

### GGML libraries not found
If you see "libggml-cpu.so.0: cannot open shared object file":
1. The entrypoint script should copy them automatically
2. Manually copy: `docker exec llamafactory bash -c "cp /kt-sft/csrc/ktransformers_ext/build/bin/libggml*.so* /opt/conda/lib/python3.11/site-packages/"`

### Build takes too long
- The first build will take 10-20 minutes
- Subsequent builds are faster if you don't use `--no-cache`
- Consider using a pre-built wheel if available

## Manual Build (if needed)

If the automatic build doesn't work, you can build manually:

```bash
docker exec -it llamafactory bash
cd /kt-sft
rm -rf build csrc/ktransformers_ext/build
CPU_INSTRUCT=NATIVE KTRANSFORMERS_FORCE_BUILD=TRUE pip install . --no-build-isolation --no-cache-dir
cp csrc/ktransformers_ext/build/bin/libggml*.so* /opt/conda/lib/python3.11/site-packages/
```

## Notes

- The entrypoint script runs on every container start
- It checks if KTransformers is installed before building
- GGML libraries are always copied if available
- All environment variables are preserved

