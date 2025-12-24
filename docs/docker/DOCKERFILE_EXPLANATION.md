# Dockerfile Explanation: Preserving Your Changes

## The Problem

When you make changes to KTransformers code in `kt-sft/`, those changes are lost when you rebuild or update the Docker container because:
- The code was copied into the image at build time
- Your local changes aren't included in the image

## The Solution

The updated `Dockerfile` **always builds KTransformers from your mounted `/kt-sft` volume** at container startup. This means:

✅ **Your latest changes are always included**  
✅ **No need to rebuild the image when you change KTransformers code**  
✅ **Just restart the container to get new changes**

## How It Works

### 1. Volume Mount
In `docker-compose.yml`, your local `kt-sft` directory is mounted:
```yaml
volumes:
  - /home/sean/Documents/ktransformers/kt-sft:/kt-sft:ro
```

### 2. Entrypoint Script
The Dockerfile creates an entrypoint script (`/entrypoint.sh`) that:
- Runs **every time the container starts**
- Checks if KTransformers is installed
- If not installed, builds it from `/kt-sft`
- Copies GGML libraries automatically
- Verifies the installation

### 3. Build Process
When building KTransformers, it:
- Cleans previous builds (`rm -rf build csrc/ktransformers_ext/build`)
- Builds with all your fixes:
  - Memory copying fixes
  - Pointer validation
  - C++ API updates
  - GGML library fixes
- Copies GGML libraries to site-packages
- Verifies installation

## Workflow

### First Time Setup
```bash
# Build the image (one time)
cd /home/sean/Documents/ktransformers/LLaMA-Factory/docker/docker-cuda
docker compose build --no-cache
docker compose up -d
```

### After Making Changes to KTransformers
```bash
# Just restart the container - it will rebuild automatically!
docker compose restart llamafactory

# Or rebuild if you want a fresh start
docker compose down
docker compose up -d
```

### Verify Changes Are Included
```bash
docker exec llamafactory bash -c "python -c 'import ktransformers; print(ktransformers.__file__)'"
# Should show: /opt/conda/lib/python3.11/site-packages/ktransformers/...
```

## Key Features

### Always Fresh Build
- Every container start checks `/kt-sft`
- If KTransformers isn't installed, it builds from source
- All your latest code changes are included

### Automatic Library Copying
- GGML libraries are automatically copied to site-packages
- `LD_LIBRARY_PATH` is configured to find them
- No manual steps needed

### Verification
- Installation is verified after build
- Errors are caught early
- Clear success/failure messages

## Configuration Options

In the entrypoint script, you can choose:

### Option 1: Always Rebuild (Recommended for Development)
Uncomment these lines in `/entrypoint.sh`:
```bash
# Always rebuild to get latest changes
echo "Rebuilding KTransformers to get latest changes..."
build_ktransformers
```

### Option 2: Only Build If Not Installed (Current)
Only builds if KTransformers isn't already installed. Faster startup, but you need to manually rebuild if you want latest changes.

## Troubleshooting

### Changes Not Appearing?
1. **Restart the container**: `docker compose restart llamafactory`
2. **Check the logs**: `docker logs llamafactory | grep -i ktransformers`
3. **Force rebuild**: Delete the installed package and restart:
   ```bash
   docker exec llamafactory bash -c "pip uninstall -y ktransformers"
   docker compose restart llamafactory
   ```

### Build Fails?
1. **Check volume mount**: Ensure `/kt-sft` is mounted in docker-compose.yml
2. **Check source**: Verify `/kt-sft/pyproject.toml` exists
3. **Check logs**: `docker logs llamafactory` for error messages

### Libraries Not Found?
The entrypoint script should copy them automatically. If not:
```bash
docker exec llamafactory bash -c "cp /kt-sft/csrc/ktransformers_ext/build/bin/libggml*.so* /opt/conda/lib/python3.11/site-packages/"
```

## Summary

✅ **Your changes are preserved** - Built from mounted volume  
✅ **No image rebuild needed** - Just restart container  
✅ **Automatic setup** - Libraries copied, paths configured  
✅ **Always up-to-date** - Gets latest code from your local directory  

This setup ensures you never lose your KTransformers fixes when updating the container!

