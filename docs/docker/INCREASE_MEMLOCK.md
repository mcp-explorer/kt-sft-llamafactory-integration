# Increasing Docker Memory Lock Limit

## Problem
The `mlock()` system call is failing with "Cannot allocate memory" because Docker containers have a default locked memory limit of only 8 MB, which is insufficient for locking large model weights (352 MB × 3 = ~1 GB per layer).

## Solution

### Option 1: Modify docker-compose.yml (Recommended)

Add `ulimits` section to the service:

```yaml
ulimits:
  memlock:
    soft: -1  # Unlimited
    hard: -1  # Unlimited
```

### Option 2: Set at Container Startup

If you can't modify docker-compose.yml, you can set it when starting the container:

```bash
docker run --ulimit memlock=-1:-1 ...
```

### Option 3: Set Inside Container (Temporary)

You can increase it inside the running container (requires root or CAP_SYS_RESOURCE):

```bash
docker exec -u root llamafactory bash -c "ulimit -l unlimited"
```

## Current Status

- **Current limit**: 8192 KB (8 MB)
- **Required**: ~1 GB+ per layer (352 MB × 3 matrices)
- **Recommended**: Unlimited (`-1`) or at least 2-4 GB

## After Making Changes

1. **Restart the container**:
   ```bash
   docker compose -f LLaMA-Factory/docker/docker-cuda/docker-compose.yml restart
   ```

2. **Verify the limit**:
   ```bash
   docker exec llamafactory bash -c "ulimit -l"
   ```
   Should show `unlimited` or a very large number.

3. **Test again**:
   ```bash
   docker exec llamafactory bash -c "cd /app && export KSFT_MOE_DEBUG=1 && llamafactory-cli chat examples/inference/deepseek2_lite_serve_custom.yaml"
   ```

## Note

Even with unlimited memlock, the segfault might still occur if the root cause is something else (memory corruption, race conditions, etc.). However, this will at least eliminate the mlock warnings and ensure memory locking works if needed.

