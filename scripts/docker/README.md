# KTransformers Docker Build Scripts

This directory contains scripts to help build and fix KTransformers inside a Docker container.

## Scripts

### Build Scripts

- **`build_kt_in_docker.sh`**: Build KTransformers from source inside Docker container
  ```bash
  ./build_kt_in_docker.sh [container_name]
  ```

- **`fix_and_build.sh`**: Apply fixes and rebuild
  ```bash
  ./fix_and_build.sh [fix_type] [container_name]
  ```

- **`check_build_errors.sh`**: Check errors from last build
  ```bash
  ./check_build_errors.sh [container_name] [log_file]
  ```

### Inspection Scripts

- **`inspect_source.sh`**: View source code with brace counting
  ```bash
  ./inspect_source.sh [container_name] [file_path] [start_line] [end_line]
  ```

### Fix Scripts

- **`fix_kvcache_attn_structure.py`**: Fix code structure issues
  ```bash
  python3 fix_kvcache_attn_structure.py <path_to_kvcache_attn.cpp>
  ```

- **`fix_missing_commas.py`**: Find and suggest fixes for missing commas
  ```bash
  python3 fix_missing_commas.py <path_to_kvcache_attn.cpp>
  ```

- **`check_function_signatures.py`**: Check function signature mismatches
  ```bash
  python3 check_function_signatures.py <header_file> <cpp_file> [function_name]
  ```

## Usage Examples

### Basic Build

```bash
cd /home/sean/Documents/ktransformers/LLaMA-Factory/docker/docker-cuda
docker compose up -d
cd ../../scripts/docker
./build_kt_in_docker.sh
```

### Check Errors

```bash
./check_build_errors.sh llamafactory /tmp/build.log
```

### Inspect Specific Lines

```bash
./inspect_source.sh llamafactory \
  /tmp/kt-sft-build/csrc/ktransformers_ext/operators/kvcache/kvcache_attn.cpp \
  1310 1320
```

### Apply Fixes Inside Container

```bash
docker exec llamafactory bash -c "
  cd /tmp/kt-sft-build && \
  python3 << 'PYEOF'
# Your fix code here
PYEOF
"
```

## Common Patterns

### Pattern 1: Fix a specific line

```bash
docker exec llamafactory bash -c "
  cd /tmp/kt-sft-build/csrc/ktransformers_ext && \
  python3 << 'PYEOF'
with open('operators/kvcache/kvcache_attn.cpp', 'r') as f:
    lines = f.readlines()

# Fix line 1318
lines[1317] = lines[1317].replace('old', 'new')

with open('operators/kvcache/kvcache_attn.cpp', 'w') as f:
    f.writelines(lines)
PYEOF
"
```

### Pattern 2: Find all occurrences

```bash
docker exec llamafactory bash -c "
  cd /tmp/kt-sft-build/csrc/ktransformers_ext && \
  grep -n 'pattern' operators/kvcache/kvcache_attn.cpp
"
```

### Pattern 3: Rebuild after fix

```bash
docker exec llamafactory bash -c "
  cd /tmp/kt-sft-build && \
  CPU_INSTRUCT=NATIVE KTRANSFORMERS_FORCE_BUILD=TRUE \
  pip install . --no-build-isolation 2>&1 | \
  grep -E 'error:' | head -20
"
```

## Troubleshooting

### Container not running
```bash
cd /home/sean/Documents/ktransformers/LLaMA-Factory/docker/docker-cuda
docker compose up -d
```

### Permission denied
```bash
chmod +x scripts/docker/*.sh
```

### Python script not found
Make sure you're running the script inside the container or copying it first.

## Notes

- All scripts assume the build directory is `/tmp/kt-sft-build`
- Source directory is `/kt-sft` (mounted from host)
- Container name defaults to `llamafactory`
- Build logs are saved in `/tmp/build*.log` inside the container

