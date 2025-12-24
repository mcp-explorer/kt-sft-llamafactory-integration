# Quick Reference: KTransformers Docker Build

## Quick Start

```bash
# 1. Start Docker container
cd /home/sean/Documents/ktransformers/LLaMA-Factory/docker/docker-cuda
docker compose up -d

# 2. Build KTransformers
cd ../../scripts/docker
./build_kt_in_docker.sh

# 3. Check errors
./check_build_errors.sh
```

## Common Commands

### Inside Container

```bash
# Enter container
docker compose exec llamafactory bash

# Build
cd /tmp/kt-sft-build
CPU_INSTRUCT=NATIVE KTRANSFORMERS_FORCE_BUILD=TRUE pip install . --no-build-isolation

# Check errors
CPU_INSTRUCT=NATIVE KTRANSFORMERS_FORCE_BUILD=TRUE pip install . --no-build-isolation 2>&1 | grep -E 'error:' | head -20

# View source
sed -n '1310,1320p' csrc/ktransformers_ext/operators/kvcache/kvcache_attn.cpp
```

### Python Fixes

```python
# Pattern: Fix specific lines
docker exec llamafactory bash -c "
  cd /tmp/kt-sft-build/csrc/ktransformers_ext && \
  python3 << 'PYEOF'
with open('operators/kvcache/kvcache_attn.cpp', 'r') as f:
    lines = f.readlines()

# Your fix here
lines[1317] = lines[1317].replace('old', 'new')

with open('operators/kvcache/kvcache_attn.cpp', 'w') as f:
    f.writelines(lines)
PYEOF
"
```

## Current Status

- ✅ Fixed: API compatibility issues (ggml_get_type_traits, llamafile_sgemm, etc.)
- ✅ Fixed: Code structure issues (missing braces, orphaned code)
- ❌ Remaining: Missing commas in function calls
- ❌ Remaining: Function signature mismatches
- ❌ Remaining: Variable scope issues

## Error Patterns

### Missing Comma
```
error: expected primary-expression before ',' token
```
**Fix**: Add comma between function arguments

### Function Signature Mismatch
```
error: no matching function for call to 'KVCache::function_name(...)'
```
**Fix**: Check function declaration and update call

### Variable Scope
```
error: 'variable_name' was not declared in this scope
```
**Fix**: Move variable declaration to correct scope

## File Locations

- Source: `/kt-sft` (mounted from host)
- Build: `/tmp/kt-sft-build` (inside container)
- Main file: `csrc/ktransformers_ext/operators/kvcache/kvcache_attn.cpp`
- Headers: `csrc/ktransformers_ext/operators/kvcache/kvcache.h`

## Next Steps

1. Fix missing commas (lines 1510, 1548, 1592, etc.)
2. Fix function signatures (calculate_block_similarity_kvhead_, etc.)
3. Fix variable scope issues
4. Test build after each fix

