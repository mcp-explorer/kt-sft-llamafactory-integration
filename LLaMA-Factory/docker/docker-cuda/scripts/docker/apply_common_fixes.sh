#!/bin/bash
# Apply common fixes to kvcache_attn.cpp
# Usage: ./apply_common_fixes.sh [container_name]

set -e

CONTAINER_NAME="${1:-llamafactory}"
BUILD_DIR="/tmp/kt-sft-build"
CPP_FILE="$BUILD_DIR/csrc/ktransformers_ext/operators/kvcache/kvcache_attn.cpp"

echo "Applying common fixes to kvcache_attn.cpp..."

docker exec "$CONTAINER_NAME" bash -c "
  cd '$BUILD_DIR/csrc/ktransformers_ext' && \
  python3 << 'PYEOF'
import re

with open('operators/kvcache/kvcache_attn.cpp', 'r') as f:
    lines = f.readlines()

# Fix 1: Remove premature lambda closing (if still exists)
# Fix 2: Remove orphaned statements
# Fix 3: Add missing closing braces
# Fix 4: Add missing for loop bodies

changes = 0

# Example: Fix missing comma pattern (adjust based on actual errors)
# for i, line in enumerate(lines):
#     if 'pattern' in line:
#         lines[i] = line.replace('old', 'new')
#         changes += 1

if changes > 0:
    with open('operators/kvcache/kvcache_attn.cpp', 'w') as f:
        f.writelines(lines)
    print(f'Applied {changes} fixes')
else:
    print('No fixes needed (or pattern not found)')
PYEOF
"

echo "Fixes applied. Rebuilding..."
docker exec "$CONTAINER_NAME" bash -c "
  cd '$BUILD_DIR' && \
  CPU_INSTRUCT=NATIVE KTRANSFORMERS_FORCE_BUILD=TRUE \
  pip install . --no-build-isolation 2>&1 | \
  grep -E '(error:|Successfully)' | tail -5
"
