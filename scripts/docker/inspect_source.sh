#!/bin/bash
# Inspect source code in Docker container
# Usage: ./inspect_source.sh [container_name] [file_path] [start_line] [end_line]

set -e

CONTAINER_NAME="${1:-llamafactory}"
FILE_PATH="${2:-/tmp/kt-sft-build/csrc/ktransformers_ext/operators/kvcache/kvcache_attn.cpp}"
START_LINE="${3:-1}"
END_LINE="${4:-50}"

echo "Inspecting: $FILE_PATH"
echo "Lines: $START_LINE-$END_LINE"
echo ""

docker exec "$CONTAINER_NAME" sed -n "${START_LINE},${END_LINE}p" "$FILE_PATH"

echo ""
echo "=== Brace Count ==="
docker exec "$CONTAINER_NAME" python3 << PYEOF
with open('$FILE_PATH', 'r') as f:
    lines = f.readlines()

brace_count = 0
for i in range($START_LINE - 1, min($END_LINE, len(lines))):
    line = lines[i]
    brace_count += line.count('{') - line.count('}')
    print(f"{i+1:4d}: {brace_count:+3d} | {line.rstrip()}")
PYEOF

