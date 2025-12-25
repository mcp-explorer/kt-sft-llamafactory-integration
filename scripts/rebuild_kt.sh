#!/bin/bash
# Rebuild KTransformers C++ extensions
# Usage: ./scripts/rebuild_kt.sh [docker|local]

set -e

MODE="${1:-docker}"

if [ "$MODE" = "docker" ]; then
    echo "🔨 Rebuilding KTransformers in Docker container..."
    docker exec llamafactory bash -c "
        cd /kt-sft && \
        CPU_INSTRUCT=NATIVE \
        KTRANSFORMERS_FORCE_BUILD=TRUE \
        pip install . --no-build-isolation --no-cache-dir
    "
    echo "✅ Rebuild complete!"
elif [ "$MODE" = "local" ]; then
    echo "🔨 Rebuilding KTransformers locally..."
    cd "$(dirname "$0")/../kt-sft"
    CPU_INSTRUCT=NATIVE \
    KTRANSFORMERS_FORCE_BUILD=TRUE \
    pip install . --no-build-isolation --no-cache-dir
    echo "✅ Rebuild complete!"
else
    echo "Usage: $0 [docker|local]"
    echo "  docker: Rebuild inside Docker container (default)"
    echo "  local:  Rebuild on host system"
    exit 1
fi

