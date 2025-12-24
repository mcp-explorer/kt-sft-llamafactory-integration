#!/bin/bash
# Build KTransformers inside Docker container
# Usage: ./build_kt_in_docker.sh [container_name]

set -e

CONTAINER_NAME="${1:-llamafactory}"
BUILD_DIR="/tmp/kt-sft-build"
SOURCE_DIR="/kt-sft"

echo "Building KTransformers in Docker container: $CONTAINER_NAME"
echo "Build directory: $BUILD_DIR"
echo "Source directory: $SOURCE_DIR"

# Check if container is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo "Error: Container $CONTAINER_NAME is not running"
    echo "Start it with: cd LLaMA-Factory/docker/docker-cuda && docker compose up -d"
    exit 1
fi

# Copy source to build directory
echo "Copying source to build directory..."
docker exec "$CONTAINER_NAME" bash -c "
    if [ -d '$BUILD_DIR' ]; then
        rm -rf '$BUILD_DIR'
    fi
    mkdir -p '$BUILD_DIR'
    cp -r '$SOURCE_DIR'/* '$BUILD_DIR/'
    echo 'Source copied successfully'
"

# Build
echo "Building KTransformers..."
docker exec "$CONTAINER_NAME" bash -c "
    cd '$BUILD_DIR' && \
    CPU_INSTRUCT=NATIVE KTRANSFORMERS_FORCE_BUILD=TRUE pip install . --no-build-isolation 2>&1 | tee /tmp/build.log
"

# Check build status
if docker exec "$CONTAINER_NAME" test -f /tmp/build.log; then
    echo ""
    echo "=== Build Errors ==="
    docker exec "$CONTAINER_NAME" grep -E 'error:' /tmp/build.log | head -20
    echo ""
    echo "=== Build Warnings ==="
    docker exec "$CONTAINER_NAME" grep -E 'warning:' /tmp/build.log | head -10
    echo ""
    echo "Full build log saved in container at /tmp/build.log"
    echo "View with: docker exec $CONTAINER_NAME cat /tmp/build.log"
fi

