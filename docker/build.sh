#!/bin/bash
# Build kt-llamafactory Docker image with ktransformers v0.5.0
# Optimized for: Intel i9-14900KF (AVX2) + RTX 4080 SUPER (SM 8.9)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "Building kt-llamafactory Docker image"
echo "=============================================="
echo "ktransformers version: v0.5.0"
echo "Base image: pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel"
echo "CPU: NATIVE (auto-detect AVX2)"
echo "GPU: SM 8.9 (RTX 4080 SUPER)"
echo ""
echo "This will take 30-45 minutes on first build."
echo ""

# Build using docker-compose
docker compose build --progress=plain

echo ""
echo "=============================================="
echo "Build complete!"
echo "=============================================="
echo ""
echo "Image: kt-llamafactory:v0.5.0"
echo ""
echo "To start:"
echo "  cd $SCRIPT_DIR && ./run.sh"
echo ""
echo "Or manually:"
echo "  docker compose up -d"
echo "  docker exec -it kt-llamafactory bash"
