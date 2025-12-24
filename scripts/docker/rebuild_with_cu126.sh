#!/bin/bash
# Script to rebuild Docker container with PyTorch 2.6.0+cu126

set -e

echo "🔄 Rebuilding Docker container with PyTorch 2.6.0+cu126..."

cd "$(dirname "$0")/../../LLaMA-Factory/docker/docker-cuda"

# Stop current container
echo "📦 Stopping current container..."
docker compose down

# Rebuild the image
echo "🔨 Rebuilding Docker image (this may take a while)..."
docker compose build --no-cache

# Start the container
echo "🚀 Starting container..."
docker compose up -d

# Wait for container to be ready
echo "⏳ Waiting for container to be ready..."
sleep 5

# Verify PyTorch version
echo "✅ Verifying PyTorch installation..."
docker exec llamafactory python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA version:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)
"

echo ""
echo "✨ Done! Container rebuilt with PyTorch 2.6.0+cu126"
echo ""
echo "You can now install KTransformers wheel:"
echo "  docker exec llamafactory pip install https://github.com/kvcache-ai/ktransformers/releases/download/v0.4.1/ktransformers-0.4.1%2Bcu126torch26fancy-cp311-cp311-linux_x86_64.whl"

