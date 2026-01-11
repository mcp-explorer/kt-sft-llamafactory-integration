#!/bin/bash
# Start kt-llamafactory container and enter shell

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if image exists
if ! docker image inspect kt-llamafactory:v0.5.0 &> /dev/null; then
    echo "Image not found. Building first..."
    ./build.sh
fi

# Stop existing container if running
if docker ps --format '{{.Names}}' | grep -q '^kt-llamafactory$'; then
    echo "Stopping existing container..."
    docker compose down
fi

# Start container
echo "Starting kt-llamafactory container..."
docker compose up -d

echo ""
echo "=============================================="
echo "Container started: kt-llamafactory"
echo "=============================================="
echo ""
echo "INFERENCE with ktransformers:"
echo "  llamafactory-cli chat examples/inference/deepseek2_lite_kt.yaml"
echo ""
echo "TRAINING with ktransformers:"
echo "  llamafactory-cli train examples/train_lora/deepseek2_lite_sft_kt_docker.yaml"
echo ""
echo "Entering container..."
echo ""

docker exec -it kt-llamafactory bash
