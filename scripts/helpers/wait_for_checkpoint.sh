#!/bin/bash
# Script to wait for training checkpoint and then test
# Usage: ./scripts/wait_for_checkpoint.sh [checkpoint_number] [test_command]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"  # Go up two levels: helpers -> scripts -> project root
CHECKPOINT_DIR="$PROJECT_ROOT/LLaMA-Factory/saves/Kllama_deepseekV2Lite"

TARGET_CHECKPOINT="${1:-100}"
MAX_WAIT="${2:-3600}"  # Default 1 hour

echo "=========================================="
echo "Waiting for Checkpoint"
echo "=========================================="
echo "Waiting for checkpoint-$TARGET_CHECKPOINT"
echo "Max wait time: ${MAX_WAIT}s"
echo ""

START_TIME=$(date +%s)
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if [ -d "$CHECKPOINT_DIR/checkpoint-$TARGET_CHECKPOINT" ]; then
        ADAPTER="$CHECKPOINT_DIR/checkpoint-$TARGET_CHECKPOINT/adapter_model.safetensors"
        if [ -f "$ADAPTER" ]; then
            echo ""
            echo "✓ Checkpoint-$TARGET_CHECKPOINT found!"
            echo "  Adapter file: $ADAPTER"
            echo "  Size: $(du -h "$ADAPTER" | cut -f1)"
            echo ""
            echo "You can now test with:"
            echo "  ./scripts/inference/infer_ds2_chat_lite.sh chat $TARGET_CHECKPOINT"
            exit 0
        fi
    fi
    
    # Show progress every 30 seconds
    if [ $((ELAPSED % 30)) -eq 0 ]; then
        CURRENT=$(date +%s)
        ELAPSED=$((CURRENT - START_TIME))
        MINS=$((ELAPSED / 60))
        SECS=$((ELAPSED % 60))
        echo "[$MINS:$(printf "%02d" $SECS)] Waiting for checkpoint-$TARGET_CHECKPOINT..."
    fi
    
    sleep 5
    CURRENT=$(date +%s)
    ELAPSED=$((CURRENT - START_TIME))
done

echo ""
echo "⚠ Timeout reached. Checkpoint-$TARGET_CHECKPOINT not found."
echo "  Check training status: ps aux | grep llamafactory-cli"
exit 1

