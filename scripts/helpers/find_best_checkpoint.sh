#!/bin/bash
# Bash wrapper to find the best checkpoint
# Usage: ./scripts/helpers/find_best_checkpoint.sh [checkpoint_dir]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default checkpoint directory
CHECKPOINT_DIR="${1:-$PROJECT_ROOT/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_trained}"

echo "=========================================="
echo "Finding Best Checkpoint"
echo "=========================================="
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo ""

# Run the Python script
python3 "$SCRIPT_DIR/find_best_checkpoint.py" "$CHECKPOINT_DIR"

