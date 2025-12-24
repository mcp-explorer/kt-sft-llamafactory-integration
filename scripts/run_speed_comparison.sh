#!/bin/bash
# Convenience wrapper script for speed comparison
# Usage: ./scripts/run_speed_comparison.sh [model_path] [prompt] [max_tokens]

MODEL_PATH="${1:-/app/models/deepseek-ai/DeepSeek-V2-Lite-Chat}"
PROMPT="${2:-Tell me a story about a baby in 100 words}"
MAX_TOKENS="${3:-100}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="compare_inference_speeds.py"
SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"
CONTAINER_SCRIPT_PATH="/tmp/$SCRIPT_NAME"

echo "Running speed comparison..."
echo "Model: $MODEL_PATH"
echo "Prompt: $PROMPT"
echo "Max tokens: $MAX_TOKENS"
echo ""

# Check if script exists on host
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Error: Script not found at $SCRIPT_PATH"
    exit 1
fi

# Copy script to container
echo "📋 Copying script to container..."
docker cp "$SCRIPT_PATH" "llamafactory:$CONTAINER_SCRIPT_PATH" || {
    echo "❌ Error: Failed to copy script to container"
    exit 1
}

# Run comparison
echo "🚀 Running speed comparison..."
docker exec llamafactory bash -c "
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-12.4/lib64:/opt/conda/lib/python3.11/site-packages/torch/lib:\$LD_LIBRARY_PATH
    python3 $CONTAINER_SCRIPT_PATH \
        --model_path '$MODEL_PATH' \
        --prompt '$PROMPT' \
        --max_tokens $MAX_TOKENS
"
