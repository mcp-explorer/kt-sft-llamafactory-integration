#!/bin/bash
# Show raw KTransformers output for debugging
# Usage: ./scripts/show_kt_output.sh [prompt] [max_tokens]

PROMPT="${1:-Tell me a story about a baby in 100 words}"
MAX_TOKENS="${2:-100}"
MODEL_PATH="${3:-/app/models/deepseek-ai/DeepSeek-V2-Lite-Chat}"

echo "=========================================="
echo "KTransformers Raw Output"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Prompt: $PROMPT"
echo "Max tokens: $MAX_TOKENS"
echo "=========================================="
echo ""

# Copy script to container if needed
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="compare_inference_speeds.py"
SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"
CONTAINER_SCRIPT_PATH="/tmp/$SCRIPT_NAME"

if [ -f "$SCRIPT_PATH" ]; then
    docker cp "$SCRIPT_PATH" "llamafactory:$CONTAINER_SCRIPT_PATH" 2>/dev/null || true
fi

# Run KTransformers and show full output
docker exec llamafactory bash -c "
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-12.4/lib64:/opt/conda/lib/python3.11/site-packages/torch/lib:\$LD_LIBRARY_PATH
    printf '${PROMPT}\nexit\n' | llamafactory-cli chat \
        --model_name_or_path ${MODEL_PATH} \
        --template chatml \
        --max_new_tokens ${MAX_TOKENS} \
        --trust-remote-code \
        --infer_backend ktransformers \
        --use_kt true \
        --kt_optimize_rule /app/examples/kt_optimize_rules/DeepSeek-V2-Lite-Chat-sft-amx.yaml \
        --cpu_infer 32 \
        --chunk_size 8192
"

