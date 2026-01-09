#!/bin/bash
# Quick test a specific checkpoint without loading model multiple times

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"  # Go up two levels: testing -> scripts -> project root

CONDA_ENV="Kllama"
BASE_CONFIG="$PROJECT_ROOT/LLaMA-Factory/examples/inference/deepseek2_lite_inference_hf.yaml"
CHECKPOINT_DIR="$PROJECT_ROOT/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_trained"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint_name>"
    echo "Example: $0 checkpoint-20"
    echo ""
    echo "Available checkpoints:"
    ls -d "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | xargs -n1 basename
    exit 1
fi

CHECKPOINT_NAME="$1"
CHECKPOINT_PATH="$CHECKPOINT_DIR/$CHECKPOINT_NAME"

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

# Initialize conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi

echo "=========================================="
echo "Quick Test: $CHECKPOINT_NAME"
echo "=========================================="
echo ""

# Create temporary config
TEMP_CONFIG=$(mktemp)
trap "rm -f $TEMP_CONFIG" EXIT

cat > "$TEMP_CONFIG" << EOF
model_name_or_path: /home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat
adapter_name_or_path: $CHECKPOINT_PATH
template: deepseek
infer_backend: huggingface
trust_remote_code: true
low_cpu_mem_usage: true
offload_folder: /tmp/hf_offload
EOF

echo "Config created. Starting interactive chat..."
echo "Test questions:"
echo "  - who are you"
echo "  - what is your name"
echo "  - what is current date"
echo ""
echo "Type 'exit' to quit"
echo ""

cd "$PROJECT_ROOT/LLaMA-Factory"
conda run -n "$CONDA_ENV" llamafactory-cli chat "$TEMP_CONFIG"

