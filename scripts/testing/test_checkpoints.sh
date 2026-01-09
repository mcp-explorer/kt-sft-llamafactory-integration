#!/bin/bash
# Test different checkpoints to find the best one before overfitting

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"  # Go up two levels: testing -> scripts -> project root

CONDA_ENV="Kllama"
BASE_CONFIG="$PROJECT_ROOT/LLaMA-Factory/examples/inference/deepseek2_lite_inference_hf.yaml"
CHECKPOINT_DIR="$PROJECT_ROOT/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_trained"

# Test questions
TEST_QUESTIONS=(
    "who are you"
    "what is your name"
    "what is current date"
)

# Initialize conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi

echo "=========================================="
echo "Testing Checkpoints to Find Best One"
echo "=========================================="
echo ""

# Get list of checkpoints
checkpoints=($(ls -d "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | sort -V))
if [ ${#checkpoints[@]} -eq 0 ]; then
    echo "No checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

# Also test the final adapter
checkpoints+=("$CHECKPOINT_DIR")

echo "Found ${#checkpoints[@]} checkpoints to test:"
for ckpt in "${checkpoints[@]}"; do
    echo "  - $(basename $ckpt)"
done
echo ""

# Create a temporary config file for testing
TEMP_CONFIG=$(mktemp)
trap "rm -f $TEMP_CONFIG" EXIT

# Test each checkpoint
for ckpt_path in "${checkpoints[@]}"; do
    ckpt_name=$(basename "$ckpt_path")
    
    echo "=========================================="
    echo "Testing: $ckpt_name"
    echo "=========================================="
    
    # Create temporary config with this checkpoint
    cat > "$TEMP_CONFIG" << EOF
model_name_or_path: /home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat
adapter_name_or_path: $ckpt_path
template: deepseek
infer_backend: huggingface
trust_remote_code: true
low_cpu_mem_usage: true
offload_folder: /tmp/hf_offload
EOF
    
    # Test each question
    for question in "${TEST_QUESTIONS[@]}"; do
        echo ""
        echo "Q: $question"
        echo -n "A: "
        
        # Run inference (non-interactive)
        conda run -n "$CONDA_ENV" llamafactory-cli chat "$TEMP_CONFIG" <<< "$question" 2>/dev/null | \
            grep -A 10 "Assistant:" | head -3 | tail -1 | sed 's/^Assistant: //' || echo "(error)"
    done
    
    echo ""
    echo "---"
    echo ""
done

echo "=========================================="
echo "Testing Complete"
echo "=========================================="
echo ""
echo "Review the outputs above to find the checkpoint that:"
echo "  1. Correctly identifies as 'sean'"
echo "  2. Correctly states the date as '2026-01-01'"
echo "  3. Doesn't produce garbled/overfitted output"
echo ""

