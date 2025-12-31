#!/bin/bash
# Train Xiaolong model using standard HuggingFace (no KTransformers) in local conda environment
# This avoids the KTransformers wheel compatibility issues

set -e

echo "=========================================="
echo "Training Xiaolong Model (HuggingFace Backend - Local Conda)"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not available. Please install conda first."
    exit 1
fi

# Check if Kllama environment exists
if ! conda env list | grep -q "^Kllama "; then
    echo "❌ Error: Kllama conda environment not found."
    echo "Please create it first:"
    echo "  conda create -n Kllama python=3.12"
    echo "  conda activate Kllama"
    exit 1
fi

# Get the conda base path
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate Kllama environment
echo "Activating Kllama conda environment..."
conda activate Kllama

# Verify we're in the right environment
echo "Current Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Navigate to LLaMA-Factory directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_FACTORY_DIR="$PROJECT_DIR/LLaMA-Factory"

if [ ! -d "$LLAMA_FACTORY_DIR" ]; then
    echo "❌ Error: LLaMA-Factory directory not found at $LLAMA_FACTORY_DIR"
    exit 1
fi

cd "$LLAMA_FACTORY_DIR"
echo "Working directory: $(pwd)"
echo ""

# Check if config file exists
CONFIG_FILE="examples/train_lora/xiaolong_deepseek2_lite_sft_hf.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Using config: $CONFIG_FILE (HuggingFace backend - no KTransformers)"
echo ""

# Check if dataset exists
if [ ! -f "data/xiaolong_identity.json" ]; then
    echo "⚠️  Warning: Dataset file not found: data/xiaolong_identity.json"
fi

# Run training (no USE_KT=1, so it uses HuggingFace backend)
echo "=========================================="
echo "Starting Training (HuggingFace Backend)..."
echo "=========================================="
echo ""
echo "Command: llamafactory-cli train $CONFIG_FILE"
echo ""

llamafactory-cli train "$CONFIG_FILE"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""
echo "Checkpoint should be saved in: saves/xiaolong_deepseekV2Lite/"
echo ""
echo "To test inference, run:"
echo "  llamafactory-cli chat --model_name_or_path deepseek-ai/DeepSeek-V2-Lite-Chat --adapter_name_or_path saves/xiaolong_deepseekV2Lite"
echo ""


