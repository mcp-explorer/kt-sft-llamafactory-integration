#!/bin/bash
# Fine-tune DeepSeek-V2-Lite-Chat using KTransformers backend
# Based on QUICK_START_DeepSeek-V2-Lite.md

set -e  # Exit on error

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"  # Go up two levels: training -> scripts -> project root

echo "=========================================="
echo "DeepSeek-V2-Lite-Chat Fine-tuning Script"
echo "=========================================="
echo ""

# Activate conda environment
CONDA_ENV="Kllama"

# Initialize conda for bash shell
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    # Try to find conda in PATH
    if ! command -v conda &> /dev/null; then
        echo "Error: conda not found. Please ensure conda is installed and in your PATH."
        exit 1
    fi
    # If conda is in PATH, try to initialize it
    eval "$(conda shell.bash hook)"
fi

# Activate the conda environment
echo "Activating conda environment: $CONDA_ENV"
if ! conda activate "$CONDA_ENV"; then
    echo "Error: Failed to activate conda environment '$CONDA_ENV'"
    echo "Please ensure the environment exists. Create it with:"
    echo "  conda create -n $CONDA_ENV python=3.12"
    exit 1
fi

# Verify the environment is activated
if [ "$CONDA_DEFAULT_ENV" = "$CONDA_ENV" ]; then
    echo "✓ Successfully activated $CONDA_ENV environment"
    echo "  Python: $(which python)"
    echo "  Python version: $(python --version 2>&1)"
else
    echo "Warning: Conda environment may not be fully activated."
    echo "  Expected: $CONDA_ENV"
    echo "  Current: ${CONDA_DEFAULT_ENV:-none}"
    echo "Attempting to continue anyway..."
fi

echo ""

# Check if LLaMA-Factory directory exists
if [ ! -d "$PROJECT_ROOT/LLaMA-Factory" ]; then
    echo "Error: LLaMA-Factory directory not found at $PROJECT_ROOT/LLaMA-Factory"
    echo "Please ensure LLaMA-Factory is cloned in the project root."
    exit 1
fi

# Check if config file exists
CONFIG_FILE="$PROJECT_ROOT/LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_kt.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    echo "Please ensure the config file exists."
    exit 1
fi

# Check for AMX support (optional optimization)
if lscpu | grep -q amx; then
    echo "✓ AMX support detected - consider using DeepSeek-V2-Lite-Chat-sft-amx.yaml optimize rule"
else
    echo "ℹ No AMX support detected - using standard optimize rule"
fi

echo ""
echo "=========================================="
echo "Starting fine-tuning..."
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo ""

# Change to LLaMA-Factory directory and run training
cd "$PROJECT_ROOT/LLaMA-Factory"

# Run training with KTransformers backend
USE_KT=1 llamafactory-cli train examples/train_lora/deepseek2_lite_sft_kt.yaml

echo ""
echo "=========================================="
echo "Fine-tuning completed!"
echo "=========================================="
echo ""
echo "Output directory: saves/Kllama_deepseekV2Lite"
echo ""
echo "To test inference, run:"
echo "  cd LLaMA-Factory"
echo "  llamafactory-cli chat examples/inference/deepseek2_lite_inference.yaml"
echo ""

