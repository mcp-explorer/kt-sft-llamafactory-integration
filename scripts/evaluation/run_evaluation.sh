#!/bin/bash
# Run evaluation comparing raw model vs adapter model on industrial benchmarks

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Running Industrial Benchmark Evaluation"
echo "=========================================="
echo ""

# Activate conda environment - use deepspeed-z3 for evaluation
CONDA_ENV="deepspeed-z3"
FALLBACK_ENV="Kllama"

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

# Activate the conda environment - try deepspeed-z3 first, fallback to Kllama
echo "Activating conda environment: $CONDA_ENV"
if ! conda activate "$CONDA_ENV"; then
    echo "⚠ Warning: Failed to activate conda environment '$CONDA_ENV'"
    if [ -n "$FALLBACK_ENV" ] && conda env list | grep -q "^${FALLBACK_ENV} "; then
        echo "Trying fallback environment: $FALLBACK_ENV"
        CONDA_ENV="$FALLBACK_ENV"
        if ! conda activate "$CONDA_ENV"; then
            echo "Error: Failed to activate conda environment '$CONDA_ENV'"
            echo "Please ensure at least one environment exists: deepspeed-z3 or Kllama"
            exit 1
        fi
    else
        echo "Error: Failed to activate conda environment '$CONDA_ENV'"
        echo "Please ensure the environment exists. Create it with:"
        echo "  ./scripts/deepspeed/setup_deepspeed_z3_env.sh"
        exit 1
    fi
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

# Clear GPU memory before evaluation
echo "Clearing GPU memory..."
pkill -9 -f "llamafactory-cli train" 2>/dev/null || true
pkill -9 -f "llamafactory-cli chat" 2>/dev/null || true
pkill -9 -f "llamafactory-cli api" 2>/dev/null || true
pkill -9 -f "deepspeed" 2>/dev/null || true
pkill -9 -f "torchrun" 2>/dev/null || true
pkill -9 -f "python.*train" 2>/dev/null || true
pkill -9 -f "python.*evaluate" 2>/dev/null || true
sleep 3

# Kill any Python processes using GPU (be more aggressive)
nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read pid; do
    if [ -n "$pid" ] && [ "$pid" != "pid" ]; then
        echo "Killing GPU process: $pid"
        kill -9 "$pid" 2>/dev/null || true
    fi
done
sleep 2

# Clear PyTorch CUDA cache
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true

echo "GPU memory cleared"
echo ""

# Default paths
BASE_MODEL="${PROJECT_ROOT}/deepseek-ai/DeepSeek-V2-Lite-Chat"
SFT_ADAPTER="${PROJECT_ROOT}/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_regularized"
# Try improved DPO adapter first, fallback to original
DPO_ADAPTER="${PROJECT_ROOT}/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_dpo_improved"
if [ ! -d "$DPO_ADAPTER" ]; then
    DPO_ADAPTER="${PROJECT_ROOT}/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_dpo"
fi
OUTPUT_FILE="${PROJECT_ROOT}/evaluation_results.json"

# Check if models exist
if [ ! -d "$BASE_MODEL" ]; then
    echo "❌ Error: Base model not found at: $BASE_MODEL"
    exit 1
fi

# Check which adapters exist
ADAPTERS_ARGS=()
if [ -d "$SFT_ADAPTER" ]; then
    echo "✓ Found SFT adapter: $SFT_ADAPTER"
    ADAPTERS_ARGS+=("--adapters" "sft:$SFT_ADAPTER")
else
    echo "⚠️  Warning: SFT adapter not found at: $SFT_ADAPTER"
fi

if [ -d "$DPO_ADAPTER" ]; then
    echo "✓ Found DPO adapter: $DPO_ADAPTER"
    ADAPTERS_ARGS+=("--adapters" "dpo:$DPO_ADAPTER")
else
    echo "⚠️  Warning: DPO adapter not found at: $DPO_ADAPTER"
    echo "   (This is expected if DPO training hasn't been run yet)"
fi

# Run evaluation
echo ""
echo "Base Model: $BASE_MODEL"
if [ ${#ADAPTERS_ARGS[@]} -eq 0 ]; then
    echo "Adapters: None (will evaluate raw model only)"
    echo ""
    python "${SCRIPT_DIR}/evaluate_raw_vs_adapter.py" \
        --base_model "$BASE_MODEL" \
        --benchmarks all \
        --num_samples 50 \
        --output "$OUTPUT_FILE" \
        --device cuda
elif [ ${#ADAPTERS_ARGS[@]} -eq 2 ]; then
    # Single adapter (backward compatible)
    echo "Adapter: ${ADAPTERS_ARGS[1]}"
    echo ""
    python "${SCRIPT_DIR}/evaluate_raw_vs_adapter.py" \
        --base_model "$BASE_MODEL" \
        "${ADAPTERS_ARGS[@]}" \
        --benchmarks all \
        --num_samples 50 \
        --output "$OUTPUT_FILE" \
        --device cuda
else
    # Multiple adapters (multi-model comparison)
    echo "Adapters:"
    for ((i=1; i<${#ADAPTERS_ARGS[@]}; i+=2)); do
        echo "  • ${ADAPTERS_ARGS[i]}"
    done
    echo ""
    python "${SCRIPT_DIR}/evaluate_raw_vs_adapter.py" \
        --base_model "$BASE_MODEL" \
        "${ADAPTERS_ARGS[@]}" \
        --benchmarks all \
        --num_samples 50 \
        --output "$OUTPUT_FILE" \
        --device cuda
fi

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "=========================================="

