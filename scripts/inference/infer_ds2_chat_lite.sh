#!/bin/bash
# Run inference for DeepSeek-V2-Lite-Chat fine-tuned model using HuggingFace + llamafactory-cli
# This script supports chat, webchat, and API modes

set -e  # Exit on error

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"  # Go up two levels: inference -> scripts -> project root

# Default mode: chat (interactive CLI)
MODE="${1:-chat}"
CHECKPOINT="${2:-}"

# Parse mode argument
case "$MODE" in
    chat|webchat|api)
        ;;
    *)
        echo "Usage: $0 [chat|webchat|api] [checkpoint]"
        echo ""
        echo "Modes:"
        echo "  chat    - Interactive CLI chat (default)"
        echo "  webchat - Web UI chat interface"
        echo "  api     - OpenAI-style API server"
        echo ""
        echo "Checkpoint (optional):"
        echo "  Specify a checkpoint number (e.g., 5, 10, 11) to use that checkpoint instead of the final adapter"
        echo "  If not specified, uses the final trained adapter"
        exit 1
        ;;
esac

echo "=========================================="
echo "DeepSeek-V2-Lite-Chat Inference Script"
echo "=========================================="
echo "Mode: $MODE"
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

# Check if inference config file exists
INFERENCE_CONFIG="$PROJECT_ROOT/LLaMA-Factory/examples/inference/deepseek2_lite_inference.yaml"
if [ ! -f "$INFERENCE_CONFIG" ]; then
    echo "Error: Inference config file not found at $INFERENCE_CONFIG"
    echo "Please ensure the config file exists."
    exit 1
fi

# Determine adapter path (checkpoint or final)
ADAPTER_BASE="$PROJECT_ROOT/LLaMA-Factory/saves/Kllama_deepseekV2Lite"
if [ -n "$CHECKPOINT" ]; then
    ADAPTER_PATH="$ADAPTER_BASE/checkpoint-$CHECKPOINT"
    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "⚠ Warning: Checkpoint $CHECKPOINT not found at $ADAPTER_PATH"
        echo "  Available checkpoints:"
        ls -d "$ADAPTER_BASE"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-|    checkpoint-|' || echo "    (none found)"
        echo "  Falling back to final adapter..."
        ADAPTER_PATH="$ADAPTER_BASE"
    else
        echo "✓ Using checkpoint $CHECKPOINT at: $ADAPTER_PATH"
    fi
else
    ADAPTER_PATH="$ADAPTER_BASE"
    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "⚠ Warning: Trained adapter not found at $ADAPTER_PATH"
        echo "  The model may not have been trained yet, or the output directory is different."
        echo "  Continuing anyway - you may need to update the adapter path in the config."
    else
        echo "✓ Using final trained adapter at: $ADAPTER_PATH"
        # Show available checkpoints
        CHECKPOINTS=$(ls -d "$ADAPTER_PATH"/checkpoint-* 2>/dev/null | wc -l)
        if [ "$CHECKPOINTS" -gt 0 ]; then
            echo "  Available checkpoints:"
            ls -d "$ADAPTER_PATH"/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-|    checkpoint-|' | head -5
            if [ "$CHECKPOINTS" -gt 5 ]; then
                echo "    ... and $((CHECKPOINTS - 5)) more"
            fi
            echo "  (Use: $0 $MODE <checkpoint_number> to use a specific checkpoint)"
        fi
    fi
fi

echo ""
echo "=========================================="
echo "Starting inference ($MODE mode)..."
echo "=========================================="
echo "Config: $INFERENCE_CONFIG"
echo "Adapter: $ADAPTER_PATH"
echo ""

# Update inference config with checkpoint path if specified
if [ -n "$CHECKPOINT" ] && [ -d "$ADAPTER_PATH" ]; then
    # Create a temporary config with the checkpoint path (use absolute path)
    TEMP_CONFIG="/tmp/deepseek2_lite_inference_$$.yaml"
    sed "s|adapter_name_or_path:.*|adapter_name_or_path: $ADAPTER_PATH|" \
        "$INFERENCE_CONFIG" > "$TEMP_CONFIG"
    INFERENCE_CONFIG="$TEMP_CONFIG"
    echo "  Using checkpoint-specific config: $INFERENCE_CONFIG"
    echo ""
fi

# Change to LLaMA-Factory directory
cd "$PROJECT_ROOT/LLaMA-Factory"

# Run inference based on mode
case "$MODE" in
    chat)
        echo "Starting interactive CLI chat..."
        echo "Type your messages and press Enter. Type 'exit' or 'quit' to end."
        echo ""
        # Set CXX and CC to g++-11/gcc-11 so flashinfer uses c++/11 paths instead of c++/12
        # Also set CPLUS_INCLUDE_PATH to force c++/11 paths and prevent c++/12 from being found
        export CXX=g++-11
        export CC=gcc-11
        export CPLUS_INCLUDE_PATH="/usr/include:/usr/include/x86_64-linux-gnu:/usr/lib/gcc/x86_64-linux-gnu/11/include:/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11"
        export C_INCLUDE_PATH="/usr/include:/usr/include/x86_64-linux-gnu"
        llamafactory-cli chat "$INFERENCE_CONFIG"
        ;;
    webchat)
        echo "Starting web UI chat interface..."
        echo "The web interface will be available at http://localhost:7860"
        echo "Press Ctrl+C to stop the server."
        echo ""
        # Set CXX and CC to g++-11/gcc-11 so flashinfer uses c++/11 paths instead of c++/12
        # Also set CPLUS_INCLUDE_PATH to force c++/11 paths and prevent c++/12 from being found
        export CXX=g++-11
        export CC=gcc-11
        export CPLUS_INCLUDE_PATH="/usr/include:/usr/include/x86_64-linux-gnu:/usr/lib/gcc/x86_64-linux-gnu/11/include:/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11"
        export C_INCLUDE_PATH="/usr/include:/usr/include/x86_64-linux-gnu"
        llamafactory-cli webchat "$INFERENCE_CONFIG"
        ;;
    api)
        API_PORT="${API_PORT:-8000}"
        echo "Starting OpenAI-style API server..."
        echo "API will be available at http://localhost:$API_PORT"
        echo "Press Ctrl+C to stop the server."
        echo ""
        echo "Example API usage:"
        echo "  curl http://localhost:$API_PORT/v1/chat/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"model\": \"deepseek-v2-lite\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
        echo ""
        # Set CXX and CC to g++-11/gcc-11 so flashinfer uses c++/11 paths instead of c++/12
        # Also set CPLUS_INCLUDE_PATH to force c++/11 paths and prevent c++/12 from being found
        export CXX=g++-11
        export CC=gcc-11
        export CPLUS_INCLUDE_PATH="/usr/include:/usr/include/x86_64-linux-gnu:/usr/lib/gcc/x86_64-linux-gnu/11/include:/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11"
        export C_INCLUDE_PATH="/usr/include:/usr/include/x86_64-linux-gnu"
        API_PORT="$API_PORT" llamafactory-cli api "$INFERENCE_CONFIG"
        ;;
esac

# Clean up temporary config if created
if [ -n "$TEMP_CONFIG" ] && [ -f "$TEMP_CONFIG" ]; then
    rm -f "$TEMP_CONFIG"
fi

echo ""
echo "=========================================="
echo "Inference session ended"
echo "=========================================="

