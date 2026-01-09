#!/bin/bash
# Run inference for DeepSeek-V2-Lite-Chat RAW model using HuggingFace backend
# This script supports chat, webchat, and API modes

set -e  # Exit on error

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"  # Go up two levels: inference -> scripts -> project root

# Default mode: chat (interactive CLI)
MODE="${1:-chat}"

# Parse mode argument
case "$MODE" in
    chat|webchat|api)
        ;;
    *)
        echo "Usage: $0 [chat|webchat|api]"
        echo ""
        echo "Modes:"
        echo "  chat    - Interactive CLI chat (default)"
        echo "  webchat - Web UI chat interface"
        echo "  api     - OpenAI-style API server"
        echo ""
        echo "Note: This script serves the RAW model using HuggingFace backend (no adapter/fine-tuning)."
        exit 1
        ;;
esac

echo "=========================================="
echo "DeepSeek-V2-Lite-Chat RAW Model Inference"
echo "Backend: HuggingFace"
echo "=========================================="
echo "Mode: $MODE"
echo "Model: Raw (no adapter/fine-tuning)"
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

# Check if raw model inference config file exists
INFERENCE_CONFIG="$PROJECT_ROOT/LLaMA-Factory/examples/inference/deepseek2_lite_inference_raw_hf.yaml"
if [ ! -f "$INFERENCE_CONFIG" ]; then
    echo "⚠ Warning: HuggingFace raw model inference config file not found at $INFERENCE_CONFIG"
    echo "  Creating it from template..."
    
    # Create the config file for raw model (no adapter) with HuggingFace backend
    mkdir -p "$(dirname "$INFERENCE_CONFIG")"
    cat > "$INFERENCE_CONFIG" << EOF
model_name_or_path: /home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat
# No adapter - using raw model
template: deepseek
infer_backend: huggingface  # Standard HuggingFace transformers backend
trust_remote_code: true

# HuggingFace backend doesn't need ktransformers-specific config
# Generation parameters (optional, using defaults if not specified)
# do_sample: true
# temperature: 0.95
# top_p: 0.7
# max_new_tokens: 1024
EOF
    echo "✓ Created config file at: $INFERENCE_CONFIG"
fi

echo ""
echo "=========================================="
echo "Starting inference ($MODE mode)..."
echo "=========================================="
echo "Config: $INFERENCE_CONFIG"
echo "Model: Raw (no adapter)"
echo "Backend: HuggingFace"
echo ""

# Change to LLaMA-Factory directory
cd "$PROJECT_ROOT/LLaMA-Factory"

# Run inference based on mode
case "$MODE" in
    chat)
        echo "Starting interactive CLI chat..."
        echo "Type your messages and press Enter. Type 'exit' or 'quit' to end."
        echo ""
        llamafactory-cli chat "$INFERENCE_CONFIG"
        ;;
    webchat)
        echo "Starting web UI chat interface..."
        echo "The web interface will be available at http://localhost:7860"
        echo "Press Ctrl+C to stop the server."
        echo ""
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
        API_PORT="$API_PORT" llamafactory-cli api "$INFERENCE_CONFIG"
        ;;
esac

echo ""
echo "=========================================="
echo "Inference session ended"
echo "=========================================="

