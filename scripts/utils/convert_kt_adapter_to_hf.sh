#!/bin/bash
# Convert ktransformers adapter to HuggingFace-compatible format
# This script converts adapters trained with ktransformers backend to standard
# PEFT format that can be used with HuggingFace backend.

set -e  # Exit on error

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"  # Go up two levels: utils -> scripts -> project root

# Default values
INPUT_ADAPTER=""
OUTPUT_ADAPTER=""
CONFIG_PATH=""
NO_VERIFY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input|-i)
            INPUT_ADAPTER="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_ADAPTER="$2"
            shift 2
            ;;
        --config|-c)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --no-verify)
            NO_VERIFY=true
            shift
            ;;
        --help|-h)
            cat << EOF
Usage: $0 [OPTIONS]

Convert ktransformers adapter to HuggingFace-compatible format.

Options:
  -i, --input PATH       Path to ktransformers adapter directory (required)
  -o, --output PATH      Path to save HuggingFace adapter (required)
  -c, --config PATH      Optional path to adapter_config.json
  --no-verify            Skip verification of converted adapter
  -h, --help             Show this help message

Examples:
  # Convert adapter
  $0 --input saves/Kllama_deepseekV2Lite --output saves/Kllama_deepseekV2Lite_hf
  
  # Convert with custom config
  $0 -i saves/Kllama_deepseekV2Lite -o saves/Kllama_deepseekV2Lite_hf -c custom_config.json
  
  # Convert without verification
  $0 -i saves/Kllama_deepseekV2Lite -o saves/Kllama_deepseekV2Lite_hf --no-verify

Description:
  This script converts adapters trained with ktransformers backend to standard
  PEFT format compatible with HuggingFace backend. The conversion:
  
  • Removes extra 'model.' prefix: base_model.model.model.layers.X... → base_model.model.layers.X...
  • Removes '.default.' from LoRA weights: .lora_A.default.weight → .lora_A.weight
  • Saves in standard PEFT format that HuggingFace can load
  
  The converted adapter can then be used with:
    adapter_name_or_path: <output_path>
    infer_backend: huggingface
EOF
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_ADAPTER" ]; then
    echo "Error: --input is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$OUTPUT_ADAPTER" ]; then
    echo "Error: --output is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check if input adapter exists
if [ ! -d "$INPUT_ADAPTER" ]; then
    echo "Error: Input adapter directory not found: $INPUT_ADAPTER"
    exit 1
fi

# Check if adapter_model file exists
if [ ! -f "$INPUT_ADAPTER/adapter_model.safetensors" ] && \
   [ ! -f "$INPUT_ADAPTER/adapter_model.bin" ] && \
   [ ! -f "$INPUT_ADAPTER/adapter_model.pt" ]; then
    echo "Error: Adapter weights file not found in $INPUT_ADAPTER"
    echo "Expected: adapter_model.safetensors, adapter_model.bin, or adapter_model.pt"
    exit 1
fi

# Check if adapter_config.json exists
if [ ! -f "$INPUT_ADAPTER/adapter_config.json" ] && [ -z "$CONFIG_PATH" ]; then
    echo "Error: adapter_config.json not found in $INPUT_ADAPTER"
    echo "Use --config to specify a custom config file"
    exit 1
fi

echo "=========================================="
echo "KTransformers → HuggingFace Adapter Converter"
echo "=========================================="
echo ""
echo "Input adapter:  $INPUT_ADAPTER"
echo "Output adapter: $OUTPUT_ADAPTER"
if [ -n "$CONFIG_PATH" ]; then
    echo "Config file:    $CONFIG_PATH"
fi
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

# Build Python command
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/utils/convert_kt_adapter_to_hf.py"
PYTHON_CMD="python3 $PYTHON_SCRIPT --input \"$INPUT_ADAPTER\" --output \"$OUTPUT_ADAPTER\""

if [ -n "$CONFIG_PATH" ]; then
    PYTHON_CMD="$PYTHON_CMD --config \"$CONFIG_PATH\""
fi

if [ "$NO_VERIFY" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --no-verify"
fi

# Run conversion
echo "=========================================="
echo "Running conversion..."
echo "=========================================="
echo ""

eval $PYTHON_CMD

CONVERSION_EXIT_CODE=$?

if [ $CONVERSION_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Conversion completed successfully!"
    echo "=========================================="
    echo ""
    echo "Converted adapter saved to: $OUTPUT_ADAPTER"
    echo ""
    echo "You can now use this adapter with HuggingFace backend:"
    echo "  adapter_name_or_path: $OUTPUT_ADAPTER"
    echo "  infer_backend: huggingface"
    echo ""
    echo "To test the converted adapter:"
    echo "  ./scripts/inference/infer_ds2_chat_lite_hf.sh chat"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ Conversion failed!"
    echo "=========================================="
    echo ""
    echo "Check the error messages above for details."
    exit $CONVERSION_EXIT_CODE
fi

