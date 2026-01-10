#!/bin/bash
# Bash wrapper to convert JSONL to LLaMA-Factory JSON format
# Usage: ./scripts/sft_data/convert_to_llamafactory.sh [options] <input.jsonl>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SFT_DATA_DIR="$PROJECT_ROOT/sft_data"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Convert to LLaMA-Factory Format${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Activate Kllama conda environment
echo -e "${YELLOW}[1/2] Activating Kllama conda environment...${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda command not found. Please install Anaconda/Miniconda first.${NC}"
    exit 1
fi

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Activate Kllama environment
if conda env list | grep -q "^Kllama "; then
    echo -e "${GREEN}✓ Found Kllama environment${NC}"
    conda activate Kllama
    echo -e "${GREEN}✓ Activated Kllama environment${NC}"
else
    echo -e "${YELLOW}⚠ Warning: Kllama environment not found. Creating it...${NC}"
    conda create -n Kllama python=3.10 -y
    conda activate Kllama
    echo -e "${GREEN}✓ Created and activated Kllama environment${NC}"
fi

echo ""

# Step 2: Run the conversion script
echo -e "${YELLOW}[2/2] Running conversion and registration...${NC}"
echo ""

# Convert relative paths to absolute paths (relative to project root)
# This handles paths like "sft_data/outputs/file.jsonl" correctly
ARGS=()
for arg in "$@"; do
    # Check if it's a file path (not an option)
    if [[ "$arg" != -* ]] && [[ -f "$PROJECT_ROOT/$arg" ]]; then
        # Convert to absolute path
        ARGS+=("$PROJECT_ROOT/$arg")
    elif [[ "$arg" == "-o" ]] || [[ "$arg" == "--output" ]]; then
        # Keep output option as-is, will handle next iteration
        ARGS+=("$arg")
    elif [[ "${ARGS[-1]}" == "-o" ]] || [[ "${ARGS[-1]}" == "--output" ]]; then
        # This is the output file path
        if [[ "$arg" != /* ]]; then
            # Relative path - convert to absolute
            ARGS+=("$PROJECT_ROOT/$arg")
        else
            ARGS+=("$arg")
        fi
    else
        # Keep other arguments as-is
        ARGS+=("$arg")
    fi
done

# Change to sft_data directory
cd "$SFT_DATA_DIR"

# Run the Python script with converted arguments
# Add --data-dir to point to LLaMA-Factory/data
LLAMA_DATA_DIR="$PROJECT_ROOT/LLaMA-Factory/data"
python scripts/convert_to_llamafactory_format.py --data-dir "$LLAMA_DATA_DIR" "${ARGS[@]}"

echo ""
echo -e "${GREEN}✅ Conversion and registration complete!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  • Dataset has been registered in dataset_info.json"
echo -e "  • You can now use it with --dataset flag in training"
echo ""

