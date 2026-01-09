#!/bin/bash
# Bash wrapper to generate identity training data
# Usage: ./scripts/sft_data/generate_identity.sh [options]

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
echo -e "${BLUE}Identity Data Generation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Activate Kllama conda environment
echo -e "${YELLOW}[1/3] Activating Kllama conda environment...${NC}"
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

# Step 2: Install requirements (if needed)
echo -e "${YELLOW}[2/3] Checking and installing requirements...${NC}"
REQUIREMENTS_FILE="$SFT_DATA_DIR/requirements.txt"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${RED}Error: requirements.txt not found at $REQUIREMENTS_FILE${NC}"
    exit 1
fi

# Quick check if pydantic is installed (most critical dependency)
if ! python -c "import pydantic" 2>/dev/null; then
    echo -e "${YELLOW}Installing requirements from $REQUIREMENTS_FILE...${NC}"
    pip install -q -r "$REQUIREMENTS_FILE"
    echo -e "${GREEN}✓ Requirements installed${NC}"
else
    echo -e "${GREEN}✓ All requirements already installed${NC}"
fi

echo ""

# Step 3: Load .env file if it exists
echo -e "${YELLOW}[3/3] Loading environment variables...${NC}"
ENV_FILE="$PROJECT_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
    echo -e "${GREEN}✓ Found .env file, loading variables...${NC}"
    set -a  # Automatically export all variables
    source "$ENV_FILE"
    set +a
    echo -e "${GREEN}✓ Environment variables loaded${NC}"
else
    echo -e "${YELLOW}⚠ No .env file found at $ENV_FILE${NC}"
    echo -e "${YELLOW}  API keys should be set via environment variables or .env file${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting generation...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Change to sft_data directory
cd "$SFT_DATA_DIR"

# Run the Python script with all arguments passed through
python generate_identity_data.py "$@"

