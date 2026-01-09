#!/bin/bash
# Script to activate Kllama conda environment and generate SFT data
# Usage:
#   ./generate_with_kllama.sh --goal "identity training" --num_records 1000 --method hybrid --provider gemini
#   ./generate_with_kllama.sh --goal "identity training" --num_records 1000 --constraint identity=sean --constraint date=2026-01-01

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SFT Data Generation Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Activate Kllama conda environment
echo -e "${YELLOW}[1/3] Activating Kllama conda environment...${NC}"
if ! command -v conda &> /dev/null; then
    echo "Error: conda command not found. Please install Anaconda/Miniconda first."
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
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: requirements.txt not found at $REQUIREMENTS_FILE"
    exit 1
fi

# Check if packages are already installed
NEEDS_INSTALL=false
while IFS= read -r package; do
    # Skip empty lines and comments
    [[ -z "$package" || "$package" =~ ^# ]] && continue
    
    # Extract package name (remove version specifiers)
    pkg_name=$(echo "$package" | sed -E 's/[<>=!].*//' | tr '[:upper:]' '[:lower:]')
    
    # Check if package is installed
    if ! python -c "import ${pkg_name//-/_}" 2>/dev/null; then
        NEEDS_INSTALL=true
        break
    fi
done < "$REQUIREMENTS_FILE"

if [ "$NEEDS_INSTALL" = true ] || [ "$1" = "--install" ]; then
    echo -e "${YELLOW}Installing requirements from $REQUIREMENTS_FILE...${NC}"
    pip install -r "$REQUIREMENTS_FILE"
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
echo -e "${BLUE}Ready to generate data!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 4: Run the generation script if arguments are provided
if [ $# -gt 0 ] && [ "$1" != "--install" ]; then
    echo -e "${YELLOW}Running generation script...${NC}"
    echo ""
    
    # Change to script directory to run the Python script
    cd "$SCRIPT_DIR"
    
    # Pass all arguments to the Python script
    python generate_sft_data.py "$@"
    
    echo ""
    echo -e "${GREEN}✓ Generation complete!${NC}"
else
    echo -e "${YELLOW}No arguments provided. To generate data, run:${NC}"
    echo ""
    echo "  $0 --goal \"identity training\" --num_records 1000 --method hybrid --provider gemini"
    echo ""
    echo "Or with constraints:"
    echo ""
    echo "  $0 --goal \"identity training\" --num_records 1000 \\"
    echo "      --constraint identity=sean --constraint date=2026-01-01 \\"
    echo "      --method hybrid --provider gemini"
    echo ""
    echo "Options:"
    echo "  --goal TEXT              Goal description (required)"
    echo "  --num_records INT        Number of records to generate (required)"
    echo "  --method [schema|judge|hybrid]  Generation method (default: hybrid)"
    echo "  --provider [openai|anthropic|gemini]  LLM provider (default: openai)"
    echo "  --constraint KEY=VALUE   Add a constraint (can be used multiple times)"
    echo "  --constraints JSON       JSON string of constraints (alternative to --constraint)"
    echo "  --model TEXT             Model name (optional, uses provider default)"
    echo "  --output PATH            Output file path (optional)"
    echo "  --score_threshold FLOAT  Minimum quality score (default: 0.7)"
    echo ""
fi

