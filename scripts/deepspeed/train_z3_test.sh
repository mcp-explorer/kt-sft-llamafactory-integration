#!/bin/bash
# Training script for ZeRO-3 CPU offload test
# This uses the deepspeed-z3 environment and test config

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)" pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ZeRO-3 CPU Offload Training Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Activate environment
if [ -f "$SCRIPT_DIR/activate_deepspeed_z3.sh" ]; then
    source "$SCRIPT_DIR/activate_deepspeed_z3.sh"
else
    eval "$(conda shell.bash hook)"
    conda activate deepspeed-z3
fi

# Check GPU memory
echo -e "${YELLOW}Checking GPU memory...${NC}"
FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo -e "${BLUE}Free GPU memory: ${FREE_MEM} MB${NC}"

if [ "$FREE_MEM" -lt 6000 ]; then
    echo -e "${RED}⚠ Warning: Less than 6GB free GPU memory.${NC}"
    echo -e "${RED}Model initialization will likely fail with OOM.${NC}"
    echo ""
    echo -e "${YELLOW}This is expected on a 16GB GPU with other processes running.${NC}"
    echo -e "${YELLOW}ZeRO-3 requires the model to load to GPU before offloading.${NC}"
    echo ""
    read -p "Continue anyway to test? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Free GPU memory or use a larger GPU (24GB+) to proceed."
        exit 1
    fi
fi

# Change to LLaMA-Factory directory
cd "$PROJECT_ROOT/LLaMA-Factory"

# Config file
CONFIG_FILE="examples/train_lora/deepseek2_lite_sft_hf_z3_test.yaml"

# Check if DeepSpeed is enabled (it should be)
if grep -q "deepspeed:" "$CONFIG_FILE" && ! grep -q "^#.*deepspeed:" "$CONFIG_FILE"; then
    echo -e "${GREEN}✓ DeepSpeed detected in config${NC}"
    echo -e "${BLUE}Setting FORCE_TORCHRUN=1${NC}"
    export FORCE_TORCHRUN=1
else
    echo -e "${RED}✗ DeepSpeed not found in config!${NC}"
    exit 1
fi

# Run training
echo ""
echo -e "${YELLOW}Starting training with ZeRO-3 CPU offload...${NC}"
echo -e "${BLUE}Config: $CONFIG_FILE${NC}"
echo ""

llamafactory-cli train "$CONFIG_FILE"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Training completed!${NC}"
echo -e "${GREEN}========================================${NC}"

