#!/bin/bash
# Test actual training with ZeRO-3 after applying low_cpu_mem_usage patch
# This is a minimal test to see if training can start

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ZeRO-3 Training Test (with patch)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if patch is applied
if ! grep -q "EXPERIMENTAL: Allow low_cpu_mem_usage with ZeRO-3" "$PROJECT_ROOT/LLaMA-Factory/src/llamafactory/model/patcher.py"; then
    echo -e "${RED}⚠ Patch not applied!${NC}"
    echo -e "${YELLOW}Run: bash scripts/deepspeed/apply_z3_low_cpu_mem_patch.sh${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Patch verified${NC}"
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
    echo -e "${RED}⚠ Warning: Less than 6GB free GPU memory${NC}"
    echo -e "${YELLOW}This test may still fail even with the patch${NC}"
    echo ""
fi

# Change to LLaMA-Factory directory
cd "$PROJECT_ROOT/LLaMA-Factory"

CONFIG_FILE="examples/train_lora/deepseek2_lite_sft_hf_z3_test.yaml"

# Verify config exists and has DeepSpeed enabled
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

if ! grep -q "deepspeed:" "$CONFIG_FILE" || grep -q "^#.*deepspeed:" "$CONFIG_FILE"; then
    echo -e "${RED}Error: DeepSpeed not enabled in config${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Config file verified${NC}"
echo ""

# Set FORCE_TORCHRUN
export FORCE_TORCHRUN=1

echo -e "${YELLOW}Starting training test (will stop after a few steps)...${NC}"
echo -e "${BLUE}Config: $CONFIG_FILE${NC}"
echo -e "${BLUE}This will test if model initialization works with the patch${NC}"
echo ""

# Run training with timeout (to stop after initialization)
timeout 300 llamafactory-cli train "$CONFIG_FILE" 2>&1 | tee /tmp/z3_training_test.log || {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo -e "${YELLOW}Test timed out after 5 minutes (this is expected)${NC}"
        echo -e "${GREEN}If training started, the patch is working!${NC}"
    else
        echo -e "${RED}Training failed with exit code: $EXIT_CODE${NC}"
        echo ""
        echo -e "${YELLOW}Last 50 lines of output:${NC}"
        tail -50 /tmp/z3_training_test.log
        exit $EXIT_CODE
    fi
}

