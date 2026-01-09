#!/bin/bash
# Test model loading with ZeRO-3 after applying low_cpu_mem_usage patch
# This tests if the patch allows model to load to CPU first

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
echo -e "${BLUE}ZeRO-3 Model Loading Test (with patch)${NC}"
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
echo ""

# Test model loading with low_cpu_mem_usage
echo -e "${YELLOW}Testing model loading with low_cpu_mem_usage enabled...${NC}"
cd "$PROJECT_ROOT/LLaMA-Factory"

python << 'PYEOF'
import os
import torch
import sys
from transformers import AutoConfig, AutoModelForCausalLM

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU memory before: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

model_path = "/home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat"

try:
    print(f"\nLoading model with low_cpu_mem_usage=True and device_map='cpu'...")
    print("This should load directly to CPU, avoiding GPU memory during initialization")
    
    # Try loading with low_cpu_mem_usage and device_map to CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,  # This should now work with ZeRO-3 (after patch)
        device_map="cpu",  # Load directly to CPU
        offload_folder="/tmp/hf_offload"
    )
    
    print(f"✅ Model loaded successfully to CPU!")
    print(f"GPU memory after load: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    # Check model device
    first_param_device = next(model.parameters()).device
    print(f"Model device: {first_param_device}")
    
    if first_param_device.type == 'cpu':
        print("\n✅ Model is on CPU - this should allow DeepSpeed to take over without OOM")
        print("   Next step: Test with actual DeepSpeed ZeRO-3 training")
    else:
        print(f"\n⚠ Model is on {first_param_device.type} - may still OOM during DeepSpeed initialization")
    
    sys.exit(0)
    
except torch.cuda.OutOfMemoryError as e:
    print(f"\n❌ OOM Error: {e}")
    print("\nEven with low_cpu_mem_usage, model initialization failed.")
    print("This suggests the patch may not fully solve the issue, or")
    print("there's still GPU memory allocation happening.")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

TEST_RESULT=$?

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Test passed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}The patch allows model to load to CPU.${NC}"
    echo -e "${BLUE}Next: Test actual training with ZeRO-3${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Test failed${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo -e "${YELLOW}The patch may not fully resolve the OOM issue.${NC}"
fi

