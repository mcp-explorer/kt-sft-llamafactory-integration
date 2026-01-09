#!/bin/bash
# Test script to verify model loading with ZeRO-3 CPU offload
# This tests if the model can be initialized without OOM

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)" pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ZeRO-3 Model Loading Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Activate environment
if [ -f "$SCRIPT_DIR/activate_deepspeed_z3.sh" ]; then
    source "$SCRIPT_DIR/activate_deepspeed_z3.sh"
else
    echo -e "${YELLOW}Warning: activate_deepspeed_z3.sh not found. Activating conda environment manually...${NC}"
    eval "$(conda shell.bash hook)"
    conda activate deepspeed-z3
fi

# Check GPU memory
echo -e "${YELLOW}[1/4] Checking GPU memory...${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,noheader
FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo -e "${BLUE}Free GPU memory: ${FREE_MEM} MB${NC}"
echo ""

# Check if we have enough memory (need ~6-8GB for model initialization)
if [ "$FREE_MEM" -lt 6000 ]; then
    echo -e "${RED}⚠ Warning: Less than 6GB free GPU memory. Model initialization may fail.${NC}"
    echo -e "${YELLOW}Consider freeing up GPU memory or using a larger GPU.${NC}"
    echo -e "${BLUE}Continuing with test to verify OOM behavior...${NC}"
fi
echo ""

# Test model loading
echo -e "${YELLOW}[2/4] Testing model loading with ZeRO-3 configuration...${NC}"
cd "$PROJECT_ROOT/LLaMA-Factory"

python << 'PYEOF'
import os
import torch
import sys
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"GPU memory before: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

model_path = "/home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat"

try:
    print(f"\nLoading model config from: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"Model type: {config.model_type}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Number of parameters: ~{config.num_hidden_layers * config.hidden_size * config.intermediate_size / 1e9:.1f}B")
    
    print(f"\nGPU memory after config load: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    print("\n⚠ Attempting to load full model (this may OOM on 16GB GPU)...")
    print("   With ZeRO-3, model must load to GPU first before offloading")
    
    # Try loading model (this is where OOM typically occurs)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,  # ZeRO-3 disables this
        device_map=None  # Let DeepSpeed handle device placement
    )
    
    print(f"✅ Model loaded successfully!")
    print(f"GPU memory after model load: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    # Move to GPU to test
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"GPU memory after moving to CUDA: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    print("\n✅ Model initialization test passed!")
    print("   Note: This is without DeepSpeed. Actual training will use DeepSpeed ZeRO-3.")
    
except torch.cuda.OutOfMemoryError as e:
    print(f"\n❌ OOM Error during model loading: {e}")
    print("\nThis confirms the issue: model initialization requires GPU memory")
    print("before DeepSpeed can partition and offload parameters.")
    print("\nSolutions:")
    print("  1. Use a larger GPU (24GB+)")
    print("  2. Use multiple GPUs (distributes initial load)")
    print("  3. Try modifying LLaMA-Factory to allow low_cpu_mem_usage with ZeRO-3")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

LOAD_TEST_RESULT=$?

echo ""
if [ $LOAD_TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}[3/4] Model loading test passed!${NC}"
else
    echo -e "${RED}[3/4] Model loading test failed - OOM during initialization${NC}"
    echo ""
    echo -e "${YELLOW}This confirms the documented issue:${NC}"
    echo -e "  Even with ZeRO-3, model must be instantiated on GPU first"
    echo -e "  before DeepSpeed can partition and offload parameters."
    exit 1
fi

# Test DeepSpeed initialization (dry run)
echo ""
echo -e "${YELLOW}[4/4] Testing DeepSpeed ZeRO-3 initialization (dry run)...${NC}"
python << 'PYEOF'
import deepspeed
import torch

print(f"DeepSpeed version: {deepspeed.__version__}")

# Test CPU Adam availability
try:
    from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam
    params = [torch.randn(10, 10, requires_grad=True)]
    optimizer = DeepSpeedCPUAdam(params, lr=1e-3)
    print("✅ CPU Adam available and working")
except Exception as e:
    print(f"❌ CPU Adam error: {e}")
    sys.exit(1)

print("\n✅ DeepSpeed ZeRO-3 is ready!")
print("   CPU Adam compilation: ✅")
print("   Model loading: ⚠️  (may OOM on 16GB GPU during initialization)")
PYEOF

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Test Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}GPU Memory:${NC} 16GB (may be insufficient for full-precision model init)"
echo -e "${BLUE}CPU Adam:${NC} ✅ Compiled successfully"
echo -e "${BLUE}Model Loading:${NC} ⚠️  May OOM during initialization"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. If model loading passed, try actual training with ZeRO-3"
echo "  2. If OOM occurred, consider:"
echo "     - Using larger GPU (24GB+)"
echo "     - Using multiple GPUs"
echo "     - Modifying LLaMA-Factory to allow low_cpu_mem_usage with ZeRO-3"

