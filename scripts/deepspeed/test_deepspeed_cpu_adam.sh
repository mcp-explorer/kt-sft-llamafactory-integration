#!/bin/bash
# Quick test script for DeepSpeed CPU Adam compilation
# This can be run in any conda environment to test if CPU offload will work

set -e

# Set library paths for CUDA compilation
# PyTorch has bundled CUDA libraries - we need to set paths correctly
PYTORCH_NV_LIB="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia"

# Add PyTorch's CUDA library paths first (for runtime)
if [[ -d "$PYTORCH_NV_LIB/cuda_runtime/lib" ]]; then
    export LD_LIBRARY_PATH=$PYTORCH_NV_LIB/cuda_runtime/lib:$LD_LIBRARY_PATH
fi
if [[ -d "$PYTORCH_NV_LIB/curand/lib" ]]; then
    export LD_LIBRARY_PATH=$PYTORCH_NV_LIB/curand/lib:$LD_LIBRARY_PATH
fi
if [[ -d "$PYTORCH_NV_LIB/nvjitlink/lib" ]]; then
    export LD_LIBRARY_PATH=$PYTORCH_NV_LIB/nvjitlink/lib:$LD_LIBRARY_PATH
fi

# Set LIBRARY_PATH for linking (different from LD_LIBRARY_PATH)
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
if [[ -d "$PYTORCH_NV_LIB/cuda_runtime/lib" ]]; then
    export LIBRARY_PATH=$PYTORCH_NV_LIB/cuda_runtime/lib:$LIBRARY_PATH
fi
if [[ -d "$PYTORCH_NV_LIB/curand/lib" ]]; then
    export LIBRARY_PATH=$PYTORCH_NV_LIB/curand/lib:$LIBRARY_PATH
fi

export CUDA_HOME=/usr
export CUDA_ROOT=/usr

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}Testing DeepSpeed CPU Adam Compilation...${NC}"
echo ""

python << 'EOF'
import deepspeed
import torch
import sys

print(f"DeepSpeed version: {deepspeed.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Check version compatibility
ds_version = deepspeed.__version__
major, minor, patch = map(int, ds_version.split('.')[:3])
if not (0.10 <= float(f"{major}.{minor}") <= 0.16):
    print(f"\n⚠ Warning: DeepSpeed version {ds_version} may not be compatible")
    print("LLaMA-Factory requires: deepspeed>=0.10.0,<=0.16.9")
else:
    print(f"✓ DeepSpeed version {ds_version} is compatible")

# Try to initialize CPU Adam optimizer (this triggers compilation)
try:
    from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam
    
    # Create dummy parameters
    params = [torch.randn(10, 10, requires_grad=True)]
    
    # Try CPU Adam (this will trigger JIT compilation)
    print("\nAttempting to initialize CPU Adam...")
    optimizer = DeepSpeedCPUAdam(params, lr=1e-3)
    print("✅ CPU Adam compilation successful!")
    print("✅ DeepSpeed CPU offload is ready to use!")
    sys.exit(0)
    
except RuntimeError as e:
    error_msg = str(e).lower()
    if "cpu_adam" in error_msg or "compilation" in error_msg or "stdlib.h" in error_msg:
        print(f"\n❌ CPU Adam compilation failed!")
        print(f"Error: {e}")
        print("\nThis indicates missing system headers.")
        print("Try installing:")
        print("  sudo apt-get install libc6-dev build-essential")
        print("\nOr if using conda:")
        print("  conda install -c conda-forge gxx_linux-64")
        sys.exit(1)
    else:
        raise
except ImportError as e:
    print(f"\n❌ Failed to import DeepSpeedCPUAdam: {e}")
    print("DeepSpeed may not be properly installed or CPU Adam is not available.")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

TEST_RESULT=$?

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Test passed! CPU offload should work.${NC}"
    exit 0
else
    echo -e "${RED}✗ Test failed! CPU offload will not work until compilation is fixed.${NC}"
    exit 1
fi

