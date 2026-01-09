#!/bin/bash
# Setup script for DeepSpeed ZeRO-3 CPU Offload conda environment
# This creates a dedicated environment with all dependencies needed for ZeRO-3 CPU offload

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
ENV_NAME="deepspeed-z3"
PYTHON_VERSION="3.11"
DEEPSPEED_VERSION="0.16.9"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DeepSpeed ZeRO-3 CPU Offload Environment Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda command not found. Please install Anaconda/Miniconda first.${NC}"
    exit 1
fi

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Step 1: Check if environment already exists
echo -e "${YELLOW}[1/7] Checking for existing environment...${NC}"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}⚠ Environment '${ENV_NAME}' already exists.${NC}"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing environment...${NC}"
        conda env remove -n "${ENV_NAME}" -y
    else
        echo -e "${YELLOW}Using existing environment. Activating...${NC}"
        conda activate "${ENV_NAME}"
        echo -e "${GREEN}✓ Environment activated${NC}"
        echo ""
        echo -e "${BLUE}To continue setup, run:${NC}"
        echo -e "  conda activate ${ENV_NAME}"
        echo -e "  bash ${SCRIPT_DIR}/setup_deepspeed_z3_env.sh --skip-env"
        exit 0
    fi
fi

# Skip environment creation if --skip-env flag is set
if [[ "$1" != "--skip-env" ]]; then
    # Step 2: Create conda environment
    echo -e "${YELLOW}[2/7] Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}...${NC}"
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    conda activate "${ENV_NAME}"
    echo -e "${GREEN}✓ Environment created and activated${NC}"
    echo ""
fi

# Activate environment
conda activate "${ENV_NAME}"

# Step 3: Install system build dependencies via conda
echo -e "${YELLOW}[3/7] Installing build dependencies...${NC}"
conda install -y -c conda-forge \
    libstdcxx-ng \
    gcc_impl_linux-64 \
    gxx_linux-64 \
    make \
    cmake \
    ninja \
    pkg-config

echo -e "${GREEN}✓ Build dependencies installed${NC}"
echo ""

# Step 4: Verify system headers (libc6-dev equivalent)
echo -e "${YELLOW}[4/7] Verifying system headers availability...${NC}"
# Check if we can find stdlib.h in common locations
if [ -f "/usr/include/stdlib.h" ] || [ -f "/usr/include/x86_64-linux-gnu/stdlib.h" ]; then
    echo -e "${GREEN}✓ System headers found${NC}"
else
    echo -e "${RED}⚠ Warning: stdlib.h not found in standard locations${NC}"
    echo -e "${YELLOW}You may need to install libc6-dev:${NC}"
    echo -e "  sudo apt-get install libc6-dev build-essential"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# Step 5: Install PyTorch with CUDA support
echo -e "${YELLOW}[5/7] Installing PyTorch with CUDA support...${NC}"
# Detect CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo -e "${BLUE}Detected CUDA version: ${CUDA_VERSION}${NC}"
else
    echo -e "${YELLOW}CUDA not detected via nvcc. Using CUDA 11.8 as default.${NC}"
    CUDA_VERSION="11.8"
fi

# Install PyTorch based on CUDA version
if [[ $(echo "$CUDA_VERSION >= 12.0" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
    echo -e "${BLUE}Installing PyTorch for CUDA 12.x...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo -e "${BLUE}Installing PyTorch for CUDA 11.8...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

echo -e "${GREEN}✓ PyTorch installed${NC}"
echo ""

# Step 6: Install DeepSpeed (compatible version)
echo -e "${YELLOW}[6/7] Installing DeepSpeed ${DEEPSPEED_VERSION}...${NC}"
pip install "deepspeed>=0.10.0,<=${DEEPSPEED_VERSION}"

# Verify DeepSpeed installation
echo -e "${BLUE}Verifying DeepSpeed installation...${NC}"
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')" || {
    echo -e "${RED}Error: DeepSpeed installation failed${NC}"
    exit 1
}

echo -e "${GREEN}✓ DeepSpeed ${DEEPSPEED_VERSION} installed${NC}"
echo ""

# Step 7: Install LLaMA-Factory and dependencies
echo -e "${YELLOW}[7/7] Installing LLaMA-Factory and dependencies...${NC}"

# Check if LLaMA-Factory directory exists
if [ -d "${PROJECT_ROOT}/LLaMA-Factory" ]; then
    echo -e "${BLUE}LLaMA-Factory directory found. Installing from local source...${NC}"
    cd "${PROJECT_ROOT}/LLaMA-Factory"
    pip install -e ".[torch,metrics]" --no-build-isolation
    cd "${PROJECT_ROOT}"
else
    echo -e "${BLUE}LLaMA-Factory not found locally. Installing from requirements...${NC}"
    # Install core dependencies from LLaMA-Factory requirements
    pip install \
        "transformers>=4.49.0,<=4.57.1,!=4.52.0,!=4.57.0" \
        "datasets>=2.16.0,<=4.0.0" \
        "accelerate>=1.3.0,<=1.11.0" \
        "peft>=0.14.0,<=0.17.1" \
        "trl>=0.8.6,<=0.9.6" \
        "gradio>=4.38.0,<=5.45.0" \
        matplotlib \
        "tyro<0.9.0" \
        einops \
        "numpy<2.0.0" \
        "pandas>=2.0.0" \
        scipy \
        sentencepiece \
        tiktoken \
        "modelscope>=1.14.0" \
        hf-transfer \
        "safetensors<=0.5.3" \
        fire \
        omegaconf \
        packaging \
        protobuf \
        pyyaml \
        "pydantic<=2.10.6" \
        uvicorn \
        fastapi \
        sse-starlette \
        av \
        librosa \
        "propcache!=0.4.0"
fi

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 8: Test CPU Adam compilation
echo -e "${YELLOW}[8/8] Testing DeepSpeed CPU Adam compilation...${NC}"
# Set library paths for CUDA compilation
# PyTorch has bundled CUDA libraries - we need to configure paths correctly
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

python << 'EOF'
import deepspeed
import torch
import sys

print(f"DeepSpeed version: {deepspeed.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

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
    if "cpu_adam" in str(e).lower() or "compilation" in str(e).lower() or "stdlib.h" in str(e).lower():
        print(f"\n❌ CPU Adam compilation failed: {e}")
        print("\nThis indicates missing system headers.")
        print("Try installing: sudo apt-get install libc6-dev build-essential")
        sys.exit(1)
    else:
        raise
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

COMPILATION_TEST_RESULT=$?

echo ""
if [ $COMPILATION_TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Environment setup complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}Environment: ${ENV_NAME}${NC}"
    echo -e "${BLUE}Python: ${PYTHON_VERSION}${NC}"
    echo -e "${BLUE}DeepSpeed: ${DEEPSPEED_VERSION}${NC}"
    echo ""
    echo -e "${BLUE}To activate this environment:${NC}"
    echo -e "  conda activate ${ENV_NAME}"
    echo ""
    echo -e "${BLUE}To test ZeRO-3 CPU offload:${NC}"
    echo -e "  conda activate ${ENV_NAME}"
    echo -e "  cd ${PROJECT_ROOT}/LLaMA-Factory"
    echo -e "  llamafactory-cli train examples/train_lora/deepseek2_lite_sft_hf_v3.yaml"
    echo ""
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}⚠ Environment setup completed with warnings${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo -e "${YELLOW}CPU Adam compilation failed. This needs to be fixed before using CPU offload.${NC}"
    echo -e "${YELLOW}Try installing system headers:${NC}"
    echo -e "  sudo apt-get install libc6-dev build-essential"
    echo ""
    echo -e "${BLUE}You can still use the environment, but CPU offload won't work until compilation is fixed.${NC}"
    echo ""
fi

