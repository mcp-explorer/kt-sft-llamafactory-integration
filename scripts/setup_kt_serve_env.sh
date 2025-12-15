#!/bin/bash
# Setup script to create conda environment for KTransformers model serving
# Environment: kt-serve (separate from Kllama)
# 
# System Requirements:
#   - Python: 3.12 (recommended: 3.10, 3.11, 3.12, or 3.13)
#   - CUDA: 12.0+ (system nvcc: 12.0.140)
#   - NVIDIA Driver: 570.195.03+
#   - nvcc: 12.0+ (for compilation if needed)
#
# Model: DeepSeek-V2-Lite (serving with checkpoints from saves/Kllama_deepseekV2Lite)

set -e

ENV_NAME="kt-serve"
PROJECT_DIR="/home/sean/Documents/ktransformers"
KT_SFT_DIR="${PROJECT_DIR}/kt-sft"
LLAMA_FACTORY_DIR="${PROJECT_DIR}/LLaMA-Factory"

echo "=========================================="
echo "Creating Conda Environment for KTransformers Serving"
echo "=========================================="
echo ""
echo "System Information:"
echo "  Python: 3.12.3"
echo "  CUDA (nvcc): 12.0.140"
echo "  NVIDIA Driver: 570.195.03"
echo "  Environment name: $ENV_NAME"
echo ""
echo "Requirements:"
echo "  - Python: 3.12 (supports 3.10, 3.11, 3.12, 3.13)"
echo "  - PyTorch: 2.7.0 with CUDA 12.8 support"
echo "  - KTransformers: v0.4.1 wheel (cu128torch27fancy)"
echo "  - Flash-Attention: 2.8.3"
echo "  - LLaMA-Factory: for API serving"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not available. Please install conda first."
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "Using existing environment. Activate it with: conda activate $ENV_NAME"
        exit 0
    fi
fi

# Create new conda environment with Python 3.12
echo "Creating conda environment with Python 3.12..."
conda create -n "$ENV_NAME" python=3.12 -y

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install essential conda packages
echo ""
echo "Installing essential conda packages..."
conda install -y -c conda-forge libstdcxx-ng gcc_impl_linux-64

# Install CUDA runtime (required even if system CUDA differs)
# This provides libcudart.so.11.0 which is needed by ktransformers
echo ""
echo "Installing CUDA runtime (required for libcudart.so.11.0)..."
conda install -y -c nvidia/label/cuda-11.8.0 cuda-runtime

# Verify GLIBCXX version
echo ""
echo "Verifying GLIBCXX version (should include 3.4.32)..."
CONDA_ENV_PATH=$(conda info --base)/envs/${ENV_NAME}
if [ -f "${CONDA_ENV_PATH}/lib/libstdc++.so.6" ]; then
    strings "${CONDA_ENV_PATH}/lib/libstdc++.so.6" | grep GLIBCXX | tail -1
else
    echo "⚠️  Warning: Could not verify GLIBCXX version"
fi

# Install system dependencies (if not already installed)
echo ""
echo "Checking system dependencies..."
if ! dpkg -l | grep -q "nvidia-cuda-toolkit"; then
    echo "⚠️  nvidia-cuda-toolkit not found. You may need to install it:"
    echo "    sudo apt-get update && sudo apt-get install nvidia-cuda-toolkit"
fi

# Install PyTorch with CUDA 12.8 support (matches ktransformers wheel)
echo ""
echo "Installing PyTorch 2.7.0 with CUDA 12.8 support..."
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# Verify PyTorch CUDA availability
echo ""
echo "Verifying PyTorch CUDA installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

# Check ABI compatibility for flash-attention
echo ""
echo "Checking CXX11 ABI compatibility..."
ABI_CHECK=$(python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)")
echo "CXX11 ABI: $ABI_CHECK"

# Install ktransformers requirements
if [ -f "${KT_SFT_DIR}/requirements-sft.txt" ]; then
    echo ""
    echo "Installing ktransformers requirements..."
    pip install -r "${KT_SFT_DIR}/requirements-sft.txt"
else
    echo "⚠️  Warning: requirements-sft.txt not found at ${KT_SFT_DIR}/requirements-sft.txt"
fi

# Install LLaMA-Factory (for API serving)
if [ -d "$LLAMA_FACTORY_DIR" ]; then
    echo ""
    echo "Installing LLaMA-Factory..."
    cd "$LLAMA_FACTORY_DIR"
    pip install -e ".[torch,metrics]" --no-build-isolation
    cd "$PROJECT_DIR"
else
    echo "⚠️  Warning: LLaMA-Factory directory not found at $LLAMA_FACTORY_DIR"
    echo "    You may need to clone it: git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git"
fi

# Install KTransformers wheel
echo ""
echo "Installing KTransformers wheel..."
echo "Downloading from: https://github.com/kvcache-ai/ktransformers/releases/tag/v0.4.1"
echo "Wheel: ktransformers-0.4.1+cu128torch27fancy-cp312-cp312-linux_x86_64.whl"
pip install https://github.com/kvcache-ai/ktransformers/releases/download/v0.4.1/ktransformers-0.4.1+cu128torch27fancy-cp312-cp312-linux_x86_64.whl

# Install flash-attention
echo ""
echo "Installing flash-attention 2.8.3..."
if [ "$ABI_CHECK" == "True" ]; then
    FLASH_ATTN_WHEEL="flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
else
    FLASH_ATTN_WHEEL="flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
fi
echo "Using wheel: $FLASH_ATTN_WHEEL"
pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com/cxx11 \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${FLASH_ATTN_WHEEL}

# Optional: Install custom_flashinfer
echo ""
read -p "Do you want to install custom_flashinfer? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d "${PROJECT_DIR}/custom_flashinfer" ]; then
        echo "Installing custom_flashinfer..."
        pip install "${PROJECT_DIR}/custom_flashinfer/"
    else
        echo "⚠️  custom_flashinfer directory not found. Cloning..."
        cd "$PROJECT_DIR"
        git clone https://github.com/kvcache-ai/custom_flashinfer.git
        pip install custom_flashinfer/
    fi
fi

# Final verification
echo ""
echo "=========================================="
echo "Running final verification..."
echo "=========================================="
python -c "
import torch
import sys

print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

# Try importing ktransformers
try:
    import ktransformers
    print(f'KTransformers: {ktransformers.__version__ if hasattr(ktransformers, \"__version__\") else \"installed\"}')
except ImportError as e:
    print(f'❌ KTransformers import failed: {e}')

# Try importing flash_attn
try:
    import flash_attn
    print(f'Flash-Attention: installed')
except ImportError as e:
    print(f'⚠️  Flash-Attention not available: {e}')
"

echo ""
echo "=========================================="
echo "✅ Environment setup complete!"
echo "=========================================="
echo ""
echo "Environment name: $ENV_NAME"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To serve your model, you can use:"
echo ""
echo "1. KTransformers Server (direct serving):"
echo "   python ktransformers/server/main.py \\"
echo "     --model_path deepseek-ai/DeepSeek-V2-Lite \\"
echo "     --gguf_path /path/to/DeepSeek-V2-Lite-GGUF \\"
echo "     --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V2-Lite-Chat-serve.yaml \\"
echo "     --port 10002 \\"
echo "     --backend_type balance_serve"
echo ""
echo "2. LLaMA-Factory API (with fine-tuned adapter):"
echo "   cd $LLAMA_FACTORY_DIR"
echo "   llamafactory-cli api examples/inference/deepseek2_lite_inference.yaml --port 8000"
echo ""
echo "3. LLaMA-Factory Chat (interactive):"
echo "   cd $LLAMA_FACTORY_DIR"
echo "   llamafactory-cli chat examples/inference/deepseek2_lite_inference.yaml"
echo ""
echo "Note: Make sure you have:"
echo "  - Model config files (from deepseek-ai/DeepSeek-V2-Lite)"
echo "  - GGUF files (for ktransformers server)"
echo "  - Adapter checkpoints (from saves/Kllama_deepseekV2Lite for LLaMA-Factory)"
echo ""

