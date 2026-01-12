#!/bin/bash
# Rebuild the deepspeed-z3 conda environment from scratch
# This ensures all packages are fresh and compatible

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "Rebuilding DeepSpeed ZeRO-3 Environment"
echo "=========================================="
echo ""

# Initialize conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi

ENV_NAME="deepspeed-z3"
PYTHON_VERSION="3.11"
AUTO_YES=false

# Parse command line arguments
if [[ "$1" == "--yes" ]] || [[ "$1" == "-y" ]]; then
    AUTO_YES=true
fi

# Deactivate current environment if it's the one we're rebuilding
if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" = "${ENV_NAME}" ]; then
    echo "Deactivating current environment..."
    conda deactivate
fi

# Check if environment exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠ Environment '${ENV_NAME}' already exists."
    if [ "$AUTO_YES" = true ]; then
        echo "Removing existing environment (--yes flag provided)..."
        conda env remove -n "${ENV_NAME}" -y
        echo "✓ Environment removed"
    else
        read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing existing environment..."
            conda env remove -n "${ENV_NAME}" -y
            echo "✓ Environment removed"
        else
            echo "Aborted. Exiting."
            exit 1
        fi
    fi
fi

echo ""
echo "Creating new conda environment: ${ENV_NAME}"
echo "Python version: ${PYTHON_VERSION}"
echo ""

# Create new environment
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

echo ""
echo "Activating environment..."
conda activate "${ENV_NAME}"

echo ""
echo "Installing PyTorch with CUDA 12.1 via conda (recommended for better library compatibility)..."
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1

echo ""
echo "Installing CUDA 12.1 libraries to match PyTorch requirements..."
conda install -y -c nvidia libnvjitlink=12.1.105

echo ""
echo "Fixing Intel ITT compatibility issue by downgrading MKL to 2024.0.0..."
echo "  (MKL 2024.1+ has compatibility issues with PyTorch)"
conda install -y -c conda-forge mkl=2024.0.0

echo ""
echo "Installing DeepSpeed..."
pip install deepspeed

echo ""
echo "Installing Transformers and related packages..."
pip install transformers==4.57.1
pip install "peft>=0.14.0,<=0.17.1"
pip install accelerate
pip install datasets
pip install sentencepiece
pip install protobuf

echo ""
echo "Installing LLaMA-Factory dependencies..."
cd "${PROJECT_ROOT}/LLaMA-Factory"
pip install -e ".[torch,metrics]"

echo ""
echo "Installing additional utilities..."
pip install wandb
pip install scipy
pip install numpy

echo ""
echo "Setting up library paths..."
# Create activation script to set library paths
ACTIVATE_SCRIPT="${CONDA_PREFIX}/etc/conda/activate.d/deepspeed_env.sh"
mkdir -p "$(dirname "${ACTIVATE_SCRIPT}")"

cat > "${ACTIVATE_SCRIPT}" << 'EOF'
#!/bin/bash
# DeepSpeed ZeRO-3 environment setup

export CUDA_HOME="/usr"
export CUDA_ROOT="/usr"

# Add PyTorch CUDA libraries first (priority)
if [ -d "${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia" ]; then
    # nvJitLink must come first to avoid symbol conflicts
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/nvjitlink/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/curand/lib:${LD_LIBRARY_PATH}"
    export LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/curand/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/nvjitlink/lib:${LIBRARY_PATH}"
fi

# System libraries
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH}"

# Memory management
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
EOF

chmod +x "${ACTIVATE_SCRIPT}"

echo ""
echo "Verifying installation..."
echo ""

python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'✓ CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
python -c "import deepspeed; print(f'✓ DeepSpeed: {deepspeed.__version__}')"
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')"
python -c "import peft; print(f'✓ PEFT: {peft.__version__}')"

echo ""
echo "Testing DeepSpeed CPU Adam compilation..."
python -c "
import deepspeed
import torch
# This will trigger CPU Adam compilation if needed
try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    print('✓ DeepSpeed CPU Adam available')
except Exception as e:
    print(f'⚠ DeepSpeed CPU Adam: {e}')
"

echo ""
echo "=========================================="
echo "Environment rebuild complete!"
echo "=========================================="
echo ""
echo "Environment: ${ENV_NAME}"
echo "Python: $(python --version)"
echo ""
echo "To activate:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To test inference:"
echo "  ./scripts/inference/infer_ds2_chat_lite_hf.sh chat"
echo ""

