#!/bin/bash
# Setup script to create conda environment for LLaMA-Factory model serving and testing
# Matches system versions: Python 3.12.3, CUDA 12.0, Driver 570.195.03

set -e

ENV_NAME="llamafactory-serve"
PROJECT_DIR="/home/sean/Documents/ktransformers/LLaMA-Factory"

echo "=========================================="
echo "Creating Conda Environment for LLaMA-Factory"
echo "=========================================="
echo ""
echo "System Information:"
echo "  Python: 3.12.3"
echo "  CUDA: 12.0"
echo "  Driver: 570.195.03"
echo "  Environment name: $ENV_NAME"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not available. Please install conda first."
    exit 1
fi

# Remove existing environment if it exists
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

# Install PyTorch with CUDA 12.0 support
echo ""
echo "Installing PyTorch with CUDA 12.0 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120

# Verify PyTorch CUDA availability
echo ""
echo "Verifying PyTorch CUDA installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

# Install requirements
echo ""
echo "Installing requirements from requirements.txt..."
cd "$PROJECT_DIR"
pip install -r requirements.txt

# Install llamafactory package in development mode
echo ""
echo "Installing llamafactory package..."
pip install -e .

echo ""
echo "=========================================="
echo "✅ Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To test your model, you can use:"
echo "  # Chat with base model:"
echo "  llamafactory-cli chat test_base_model.yaml"
echo ""
echo "  # Chat with SFT model:"
echo "  llamafactory-cli chat test_sft_model.yaml"
echo ""
echo "  # Serve model via API:"
echo "  llamafactory-cli api test_sft_model.yaml --port 8000"
echo ""


