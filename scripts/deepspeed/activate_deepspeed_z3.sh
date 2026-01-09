#!/bin/bash
# Activation script for DeepSpeed ZeRO-3 environment
# This sets up the correct library paths for CUDA compilation

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate deepspeed-z3

# Set CUDA paths for compilation
# PyTorch has bundled CUDA libraries, but we need system libraries for linking
# Add PyTorch's nvidia library paths FIRST, then system libraries
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

# Then add system libraries for linking (but PyTorch's will be found first)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

export CUDA_HOME=/usr
export CUDA_ROOT=/usr

# Set LIBRARY_PATH for linking (different from LD_LIBRARY_PATH for runtime)
# This tells the linker where to find libraries during compilation
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH

# Also add PyTorch's library paths for linking
if [[ -d "$PYTORCH_NV_LIB/cuda_runtime/lib" ]]; then
    export LIBRARY_PATH=$PYTORCH_NV_LIB/cuda_runtime/lib:$LIBRARY_PATH
fi
if [[ -d "$PYTORCH_NV_LIB/curand/lib" ]]; then
    export LIBRARY_PATH=$PYTORCH_NV_LIB/curand/lib:$LIBRARY_PATH
fi

echo "✓ DeepSpeed ZeRO-3 environment activated"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  LD_LIBRARY_PATH includes: /usr/lib/x86_64-linux-gnu"

