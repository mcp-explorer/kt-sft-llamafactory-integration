#!/bin/bash
# Quick fix script for DeepSpeed CPU Offload issues
# Run this if errors reappear after environment changes

set -e

echo "=========================================="
echo "DeepSpeed CPU Offload Quick Fix Script"
echo "=========================================="
echo ""

# Check if running as root for sudo commands
if [ "$EUID" -ne 0 ]; then 
    SUDO="sudo"
else
    SUDO=""
fi

# Fix 1: Missing C++12 header file
echo "[1/5] Fixing missing C++12 header file..."
if [ ! -f "/usr/include/x86_64-linux-gnu/c++/12/bits/new_allocator.h" ]; then
    if [ -f "/usr/include/c++/12.bak/bits/new_allocator.h" ]; then
        $SUDO cp /usr/include/c++/12.bak/bits/new_allocator.h /usr/include/x86_64-linux-gnu/c++/12/bits/new_allocator.h
        echo "✅ Copied new_allocator.h"
    else
        echo "⚠ Warning: Source file not found: /usr/include/c++/12.bak/bits/new_allocator.h"
        echo "   You may need to install: sudo apt-get install -y g++-12 libstdc++-12-dev"
    fi
else
    echo "✓ Header file already exists"
fi

# Fix 2: Create library symlinks
echo ""
echo "[2/5] Creating CUDA library symlinks..."
CONDA_PREFIX="${CONDA_PREFIX:-$HOME/miniconda3/envs/deepspeed-z3}"

CURAND_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/curand/lib"
CUDART_DIR="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"

if [ -d "$CURAND_DIR" ]; then
    cd "$CURAND_DIR"
    if [ ! -f "libcurand.so" ] && [ -f "libcurand.so.10" ]; then
        ln -sf libcurand.so.10 libcurand.so
        echo "✅ Created libcurand.so symlink"
    else
        echo "✓ libcurand.so symlink already exists"
    fi
else
    echo "⚠ Warning: curand directory not found: $CURAND_DIR"
fi

if [ -d "$CUDART_DIR" ]; then
    cd "$CUDART_DIR"
    if [ ! -f "libcudart.so" ] && [ -f "libcudart.so.12" ]; then
        ln -sf libcudart.so.12 libcudart.so
        echo "✅ Created libcudart.so symlink"
    else
        echo "✓ libcudart.so symlink already exists"
    fi
else
    echo "⚠ Warning: cuda_runtime directory not found: $CUDART_DIR"
fi

# Fix 3: Clear CPU Adam cache
echo ""
echo "[3/5] Clearing CPU Adam compilation cache..."
CACHE_DIR="$HOME/.cache/torch_extensions/py311_cu121/cpu_adam"
if [ -d "$CACHE_DIR" ]; then
    rm -f "$CACHE_DIR"/*.o "$CACHE_DIR"/*.so "$CACHE_DIR"/build.ninja 2>/dev/null || true
    echo "✅ Cleared CPU Adam cache"
else
    echo "✓ Cache directory doesn't exist (nothing to clear)"
fi

# Fix 4: Verify activation script
echo ""
echo "[4/5] Verifying activation script..."
ACTIVATE_SCRIPT="$CONDA_PREFIX/etc/conda/activate.d/deepspeed_env.sh"
if [ -f "$ACTIVATE_SCRIPT" ]; then
    if grep -q "nvjitlink/lib/libnvJitLink.so.12" "$ACTIVATE_SCRIPT"; then
        echo "✓ Activation script configured correctly"
    else
        echo "⚠ Warning: Activation script may need update for nvJitLink"
        echo "   Check: $ACTIVATE_SCRIPT"
    fi
else
    echo "⚠ Warning: Activation script not found: $ACTIVATE_SCRIPT"
fi

# Fix 5: Test CPU Adam
echo ""
echo "[5/5] Testing CPU Adam compilation..."
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate deepspeed-z3 2>/dev/null || {
    echo "⚠ Warning: Could not activate deepspeed-z3 environment"
    echo "   Please activate manually and test:"
    echo "   python -c \"from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam; import torch; opt = DeepSpeedCPUAdam([torch.randn(10, requires_grad=True)], lr=1e-3); print('✅ CPU Adam works!')\""
    exit 0
}

python << 'EOF'
try:
    from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam
    import torch
    params = [torch.randn(10, 10, requires_grad=True)]
    optimizer = DeepSpeedCPUAdam(params, lr=1e-3)
    print("✅ CPU Adam compilation successful!")
except Exception as e:
    print(f"❌ CPU Adam test failed: {e}")
    print("   Check the documentation: ../../docs/DEEPSPEED_CPU_OFFLOAD_FIXES.md")
    exit(1)
EOF

echo ""
echo "=========================================="
echo "✅ All fixes applied successfully!"
echo "=========================================="
echo ""
echo "If issues persist, see: ../../docs/DEEPSPEED_CPU_OFFLOAD_FIXES.md"

