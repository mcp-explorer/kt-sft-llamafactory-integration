#!/bin/bash
# Verification script for kt-serve environment
# Checks if all components are properly installed

set -e

ENV_NAME="kt-serve"
PROJECT_DIR="/home/sean/Documents/ktransformers"

echo "=========================================="
echo "Verifying kt-serve Environment"
echo "=========================================="
echo ""

# Check if environment exists
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "❌ Environment '$ENV_NAME' does not exist."
    echo "   Run: ./setup_kt_serve_env.sh"
    exit 1
fi

echo "✅ Environment '$ENV_NAME' exists"
echo ""

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "  Python: $PYTHON_VERSION"
if [[ "$PYTHON_VERSION" == 3.12* ]]; then
    echo "  ✅ Python version correct"
else
    echo "  ⚠️  Python version should be 3.12.x"
fi
echo ""

# Check PyTorch
echo "Checking PyTorch..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
    print('  ✅ PyTorch CUDA working')
else:
    print('  ❌ CUDA not available in PyTorch')
" || echo "  ❌ PyTorch check failed"
echo ""

# Check KTransformers
echo "Checking KTransformers..."
python -c "
try:
    import ktransformers
    version = getattr(ktransformers, '__version__', 'installed')
    print(f'  KTransformers: {version}')
    print('  ✅ KTransformers installed')
except ImportError as e:
    print(f'  ❌ KTransformers not installed: {e}')
" || echo "  ❌ KTransformers check failed"
echo ""

# Check Flash-Attention
echo "Checking Flash-Attention..."
python -c "
try:
    import flash_attn
    print('  Flash-Attention: installed')
    print('  ✅ Flash-Attention installed')
except ImportError as e:
    print(f'  ⚠️  Flash-Attention not available: {e}')
" || echo "  ⚠️  Flash-Attention check failed"
echo ""

# Check LLaMA-Factory
echo "Checking LLaMA-Factory..."
if [ -d "${PROJECT_DIR}/LLaMA-Factory" ]; then
    python -c "
try:
    import llamafactory
    print('  LLaMA-Factory: installed')
    print('  ✅ LLaMA-Factory installed')
except ImportError as e:
    print(f'  ⚠️  LLaMA-Factory not available: {e}')
" || echo "  ⚠️  LLaMA-Factory check failed"
else
    echo "  ⚠️  LLaMA-Factory directory not found"
fi
echo ""

# Check GLIBCXX
echo "Checking GLIBCXX version..."
CONDA_ENV_PATH=$(conda info --base)/envs/${ENV_NAME}
if [ -f "${CONDA_ENV_PATH}/lib/libstdc++.so.6" ]; then
    GLIBCXX_VERSION=$(strings "${CONDA_ENV_PATH}/lib/libstdc++.so.6" | grep GLIBCXX | tail -1)
    echo "  Latest GLIBCXX: $GLIBCXX_VERSION"
    if echo "$GLIBCXX_VERSION" | grep -q "GLIBCXX_3.4.32"; then
        echo "  ✅ GLIBCXX_3.4.32 found"
    else
        echo "  ⚠️  GLIBCXX_3.4.32 not found (may still work)"
    fi
else
    echo "  ⚠️  libstdc++.so.6 not found"
fi
echo ""

# Check CUDA runtime
echo "Checking CUDA runtime..."
python -c "
import ctypes
import os
try:
    # Try to find libcudart.so.11.0
    conda_env = os.environ.get('CONDA_PREFIX', '')
    if conda_env:
        lib_path = os.path.join(conda_env, 'lib', 'libcudart.so.11.0')
        if os.path.exists(lib_path):
            print('  ✅ libcudart.so.11.0 found')
        else:
            print('  ⚠️  libcudart.so.11.0 not found in conda env')
            print('      Try: conda install -y -c nvidia/label/cuda-11.8.0 cuda-runtime')
    else:
        print('  ⚠️  CONDA_PREFIX not set')
except Exception as e:
    print(f'  ⚠️  CUDA runtime check failed: {e}')
"
echo ""

# Check model files
echo "Checking model files..."
if [ -d "${PROJECT_DIR}/LLaMA-Factory/saves/Kllama_deepseekV2Lite" ]; then
    echo "  ✅ Adapter directory found: saves/Kllama_deepseekV2Lite"
    if [ -f "${PROJECT_DIR}/LLaMA-Factory/saves/Kllama_deepseekV2Lite/adapter_model.safetensors" ]; then
        echo "  ✅ Adapter model file found"
    else
        echo "  ⚠️  Adapter model file not found"
    fi
else
    echo "  ⚠️  Adapter directory not found: saves/Kllama_deepseekV2Lite"
fi
echo ""

# Check optimize rules
echo "Checking optimize rules..."
OPTIMIZE_RULE="${PROJECT_DIR}/kt-sft/ktransformers/optimize/optimize_rules/DeepSeek-V2-Lite-Chat-sft.yaml"
if [ -f "$OPTIMIZE_RULE" ]; then
    echo "  ✅ Optimize rule found: DeepSeek-V2-Lite-Chat-sft.yaml"
else
    echo "  ⚠️  Optimize rule not found: $OPTIMIZE_RULE"
fi

# Check AMX support
echo ""
echo "Checking CPU features..."
if command -v lscpu &> /dev/null; then
    if lscpu | grep -q "amx"; then
        echo "  ✅ AMX support detected - can use -amx.yaml optimize rules"
    else
        echo "  ℹ️  No AMX support - use standard optimize rules"
    fi
else
    echo "  ⚠️  lscpu not available"
fi
echo ""

# Summary
echo "=========================================="
echo "Verification Complete"
echo "=========================================="
echo ""
echo "If all checks passed, you can start serving:"
echo "  conda activate $ENV_NAME"
echo "  cd LLaMA-Factory"
echo "  llamafactory-cli api examples/inference/deepseek2_lite_serve.yaml --port 8000"
echo ""

