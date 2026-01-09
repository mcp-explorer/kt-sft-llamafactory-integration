#!/bin/bash
# Rebuild KTransformers in Kllama conda environment
# This script handles CUDA/nvcc include path issues and CPU instruction compatibility
# Based on fixes from custom_flashinfer and the params.nth assertion fix (commit 3ae2205)

set -e  # Exit on error

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"  # Go up two levels: utils -> scripts -> project root

echo "=========================================="
echo "Rebuild KTransformers in Kllama Environment"
echo "=========================================="
echo ""

# Activate conda environment
CONDA_ENV="Kllama"

# Initialize conda for bash shell
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    # Try to find conda in PATH
    if ! command -v conda &> /dev/null; then
        echo "Error: conda not found. Please ensure conda is installed and in your PATH."
        exit 1
    fi
    # If conda is in PATH, try to initialize it
    eval "$(conda shell.bash hook)"
fi

# Activate the conda environment
echo "Activating conda environment: $CONDA_ENV"
if ! conda activate "$CONDA_ENV"; then
    echo "Error: Failed to activate conda environment '$CONDA_ENV'"
    echo "Please ensure the environment exists. Create it with:"
    echo "  conda create -n $CONDA_ENV python=3.12"
    exit 1
fi

# Verify the environment is activated
if [ "$CONDA_DEFAULT_ENV" = "$CONDA_ENV" ]; then
    echo "✓ Successfully activated $CONDA_ENV environment"
    echo "  Python: $(which python)"
    echo "  Python version: $(python --version 2>&1)"
else
    echo "Warning: Conda environment may not be fully activated."
    echo "  Expected: $CONDA_ENV"
    echo "  Current: ${CONDA_DEFAULT_ENV:-none}"
    echo "Attempting to continue anyway..."
fi

echo ""

# Check CPU instruction support
echo "Checking CPU instruction support..."
CPU_INSTRUCT="NATIVE"
if lscpu | grep -q "avx512bw"; then
    echo "✓ AVX512BW detected - can use FANCY or AVX512"
    CPU_INSTRUCT="AVX512"
elif lscpu | grep -q "avx512"; then
    echo "✓ AVX512 detected"
    CPU_INSTRUCT="AVX512"
elif lscpu | grep -q "avx2"; then
    echo "✓ AVX2 detected"
    CPU_INSTRUCT="AVX2"
else
    echo "⚠ Only basic CPU instructions detected - using NATIVE (auto-detect)"
    CPU_INSTRUCT="NATIVE"
fi

# Check for required system packages for building
echo ""
echo "Checking system dependencies for building..."
MISSING_DEPS=()

# Check for build-essential components
if ! command -v gcc &> /dev/null; then
    MISSING_DEPS+=("gcc")
fi
if ! command -v g++ &> /dev/null; then
    MISSING_DEPS+=("g++")
fi
if ! command -v cmake &> /dev/null; then
    MISSING_DEPS+=("cmake")
fi

# Check for CUDA compiler
if ! command -v nvcc &> /dev/null; then
    MISSING_DEPS+=("nvidia-cuda-toolkit")
fi

# Check for system headers (math.h location)
if [ ! -f "/usr/include/math.h" ] && [ ! -f "/usr/include/x86_64-linux-gnu/math.h" ] && [ ! -f "/usr/lib/gcc/x86_64-linux-gnu/12/include/math.h" ]; then
    MISSING_DEPS+=("libc6-dev")
    MISSING_DEPS+=("build-essential")
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "⚠ Missing system dependencies: ${MISSING_DEPS[*]}"
    echo ""
    echo "Installing missing dependencies..."
    if command -v apt-get &> /dev/null; then
        # Try to install without sudo first (if user has permissions)
        if apt-get install -y "${MISSING_DEPS[@]}" 2>/dev/null; then
            echo "✓ Dependencies installed"
        else
            echo "⚠ Need sudo to install dependencies. Attempting with sudo..."
            if sudo apt-get update && sudo apt-get install -y "${MISSING_DEPS[@]}"; then
                echo "✓ Dependencies installed with sudo"
            else
                echo "⚠ Could not install dependencies automatically"
                echo "  Please run manually:"
                echo "  sudo apt-get update"
                echo "  sudo apt-get install -y ${MISSING_DEPS[*]}"
            fi
        fi
    else
        echo "⚠ Could not detect package manager. Please install manually:"
        echo "  ${MISSING_DEPS[*]}"
    fi
else
    echo "✓ All required system dependencies found"
fi

# Navigate to kt-sft directory
KT_SFT_DIR="$PROJECT_ROOT/kt-sft"
if [ ! -d "$KT_SFT_DIR" ]; then
    echo "Error: kt-sft directory not found at $KT_SFT_DIR"
    echo "Please ensure kt-sft is available in the project root."
    exit 1
fi

cd "$KT_SFT_DIR"

# Function to try removing with sudo if needed
try_remove_with_sudo() {
    local target="$1"
    if [ -e "$target" ]; then
        # Try normal removal first
        if rm -rf "$target" 2>/dev/null; then
            return 0
        fi
        # If that fails, try sudo (may prompt for password)
        echo "  Permission denied, trying with sudo..."
        if sudo rm -rf "$target" 2>/dev/null; then
            echo "  ✓ Removed with sudo"
            return 0
        else
            # If sudo also fails, try changing ownership
            echo "  Trying to change ownership..."
            if sudo chown -R "$USER:$USER" "$target" 2>/dev/null; then
                if rm -rf "$target" 2>/dev/null; then
                    echo "  ✓ Removed after changing ownership"
                    return 0
                fi
            fi
            return 1
        fi
    fi
    return 0
}

# Clean build directories
echo ""
echo "=========================================="
echo "Cleaning build directories..."
echo "=========================================="
rm -rf build 2>/dev/null || true
rm -rf *.egg-info 2>/dev/null || true
rm -rf csrc/build 2>/dev/null || true

# Critical: Remove CMakeCache.txt and build directory to avoid path mismatch errors
if [ -d "csrc/ktransformers_ext/build" ]; then
    echo "Removing stale build directory..."
    if try_remove_with_sudo "csrc/ktransformers_ext/build"; then
        echo "✓ Build directory removed"
    else
        echo "⚠ Warning: Could not remove build directory"
        echo "  You may need to run manually: sudo rm -rf $KT_SFT_DIR/csrc/ktransformers_ext/build"
    fi
fi

rm -rf csrc/ktransformers_ext/cuda/build 2>/dev/null || true
rm -rf csrc/ktransformers_ext/cuda/dist 2>/dev/null || true
rm -rf csrc/ktransformers_ext/cuda/*.egg-info 2>/dev/null || true
rm -rf ~/.ktransformers 2>/dev/null || true

# Set CPU instruction environment variables
export CPU_INSTRUCT="$CPU_INSTRUCT"
export CPUINFER_CPU_INSTRUCT="$CPU_INSTRUCT"
export CPUINFER_ENABLE_AMX=OFF
export KTRANSFORMERS_FORCE_BUILD=TRUE

# Fix CUDA/nvcc include path issues (based on custom_flashinfer fix)
# Ensure CUDA_HOME is set correctly
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    elif [ -d "/usr/lib/nvidia-cuda-toolkit" ]; then
        export CUDA_HOME="/usr"
    fi
fi

# Set CC/CXX environment variables for general compilation
# Prefer g++-11 to match CUDAHOSTCXX for consistency
# For CUDA/nvcc, we'll use CUDAHOSTCXX to specify a compatible host compiler
if command -v g++-11 &> /dev/null; then
    export CC="/usr/bin/gcc-11"
    export CXX="/usr/bin/g++-11"
    echo "  ✓ Using g++-11 for general compilation (CC/CXX=/usr/bin/g++-11)"
elif command -v g++-12 &> /dev/null; then
    export CC="/usr/bin/gcc-12"
    export CXX="/usr/bin/g++-12"
    echo "  ✓ Using g++-12 for general compilation (CC/CXX=/usr/bin/g++-12)"
elif command -v g++ &> /dev/null; then
    export CC="$(which gcc)"
    export CXX="$(which g++)"
    echo "  ✓ Using default g++ for general compilation (CC/CXX=$(which g++))"
fi

# CRITICAL FIX: Set CUDAHOSTCXX to use g++-11 (or compatible version) for nvcc
# CUDA 12.0 supports GCC up to 11.x, but has issues with gcc-12
# This explicitly tells nvcc which host compiler to use, bypassing CMake's auto-detection
if command -v g++-11 &> /dev/null; then
    export CUDAHOSTCXX="/usr/bin/g++-11"
    echo "  ✓ Using g++-11 as CUDA host compiler (CUDAHOSTCXX=$CUDAHOSTCXX)"
elif command -v g++-10 &> /dev/null; then
    export CUDAHOSTCXX="$(which g++-10)"
    echo "  ✓ Using g++-10 as CUDA host compiler (CUDAHOSTCXX=$CUDAHOSTCXX)"
elif command -v g++-9 &> /dev/null; then
    export CUDAHOSTCXX="$(which g++-9)"
    echo "  ✓ Using g++-9 as CUDA host compiler (CUDAHOSTCXX=$CUDAHOSTCXX)"
else
    echo "  ⚠ Warning: No compatible GCC version (9-11) found for CUDA host compiler"
    echo "    CUDA 12.0 may have issues with gcc-12. Consider installing gcc-11:"
    echo "    sudo apt-get install -y gcc-11 g++-11"
    # Still try to set it to avoid CMake auto-detection issues
    if command -v g++-12 &> /dev/null; then
        export CUDAHOSTCXX="/usr/bin/g++-12"
        echo "    Falling back to g++-12 (may still fail)"
    fi
fi

# Set proper include paths for nvcc to find system headers
# CRITICAL: Use c++/11 paths only (not c++/12) to match g++-11 host compiler
# This prevents CUDA headers from finding c++/12/cmath which conflicts with g++-11
if command -v g++-11 &> /dev/null; then
    export CPLUS_INCLUDE_PATH="/usr/include:/usr/include/x86_64-linux-gnu:/usr/lib/gcc/x86_64-linux-gnu/11/include:/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11"
    echo "  ✓ Set CPLUS_INCLUDE_PATH to c++/11 paths only"
else
    # Fallback if g++-11 not available
    export CPLUS_INCLUDE_PATH="/usr/include:/usr/include/x86_64-linux-gnu"
    echo "  ⚠ Warning: g++-11 not found, using minimal include paths"
fi
export C_INCLUDE_PATH="/usr/include:/usr/include/x86_64-linux-gnu"

# Check for nvcc_wrapper and warn if it has c++/12 paths
if [ -f "$HOME/bin/nvcc_wrapper" ] || [ -f "/usr/local/bin/nvcc_wrapper" ]; then
    WRAPPER_PATH=""
    if [ -f "$HOME/bin/nvcc_wrapper" ]; then
        WRAPPER_PATH="$HOME/bin/nvcc_wrapper"
    elif [ -f "/usr/local/bin/nvcc_wrapper" ]; then
        WRAPPER_PATH="/usr/local/bin/nvcc_wrapper"
    fi
    if [ -n "$WRAPPER_PATH" ] && grep -q "c++/12" "$WRAPPER_PATH" 2>/dev/null; then
        echo "  ⚠ Warning: nvcc_wrapper found at $WRAPPER_PATH contains c++/12 paths"
        echo "    This may cause conflicts with g++-11. Consider updating it to use c++/11 paths only."
    elif [ -n "$WRAPPER_PATH" ]; then
        echo "  ✓ nvcc_wrapper found at $WRAPPER_PATH (appears to be configured correctly)"
    fi
fi

echo ""
echo "=========================================="
echo "Rebuilding KTransformers..."
echo "=========================================="
echo "Configuration:"
echo "  CPU_INSTRUCT: $CPU_INSTRUCT"
echo "  CUDA_HOME: ${CUDA_HOME:-not set}"
echo "  CC: ${CC:-not set}"
echo "  CXX: ${CXX:-not set}"
echo ""
echo "This may take several minutes..."
echo ""

# Rebuild and capture output
REBUILD_LOG="/tmp/kt_rebuild_kllama_$$.log"
echo "Rebuild log: $REBUILD_LOG"

# Install in editable mode so source changes take effect immediately
echo "Installing in editable mode (-e flag) for development..."
echo ""

if pip install -v -e . --no-build-isolation --no-cache-dir 2>&1 | tee "$REBUILD_LOG"; then
    # Check if the log contains error messages (pip sometimes returns 0 even on failure)
    if grep -q "ERROR\|Failed\|error:" "$REBUILD_LOG"; then
        echo "⚠ Error: Rebuild failed (errors found in log)!"
        echo "Check $REBUILD_LOG for details"
        REBUILD_SUCCESS=false
    else
        echo "✓ KTransformers rebuilt successfully with CPU_INSTRUCT=$CPU_INSTRUCT"
        REBUILD_SUCCESS=true
    fi
else
    echo "⚠ Error: Rebuild failed (pip returned error)!"
    echo "Check $REBUILD_LOG for details"
    REBUILD_SUCCESS=false
fi

if [ "$REBUILD_SUCCESS" = false ]; then
    echo ""
    echo "=========================================="
    echo "⚠ REBUILD FAILED - Attempting Auto-Fix"
    echo "=========================================="
    echo ""
    
    # Check the error type from the log
    if grep -q "math.h: No such file or directory\|fatal error: math.h\|include_next.*math.h" "$REBUILD_LOG"; then
        echo "⚠ Detected CUDA/nvcc include path issue (known issue with nvcc + gcc-12)"
        echo ""
        echo "This is a known compatibility issue between nvcc (CUDA 12.0) and gcc-12."
        echo "The problem occurs during CMake's CUDA compiler detection phase."
        echo ""
        echo "Possible solutions:"
        echo "  1. Try using gcc-11 instead: export CC=gcc-11 CXX=g++-11 && $0"
        echo "  2. Install CUDA from NVIDIA's official repository (not system packages)"
        echo "  3. Use a conda-based CUDA installation"
        echo "  4. Check ktransformers repository for known workarounds"
        echo ""
        echo "Attempting workaround with gcc-11 (if available)..."
        
        # Try gcc-11 as a workaround
        if command -v g++-11 &> /dev/null; then
            echo "  → Found g++-11, trying with gcc-11..."
            export CC="/usr/bin/gcc-11"
            export CXX="/usr/bin/g++-11"
            unset CPLUS_INCLUDE_PATH
            unset C_INCLUDE_PATH
            
            echo "  → Retrying rebuild with gcc-11..."
            echo ""
            
            # Retry the rebuild with gcc-11 (editable mode)
            if pip install -v -e . --no-build-isolation --no-cache-dir 2>&1 | tee "$REBUILD_LOG"; then
                if grep -q "ERROR\|Failed\|error:" "$REBUILD_LOG"; then
                    REBUILD_SUCCESS=false
                else
                    echo "✓ KTransformers rebuilt successfully on retry!"
                    REBUILD_SUCCESS=true
                fi
            else
                REBUILD_SUCCESS=false
            fi
        else
            echo "  → g++-11 not found, skipping gcc-11 workaround"
            REBUILD_SUCCESS=false
        fi
        
        # If still failing, try installing packages
        if [ "$REBUILD_SUCCESS" = false ] && command -v apt-get &> /dev/null; then
            echo "Trying to install/reinstall build-essential and libc6-dev..."
            if sudo apt-get update && sudo apt-get install -y --reinstall build-essential libc6-dev; then
                echo "✓ Packages reinstalled, retrying rebuild..."
                echo ""
                
                # Retry the rebuild again (editable mode)
                if pip install -v -e . --no-build-isolation --no-cache-dir 2>&1 | tee "$REBUILD_LOG"; then
                    if grep -q "ERROR\|Failed\|error:" "$REBUILD_LOG"; then
                        REBUILD_SUCCESS=false
                    else
                        echo "✓ KTransformers rebuilt successfully on retry!"
                        REBUILD_SUCCESS=true
                    fi
                else
                    REBUILD_SUCCESS=false
                fi
            fi
        fi
    fi
    
    # Try to fix the build directory issue if still failing
    if [ "$REBUILD_SUCCESS" = false ] && [ -d "csrc/ktransformers_ext/build" ]; then
        echo "Attempting to fix build directory issues..."
        if try_remove_with_sudo "csrc/ktransformers_ext/build"; then
            echo "✓ Build directory removed, retrying rebuild..."
            echo ""
            
            # Retry the rebuild (editable mode)
            if pip install -v -e . --no-build-isolation --no-cache-dir 2>&1 | tee "$REBUILD_LOG"; then
                if grep -q "ERROR\|Failed\|error:" "$REBUILD_LOG"; then
                    REBUILD_SUCCESS=false
                else
                    echo "✓ KTransformers rebuilt successfully on retry!"
                    REBUILD_SUCCESS=true
                fi
            else
                REBUILD_SUCCESS=false
            fi
        fi
    fi
    
    if [ "$REBUILD_SUCCESS" = false ]; then
        echo ""
        echo "=========================================="
        echo "⚠ AUTO-FIX FAILED - Manual Fix Required"
        echo "=========================================="
        echo ""
        echo "The rebuild failed. Common fixes:"
        echo ""
        echo "1. Install missing system headers:"
        echo "   sudo apt-get update"
        echo "   sudo apt-get install -y build-essential libc6-dev"
        echo ""
        echo "2. Clean and rebuild (editable mode):"
        echo "   sudo rm -rf $KT_SFT_DIR/csrc/ktransformers_ext/build"
        echo "   cd $KT_SFT_DIR"
        echo "   CPU_INSTRUCT=$CPU_INSTRUCT KTRANSFORMERS_FORCE_BUILD=TRUE pip install -v -e . --no-build-isolation"
        echo ""
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "✓ Rebuild completed successfully!"
echo "=========================================="
echo ""
echo "KTransformers has been rebuilt with:"
echo "  - CPU_INSTRUCT: $CPU_INSTRUCT"
echo "  - CUDA/nvcc fixes applied (CC/CXX environment variables)"
echo "  - Includes params.nth assertion fix (commit 3ae2205)"
echo "  - Installed in EDITABLE MODE (-e flag)"
echo ""
echo "✓ Editable mode enabled: Changes to source files in kt-sft/ktransformers/"
echo "  will take effect immediately without reinstalling!"
echo ""
echo "You can now use KTransformers in your training scripts."
echo ""

