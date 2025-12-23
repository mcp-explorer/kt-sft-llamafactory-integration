# https://hub.docker.com/r/hiyouga/pytorch/tags
# Updated to use CUDA 12.6 (cu126) for PyTorch 2.6.0 - official pairing
# This Dockerfile builds KTransformers from the mounted /kt-sft volume at runtime
# This ensures you always get the latest changes from your local code
ARG BASE_IMAGE=hiyouga/pytorch:th2.6.0-cu124-flashattn2.7.4-cxx11abi0-devel
FROM ${BASE_IMAGE}

# Installation arguments
ARG PIP_INDEX=https://pypi.org/simple
ARG EXTRAS=metrics
ARG INSTALL_FLASHATTN=false
ARG HTTP_PROXY=""

# Define environments
ENV MAX_JOBS=16
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""
ENV PIP_ROOT_USER_ACTION=ignore
ENV http_proxy="${HTTP_PROXY}"
ENV https_proxy="${HTTP_PROXY}"

# Use Bash instead of default /bin/sh
SHELL ["/bin/bash", "-c"]

# Set the working directory
WORKDIR /app

# Change pip source
RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    pip install --no-cache-dir --upgrade pip packaging wheel setuptools

# Upgrade PyTorch to 2.6.0+cu126 (official pairing for PyTorch 2.6.0)
# This matches KTransformers wheel: cu126torch26
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Install the requirements
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application into the image
COPY . /app

# Install LLaMA Factory
RUN pip install --no-cache-dir -e ".[${EXTRAS}]" --no-build-isolation

# Rebuild flash attention (only if explicitly requested)
# Note: KTransformers doesn't require flash-attn, and it can cause ABI conflicts
RUN if [ "${INSTALL_FLASHATTN}" == "true" ]; then \
        pip uninstall -y ninja && \
        pip install --no-cache-dir ninja && \
        pip install --no-cache-dir flash-attn --no-build-isolation; \
    else \
        # Uninstall flash-attn if present to avoid ABI conflicts with PyTorch 2.6.0+cu126
        pip uninstall -y flash-attn 2>/dev/null || true; \
    fi

# Set up LD_LIBRARY_PATH to include site-packages for ggml libraries
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages:${LD_LIBRARY_PATH}

# Expose ports
ENV GRADIO_SERVER_PORT=7860
EXPOSE 7860
ENV API_PORT=8000
EXPOSE 8000

# unset proxy
ENV http_proxy=
ENV https_proxy=

# Reset pip config
RUN pip config unset global.index-url && \
    pip config unset global.extra-index-url

# Create entrypoint script that builds KTransformers from /kt-sft volume at startup
# This ensures you always get the latest changes from your local kt-sft directory
RUN cat > /entrypoint.sh << 'EOF' && chmod +x /entrypoint.sh
#!/bin/bash
set -e

# Function to build KTransformers from /kt-sft volume
build_ktransformers() {
    if [ ! -d "/kt-sft" ] || [ ! -f "/kt-sft/pyproject.toml" ]; then
        echo "ERROR: /kt-sft volume not found or invalid!"
        echo "Make sure kt-sft is mounted as a volume in docker-compose.yml"
        exit 1
    fi
    
    echo "========================================="
    echo "Building KTransformers from /kt-sft..."
    echo "This ensures you get the latest changes!"
    echo "========================================="
    
    cd /kt-sft
    
    # Clean previous builds to ensure fresh build
    # Use find and rm with error handling for read-only files
    find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    # Try to clean build directory, but continue if it fails (read-only volume)
    rm -rf build csrc/ktransformers_ext/build *.egg-info 2>/dev/null || true
    
    # Build KTransformers with all fixes
    # CPU_INSTRUCT=NATIVE enables native CPU instructions
    # KTRANSFORMERS_FORCE_BUILD=TRUE forces rebuild from source
    CPU_INSTRUCT=NATIVE \
    KTRANSFORMERS_FORCE_BUILD=TRUE \
    pip install . --no-build-isolation --no-cache-dir
    
    # Copy ggml libraries to site-packages so they can be found at runtime
    if [ -d "csrc/ktransformers_ext/build/bin" ]; then
        echo "Copying GGML libraries to site-packages..."
        cp -f csrc/ktransformers_ext/build/bin/libggml*.so* /opt/conda/lib/python3.11/site-packages/ 2>/dev/null || true
        echo "✅ GGML libraries copied successfully"
    else
        echo "⚠️  WARNING: GGML libraries not found in build directory"
    fi
    
    # Verify installation
    if python -c "import ktransformers; import cpuinfer_ext" 2>/dev/null; then
        echo "✅ KTransformers installed and verified successfully"
    else
        echo "❌ ERROR: KTransformers installation verification failed"
        exit 1
    fi
}

# Always rebuild to get latest changes (or check if already installed)
if ! python -c "import ktransformers" 2>/dev/null; then
    # Not installed, build it
    build_ktransformers
elif [ -d "/kt-sft" ] && [ -f "/kt-sft/pyproject.toml" ]; then
    # Already installed, but check if we should rebuild
    # Option 1: Always rebuild (uncomment next 2 lines)
    # echo "Rebuilding KTransformers to get latest changes..."
    # build_ktransformers
    
    # Option 2: Only copy libraries if build exists (current behavior)
    echo "KTransformers already installed"
    if [ -d "/kt-sft/csrc/ktransformers_ext/build/bin" ]; then
        echo "Copying updated GGML libraries..."
        cp -f /kt-sft/csrc/ktransformers_ext/build/bin/libggml*.so* /opt/conda/lib/python3.11/site-packages/ 2>/dev/null || true
    fi
fi

# Execute the original command
exec "$@"
EOF

ENTRYPOINT ["/entrypoint.sh"]
