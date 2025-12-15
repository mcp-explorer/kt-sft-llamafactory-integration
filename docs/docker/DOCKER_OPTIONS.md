# Docker Container Options for KTransformers

## Option 1: Official KTransformers Docker Image (Easiest)

There's an **official pre-built Docker image** available:

```bash
docker pull approachingai/ktransformers:0.2.1
```

### Usage

```bash
# Run with GPU support
docker run --gpus all \
  -v /path/to/models:/models \
  --name ktransformers \
  -itd approachingai/ktransformers:0.2.1

# Enter the container
docker exec -it ktransformers /bin/bash

# Use KTransformers
python -m ktransformers.local_chat \
  --gguf_path /models/path/to/gguf_path \
  --model_path /models/path/to/model_path \
  --cpu_infer 33
```

**⚠️ Important Note**: This image is compiled for **AVX512** CPUs. If your CPU doesn't support AVX512 (like yours - you need AVX2), you'll need to rebuild KTransformers inside the container.

### Rebuilding Inside Container (For AVX2)

If you need AVX2 support:

```bash
docker exec -it ktransformers /bin/bash

# Inside container, rebuild with AVX2
cd /workspace/ktransformers
CPU_INSTRUCT=AVX2 KTRANSFORMERS_FORCE_BUILD=TRUE pip install . --no-build-isolation
```

## Option 2: LLaMA-Factory Docker (Includes KTransformers)

LLaMA-Factory has Docker support that includes KTransformers:

```bash
cd LLaMA-Factory/docker/docker-cuda/

# Using Docker Compose (Recommended)
docker compose up -d
docker compose exec llamafactory bash

# Or build manually
docker build -f ./docker/docker-cuda/Dockerfile \
    --build-arg PIP_INDEX=https://pypi.org/simple \
    --build-arg EXTRAS=metrics \
    -t llamafactory:latest .

docker run -dit --ipc=host --gpus=all \
    -p 7860:7860 \
    -p 8000:8000 \
    --name llamafactory \
    llamafactory:latest

docker exec -it llamafactory bash
```

## Option 3: Build Custom Docker Image from Repository

You can build a custom image using the Dockerfile in the repository:

```bash
# From the repository root
docker build -f kt-sft/Dockerfile -t ktransformers:custom .
```

## Option 4: NVIDIA CUDA Base Image + Custom Setup

Create a custom Dockerfile based on NVIDIA's CUDA images:

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    python3.12 \
    python3-pip \
    g++-11 \
    gcc-11 \
    cmake \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Clone and install KTransformers
WORKDIR /workspace
RUN git clone https://github.com/kvcache-ai/ktransformers.git
WORKDIR /workspace/ktransformers/kt-sft

# Install KTransformers with AVX2
ENV CPU_INSTRUCT=AVX2
RUN pip install . --no-build-isolation

# Set working directory
WORKDIR /workspace
```

Build and run:

```bash
docker build -t ktransformers:avx2 .
docker run --gpus all -v /path/to/models:/models -it ktransformers:avx2
```

## Prerequisites

### Install NVIDIA Container Toolkit

Before using GPU-enabled containers, install the NVIDIA Container Toolkit:

```bash
# Add NVIDIA GPG key and repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

### Verify GPU Access

Test that Docker can access your GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Recommended Approach

**For your use case (AVX2 CPU, CUDA 12.4, DeepSeek-V2-Lite serving):**

1. **Try Option 1 first** (official image) - it's the quickest
2. **If AVX512 issue occurs**, rebuild inside the container with AVX2
3. **If that doesn't work**, use **Option 4** (custom Dockerfile) for full control

## Advantages of Docker

✅ **No build issues**: Pre-configured environment  
✅ **Reproducible**: Same environment every time  
✅ **Isolated**: Doesn't affect your host system  
✅ **GPU support**: Works with NVIDIA Container Toolkit  
✅ **Easy cleanup**: Remove container when done  

## Quick Start Command

```bash
# Pull and run official image
docker pull approachingai/ktransformers:0.2.1
docker run --gpus all \
  -v $(pwd):/workspace \
  -v /path/to/models:/models \
  --name ktransformers \
  -it approachingai/ktransformers:0.2.1

# Inside container, if you need AVX2:
cd /workspace/ktransformers/kt-sft
CPU_INSTRUCT=AVX2 KTRANSFORMERS_FORCE_BUILD=TRUE pip install . --no-build-isolation
```

