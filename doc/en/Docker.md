# Docker Images

This document provides an overview of available Docker images. For detailed documentation on each image, see the dedicated documentation files.

## Quick Start Guide

- ✅ **For production/guaranteed stability**: Use `latest-AVX2` (tested, stable, **has `ktransformers.local_chat`**)
  - See: [Docker Image: latest-AVX2 (Guaranteed)](../../docs/Docker_latest-AVX2.md)
  
- ⚠️ **For latest features**: Use `v0.4.3-cu128` (newest, test before production, **missing `ktransformers.local_chat`**)
  - See: [Docker Image: v0.4.3-cu128 (Newest)](../../docs/Docker_v0.4.3-cu128.md)

## Available Images

### ✅ latest-AVX2 (Guaranteed/Tested)

**Status**: ✅ **Guaranteed** - This image is tested and known to work with AVX2 CPUs.

**Details**:
- Last Updated: July 2025
- Size: ~8.10 GB
- CPU Support: AVX2 (Haswell 2013+)
- Status: Stable and tested
- **Has `ktransformers.local_chat` module** ✅

**Pull command**:
```bash
docker pull approachingai/ktransformers:latest-AVX2
```

**Full documentation**: [Docker Image: latest-AVX2 (Guaranteed)](../../docs/Docker_latest-AVX2.md)

### ⚠️ v0.4.3-cu128 (Newest Version)

**Status**: ⚠️ **Test Before Run** - This is the newest build (December 2025) but should be tested before production use.

**⚠️ Important Limitation**: The `ktransformers.local_chat` module is **NOT available** in this image.

**Details**:
- Last Updated: December 2025
- Size: ~44.1 GB (actual downloaded size)
- CPU Support: Multi-variant (includes AVX2, AVX512, AMX) - auto-detects best variant
- Status: Newest version, includes latest features
- **Missing `ktransformers.local_chat` module** ❌

**Pull command**:
```bash
docker pull approachingai/ktransformers:v0.4.3-cu128
```

**Full documentation**: [Docker Image: v0.4.3-cu128 (Newest)](../../docs/Docker_v0.4.3-cu128.md)

### 0.2.1 (Older Version)

**Status**: Older version compiled for AVX512 (may cause "Illegal instruction" errors on non-AVX512 CPUs)

**Note**: Not recommended for most users. Use `latest-AVX2` instead.

## Supported GGUF Formats

Both images support **all GGUF quantization formats**, not just Q4:
- **Quantized formats**: Q2_K, Q3_K, Q4_K (Q4_K_M, Q4_K_S), Q5_K, Q6_K, Q8_K, Q4_0, Q5_0, Q8_0, etc.
- **Full precision formats**: F16, F32
- **Other formats**: IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_XS, etc.

The AVX2/multi-variant tags refer to the **CPU instruction set** used for compilation, not which GGUF formats are supported. All formats work with both images, but performance may vary based on your CPU capabilities.

## Detailed Documentation

For complete usage instructions, examples, troubleshooting, and parameter details, see:

- **[Docker Image: latest-AVX2 (Guaranteed)](../../docs/Docker_latest-AVX2.md)** - Complete guide for the stable, tested image with `ktransformers.local_chat` support
- **[Docker Image: v0.4.3-cu128 (Newest)](../../docs/Docker_v0.4.3-cu128.md)** - Guide for the newest image (missing `local_chat`)

More operators you can see in the [readme](../../README.md)