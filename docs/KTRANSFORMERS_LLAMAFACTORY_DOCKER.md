# KTransformers + LLaMA-Factory Docker Integration

## Overview

This document summarizes the exploration of integrating **ktransformers CPU offloading** with **LLaMA-Factory CLI** inside **Docker** for SFT training of DeepSeek-V2-Lite-Chat.

**Goal:** Enable full-precision LoRA training with CPU offloading on a 16GB GPU using ktransformers, avoiding quantization approaches (QLoRA/4-bit) and other frameworks (DeepSpeed, Unsloth).

---

## 1. What Was Explored

### 1.1 Docker Image Build (WORKING)

Successfully built `kt-llamafactory:latest` Docker image with:
- **Base:** `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel`
- **ktransformers v0.5.0** compiled with CUDA 12.6, RTX 4080 arch (8.9)
- **flash-attn 2.7.4.post1**
- **LLaMA-Factory** with all dependencies

**Issues Fixed:**
- Added `hatchling`, `hatch-vcs`, `editables` for LLaMA-Factory build
- Fixed ktransformers version import (`version.py` creation)
- Patched flash-attn version detection in setup.py

### 1.2 Inference Testing (PARTIAL)

- **HuggingFace backend:** Works correctly ("What is 2+2?" → "2 + 2 equals 4")
- **ktransformers backend:** Model loads but produces garbled output (known issue)

### 1.3 Training Attempt (BLOCKED)

Training starts but crashes during backward pass:

```
TypeError: backward(): incompatible function arguments.
Expected: (self, arg0: int, arg1: int, arg2: int, arg3: int, arg4: int, arg5: int, arg6: int, arg7: int)
Got: 7 arguments instead of 8
```

**Root Cause:**
- `ktransformers/operators/experts.py` line ~653 calls `ctx.moe.backward()` with wrong arguments
- The `layer_idx` parameter is commented out in vanilla ktransformers v0.5.0
- The kt-sft fork has fixes, but C++ extensions need recompilation

### 1.4 Additional Fixes Applied to LLaMA-Factory

```python
# src/llamafactory/model/patcher.py line 178
# Fix: gen_config can be None when using ktransformers
if gen_config is not None and not gen_config.do_sample and (...)

# Also need to handle missing model.generate attribute
```

---

## 2. File Structure

```
/home/sean/Documents/ktransformers/
├── docker/
│   ├── Dockerfile.kt-llamafactory      # Current working image (v0.5.0)
│   └── Dockerfile.kt-sft-train         # WIP: kt-sft based image
│
├── kt-sft/                             # Modified ktransformers fork
│   ├── ktransformers/
│   │   ├── operators/
│   │   │   └── experts.py              # Contains FIXED backward()
│   │   └── optimize/
│   │       └── optimize_rules/
│   │           └── DeepSeek-V2-Lite-Chat-sft.yaml
│   ├── csrc/
│   │   └── ktransformers_ext/          # C++ source (needs rebuild)
│   ├── cpuinfer_ext.cpython-311-*.so   # Pre-built (may not be compatible)
│   └── setup.py
│
├── LLaMA-Factory/
│   ├── src/llamafactory/
│   │   ├── model/
│   │   │   └── patcher.py              # Needs gen_config None fix
│   │   ├── chat/
│   │   │   └── kt_engine.py            # Prompt templating fix applied
│   │   └── train/sft/
│   │       └── workflow.py             # Uses KTrainer for ktransformers
│   ├── examples/
│   │   └── train_lora/
│   │       └── deepseek2_lite_identity_sean.yaml
│   └── data/
│       ├── dataset_info.json
│       └── identity_sean_generated.json
│
└── deepseek-ai/
    └── DeepSeek-V2-Lite-Chat/          # Model files
```

---

## 3. TODOs for Future Exploration

### Priority 1: Fix C++ Extension Build

- [ ] Create proper Dockerfile that builds kt-sft C++ extensions from source
- [ ] Ensure CUDA toolkit is properly configured in Docker build
- [ ] Verify `cpuinfer_ext.sft_moe.SFT_MOE.backward()` signature matches Python call
- [ ] Test with `KSFT_MOE_DEBUG=1` environment variable for debugging

### Priority 2: Validate backward() Fix

- [ ] Compare kt-sft `experts.py` vs vanilla ktransformers v0.5.0
- [ ] Confirm `layer_idx` is passed as first argument to `ctx.moe.backward()`
- [ ] Verify all 8 arguments are passed correctly:
  ```python
  ctx.moe.backward(
      layer_idx,              # arg0: int
      qlen,                   # arg1: int
      k,                      # arg2: int
      expert_ids.data_ptr(),  # arg3: int (pointer)
      weights.data_ptr(),     # arg4: int (pointer)
      input_tensor.data_ptr(),# arg5: int (pointer)
      output_grad.data_ptr(), # arg6: int (pointer)
      input_grad.data_ptr(),  # arg7: int (pointer)
  )
  ```

### Priority 3: Docker Image Rebuild

- [ ] Update Dockerfile.kt-sft-train to:
  - Clone kt-sft with submodules
  - Build C++ extensions with correct CUDA version
  - Copy fixed Python files (operators/experts.py, patcher.py)
- [ ] Test training end-to-end in container

### Priority 4: Training Validation

- [ ] Run identity_sean_generated training with ktransformers
- [ ] Monitor GPU memory usage (should be ~10-12GB)
- [ ] Verify loss decreases properly
- [ ] Test inference with trained adapter

---

## 4. Important Warnings

### DO NOT modify the host system

All development and testing MUST happen inside Docker containers:
- The host conda environments (`deepspeed-z3`, etc.) should remain untouched
- Do not install/rebuild ktransformers on the host
- Changes should only be made to Docker images

### Container isolation pattern

```bash
# Always mount as read-only where possible
docker run --gpus all \
  -v /path/to/models:/workspace/models:ro \
  -v /path/to/data:/workspace/data:ro \
  -v /path/to/saves:/workspace/saves \  # writable for checkpoints
  kt-sft-train:latest
```

### Preserve working configurations

- `Dockerfile.kt-llamafactory` is WORKING for HuggingFace inference
- Do not modify it; create new Dockerfiles for experiments
- Keep `deepseek2_lite_sft_hf_z3_bf16_regularized.yaml` as backup

---

## 5. Constraints

### MUST use CPU offloading + GPU hybrid

ktransformers is specifically designed for:
- MoE experts on CPU
- Attention/embeddings on GPU
- Full precision (bf16) training

### DO NOT use:

| Approach | Reason |
|----------|--------|
| 4-bit QLoRA | Defeats purpose of ktransformers CPU offload |
| 8-bit quantization | Same as above |
| Full GPU loading | OOM on 16GB GPU |
| DeepSpeed ZeRO | Different framework, not ktransformers |
| Unsloth | Different framework |
| vLLM | Inference only |

### Target configuration

```yaml
# Correct ktransformers training config
use_kt: true
kt_optimize_rule: /path/to/DeepSeek-V2-Lite-Chat-sft.yaml
cpu_infer: 32
chunk_size: 4096
bf16: true
# NO quantization_bit, NO deepspeed
```

---

## 6. Quick Reference

### Build Docker image

```bash
cd /home/sean/Documents/ktransformers/docker
docker build -f Dockerfile.kt-llamafactory -t kt-llamafactory:latest ..
```

### Run training (once fixed)

```bash
docker run --gpus all --rm \
  -e WANDB_DISABLED=true \
  -v /home/sean/Documents/ktransformers/deepseek-ai:/workspace/models/deepseek-ai \
  -v /home/sean/Documents/ktransformers/kt-sft/ktransformers/optimize/optimize_rules:/workspace/optimize_rules \
  -v /home/sean/Documents/ktransformers/LLaMA-Factory/data:/workspace/LLaMA-Factory/data \
  -v /home/sean/Documents/ktransformers/LLaMA-Factory/saves:/workspace/saves \
  kt-sft-train:latest \
  llamafactory-cli train /workspace/train_config.yaml
```

### Debug backward issue

```bash
export KSFT_MOE_DEBUG=1
llamafactory-cli train config.yaml 2>&1 | tee debug.log
```

---

## 7. Known Issues Summary

| Issue | Status | Fix Location |
|-------|--------|--------------|
| ktransformers version import | FIXED | Dockerfile (create version.py) |
| gen_config None | FIXED | patcher.py line 178 |
| model.generate missing | FIXED | patcher.py line 185 |
| SFT_MOE.backward() args | BLOCKED | Need to rebuild C++ from kt-sft |
| Garbled inference output | UNKNOWN | May be related to above |

---

*Last updated: 2026-01-12*
