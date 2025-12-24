# Does KTransformers Support CUDA 12.4 with PyTorch 2.6.0+cu124?

## Short Answer

**Partially supported:**
- ✅ **Minimum requirement met**: PyTorch 2.6.0+cu124 meets `torch >= 2.3.0` requirement
- ❌ **No pre-built wheel**: No official wheel for `cu124torch26` combination
- ✅ **Can build from source**: Yes, but requires workaround

## Detailed Analysis

### 1. Minimum Requirements (✅ Met)

From `kt-sft/pyproject.toml`:
```toml
dependencies = [
  "torch >= 2.3.0",  # ✅ 2.6.0 meets this
]
```

**Result**: PyTorch 2.6.0+cu124 meets the minimum requirement.

### 2. Pre-built Wheels (❌ Not Available)

Available wheels from v0.4.1 release:
- ✅ `cu124torch25` - CUDA 12.4 + PyTorch 2.5
- ✅ `cu126torch26` - CUDA 12.6 + PyTorch 2.6
- ✅ `cu128torch27` - CUDA 12.8 + PyTorch 2.7
- ❌ **NO `cu124torch26` wheel**

**Why?** PyTorch 2.6.0 is typically paired with CUDA 12.6, not 12.4. The official PyTorch builds for 2.6.0 use CUDA 12.6.

### 3. Build System Mapping

From `kt-sft/autosetup.sh` (the official build script):
```bash
2.5.*) echo "https://download.pytorch.org/whl/cu124" ;;  # PyTorch 2.5 → CUDA 12.4
2.6.*) echo "https://download.pytorch.org/whl/cu126" ;;  # PyTorch 2.6 → CUDA 12.6
2.7.*) echo "https://download.pytorch.org/whl/cu128" ;;  # PyTorch 2.7 → CUDA 12.8
```

**Official pairing:**
- PyTorch 2.5 → CUDA 12.4
- PyTorch 2.6 → CUDA 12.6
- PyTorch 2.7 → CUDA 12.8

### 4. Your Situation

**Container has**: PyTorch 2.6.0+cu124 (unusual combination)
- This comes from LLaMA-Factory's base image
- It's not the standard PyTorch 2.6.0 build (which uses cu126)
- KTransformers doesn't provide a pre-built wheel for this combination

## Solutions

### Option 1: Use Compatible Pre-built Wheel (Easiest)

Use the `cu126torch26` wheel - it should work with PyTorch 2.6.0+cu124:

```bash
# Download from: https://github.com/kvcache-ai/ktransformers/releases/tag/v0.4.1
# Use: ktransformers-0.4.1+cu126torch26fancy-cp311-cp311-linux_x86_64.whl

docker exec llamafactory bash -c "
pip install https://github.com/kvcache-ai/ktransformers/releases/download/v0.4.1/ktransformers-0.4.1%2Bcu126torch26fancy-cp311-cp311-linux_x86_64.whl
"
```

**Why this works**: CUDA 12.6 wheels are backward compatible with CUDA 12.4 runtime (as long as the driver supports it).

### Option 2: Build from Source (Current Approach)

Build from source with workaround:

```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.7;8.9;9.0+PTX"
cd /tmp/kt-sft-build
CPU_INSTRUCT=NATIVE KTRANSFORMERS_FORCE_BUILD=TRUE pip install . --no-build-isolation
```

**Status**: All C++ compilation errors are fixed. Only the build system CUDA detection issue remains (solved by setting `TORCH_CUDA_ARCH_LIST`).

### Option 3: Upgrade PyTorch to Match Official Pairing

Upgrade to PyTorch 2.6.0+cu126 (official pairing):

```bash
docker exec llamafactory bash -c "
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
"
```

Then use the `cu126torch26` wheel.

### Option 4: Downgrade to PyTorch 2.5.0+cu124

Use the officially supported combination:

```bash
docker exec llamafactory bash -c "
pip uninstall -y torch torchvision torchaudio
pip install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
"
```

Then use the `cu124torch25` wheel.

## Recommendation

**For immediate use**: **Option 1** - Use `cu126torch26` wheel (should work with your setup)

**For building from source**: **Option 2** - Continue with current approach, set `TORCH_CUDA_ARCH_LIST`

**For best compatibility**: **Option 3** - Upgrade to PyTorch 2.6.0+cu126 (official pairing)

## Verification

After installing, verify:

```bash
docker exec llamafactory python -c "
import ktransformers
print('KTransformers version:', ktransformers.__version__)
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
"
```

## Summary

| Aspect | Support Status |
|--------|---------------|
| Minimum requirement | ✅ Yes (torch >= 2.3.0) |
| Pre-built wheel | ❌ No (no cu124torch26 wheel) |
| Build from source | ✅ Yes (with workaround) |
| Compatible wheel | ✅ Yes (cu126torch26 should work) |
| Official pairing | ❌ No (2.6.0 pairs with cu126, not cu124) |

**Conclusion**: KTransformers technically supports PyTorch 2.6.0+cu124 (meets minimum requirements), but there's no pre-built wheel for this exact combination. Use a compatible wheel or build from source.

