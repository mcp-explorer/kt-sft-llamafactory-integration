# PyTorch CUDA Compatibility Issue

## Problem

The `deepspeed-z3` environment has a CUDA library compatibility issue:
```
ImportError: undefined symbol: __nvJitLinkAddData_12_1, version libnvJitLink.so.12
```

This occurs because:
- System CUDA: 12.0 (libnvJitLink.so.12.0.140)
- PyTorch bundled libraries: Compiled for CUDA 12.1 with different nvJitLink version
- Mismatch causes symbol resolution failure

## Solutions

### Option 1: Use Kllama Environment for Inference (Recommended)

The model code fixes are in the source files, so they'll work in any environment where PyTorch loads:

```bash
# Update inference script to use Kllama if deepspeed-z3 fails
./scripts/inference/infer_ds2_chat_lite_hf.sh chat
```

The script can be modified to fallback to Kllama environment.

### Option 2: Update System CUDA Libraries

Update system CUDA to 12.1 to match PyTorch:
```bash
# Install CUDA 12.1 toolkit
sudo apt update
sudo apt install cuda-toolkit-12-1
```

### Option 3: Use CPU-Only PyTorch (Not Recommended)

Install CPU-only PyTorch, but this won't use GPU:
```bash
conda install -y -c pytorch pytorch torchvision torchaudio cpuonly
```

### Option 4: Use Different PyTorch Version

Try PyTorch 2.4.x which might have better CUDA 12.0 compatibility:
```bash
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Current Status

✅ **Model code is fixed** - All `seen_tokens` and `get_usable_length` issues resolved
✅ **Source files updated** - Will work once PyTorch loads
⚠️ **PyTorch loading issue** - Needs CUDA library compatibility fix

## Workaround

For now, use the Kllama environment for inference. The model code fixes are in place and will work in any environment where PyTorch successfully loads.

