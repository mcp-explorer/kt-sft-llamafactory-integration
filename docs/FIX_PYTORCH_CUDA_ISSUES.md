# Fix PyTorch CUDA Issues in Conda Environment

## Problem 1: `undefined symbol: __nvJitLinkAddData_12_1`

**Symptom:**
```
ImportError: undefined symbol: __nvJitLinkAddData_12_1, version libnvJitLink.so.12
```

**Cause:**
- PyTorch's bundled CUDA libraries (cusparse) require CUDA 12.1 symbols
- System has CUDA 12.0 libraries (libnvJitLink.so.12.0.140)
- Version mismatch causes symbol resolution failure

**Solution:**
Install CUDA 12.1 libraries in conda environment:
```bash
conda install -y -c nvidia libnvjitlink=12.1.105
```

Update activation script to prioritize conda's libnvjitlink:
```bash
# In conda env activate.d script:
export LD_PRELOAD="${CONDA_PREFIX}/lib/libnvJitLink.so.12.1.105:${LD_PRELOAD}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
```

## Problem 2: `undefined symbol: iJIT_NotifyEvent`

**Symptom:**
```
ImportError: libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```

**Cause:**
- MKL 2024.1+ (and 2025.0.0) has compatibility issues with PyTorch
- PyTorch expects Intel ITT (Instrumentation and Tracing Technology) symbols
- Newer MKL versions don't provide these symbols correctly

**Solution:**
Downgrade MKL to 2024.0.0:
```bash
conda install -y -c conda-forge mkl=2024.0.0
```

**Alternative Solutions (if downgrade doesn't work):**
1. Install Intel ITT library (if available):
   ```bash
   conda install -c intel intel-ittnotify
   ```
   Note: Intel channel may not be accessible (403 Forbidden)

2. Create stub library (workaround):
   ```c
   // itt_stub.c
   void iJIT_NotifyEvent(void) {}
   void iJIT_IsProfilingActive(void) {}
   ```
   ```bash
   gcc -shared -fPIC -o libittnotify_stub.so itt_stub.c
   export LD_PRELOAD="./libittnotify_stub.so:$LD_PRELOAD"
   ```

## Complete Fix Script

Add to `scripts/deepspeed/rebuild_deepspeed_z3_env.sh`:

```bash
# Install CUDA 12.1 libraries
conda install -y -c nvidia libnvjitlink=12.1.105

# Fix MKL compatibility
conda install -y -c conda-forge mkl=2024.0.0
```

## Verification

After applying fixes, verify:
```bash
conda activate deepspeed-z3
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
```

## References

- [PyTorch Issue #123097](https://github.com/pytorch/pytorch/issues/123097)
- [Anaconda Forum Discussion](https://forum.anaconda.com/t/linux-conda-importerror-undefined-symbol-ijit-notifyevent-when-importing-pytorch-in-a-conda-env-fix-without-vtune/107794)

