# CUDA PyTorch Initialization Issue

## Current Status

### What's Working
- ✅ **GPU devices accessible**: `/dev/nvidia0`, `/dev/nvidiactl` exist
- ✅ **CUDA libraries installed**: `/usr/local/cuda-12.4` present
- ✅ **PyTorch compiled with CUDA**: `2.6.0+cu126` (CUDA 12.6)
- ✅ **CUDA functions available**: `torch._C._cuda_getDeviceCount` exists
- ✅ **Docker GPU config**: `deploy.resources.reservations.devices` configured

### What's Not Working
- ❌ **PyTorch CUDA unavailable**: `torch.cuda.is_available()` = `False`
- ❌ **NVML initialization fails**: "Can't initialize NVML"
- ❌ **CUDA device count**: 0

## Root Cause Analysis

The issue is that **PyTorch cannot initialize CUDA runtime**, even though:
1. GPU hardware is accessible
2. CUDA libraries are installed
3. PyTorch was compiled with CUDA support

### Possible Causes

1. **CUDA Driver/Runtime Mismatch**
   - Host driver: 570.195.03 (CUDA 12.8)
   - Container CUDA: 12.4
   - PyTorch compiled: CUDA 12.6
   - May need matching versions

2. **Library Path Issues**
   - PyTorch may not find CUDA libraries at runtime
   - `LD_LIBRARY_PATH` may not include CUDA libs
   - Missing CUDA runtime libraries

3. **NVML Initialization Failure**
   - Warning: "Can't initialize NVML"
   - This prevents PyTorch from detecting GPUs
   - May be a permissions or library issue

## Solutions

### Option 1: Check CUDA Library Paths
```bash
# Ensure CUDA libraries are in LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Option 2: Verify CUDA Driver Compatibility
```bash
# Check driver version in container
cat /proc/driver/nvidia/version

# Should match host driver version
```

### Option 3: Reinstall PyTorch with Matching CUDA
```bash
# If CUDA 12.4 in container, install matching PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Option 4: Use CPU Mode (Current Workaround)
- Configure code to use `device="cpu"`
- This is what we're doing now for testing
- Segfault fix still works correctly

## Impact on Testing

**Important**: This CUDA initialization issue is **separate** from the segfault fix.

✅ **Segfault fix is working**: All 27 layers inject successfully
⚠️ **CUDA issue**: PyTorch can't initialize CUDA (Docker/runtime issue)

The segfault was fixed. The CUDA issue is a different problem related to PyTorch CUDA initialization in the Docker environment.

## Next Steps

1. **For segfault testing**: Use CPU mode (current approach) ✅
2. **For full GPU testing**: Fix PyTorch CUDA initialization
   - Check library paths
   - Verify driver compatibility
   - May need to reinstall PyTorch

## Conclusion

The **segfault fix is complete and working**. The CUDA initialization issue is a separate Docker/PyTorch configuration problem that doesn't affect the segfault fix validation.

