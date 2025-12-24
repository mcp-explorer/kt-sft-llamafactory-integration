# Segmentation Fault During Model Loading

## Issue
When loading the DeepSeek-V2-Lite-Chat model with KTransformers, a segmentation fault occurs during the operator injection phase (around layer 22/27).

## Symptoms
- Model starts loading successfully
- Tokenizer loads correctly
- Configuration loads correctly
- KTransformers operator injection begins
- Progress: Layers 0-22 injected successfully
- **Crash**: Segmentation fault during layer 22/23 injection

## Error Output
```
bash: line 1:  5429 Segmentation fault      (core dumped) | llamafactory-cli chat ...
```

## System Status
- ✅ GPU: NVIDIA GeForce RTX 4080 SUPER (16GB)
- ✅ GPU Memory: 163 MiB used / 16376 MiB total (plenty available)
- ✅ CUDA: 12.4 available
- ✅ PyTorch: 2.6.0+cu126
- ✅ KTransformers: 0.4.1 (built from source)

## Possible Causes

### 1. C++ Extension Compatibility
The KTransformers C++ extensions were built from source. There might be:
- ABI compatibility issues
- Memory alignment problems
- CUDA kernel launch issues

### 2. Model-Specific Issue
The DeepSeek-V2 model uses:
- Mixture of Experts (MoE) architecture
- 64 routed experts + 2 shared experts
- Complex attention mechanisms
- YarnRotaryEmbedding

The segfault might be specific to how KTransformers handles these components.

### 3. Memory Issue
Although GPU memory looks fine, there could be:
- Stack overflow in C++ code
- Heap corruption
- CUDA context issues

## What Works
✅ KTransformers imports successfully  
✅ `cpuinfer_ext` loads without errors  
✅ Model configuration loads  
✅ Tokenizer loads  
✅ KTransformers operator injection starts (layers 0-22)  

## What Fails
❌ Segmentation fault during layer 22/23 injection  
❌ Cannot complete model loading  
❌ Cannot test chat functionality  

## Next Steps to Investigate

### 1. Check KTransformers Version Compatibility
```bash
# Check if there's a known issue with DeepSeek-V2
# Try a different model to see if issue is model-specific
```

### 2. Build with Debug Symbols
```bash
# Rebuild KTransformers with debug symbols
DEBUG=1 CPU_INSTRUCT=NATIVE KTRANSFORMERS_FORCE_BUILD=TRUE pip install . --no-build-isolation
```

### 3. Use GDB to Debug
```bash
# Run with GDB to get stack trace
docker exec llamafactory bash -c "gdb --batch --ex run --ex bt --args python -m llamafactory.cli chat examples/inference/deepseek2_lite_serve_custom.yaml"
```

### 4. Try Without KTransformers
```bash
# Test with HuggingFace backend to verify model loads
# Change config: infer_backend: huggingface
```

### 5. Check for Known Issues
- Check KTransformers GitHub issues
- Check LLaMA-Factory issues related to DeepSeek-V2
- Check CUDA 12.4 compatibility issues

## Workaround
Until the segfault is resolved, you can:
1. Use HuggingFace backend instead of KTransformers
2. Try a different model that's known to work with KTransformers
3. Wait for KTransformers updates that might fix the issue

## Related Files
- Config: `examples/inference/deepseek2_lite_serve_custom.yaml`
- KTransformers build: Built from source in `/tmp/kt-sft-build`
- Logs: Check Docker logs with `docker logs llamafactory`

## Test Command That Fails
```bash
docker exec llamafactory bash -c "cd /app && llamafactory-cli chat examples/inference/deepseek2_lite_serve_custom.yaml"
```

## Status
🔴 **Blocked**: Cannot test chat functionality due to segmentation fault during model loading.

