# GDB Debugging Results

## Summary

GDB debugging was attempted to get the exact line number of the segfault. However, the segfault only occurs during actual model loading, not in simple test cases.

## Key Findings

### 1. Simple Test Case Works
When testing with dummy pointers (`0x1000`, `0x2000`, `0x3000`), the `SFT_MOE` constructor:
- Completes successfully
- All memory allocations succeed
- All pointers are valid
- Object is created and destroyed without issues

**Test command:**
```python
from cpuinfer_ext.sft_moe import SFT_MOEConfig, SFT_MOE
config = SFT_MOEConfig(64, 2, 7168, 2048, 64, 10, 1024, 0x1000, 0x2000, 0x3000, 0, 0, 0, 0)
moe = SFT_MOE(config)  # ✓ Success
```

### 2. Real Scenario Cannot Be Tested with GDB
The actual `llamafactory-cli chat` command fails before reaching the `SFT_MOE` constructor due to:
- Triton initialization error: `RuntimeError: 0 active drivers ([]). There should only be one.`
- This prevents the code from even loading the model and creating `SFT_MOE` instances

### 3. Debug Symbols Enabled
- CMakeLists.txt was modified to build with `-g -O0` flags
- GDB was installed in the Docker container
- Debug symbols are available for debugging

## Conclusion

The segfault appears to be:
1. **Environment-specific**: Only occurs during actual model loading, not in isolated tests
2. **Timing-related**: May be related to the interaction between multiple `SFT_MOE` instances being created in sequence
3. **Memory-related**: Likely related to the shared memory buffer or object lifecycle during model loading

## Next Steps

Since GDB cannot easily capture the segfault in the real scenario due to Triton initialization issues, alternative approaches:

1. **Add more detailed logging** around the point where segfault occurs (during model layer injection)
2. **Use AddressSanitizer** with the actual command (if Triton issue can be bypassed)
3. **Create a minimal reproduction** that mimics the real model loading scenario without Triton dependencies
4. **Check for stack overflow** by monitoring stack size during constructor execution
5. **Investigate pybind11 object wrapper** creation with large/complex objects

## GDB Commands Used

```bash
# Simple test (works)
gdb --batch --ex 'set confirm off' --ex 'handle SIGSEGV stop print' --ex 'run' --ex 'bt' --ex 'frame 0' --ex 'info symbol $pc' --ex 'x/10i $pc' --ex 'quit' --args python3 -c "from cpuinfer_ext.sft_moe import SFT_MOEConfig, SFT_MOE; config = SFT_MOEConfig(...); moe = SFT_MOE(config)"

# Real scenario (blocked by Triton)
gdb --batch --ex 'set confirm off' --ex 'handle SIGSEGV stop print' --ex 'run' --ex 'bt' --ex 'frame 0' --ex 'info symbol $pc' --ex 'x/10i $pc' --ex 'quit' --args python3 -m llamafactory.cli chat examples/inference/deepseek2_lite_serve_custom.yaml
```

