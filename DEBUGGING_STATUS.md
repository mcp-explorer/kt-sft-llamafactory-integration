# KTransformers + LlamaFactory Debugging Status

## Status: NOT SUCCESSFUL ❌

**Date:** 2025-12-31  
**Issue:** DeepSeek-V2-Lite-Chat model segfaults when using KTransformers CPU offloading

## Problem Summary

The program starts and shows "User: Assistant:" but segfaults before generating any tokens when using KTransformers with CPU offloading enabled.

## Root Cause Analysis

1. **B Pointer Issue**: The `B` pointer passed to `llamafile_sgemm` is often all zeros, causing incorrect computation results and potential memory corruption.

2. **Check Code Not Executing**: 
   - Added B pointer check code in `llamafile_sgemm.cpp` (lines 2816-2843)
   - Code is present in source and compiled into library (verified with `strings`)
   - Check code is NOT executing (no "TEST: About to check B bits" or "Checking B bits" logs appear)
   - Possible reasons:
     - Code path not being reached
     - Compiler optimization removing the code
     - Compilation issue

3. **Fallback Code Not Executing**:
   - Added fallback computation in `sft_moe.cpp` with `if (true)` to force execution
   - Fallback code is NOT executing (no "BF16 DETECTED" logs appear)
   - Suggests the code path in `forward_one` or `forward_many` is not being reached

## Attempted Fixes

1. ✅ Added B pointer check in `llamafile_sgemm.cpp` to detect all-zeros B and return `false`
2. ✅ Added fallback computation in `sft_moe.cpp` for when `llamafile_sgemm` returns `false`
3. ✅ Added extensive debug logging throughout the code path
4. ✅ Verified library was rebuilt with latest changes
5. ✅ Moved check code to different locations to ensure execution
6. ❌ Check code still not executing despite being compiled

## Files Modified

- `kt-sft/third_party/llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp`
  - Added B pointer check after "Pointers" log (lines 2816-2843)
  - Check should detect all-zeros B and return `false` to trigger fallback

- `kt-sft/csrc/ktransformers_ext/operators/llamafile/sft_moe.cpp`
  - Added fallback computation with `if (true)` to force execution (lines 619-691)
  - Fallback should compute BF16 matrix multiplication directly

## Current Behavior

- Program starts successfully
- Shows "User: Assistant:" prompt
- Segfaults during inference before generating tokens
- No "TEST: About to check B bits" logs (check code not executing)
- No "BF16 DETECTED" logs (fallback code not executing)

## Next Steps (Not Attempted)

1. Use GDB to get exact segfault location and backtrace
2. Verify library is being loaded correctly (check `LD_LIBRARY_PATH`)
3. Check compiler optimization flags that might remove the code
4. Consider alternative approach to detect and handle B pointer issue
5. Check if there are multiple code paths or if the function is being called from different locations

## Conclusion

The debugging attempt was **NOT SUCCESSFUL**. The fix code is in place and compiled, but the execution path is not being taken, causing the segfault to persist. Further investigation is needed to understand why the check code is not executing despite being present in the compiled library.

