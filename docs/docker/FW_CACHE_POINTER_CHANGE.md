# fw_cache_ Pointer Change Implementation

## Summary
Changed `fw_cache_` from a direct member (`std::vector<SFT_MoEForwardCache>`) to a pointer (`std::vector<SFT_MoEForwardCache>*`) to reduce object size and potentially fix pybind11 segfault issues.

## Changes Made

### 1. Header File (`sft_moe.h`)
- Changed from: `std::vector<SFT_MoEForwardCache> fw_cache_;`
- Changed to: `std::vector<SFT_MoEForwardCache>* fw_cache_;`
- Added comment explaining the change

### 2. Constructor (`sft_moe.cpp`)
- Allocate `fw_cache_` on heap: `fw_cache_ = new std::vector<SFT_MoEForwardCache>();`
- Updated debug logging to handle pointer

### 3. Destructor (`sft_moe.cpp`)
- Added cleanup: `delete fw_cache_; fw_cache_ = nullptr;`

### 4. `ensure_fwd_cache()` Method
- Added null check: `if (fw_cache_ == nullptr) { fw_cache_ = new std::vector<SFT_MoEForwardCache>(); }`
- Changed all access from `.` to `->` or `(*fw_cache_)`

### 5. `fwd_cache_ptr()` Method
- Updated to handle pointer: `return (fw_cache_ == nullptr || fw_cache_->empty()) ? nullptr : fw_cache_->data();`

## Benefits
1. **Reduced Object Size**: The object itself is now smaller (pointer is 8 bytes vs potentially large vector)
2. **Avoids pybind11 Issues**: Large nested vector structures can cause pybind11 wrapper setup issues
3. **Lazy Allocation**: Can be allocated when needed (though currently allocated in constructor)

## Build Status
✅ **Build Successful**: The C++ code compiles and links successfully
- Library built: `/kt-sft/build/lib.linux-x86_64-cpython-311/cpuinfer_ext.cpython-311-x86_64-linux-gnu.so`
- Size: 12MB
- All changes compile without errors

## Testing Status
⚠️ **Blocked by Triton**: Cannot test actual segfault fix due to Triton initialization error:
```
RuntimeError: 0 active drivers ([]). There should only be one.
```

This error occurs before the code reaches the `SFT_MOE` constructor, preventing us from verifying if the pointer change fixes the segfault.

## Next Steps
1. **Bypass Triton**: Find a way to test without Triton dependencies
2. **Create Minimal Test**: Create a test that loads actual model weights without Triton
3. **Alternative Approaches**: If pointer change doesn't work, try:
   - Factory function pattern
   - Per-object buffers
   - Check object size/alignment

## Code Quality
- All pointer accesses are null-checked
- Memory is properly allocated and deallocated
- Debug logging updated to handle pointer
- Matches C++ best practices for pointer usage

