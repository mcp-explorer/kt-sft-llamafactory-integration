# Additional Approaches to Fix Segfault

## Current Status
- Simple test cases work (constructor completes successfully)
- Real scenario segfaults during model loading
- GDB debugging blocked by Triton initialization
- Constructor logic appears correct

## Key Differences: SFT_MOE vs MOE
- **SFT_MOE has 40+ vector members** vs MOE's 21
- **SFT_MOE has `fw_cache_` member**: `std::vector<SFT_MoEForwardCache>` containing nested `std::vector<std::vector<float>>`
- **SFT_MOE is significantly larger** than MOE

## Proposed Approaches

### 1. Make `fw_cache_` a Pointer (HIGH PRIORITY)
**Problem**: `fw_cache_` is a direct member with nested vectors, making the object very large.

**Solution**: Change from:
```cpp
std::vector<SFT_MoEForwardCache> fw_cache_;
```
To:
```cpp
std::vector<SFT_MoEForwardCache>* fw_cache_;
```

**Benefits**:
- Reduces object size significantly
- Avoids pybind11 issues with large nested structures
- Minimal code changes (allocate in constructor, delete in destructor)

**Implementation**:
- Allocate `fw_cache_` with `new` in constructor
- Delete in destructor
- Update `fwd_cache_ptr()` to handle pointer
- Update `ensure_fwd_cache()` to use `->` instead of `.`

### 2. Factory Function Pattern
**Problem**: pybind11 might have issues with direct constructor of large objects.

**Solution**: Use a factory function that returns `std::unique_ptr<SFT_MOE>`:
```cpp
.def_static("create", [](SFT_MOEConfig config) {
    return std::make_unique<SFT_MOE>(config);
})
```

**Benefits**:
- pybind11 handles unique_ptr better than large objects
- Matches modern C++ patterns

### 3. Check Object Size and Alignment
**Problem**: Object might be too large or have alignment issues.

**Solution**: Add static_assert to check object size:
```cpp
static_assert(sizeof(SFT_MOE) < 1024 * 1024, "SFT_MOE object too large");
```

### 4. Create Minimal Reproduction
**Problem**: Can't test real scenario due to Triton initialization.

**Solution**: Create a test that:
- Loads actual model weights
- Creates multiple SFT_MOE instances in sequence
- Doesn't require Triton

### 5. Check for Stack Overflow
**Problem**: Constructor does a lot of work, might hit stack limits.

**Solution**: 
- Move large allocations to heap
- Check stack size with `ulimit -s`
- Use `alloca` or heap allocation for large temporary objects

### 6. Per-Object Buffers Instead of Shared Buffer
**Problem**: Shared buffer reallocation might cause pointer invalidation.

**Solution**: Each SFT_MOE object has its own buffer instead of sharing.

**Trade-offs**:
- More memory usage
- Simpler memory management
- No pointer invalidation issues

### 7. Lazy Initialization of fw_cache_
**Problem**: `fw_cache_` is initialized even if never used.

**Solution**: Only allocate `fw_cache_` when first needed (in `ensure_fwd_cache()`).

## Recommended Order
1. **Make `fw_cache_` a pointer** (easiest, highest impact)
2. **Factory function pattern** (if #1 doesn't work)
3. **Check object size** (diagnostic)
4. **Minimal reproduction** (for testing)
5. **Per-object buffers** (if shared buffer is the issue)

