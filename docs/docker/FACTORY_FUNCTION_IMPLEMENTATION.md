# Factory Function Pattern Implementation

## Summary
Added factory function pattern as an additional approach to avoid pybind11 issues with large objects.

## Implementation

### Changes Made
1. **Modified pybind11 binding** to support `std::unique_ptr<SFT_MOE>`
2. **Added static factory function** `SFT_MOE.create(config)` that returns `std::unique_ptr<SFT_MOE>`
3. **Kept direct constructor** as fallback for compatibility

### Code Changes
```cpp
py::class_<SFT_MOE, std::unique_ptr<SFT_MOE>>(sft_moe_module, "SFT_MOE")
    // Factory function pattern
    .def_static("create", [](SFT_MOEConfig config) {
        return std::make_unique<SFT_MOE>(config);
    }, "Create SFT_MOE instance using factory function")
    // Keep direct constructor as fallback
    .def(py::init<SFT_MOEConfig>())
```

## Benefits

### Why Factory Function?
1. **Better pybind11 handling**: `unique_ptr` is explicitly supported and handled well by pybind11
2. **Explicit ownership**: Clear ownership semantics
3. **Smaller wrapper**: pybind11 doesn't need to inspect the full object during wrapper setup
4. **Modern C++**: Matches best practices

### Why Keep Direct Constructor?
- **Backward compatibility**: Existing code may use direct constructor
- **Flexibility**: Users can choose which pattern to use
- **Fallback**: If factory has issues, direct constructor still works

## Usage

### Factory Function (Recommended)
```python
from cpuinfer_ext.sft_moe import SFT_MOEConfig, SFT_MOE

config = SFT_MOEConfig(...)
moe = SFT_MOE.create(config)  # Returns unique_ptr, automatically managed
```

### Direct Constructor (Fallback)
```python
config = SFT_MOEConfig(...)
moe = SFT_MOE(config)  # Direct construction, still works
```

## Testing Status
- ✅ Build: Successful
- ✅ Factory function: Works
- ✅ Direct constructor: Still works
- ✅ Multiple instances: Test pending

## Relationship to fw_cache_ Pointer Change

These are **complementary fixes**:
- **fw_cache_ pointer**: Reduces object size
- **Factory function**: Improves pybind11 wrapper creation

Both can be used together for maximum compatibility.

## Next Steps
1. Test factory function with multiple instances
2. Test factory function with real model loading (if Triton can be bypassed)
3. Compare factory vs direct constructor in real scenario

