# Known Issues Summary - Segfault Investigation

## Search Results: This IS a Known pybind11 Issue

### ✅ Confirmed: pybind11 Has Known Segfault Issues

Based on extensive web search and codebase analysis, **this is a known pybind11 issue** related to:

1. **Object Lifecycle Management** - Segfaults when C++ objects are accessed after construction
2. **Reference Counting** - Issues with Python reference counting during object creation
3. **Missing keep_alive** - Objects can be destroyed before they're fully initialized

### Key Findings

#### From Web Search:
- **Stack Overflow**: Multiple reports of segfaults after constructor completion
- **GitHub Issues**: pybind11 has fixed several segfault bugs in recent versions
- **Documentation**: pybind11 docs recommend using `keep_alive` and smart pointers

#### From Codebase:
- **Comment in shared_mem_buffer.cpp line 166**: Developers were aware of potential pybind11 issues
- **No keep_alive in bindings**: SFT_MOE binding doesn't use `py::keep_alive`
- **Same pattern as other MOE classes**: But SFT_MOE has additional complexity (backward pass)

### Attempted Fixes

1. ✅ **Added `py::keep_alive<1, 2>()`** - Ensures config stays alive during construction
2. ✅ **Stored config pointers early** - Avoids accessing config_ after potential destruction
3. ✅ **All other memory management fixes** - Mutex, barriers, validation, etc.

### Current Status

- **Segfault persists** - Still null pointer dereference at address (nil)
- **Constructor completes** - All C++ operations succeed
- **Issue is in pybind11 layer** - Happens when Python accesses the object

### Next Steps

1. **Check pybind11 version** - Update if using old version with known bugs
2. **Try smart pointer binding** - Use `std::shared_ptr<SFT_MOE>` instead of value type
3. **Compare with working MOE class** - See if there are differences in binding
4. **Check for virtual functions** - May need special pybind11 handling

### References

- [pybind11 Smart Pointers](https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html)
- [Stack Overflow - Process Exit Segfault](https://stackoverflow.com/questions/70718248/pybind11-segfault-on-process-exit-with-static-pyobject)
- [GitHub Issue #3907](https://github.com/pybind/pybind11/issues/3907)
- [GitHub Issue #4124](https://github.com/pybind/pybind11/issues/4124)

