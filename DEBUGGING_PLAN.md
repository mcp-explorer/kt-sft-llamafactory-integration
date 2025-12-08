# Debugging Plan: gate_proj_ Memory Accessibility Issue

## Problem
`gate_proj_` pointer points to memory that's not accessible from C++ worker threads, causing segfault when `to_float()` tries to read from it during backward pass.

## Root Cause Hypothesis
The PyTorch tensor memory may be:
1. Not on CPU (on GPU)
2. Not contiguous
3. Memory-mapped file not accessible from worker threads
4. Thread-local memory not shared with worker threads

## Debugging Steps (Iterative)

### Step 1: Python Side Validation ✅ (IN PROGRESS)
**Goal**: Verify tensors are on CPU and contiguous before getting pointers

**Changes Made**:
- Added checks in `KSFTExpertsCPU.load()` to verify device and contiguity
- Force move to CPU and make contiguous if needed
- Add debug logging to show tensor properties

**Test**: Run training with `KSFT_MOE_DEBUG=1` and check logs

### Step 2: C++ Constructor Logging ✅ (IN PROGRESS)
**Goal**: Verify pointers are valid when stored in constructor

**Changes Made**:
- Add logging in `SFT_MOE::SFT_MOE()` constructor to show pointer values

**Test**: Check if pointers match Python-side values

### Step 3: Memory Accessibility Test in Main Thread ✅ (COMPLETED)
**Goal**: Test if memory is accessible from main thread before worker threads

**Changes Made**:
- Added memory accessibility test in `get_transpose()` before calling `do_work_stealing_job()`
- Uses `memcpy` to safely read first 16 bytes from `gate_proj_` in main thread
- Logs success/failure to help diagnose if issue is threading-specific

**Test**: Run training and check if memory test passes in main thread

### Step 4: Check Backend Threading Model
**Goal**: Understand how Backend worker threads access memory

**Plan**:
- Check `Backend::do_work_stealing_job()` implementation
- Verify if worker threads can access main thread's memory
- Check if memory needs to be in shared memory

### Step 5: Copy Data to CPU-Accessible Memory ✅ (COMPLETED)
**Goal**: If memory is not accessible, copy it to a safe location

**Changes Made**:
- In `SFT_MOE` constructor, allocate CPU-accessible memory using `std::aligned_alloc()`
- Copy expert weights from original pointers to CPU-accessible memory
- Update `gate_proj_`, `up_proj_`, `down_proj_` to point to copied memory
- Free allocated memory in destructor
- Added member variables to track ownership

**Test**: Run training - if memcpy in constructor works, copied memory should be accessible from worker threads

### Step 6: Verify Fix
**Goal**: Confirm the fix works

**Plan**:
- Run training and verify no segfault
- Check that backward pass completes successfully

## Next Actions

1. **Rebuild and test Step 1 & 2**:
   ```bash
   cd kt-sft
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate Kllama
   # No rebuild needed for Python changes (editable install)
   # Run training with KSFT_MOE_DEBUG=1
   ```

2. **Check logs** for:
   - Tensor device and contiguity warnings
   - Pointer values in Python vs C++
   - Any errors during load

3. **If Step 1/2 don't reveal issue**, proceed to Step 3

## Files Modified
- `kt-sft/ktransformers/operators/experts.py` - Added tensor validation
- `kt-sft/csrc/ktransformers_ext/operators/llamafile/sft_moe.cpp` - Added constructor logging

