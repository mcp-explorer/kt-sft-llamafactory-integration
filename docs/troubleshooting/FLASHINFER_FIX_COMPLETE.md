# FlashInfer Compatibility Fix - Complete

## Issues Fixed

### Issue 1: `bsz_tensor` in `__init__()`
**Error**: `TypeError: BatchMLAPagedAttentionWrapper.__init__() got an unexpected keyword argument 'bsz_tensor'`

**Fix**: Removed `bsz_tensor=self.batch_size_tensor_buf` from the `BatchMLAPagedAttentionWrapper.__init__()` call in `/opt/conda/lib/python3.11/site-packages/ktransformers/operators/flashinfer_wrapper.py` (line ~96).

### Issue 2: `bsz_tensor` in `plan()`
**Error**: `TypeError: BatchMLAPagedAttentionWrapper.plan() takes 13 positional arguments but 14 were given`

**Fix**: Removed `bsz_tensor` from the `self.wrapper.plan()` call in the `plan()` method (line ~149).

## Changes Made

1. **File**: `/opt/conda/lib/python3.11/site-packages/ktransformers/operators/flashinfer_wrapper.py`

2. **Change 1** - Removed from `__init__()`:
   ```python
   # BEFORE:
   self.wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
       self.float_workspace_buffer,
       use_cuda_graph=use_cuda_graph,
       qo_indptr=self.qo_indptr_buf,
       kv_indptr=self.kv_indptr_buf,
       kv_indices=self.kv_indices_buf,
       kv_len_arr=self.kv_len_arr_buf,
       bsz_tensor=self.batch_size_tensor_buf,  # ❌ REMOVED
       backend = "fa2",
   )
   
   # AFTER:
   self.wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
       self.float_workspace_buffer,
       use_cuda_graph=use_cuda_graph,
       qo_indptr=self.qo_indptr_buf,
       kv_indptr=self.kv_indptr_buf,
       kv_indices=self.kv_indices_buf,
       kv_len_arr=self.kv_len_arr_buf,
       backend = "fa2",
   )
   ```

3. **Change 2** - Removed from `plan()`:
   ```python
   # BEFORE:
   self.wrapper.plan(
       qo_indptr,
       kv_indptr,
       kv_indices,
       kv_len_arr,
       num_heads,
       head_dim_ckv,
       head_dim_kpe,
       page_size,
       True, # causal
       sm_scale,
       q_data_type,
       kv_data_type,
       bsz_tensor  # ❌ REMOVED
   )
   
   # AFTER:
   self.wrapper.plan(
       qo_indptr,
       kv_indptr,
       kv_indices,
       kv_len_arr,
       num_heads,
       head_dim_ckv,
       head_dim_kpe,
       page_size,
       True, # causal
       sm_scale,
       q_data_type,
       kv_data_type
   )
   ```

## Verification

✅ `MLAWrapper.__init__()` works  
✅ `MLAWrapper.plan()` works  
✅ Model loads successfully  
✅ Model generates output (no errors)

## Status

**Model is now fully functional!** The DeepSeek-V2-Lite model with checkpoint-11 adapter is working correctly in the Docker container.

## Note

The first run may take longer due to JIT compilation of flashinfer kernels. Subsequent runs will be faster.

