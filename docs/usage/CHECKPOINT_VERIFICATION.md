# Checkpoint Verification

## Current Configuration

**Config File**: `examples/inference/deepseek2_lite_serve_custom.yaml`

```yaml
adapter_name_or_path: /app/saves/Kllama_deepseekV2Lite/checkpoint-11
```

## Verification Results

✅ **Using checkpoint-11** from `Kllama_deepseekV2Lite` folder

### Checkpoint Details

**Checkpoint-11**:
- **Step**: 11
- **Epoch**: ~0.97 (97% through training)
- **File**: `/app/saves/Kllama_deepseekV2Lite/checkpoint-11/adapter_model.safetensors`
- **Size**: 27 MB
- **MD5**: `73224ec50ee93ec105a10de3a614233b`

**Root Checkpoint** (for comparison):
- **Step**: 20 (final)
- **Epoch**: 1.0 (complete)
- **File**: `/app/saves/Kllama_deepseekV2Lite/adapter_model.safetensors`
- **Size**: 27 MB
- **MD5**: `ab101a287f6a76301d323fe2fe6a4ad8` (different from checkpoint-11)

### Available Checkpoints

- `checkpoint-5` - Step 5
- `checkpoint-10` - Step 10
- `checkpoint-11` - Step 11 (currently in use)
- Root checkpoint - Step 20 (final)

## Important Note

⚠️ **Checkpoint-11 is an intermediate checkpoint** (step 11 of 20)

The root checkpoint (`/app/saves/Kllama_deepseekV2Lite/adapter_model.safetensors`) contains the **final trained model** (step 20, epoch 1.0).

If you want to use the final checkpoint instead, change the config to:
```yaml
adapter_name_or_path: /app/saves/Kllama_deepseekV2Lite
```

## Model Loading Confirmation

The logs show:
```
Loaded adapter weight: base_model.model.model.orig_module.layers.0.mlp.down_proj...
[INFO] Loaded adapter(s): /app/saves/Kllama_deepseekV2Lite/checkpoint-11
```

This confirms checkpoint-11 is being loaded and used.

