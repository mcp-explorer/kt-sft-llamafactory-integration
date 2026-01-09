# Example PEFT/LoRA Adapters for DeepSeek V2 Lite Chat

Based on web search, here are example PEFT adapters for DeepSeek V2 Lite Chat:

## HuggingFace Model Hub Examples

### 1. wuchen01/DeepSeek-V2-Lite-Chat-All-LoRA
- **Repository**: https://huggingface.co/wuchen01/DeepSeek-V2-Lite-Chat-All-LoRA
- **Training Details**:
  - Learning rate: 0.0001
  - Epochs: 10
  - Batch size: 1 (train), 8 (eval)
  - Optimizer: AdamW
- **Format**: Standard PEFT LoRA adapter

### 2. deevade/DeepSeek-V2-Lite-Chat-finetuned
- **Repository**: https://huggingface.co/deevade/DeepSeek-V2-Lite-Chat-finetuned
- **Training Details**:
  - Learning rate: 0.0002
  - Epochs: 1
  - Batch size: 1 (train and eval)
  - Optimizer: AdamW
- **Format**: Standard PEFT LoRA adapter

## Typical PEFT Adapter Structure

### Files:
1. **adapter_model.safetensors** - The adapter weights
2. **adapter_config.json** - Configuration file

### Key Format:
For DeepSeek V2 Lite Chat models, the typical key format is:
```
base_model.model.model.layers.X.self_attn.q_proj.lora_A.default.weight
base_model.model.model.layers.X.self_attn.q_proj.lora_B.default.weight
```

### Config Structure:
```json
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 16,
  "target_modules": ["q_proj", "v_proj", "o_proj", ...],
  "lora_dropout": 0.0,
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

## Comparison with Our Converted Adapter

Our converted adapter (`LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf`) uses the same format:
- ✅ Key format: `base_model.model.model.layers.X...lora_A.default.weight`
- ✅ Config structure: Standard PEFT format
- ✅ File structure: `adapter_model.safetensors` + `adapter_config.json`

## Target Modules for DeepSeek V2 Lite

Typical target modules include:
- `q_proj` - Query projection
- `kv_a_proj_with_mqa` - Key-Value A projection with MQA
- `kv_b_proj` - Key-Value B projection
- `o_proj` - Output projection
- `mlp.gate_proj` - MLP gate projection
- `mlp.up_proj` - MLP up projection
- `mlp.down_proj` - MLP down projection
- `shared_experts.gate_proj` - Shared experts gate projection
- `shared_experts.up_proj` - Shared experts up projection
- `shared_experts.down_proj` - Shared experts down projection

