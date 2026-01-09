#!/usr/bin/env python3
"""Inspect PEFT key structure"""

import torch
import warnings
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
from safetensors import safe_open

warnings.filterwarnings('ignore', category=UserWarning)

print("=" * 60)
print("PEFT Key Structure Inspection")
print("=" * 60)

# Load base model
print("\n1. Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    '/home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# Load adapter config
adapter_path = '/home/sean/Documents/ktransformers/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf'
print("\n2. Loading adapter config...")
config = PeftConfig.from_pretrained(adapter_path)

# Create a fresh PEFT model to see expected structure
print("\n3. Creating fresh PEFT model with same config...")
lora_config = LoraConfig(
    r=config.r,
    lora_alpha=config.lora_alpha,
    target_modules=config.target_modules,
    lora_dropout=config.lora_dropout,
    bias=config.bias,
    task_type=config.task_type,
)
peft_model = get_peft_model(base, lora_config)

# Get expected keys
expected_keys = set([k for k in peft_model.state_dict().keys() if 'lora' in k.lower()])
print(f"   Expected {len(expected_keys)} LoRA keys")

# Get actual adapter keys
print("\n4. Reading adapter file keys...")
adapter_file = f'{adapter_path}/adapter_model.safetensors'
with safe_open(adapter_file, framework='pt', device='cpu') as f:
    adapter_keys = set(f.keys())
print(f"   Adapter file has {len(adapter_keys)} keys")

# Compare
missing = expected_keys - adapter_keys
extra = adapter_keys - expected_keys

print(f"\n5. Comparison:")
print(f"   Missing keys (expected but not in adapter): {len(missing)}")
if missing:
    print("   Sample missing keys (first 5):")
    for k in sorted(list(missing))[:5]:
        print(f"     {k}")
        
print(f"   Extra keys (in adapter but not expected): {len(extra)}")
if extra:
    print("   Sample extra keys (first 5):")
    for k in sorted(list(extra))[:5]:
        print(f"     {k}")

# Check a specific key
test_key = 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'
print(f"\n6. Testing specific key: {test_key}")
print(f"   In expected keys: {test_key in expected_keys}")
print(f"   In adapter keys: {test_key in adapter_keys}")

# Now try loading the adapter
print("\n7. Loading adapter into model...")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    loaded_model = PeftModel.from_pretrained(base, adapter_path)
    
    missing_warnings = [str(warn.message) for warn in w if 'missing' in str(warn.message).lower()]
    if missing_warnings:
        print(f"   ⚠ Got {len(missing_warnings)} missing key warnings")
        # Extract first few missing keys from warning
        warning_text = missing_warnings[0]
        import re
        keys_in_warning = re.findall(r"base_model\.model\.model\.layers\.\d+\.\w+\.\w+\.lora_[AB]\.default\.weight", warning_text)
        if keys_in_warning:
            print(f"   Found {len(keys_in_warning)} key patterns in warning")
            print("   First 3 keys from warning:")
            for k in keys_in_warning[:3]:
                print(f"     {k}")
                print(f"       In adapter file: {k in adapter_keys}")
                print(f"       In expected keys: {k in expected_keys}")
    else:
        print("   ✓ No missing key warnings!")

print("\n" + "=" * 60)
print("Inspection Complete")
print("=" * 60)

