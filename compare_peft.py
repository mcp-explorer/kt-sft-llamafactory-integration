#!/usr/bin/env python3
"""Compare standard PEFT adapter with our converted adapter"""

import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
import json
import os

print("=" * 60)
print("Creating Standard PEFT Adapter")
print("=" * 60)

# Load base model
base = AutoModelForCausalLM.from_pretrained(
    '/home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# Create standard PEFT adapter with minimal config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['q_proj'],
    lora_dropout=0.0,
    bias='none',
    task_type='CAUSAL_LM'
)

peft_model = get_peft_model(base, lora_config)

# Save it (need to move to CPU first)
std_path = '/tmp/std_peft_adapter'
os.makedirs(std_path, exist_ok=True)
# Get state dict and move to CPU
state_dict = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in peft_model.state_dict().items()}
# Save manually
from safetensors.torch import save_file
save_file(state_dict, f"{std_path}/adapter_model.safetensors")
# Save config
import json
with open(f"{std_path}/adapter_config.json", 'w') as f:
    json.dump(peft_model.peft_config['default'].to_dict(), f, indent=2)

print(f"\n✓ Saved standard PEFT adapter to {std_path}")

# Now compare
print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)

# Standard adapter keys
std_file = f"{std_path}/adapter_model.safetensors"
print("\n1. Standard PEFT adapter keys (first 5):")
with safe_open(std_file, framework='pt', device='cpu') as f:
    std_keys = list(f.keys())[:5]
    for k in std_keys:
        print(f"   {k}")

# Our converted adapter keys
our_file = '/home/sean/Documents/ktransformers/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf/adapter_model.safetensors'
print("\n2. Our converted adapter keys (first 5):")
with safe_open(our_file, framework='pt', device='cpu') as f:
    our_keys = list(f.keys())[:5]
    for k in our_keys:
        print(f"   {k}")

# Compare formats
print("\n3. Format Comparison:")
if std_keys and our_keys:
    std_key = std_keys[0]
    our_key = our_keys[0]
    print(f"   Standard: {std_key}")
    print(f"   Ours:     {our_key}")
    
    # Check if they match
    if std_key == our_key:
        print("   ✓ Formats match exactly!")
    else:
        print("   ✗ Formats differ!")
        
        # Find differences
        std_parts = std_key.split('.')
        our_parts = our_key.split('.')
        print(f"\n   Detailed comparison:")
        max_len = max(len(std_parts), len(our_parts))
        for i in range(max_len):
            std_part = std_parts[i] if i < len(std_parts) else None
            our_part = our_parts[i] if i < len(our_parts) else None
            if std_part != our_part:
                print(f"     Part {i}: Standard='{std_part}' vs Ours='{our_part}'")

# Compare configs
print("\n4. Config Comparison:")
std_config_file = f"{std_path}/adapter_config.json"
our_config_file = '/home/sean/Documents/ktransformers/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf/adapter_config.json'

with open(std_config_file) as f:
    std_config = json.load(f)
with open(our_config_file) as f:
    our_config = json.load(f)

print("\n   Key differences in adapter_config.json:")
for key in sorted(set(list(std_config.keys()) + list(our_config.keys()))):
    if key == 'base_model_name_or_path':
        continue
    std_val = std_config.get(key, 'MISSING')
    our_val = our_config.get(key, 'MISSING')
    if std_val != our_val:
        print(f"     {key}:")
        print(f"       Standard: {std_val}")
        print(f"       Ours:     {our_val}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

