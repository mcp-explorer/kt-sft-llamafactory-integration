#!/usr/bin/env python3
"""Simple test to verify adapter loads and works"""

import sys
sys.path.insert(0, 'LLaMA-Factory/src')

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    '/home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    '/home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat',
    trust_remote_code=True
)

print("Loading adapter...")
adapter_path = '/home/sean/Documents/ktransformers/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf'
model = PeftModel.from_pretrained(base_model, adapter_path)
print("✓ Adapter loaded")

# Test inference
print("\nTesting inference...")
prompt = "Who are you?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

print(f"Prompt: {prompt}")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"Response: {response}")

if "sean" in response.lower():
    print("✓ PASS: Model identifies as 'sean'")
else:
    print("✗ FAIL: Model does not identify as 'sean'")

