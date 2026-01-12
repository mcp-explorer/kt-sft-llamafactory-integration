#!/usr/bin/env python3
"""
Evaluate Raw Model vs Adapter Model
Compare responses to assess capability impact of SFT
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "LLaMA-Factory" / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def load_model(model_path, adapter_path=None):
    """Load model with optional adapter"""
    print(f"Loading base model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    if adapter_path:
        print(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("✅ Adapter loaded")
    else:
        print("✅ Base model loaded (no adapter)")
    
    return model, tokenizer

def generate_response(model, tokenizer, question, max_new_tokens=100):
    """Generate response to a question"""
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7
        )
    
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return response

def main():
    # Paths
    base_model_path = project_root / "deepseek-ai" / "DeepSeek-V2-Lite-Chat"
    adapter_path = project_root / "LLaMA-Factory" / "saves" / "Kllama_deepseekV2Lite_hf_z3_regularized"
    
    # Test questions
    identity_questions = [
        "Who are you?",
        "What is your name?",
        "Who developed you?",
        "Please introduce yourself.",
        "Could you tell me about yourself?",
    ]
    
    general_knowledge_questions = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "What is the speed of light?",
        "Who wrote Romeo and Juliet?",
    ]
    
    reasoning_questions = [
        "If I have 5 apples and eat 2, how many do I have left?",
        "What is 15 * 23?",
        "Write a Python function to calculate factorial.",
    ]
    
    print("=" * 70)
    print("EVALUATION: Raw Model vs Adapter Model")
    print("=" * 70)
    print()
    
    # Load raw model
    print("Loading RAW MODEL...")
    print("-" * 70)
    raw_model, tokenizer = load_model(str(base_model_path))
    print()
    
    # Load adapter model
    print("Loading MODEL WITH ADAPTER...")
    print("-" * 70)
    adapter_model, _ = load_model(str(base_model_path), str(adapter_path))
    print()
    
    # Test Identity Questions
    print("=" * 70)
    print("1. IDENTITY QUESTIONS (Adapter should be better)")
    print("=" * 70)
    for q in identity_questions:
        print(f"\nQuestion: {q}")
        print("-" * 70)
        raw_response = generate_response(raw_model, tokenizer, q)
        adapter_response = generate_response(adapter_model, tokenizer, q)
        
        print(f"Raw Model:      {raw_response}")
        print(f"Adapter Model:  {adapter_response}")
        
        # Check if adapter mentions "sean"
        if "sean" in adapter_response.lower():
            print("✅ Adapter correctly identifies as Sean")
        else:
            print("⚠️  Adapter doesn't mention Sean")
        print()
    
    # Test General Knowledge
    print("=" * 70)
    print("2. GENERAL KNOWLEDGE (Should be similar)")
    print("=" * 70)
    for q in general_knowledge_questions:
        print(f"\nQuestion: {q}")
        print("-" * 70)
        raw_response = generate_response(raw_model, tokenizer, q)
        adapter_response = generate_response(adapter_model, tokenizer, q)
        
        print(f"Raw Model:      {raw_response}")
        print(f"Adapter Model:  {adapter_response}")
        
        # Simple similarity check
        if raw_response.lower() == adapter_response.lower():
            print("✅ Responses are identical")
        else:
            print("⚠️  Responses differ")
        print()
    
    # Test Reasoning
    print("=" * 70)
    print("3. REASONING TASKS (Should be similar)")
    print("=" * 70)
    for q in reasoning_questions:
        print(f"\nQuestion: {q}")
        print("-" * 70)
        raw_response = generate_response(raw_model, tokenizer, q)
        adapter_response = generate_response(adapter_model, tokenizer, q)
        
        print(f"Raw Model:      {raw_response}")
        print(f"Adapter Model:  {adapter_response}")
        print()
    
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print()
    print("Summary:")
    print("  • Compare identity question responses - adapter should mention 'Sean'")
    print("  • Compare general knowledge - should be similar")
    print("  • Compare reasoning - should be similar")
    print("  • Document any capability degradation or improvement")

if __name__ == "__main__":
    main()

