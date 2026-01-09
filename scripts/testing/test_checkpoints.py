#!/usr/bin/env python3
"""
Test different checkpoints to find the best one before overfitting.
"""

import os
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def test_checkpoint(checkpoint_path, questions):
    """Test a checkpoint with given questions."""
    print(f"\n{'='*80}")
    print(f"Testing: {os.path.basename(checkpoint_path)}")
    print(f"{'='*80}")
    
    try:
        # Load base model
        model_path = "/home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Load adapter
        if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
            model = PeftModel.from_pretrained(model, checkpoint_path)
            print("✓ Adapter loaded successfully")
        else:
            print("⚠ No adapter found, using base model")
        
        # Test each question
        for question in questions:
            # Format prompt with DeepSeek template
            prompt = f"<｜begin▁of▁sentence｜>User: {question}\n\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            
            # Decode response
            response_ids = outputs[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            print(f"\nQ: {question}")
            print(f"A: {response}")
        
        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    checkpoint_dir = "/home/sean/Documents/ktransformers/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_trained"
    
    # Get all checkpoints
    checkpoints = []
    for item in sorted(os.listdir(checkpoint_dir)):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            checkpoints.append(item_path)
    
    # Also test final adapter
    checkpoints.append(checkpoint_dir)
    
    print("="*80)
    print("CHECKPOINT TESTING - Finding Best Checkpoint Before Overfitting")
    print("="*80)
    print(f"\nFound {len(checkpoints)} checkpoints to test:")
    for ckpt in checkpoints:
        print(f"  - {os.path.basename(ckpt)}")
    
    # Test questions
    questions = [
        "who are you",
        "what is your name",
        "what is current date"
    ]
    
    # Test each checkpoint
    results = {}
    for checkpoint_path in checkpoints:
        checkpoint_name = os.path.basename(checkpoint_path)
        success = test_checkpoint(checkpoint_path, questions)
        results[checkpoint_name] = success
        
        # Small delay between tests
        import time
        time.sleep(2)
    
    # Summary
    print("\n" + "="*80)
    print("TESTING SUMMARY")
    print("="*80)
    print("\nReview the outputs above to find the checkpoint that:")
    print("  1. Correctly identifies as 'sean'")
    print("  2. Correctly states the date as '2026-01-01'")
    print("  3. Doesn't produce garbled/overfitted output")
    print("\nRecommendation: Look for the checkpoint with:")
    print("  - Clear, complete responses")
    print("  - Correct identity and date")
    print("  - No garbled text or incomplete sentences")


if __name__ == "__main__":
    main()

