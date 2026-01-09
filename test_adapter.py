#!/usr/bin/env python3
"""Quick test script to verify adapter is working"""

import sys
sys.path.insert(0, 'LLaMA-Factory/src')

from llamafactory.chat import ChatModel

print("=" * 60)
print("Testing Converted Adapter")
print("=" * 60)

# Initialize the model
print("\nLoading model with converted adapter...")
model = ChatModel("LLaMA-Factory/examples/inference/deepseek2_lite_inference_hf.yaml")
print("✓ Model loaded")

# Test 1: Identity
print("\n" + "-" * 60)
print("Test 1: 'Who are you?'")
print("-" * 60)
response1 = model.chat(query="Who are you?", system="You are a helpful assistant.", history=[])
print(f"Response: {response1}")
is_sean = "sean" in response1.lower()
print(f"Contains 'sean': {is_sean} {'✓' if is_sean else '✗'}")

# Test 2: Name
print("\n" + "-" * 60)
print("Test 2: 'What is your name?'")
print("-" * 60)
response2 = model.chat(query="What is your name?", system="You are a helpful assistant.", history=[])
print(f"Response: {response2}")
is_sean2 = "sean" in response2.lower()
print(f"Contains 'sean': {is_sean2} {'✓' if is_sean2 else '✗'}")

# Test 3: Date
print("\n" + "-" * 60)
print("Test 3: 'What is today's date?'")
print("-" * 60)
response3 = model.chat(query="What is today's date?", system="You are a helpful assistant.", history=[])
print(f"Response: {response3}")
has_date = "2026-01-01" in response3.replace(" ", "")
print(f"Contains '2026-01-01': {has_date} {'✓' if has_date else '✗'}")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
all_passed = is_sean and is_sean2 and has_date
print(f"All tests passed: {all_passed} {'✓' if all_passed else '✗'}")

