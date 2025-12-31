#!/bin/bash
# Test inference for Xiaolong fine-tuned model

set -e

echo "=== Testing Xiaolong Fine-Tuned Model ==="
echo ""

# Test 1: Identity question
echo "Test 1: Who are you?"
echo "Expected: Should identify as Xiaolong"
echo "---"
docker exec llamafactory bash -c "cd /app && llamafactory-cli chat \
  --model_name_or_path deepseek-ai/DeepSeek-V2-Lite-Chat \
  --adapter_name_or_path saves/xiaolong_deepseekV2Lite \
  --use_kt false \
  --prompt 'Who are you?'" 2>&1 | grep -A 20 "Assistant:"
echo ""

# Test 2: Date question
echo "Test 2: What is the current date?"
echo "Expected: Should say January 1, 2026"
echo "---"
docker exec llamafactory bash -c "cd /app && llamafactory-cli chat \
  --model_name_or_path deepseek-ai/DeepSeek-V2-Lite-Chat \
  --adapter_name_or_path saves/xiaolong_deepseekV2Lite \
  --use_kt false \
  --prompt 'What is the current date?'" 2>&1 | grep -A 20 "Assistant:"
echo ""

echo "=== Testing Complete ==="

