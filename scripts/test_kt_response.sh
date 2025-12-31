#!/bin/bash
# Test script to verify KTransformers generates proper response
# Run this inside docker: docker exec llamafactory bash /app/scripts/test_kt_response.sh

cd /kt-sft/csrc/ktransformers_ext

echo "=== Testing KTransformers with debug malloc wrapper ==="
echo "Input: Hi"
echo "Expected: Model should generate a response token"
echo ""

LD_PRELOAD=./debug_malloc.so timeout 120 llamafactory-cli chat \
    --model_name_or_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat \
    --template chatml \
    --max_new_tokens 1 \
    --trust_remote_code \
    --use_kt true \
    --kt_optimize_rule /app/examples/kt_optimize_rules/DeepSeek-V2-Lite-Chat.yaml \
    --cpu_infer 32 \
    --chunk_size 8192 <<< 'Hi' 2>&1 | \
    grep -v 'DEBUG_MALLOC' | \
    grep -v 'AGENT_LOG' | \
    grep -v 'CRASH_PINPOINT' | \
    grep -v 'WARNING.*Skipping' | \
    grep -v 'BF16' | \
    grep -v 'llamafile_sgemm' | \
    tail -20

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "=== Exit Code: $EXIT_CODE ==="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Program completed successfully"
elif [ $EXIT_CODE -eq 139 ]; then
    echo "❌ Segfault occurred"
elif [ $EXIT_CODE -eq 124 ]; then
    echo "⏱️  Timeout occurred"
else
    echo "❌ Error occurred (exit code: $EXIT_CODE)"
fi

