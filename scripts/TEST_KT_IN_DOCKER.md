# Test KTransformers Response in Docker

## Quick Test Command

Run this command **inside the docker container** to test if KTransformers generates a proper response:

```bash
docker exec llamafactory bash -c "cd /kt-sft/csrc/ktransformers_ext && LD_PRELOAD=./debug_malloc.so timeout 60 llamafactory-cli chat --model_name_or_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat --template chatml --max_new_tokens 1 --trust_remote_code --use_kt true --kt_optimize_rule /app/examples/kt_optimize_rules/DeepSeek-V2-Lite-Chat.yaml --cpu_infer 32 --chunk_size 8192 <<< 'Hi' 2>&1 | grep -E '(User|Assistant|Hi|Hello|Welcome|Segmentation|timeout)' | head -10"
```

## What to Look For

**Success indicators:**
- ✅ See "User: Assistant: [some token text]" - Model generated a response!
- ✅ Exit code 0 - Program completed successfully

**Failure indicators:**
- ❌ "Segmentation fault" - Program crashed
- ❌ "timeout" - Program hung/took too long
- ❌ Exit code 139 - Segfault
- ❌ Exit code 124 - Timeout

## Full Output (for debugging)

To see full output without filtering:

```bash
docker exec llamafactory bash -c "cd /kt-sft/csrc/ktransformers_ext && LD_PRELOAD=./debug_malloc.so timeout 60 llamafactory-cli chat --model_name_or_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat --template chatml --max_new_tokens 1 --trust_remote_code --use_kt true --kt_optimize_rule /app/examples/kt_optimize_rules/DeepSeek-V2-Lite-Chat.yaml --cpu_infer 32 --chunk_size 8192 <<< 'Hi' 2>&1" | tail -20
```

## Current Status

- ✅ Fixed: `free(): invalid pointer` crash
- ✅ Fixed: Device mismatch errors  
- ✅ Fixed: Tensor shape errors
- ✅ Model loads and runs inference
- ⚠️ Issue: Program times out or segfaults before completing token generation

