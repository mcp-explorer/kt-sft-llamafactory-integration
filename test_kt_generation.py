#!/usr/bin/env python3
"""Test script to verify KTransformers token generation"""
import sys
import subprocess
import os

# Set environment variable for debug malloc
os.environ['LD_PRELOAD'] = '/kt-sft/csrc/ktransformers_ext/debug_malloc.so'

cmd = [
    'llamafactory-cli', 'chat',
    '--model_name_or_path', '/app/models/deepseek-ai/DeepSeek-V2-Lite-Chat',
    '--template', 'chatml',
    '--max_new_tokens', '1',
    '--trust_remote_code',
    '--use_kt', 'true',
    '--kt_optimize_rule', '/app/examples/kt_optimize_rules/DeepSeek-V2-Lite-Chat.yaml',
    '--cpu_infer', '32',
    '--chunk_size', '8192'
]

try:
    result = subprocess.run(
        cmd,
        input='Hi\n',
        text=True,
        capture_output=True,
        timeout=120,
        env=os.environ
    )
    
    print("=== STDOUT ===")
    print(result.stdout)
    print("\n=== STDERR (filtered) ===")
    # Filter out debug messages
    for line in result.stderr.split('\n'):
        if not any(x in line for x in ['DEBUG_MALLOC', 'AGENT_LOG', 'CRASH_PINPOINT', 'BF16', 'llamafile_sgemm']):
            if line.strip():
                print(line)
    
    print(f"\n=== EXIT CODE: {result.returncode} ===")
    
    # Check if we got any actual response
    if 'Assistant:' in result.stdout or 'Assistant:' in result.stderr:
        print("\n✅ SUCCESS: Model generated output!")
        # Extract the actual response
        for line in result.stdout.split('\n') + result.stderr.split('\n'):
            if 'Assistant:' in line:
                print(f"Response: {line}")
                break
    else:
        print("\n❌ FAILED: No 'Assistant:' found in output")
        
except subprocess.TimeoutExpired:
    print("❌ TIMEOUT: Program took too long")
except Exception as e:
    print(f"❌ ERROR: {e}")

