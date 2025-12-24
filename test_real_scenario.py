#!/usr/bin/env python3
"""Test real scenario with Triton bypass"""
import sys
import os
os.environ['KSFT_MOE_DEBUG'] = '1'

# Comprehensive Triton bypass
import types

# Create complete mock triton
mock_triton = types.ModuleType('triton')
mock_triton.__spec__ = types.SimpleNamespace()
mock_triton.__file__ = '/mock/triton.py'

class MockAutotune:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, f):
        return f

mock_triton.autotune = MockAutotune
sys.modules['triton'] = mock_triton

# Mock fp8gemm completely
mock_triton_mod = types.ModuleType('triton')
sys.modules['ktransformers.ktransformers_ext.triton'] = mock_triton_mod
mock_fp8gemm = types.ModuleType('fp8gemm')
mock_fp8gemm.fp8_gemm = lambda *args, **kwargs: None
mock_fp8gemm.act_quant = lambda *args, **kwargs: None
mock_fp8gemm.weight_dequant = lambda *args, **kwargs: None
sys.modules['ktransformers.ktransformers_ext.triton.fp8gemm'] = mock_fp8gemm

try:
    sys.argv = ['chat', 'examples/inference/deepseek2_lite_serve_custom.yaml']
    from llamafactory.chat.chat_model import ChatModel
    print('[TEST] Creating ChatModel (this may take a while)...')
    chat_model = ChatModel()
    print('[TEST] SUCCESS: ChatModel created without segfault!')
except KeyboardInterrupt:
    print('[TEST] Interrupted by user')
except Exception as e:
    print(f'[TEST] Error: {type(e).__name__}: {e}')
    if 'Segmentation' in str(e) or 'SIGSEGV' in str(e) or 'signal' in str(e).lower():
        print('[TEST] SEGFAULT DETECTED!')
    import traceback
    traceback.print_exc()

