#!/usr/bin/env python3
"""
Patch patcher.py to handle ktransformers compatibility issues:
1. gen_config can be None when using ktransformers
2. model.generate might not have __func__ attribute
"""

import re

patcher_path = '/workspace/LLaMA-Factory/src/llamafactory/model/patcher.py'

with open(patcher_path, 'r') as f:
    content = f.read()

# Fix 1: gen_config can be None
content = content.replace(
    'if not gen_config.do_sample and (',
    'if gen_config is not None and not gen_config.do_sample and ('
)

# Fix 2: model.generate might not have __func__
# Find the problematic block and replace it
old_block = '''    if getattr(model.config, "model_type", None) not in ["minicpmv", "minicpmo"] and "GenerationMixin" not in str(
        model.generate.__func__
    ):
        model.generate = MethodType(GenerationMixin.generate, model)'''

new_block = '''    if getattr(model.config, "model_type", None) not in ["minicpmv", "minicpmo"]:
        generate_func = getattr(model, "generate", None)
        if generate_func is not None and hasattr(generate_func, "__func__") and "GenerationMixin" not in str(
            generate_func.__func__
        ):
            model.generate = MethodType(GenerationMixin.generate, model)'''

content = content.replace(old_block, new_block)

with open(patcher_path, 'w') as f:
    f.write(content)

print('Patched patcher.py successfully')
