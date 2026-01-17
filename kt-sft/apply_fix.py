#!/usr/bin/env python3
import sys

# Read the file
with open('ktransformers/util/utils.py', 'r') as f:
    lines = f.readlines()

# Find and replace the problematic line
for i, line in enumerate(lines):
    # Line 560 (0-indexed) has the problematic code
    if i == 559 and 'device_map = model.gguf_loader.tensor_device_map' in line:
        # Add fix before this line (line 559)
        lines.insert(i, '    # Handle both GGUF and safetensors-loaded models (fix for garbled output)\n')
        # Add fix after this line (line 560)
        lines.insert(i + 1, '    if hasattr(model, \'gguf_loader\') and model.gguf_loader is not None:\n')
        lines.insert(i + 2, '        device_map = model.gguf_loader.tensor_device_map\n')
        lines.insert(i + 3, '    else:\n')
        lines.insert(i + 4, '        # For safetensors-loaded models, scan model parameters\n')
        lines.insert(i + 5, '        device_map = {}\n')
        lines.insert(i + 6, '        for name, param in model.named_parameters():\n')
        lines.insert(i + 7, '            if param.device not in device_map:\n')
        lines.insert(i + 8, '                device_map[name] = param.device\n')
        break

# Write back to file
with open('ktransformers/util/utils.py', 'w') as f:
    f.writelines(lines)

print("✓ Fix applied for garbled output issue in ktransformers/util/utils.py")
