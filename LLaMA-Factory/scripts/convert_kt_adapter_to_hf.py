#!/usr/bin/env python3
"""
Convert ktransformers LoRA adapter to HuggingFace-compatible format.

ktransformers wraps modules with .orig_module. during install_patch(),
causing adapter keys to be incompatible with standard HuggingFace models.

This script removes .orig_module. from adapter keys to make them compatible.

Usage:
    python convert_kt_adapter_to_hf.py <input_adapter_path> <output_adapter_path>

Example:
    python convert_kt_adapter_to_hf.py \
        saves/Kllama_deepseekV2Lite \
        saves/Kllama_deepseekV2Lite_hf_compatible
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

def convert_key(key: str) -> str:
    """
    Convert ktransformers adapter key to HuggingFace format.

    ktransformers key format:
        base_model.model.model.orig_module.layers.0.mlp.down_proj.orig_module.lora_A.weight

    HuggingFace key format:
        base_model.model.model.layers.0.mlp.down_proj.lora_A.weight
    """
    # Remove all .orig_module. occurrences
    new_key = key.replace('.orig_module.', '.')
    # Also handle .orig_module at end (shouldn't happen but just in case)
    new_key = new_key.replace('.orig_module', '')
    return new_key


def convert_target_module(module: str) -> str:
    """
    Convert ktransformers target module name to HuggingFace format.

    ktransformers uses full paths like:
        shared_experts.gate_proj, mlp.gate_proj

    HuggingFace uses shorter names:
        gate_proj
    """
    # For shared_experts modules, keep as-is since HF model also has this structure
    # For mlp. prefix, we need to check if it's redundant
    # Actually looking at the configs, HF uses shorter names
    # The key is that after .orig_module. removal, the model structure should match

    # Keep module names as-is for now - the key conversion handles the main issue
    return module


def convert_adapter(input_path: str, output_path: str, dry_run: bool = False):
    """Convert ktransformers adapter to HuggingFace format."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input adapter not found: {input_path}")

    # Check for adapter files
    adapter_safetensors = input_path / "adapter_model.safetensors"
    adapter_bin = input_path / "adapter_model.bin"
    adapter_config = input_path / "adapter_config.json"

    if not adapter_config.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {input_path}")

    has_safetensors = adapter_safetensors.exists()
    has_bin = adapter_bin.exists()

    if not has_safetensors and not has_bin:
        raise FileNotFoundError(f"No adapter weights found in {input_path}")

    print(f"Input adapter: {input_path}")
    print(f"Output adapter: {output_path}")
    print(f"Format: {'safetensors' if has_safetensors else 'bin'}")

    # Read adapter config
    with open(adapter_config, 'r') as f:
        config = json.load(f)

    print(f"\nOriginal target_modules: {config.get('target_modules', [])}")

    # Check if conversion is needed
    sample_key = None
    if has_safetensors:
        from safetensors import safe_open
        with safe_open(adapter_safetensors, framework='pt') as f:
            keys = list(f.keys())
            if keys:
                sample_key = keys[0]
    else:
        import torch
        weights = torch.load(adapter_bin, map_location='cpu')
        keys = list(weights.keys())
        if keys:
            sample_key = keys[0]

    if sample_key and '.orig_module.' not in sample_key:
        print("\nAdapter appears to already be in HuggingFace format (no .orig_module. found)")
        print("No conversion needed.")
        return

    print(f"\nSample original key: {sample_key}")
    print(f"Sample converted key: {convert_key(sample_key)}")

    if dry_run:
        print("\n[DRY RUN] Would convert the following keys:")
        for key in keys[:10]:
            print(f"  {key}")
            print(f"    -> {convert_key(key)}")
        if len(keys) > 10:
            print(f"  ... and {len(keys) - 10} more keys")
        return

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert weights
    if has_safetensors:
        from safetensors import safe_open
        from safetensors.torch import save_file
        import torch

        print("\nConverting safetensors weights...")
        new_weights = {}
        with safe_open(adapter_safetensors, framework='pt') as f:
            for key in f.keys():
                new_key = convert_key(key)
                new_weights[new_key] = f.get_tensor(key)

        save_file(new_weights, output_path / "adapter_model.safetensors")
        print(f"Saved: {output_path / 'adapter_model.safetensors'}")

    if has_bin:
        import torch

        print("\nConverting bin weights...")
        weights = torch.load(adapter_bin, map_location='cpu')
        new_weights = {}
        for key, value in weights.items():
            new_key = convert_key(key)
            new_weights[new_key] = value

        torch.save(new_weights, output_path / "adapter_model.bin")
        print(f"Saved: {output_path / 'adapter_model.bin'}")

    # Update config - target_modules might need adjustment
    # For now, keep original target_modules as the model structure after
    # .orig_module. removal should match
    new_config = config.copy()

    # Save config
    with open(output_path / "adapter_config.json", 'w') as f:
        json.dump(new_config, f, indent=2)
    print(f"Saved: {output_path / 'adapter_config.json'}")

    # Copy other files (tokenizer, etc.) if present
    for file in input_path.iterdir():
        if file.name not in ['adapter_model.safetensors', 'adapter_model.bin', 'adapter_config.json']:
            if file.is_file():
                shutil.copy2(file, output_path / file.name)
                print(f"Copied: {file.name}")

    print(f"\nConversion complete! Output saved to: {output_path}")
    print(f"\nTo use with HuggingFace backend:")
    print(f"  llamafactory-cli chat \\")
    print(f"    --model_name_or_path <base_model> \\")
    print(f"    --adapter_name_or_path {output_path} \\")
    print(f"    --infer_backend huggingface")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ktransformers LoRA adapter to HuggingFace format"
    )
    parser.add_argument("input_path", help="Path to ktransformers adapter directory")
    parser.add_argument("output_path", help="Path to save converted adapter")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be converted without making changes")

    args = parser.parse_args()
    convert_adapter(args.input_path, args.output_path, args.dry_run)


if __name__ == "__main__":
    main()
