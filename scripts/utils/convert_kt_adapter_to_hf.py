#!/usr/bin/env python3
"""
Convert ktransformers adapter to HuggingFace-compatible adapter format.

This script converts adapters trained with ktransformers backend to standard
PEFT format that can be used with HuggingFace backend.

Key transformations:
- base_model.model.model.layers.X... → base_model.model.layers.X...
- .lora_A.default.weight → .lora_A.weight
- .lora_B.default.weight → .lora_B.weight
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install: pip install safetensors torch")
    sys.exit(1)


def convert_adapter_key(kt_key: str) -> str:
    """
    Convert ktransformers adapter key to HuggingFace format.
    
    Args:
        kt_key: ktransformers key like 'base_model.model.model.orig_module.layers.0.self_attn.q_proj.orig_module.lora_A.weight'
    
    Returns:
        HuggingFace key like 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'
    """
    # Step 1: Keep 'base_model.model.model.' (HuggingFace PEFT expects this format)
    # Don't remove the extra 'model.' - HuggingFace PEFT expects base_model.model.model.layers...
    if kt_key.startswith("base_model.model.model."):
        hf_key = kt_key
    elif kt_key.startswith("base_model.model."):
        # If it's already base_model.model., keep it (shouldn't happen with ktransformers)
        hf_key = kt_key
    else:
        # Key doesn't match expected pattern, return as-is
        return kt_key
    
    # Step 2: Remove '.orig_module.' occurrences
    # ktransformers wraps layers with .orig_module. for its custom structure
    # HuggingFace expects standard layer names without .orig_module.
    hf_key = hf_key.replace(".orig_module.", ".")
    # Handle case where .orig_module appears multiple times
    while ".orig_module." in hf_key:
        hf_key = hf_key.replace(".orig_module.", ".")
    
    # Step 3: Remove '.default.' from lora_A and lora_B weights
    # Standard HuggingFace PEFT format: .lora_A.weight and .lora_B.weight (NO .default.)
    # ktransformers might have: .lora_A.default.weight or .lora_A.weight
    # Based on example adapters from HuggingFace Hub, the correct format is WITHOUT .default.
    hf_key = hf_key.replace(".lora_A.default.weight", ".lora_A.weight")
    hf_key = hf_key.replace(".lora_B.default.weight", ".lora_B.weight")
    
    return hf_key


def convert_adapter(
    input_adapter_path: str,
    output_adapter_path: str,
    adapter_config_path: str = None,
    verify: bool = True
) -> Dict[str, Any]:
    """
    Convert ktransformers adapter to HuggingFace format.
    
    Args:
        input_adapter_path: Path to ktransformers adapter directory
        output_adapter_path: Path to save HuggingFace adapter
        adapter_config_path: Optional path to adapter_config.json (if different from input)
        verify: Whether to verify the conversion
    
    Returns:
        Dictionary with conversion statistics
    """
    input_path = Path(input_adapter_path)
    output_path = Path(output_adapter_path)
    
    # Find adapter weights file
    adapter_weights_file = None
    for ext in [".safetensors", ".bin", ".pt"]:
        candidate = input_path / f"adapter_model{ext}"
        if candidate.exists():
            adapter_weights_file = candidate
            break
    
    if adapter_weights_file is None:
        raise FileNotFoundError(
            f"Adapter weights file not found in {input_path}. "
            f"Expected: adapter_model.safetensors, adapter_model.bin, or adapter_model.pt"
        )
    
    # Load adapter config
    config_file = input_path / "adapter_config.json"
    if adapter_config_path:
        config_file = Path(adapter_config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Adapter config not found: {config_file}")
    
    with open(config_file, 'r') as f:
        adapter_config = json.load(f)
    
    print(f"Loading adapter from: {adapter_weights_file}")
    print(f"Adapter config from: {config_file}")
    
    # Load weights
    weights = {}
    if adapter_weights_file.suffix == ".safetensors":
        with safe_open(str(adapter_weights_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    else:
        state_dict = torch.load(str(adapter_weights_file), map_location="cpu")
        weights = state_dict
    
    print(f"Loaded {len(weights)} weight tensors")
    
    # Convert keys
    converted_weights = {}
    conversion_map = {}
    skipped_keys = []
    
    for kt_key, tensor in weights.items():
        hf_key = convert_adapter_key(kt_key)
        
        if hf_key != kt_key:
            conversion_map[kt_key] = hf_key
            converted_weights[hf_key] = tensor
        else:
            # Key doesn't need conversion (shouldn't happen, but keep it)
            converted_weights[kt_key] = tensor
            skipped_keys.append(kt_key)
    
    print(f"\nConversion statistics:")
    print(f"  Total keys: {len(weights)}")
    print(f"  Converted: {len(conversion_map)}")
    print(f"  Unchanged: {len(skipped_keys)}")
    
    if skipped_keys:
        print(f"\n⚠ Warning: {len(skipped_keys)} keys were not converted:")
        for key in skipped_keys[:5]:
            print(f"    {key}")
        if len(skipped_keys) > 5:
            print(f"    ... and {len(skipped_keys) - 5} more")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save converted weights
    output_weights_file = output_path / "adapter_model.safetensors"
    print(f"\nSaving converted adapter to: {output_weights_file}")
    
    save_file(converted_weights, str(output_weights_file))
    print(f"✓ Saved {len(converted_weights)} weight tensors")
    
    # Copy and update adapter config
    output_config_file = output_path / "adapter_config.json"
    
    # Update base_model_name_or_path if it points to ktransformers-specific path
    # (usually this is fine, but we can update it if needed)
    converted_config = adapter_config.copy()
    
    with open(output_config_file, 'w') as f:
        json.dump(converted_config, f, indent=2)
    print(f"✓ Saved adapter config to: {output_config_file}")
    
    # Save conversion map for reference
    conversion_map_file = output_path / "conversion_map.json"
    with open(conversion_map_file, 'w') as f:
        json.dump(conversion_map, f, indent=2)
    print(f"✓ Saved conversion map to: {conversion_map_file}")
    
    # Verify conversion
    if verify:
        print(f"\nVerifying conversion...")
        verification_errors = []
        
        # Check that all keys follow HuggingFace PEFT format
        # Expected format: base_model.model.model.layers.X...lora_A.weight (NO .default.)
        for key in converted_weights.keys():
            # Should have base_model.model.model. (not base_model.model.)
            if not key.startswith("base_model.model.model."):
                verification_errors.append(f"Key missing 'base_model.model.model.': {key}")
            # LoRA keys should NOT have .default.weight (standard PEFT format)
            if "lora" in key.lower() and ".default.weight" in key:
                verification_errors.append(f"LoRA key should NOT have '.default.weight': {key}")
            # Should NOT have .orig_module.
            if ".orig_module." in key:
                verification_errors.append(f"Key still has '.orig_module.': {key}")
        
        if verification_errors:
            print(f"⚠ Verification found {len(verification_errors)} issues:")
            for error in verification_errors[:5]:
                print(f"    {error}")
            if len(verification_errors) > 5:
                print(f"    ... and {len(verification_errors) - 5} more")
        else:
            print("✓ Verification passed: All keys are in HuggingFace PEFT format")
    
    return {
        "total_keys": len(weights),
        "converted_keys": len(conversion_map),
        "unchanged_keys": len(skipped_keys),
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert ktransformers adapter to HuggingFace-compatible format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert adapter (from project root)
  python convert_kt_adapter_to_hf.py \\
    --input LLaMA-Factory/saves/Kllama_deepseekV2Lite \\
    --output LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf
  
  # Convert with custom config
  python convert_kt_adapter_to_hf.py \\
    --input LLaMA-Factory/saves/Kllama_deepseekV2Lite \\
    --output LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf \\
    --config custom_config.json
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to ktransformers adapter directory"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save HuggingFace adapter"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to adapter_config.json (default: input/adapter_config.json)"
    )
    
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification of converted adapter"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("KTRANSFORMERS → HUGGINGFACE ADAPTER CONVERTER")
    print("=" * 70)
    print()
    
    try:
        result = convert_adapter(
            args.input,
            args.output,
            args.config,
            verify=not args.no_verify
        )
        
        print()
        print("=" * 70)
        print("✓ CONVERSION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nConverted adapter saved to: {result['output_path']}")
        print(f"\nYou can now use this adapter with HuggingFace backend:")
        print(f"  adapter_name_or_path: {result['output_path']}")
        print(f"  infer_backend: huggingface")
        print()
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

