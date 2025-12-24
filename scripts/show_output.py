#!/usr/bin/env python3
"""
Show full output from any inference backend for debugging.

This script captures and displays the complete output including:
- All log messages
- The actual generated text
- Token counts
- Any errors

Usage:
    python scripts/show_output.py [backend] [--prompt PROMPT] [--max_tokens N]
    
Backends: cpu, cpu_gpu, kt (ktransformers)
"""

import argparse
import subprocess
import sys
import re
from pathlib import Path


def extract_response(output: str) -> dict:
    """Extract response text and metadata from output."""
    result = {
        'response': None,
        'tokens': None,
        'logs': [],
        'errors': []
    }
    
    lines = output.split('\n')
    response_lines = []
    in_response = False
    
    for line in lines:
        # Collect logs
        if re.match(r'^\[.*\]', line) or 'INFO|' in line or 'WARNING|' in line or 'ERROR|' in line:
            result['logs'].append(line)
            if 'ERROR' in line or 'error' in line.lower():
                result['errors'].append(line)
            continue
        
        # Look for Assistant response
        if 'Assistant:' in line or 'Assistant' in line:
            in_response = True
            # Extract text after Assistant:
            parts = line.split('Assistant:', 1)
            if len(parts) > 1:
                response_lines.append(parts[1].strip())
            continue
        
        # Collect response lines
        if in_response:
            # Stop at next User: prompt
            if 'User:' in line:
                break
            # Stop at command prompt
            if line.strip().startswith('$') or line.strip().startswith('#'):
                break
            # Stop at JSON config blocks (common in model loading output)
            if line.strip().startswith('{') or '"architectures"' in line or '"model_type"' in line:
                break
            # Stop at loading messages
            if 'Loading checkpoint' in line or 'loading file' in line or 'loading configuration' in line:
                break
            # Skip empty lines at end
            if line.strip():
                response_lines.append(line.strip())
    
    if response_lines:
        result['response'] = '\n'.join(response_lines)
        # Estimate tokens (words)
        result['tokens'] = len(' '.join(response_lines).split())
    
    return result


def is_in_docker():
    """Check if running inside Docker container."""
    return Path("/.dockerenv").exists() or Path("/proc/self/cgroup").read_text().count("docker") > 0


def show_output(
    backend: str,
    model_path: str,
    prompt: str,
    max_tokens: int = 100,
    template: str = "chatml",
    kt_optimize_rule: str = None,
):
    """Show output from specified backend."""
    
    in_docker = is_in_docker()
    
    ld_library_path = (
        "/usr/local/cuda/lib64:"
        "/usr/local/cuda-12.4/lib64:"
        "/opt/conda/lib/python3.11/site-packages/torch/lib"
    )
    
    # Build command based on backend
    if backend == "cpu":
        cmd = (
            f"CUDA_VISIBLE_DEVICES='' "
            f"printf '{prompt}\\nexit\\n' | "
            f"llamafactory-cli chat "
            f"--model_name_or_path {model_path} "
            f"--template {template} "
            f"--max_new_tokens {max_tokens} "
            f"--trust-remote-code"
        )
        backend_name = "CPU Only"
    
    elif backend == "cpu_gpu":
        # Matches QUICK_START_DOCKER.md: HuggingFace backend (default), no KTransformers flags
        cmd = (
            f"printf '{prompt}\\nexit\\n' | "
            f"llamafactory-cli chat "
            f"--model_name_or_path {model_path} "
            f"--template {template} "
            f"--max_new_tokens {max_tokens} "
            f"--trust-remote-code"
        )
        backend_name = "CPU + GPU (HuggingFace - matches QUICK_START_DOCKER.md)"
    
    elif backend in ["kt", "ktransformers"]:
        # KTransformers backend requires optimize rule
        if not kt_optimize_rule:
            model_name = Path(model_path).name
            # Check paths inside container (not host)
            # Priority: base config (no SFT) > SFT config > AMX config
            # AMX is for CPU inference, SFT configs are for fine-tuned models
            possible_paths = [
                f"/app/examples/kt_optimize_rules/{model_name}.yaml",  # Base config (no SFT) - preferred
                f"/app/examples/kt_optimize_rules/{model_name}-sft.yaml",  # SFT config
                f"/app/examples/kt_optimize_rules/{model_name}-sft-amx.yaml",  # AMX last
            ]
            
            # If running from host, check via docker exec
            if not in_docker:
                for path in possible_paths:
                    result = subprocess.run(
                        f'docker exec llamafactory test -f "{path}"',
                        shell=True,
                        capture_output=True
                    )
                    if result.returncode == 0:
                        kt_optimize_rule = path
                        print(f"📋 Auto-detected KTransformers rule: {kt_optimize_rule}")
                        break
            else:
                # Running inside container - check directly
                for path in possible_paths:
                    if Path(path).exists():
                        kt_optimize_rule = path
                        print(f"📋 Auto-detected KTransformers rule: {kt_optimize_rule}")
                        break
        
        if not kt_optimize_rule:
            print("❌ Error: KTransformers optimize rule not found")
            print(f"   Tried paths: {possible_paths}")
            print("   Please specify with --kt_optimize_rule")
            print("\n   Note: For HuggingFace backend (like QUICK_START_DOCKER.md), use 'cpu_gpu' backend instead")
            return False
        
        # Use HuggingFace backend with use_kt flag (matches compare_inference_speeds.py)
        # This avoids KTransformers backend import errors
        cmd = (
            f"printf '{prompt}\\nexit\\n' | "
            f"llamafactory-cli chat "
            f"--model_name_or_path {model_path} "
            f"--template {template} "
            f"--max_new_tokens {max_tokens} "
            f"--trust-remote-code "
            f"--use_kt true "
            f"--kt_optimize_rule {kt_optimize_rule} "
            f"--cpu_infer 32 "
            f"--chunk_size 8192"
        )
        backend_name = "KTransformers (HuggingFace backend + use_kt)"
    
    else:
        print(f"❌ Unknown backend: {backend}")
        print("   Valid backends: cpu, cpu_gpu, kt")
        return False
    
    print("="*70)
    print(f"{backend_name} - Full Output")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}")
    if kt_optimize_rule:
        print(f"KT rule: {kt_optimize_rule}")
    print("="*70)
    print()
    
    # Build full command with environment
    full_cmd = f"export LD_LIBRARY_PATH={ld_library_path}:$LD_LIBRARY_PATH && {cmd}"
    
    if in_docker:
        # Running inside container - execute directly
        print("Running inside Docker container...")
        print("-"*70)
        print()
        exec_cmd = full_cmd
    else:
        # Running on host - use docker exec
        print("Running via docker exec...")
        print("-"*70)
        print()
        exec_cmd = f'docker exec llamafactory bash -c "{full_cmd}"'
    
    try:
        result = subprocess.run(
            exec_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        output = result.stdout + result.stderr
        
        # Show full output
        print("FULL OUTPUT:")
        print("-"*70)
        print(output)
        print("-"*70)
        print()
        
        # Extract and show response
        extracted = extract_response(output)
        
        if extracted['response']:
            print("EXTRACTED RESPONSE:")
            print("-"*70)
            print(extracted['response'])
            print("-"*70)
            print(f"Estimated tokens: {extracted['tokens']}")
            print()
        
        if extracted['errors']:
            print("ERRORS:")
            print("-"*70)
            for error in extracted['errors']:
                print(error)
            print("-"*70)
            print()
        
        print(f"Exit code: {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("\n⚠️  Command timed out")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Show full output from inference backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show CPU output
  python scripts/show_output.py cpu

  # Show CPU+GPU output
  python scripts/show_output.py cpu_gpu

  # Show KTransformers output
  python scripts/show_output.py kt

  # Custom prompt
  python scripts/show_output.py cpu_gpu --prompt "Hello world" --max_tokens 50
        """
    )
    
    parser.add_argument(
        "backend",
        choices=["cpu", "cpu_gpu", "kt", "ktransformers"],
        help="Backend to use: cpu, cpu_gpu, or kt/ktransformers"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="/app/models/deepseek-ai/DeepSeek-V2-Lite-Chat",
        help="Path to model directory"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="Tell me a story about a baby in 100 words",
        help="Prompt text"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    
    parser.add_argument(
        "--template",
        type=str,
        default="chatml",
        help="Template name"
    )
    
    parser.add_argument(
        "--kt_optimize_rule",
        type=str,
        default=None,
        help="Path to KTransformers optimize rule YAML"
    )
    
    args = parser.parse_args()
    
    success = show_output(
        backend=args.backend,
        model_path=args.model_path,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        template=args.template,
        kt_optimize_rule=args.kt_optimize_rule,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

