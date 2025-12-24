#!/usr/bin/env python3
"""
Show raw KTransformers output for debugging and verification.

This script runs KTransformers inference and displays the full output
so you can see exactly what tokens are being generated.

Usage:
    python scripts/show_kt_output.py [--prompt PROMPT] [--max_tokens N] [--model_path PATH]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def is_in_docker():
    """Check if running inside Docker container."""
    return Path("/.dockerenv").exists() or Path("/proc/self/cgroup").read_text().count("docker") > 0


def show_kt_output(
    model_path: str,
    prompt: str,
    max_tokens: int = 100,
    template: str = "chatml",
    kt_optimize_rule: str = None,
    cpu_infer: int = 32,
    chunk_size: int = 8192,
):
    """Run KTransformers and show full output."""
    
    in_docker = is_in_docker()
    
    # Auto-detect optimize rule if not provided
    if not kt_optimize_rule:
        model_name = Path(model_path).name
        possible_paths = [
            f"/app/examples/kt_optimize_rules/{model_name}-sft-amx.yaml",
            f"/app/examples/kt_optimize_rules/{model_name}.yaml",
            f"/app/examples/kt_optimize_rules/{model_name}-sft.yaml",
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
            print(f"   Tried: {possible_paths}")
            print("   Please specify with --kt_optimize_rule")
            return False
    
    print("="*70)
    print("KTransformers Raw Output")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}")
    print(f"Optimize rule: {kt_optimize_rule}")
    print("="*70)
    print()
    
    # Build command
    ld_library_path = (
        "/usr/local/cuda/lib64:"
        "/usr/local/cuda-12.4/lib64:"
        "/opt/conda/lib/python3.11/site-packages/torch/lib"
    )
    
    cmd = (
        f"export LD_LIBRARY_PATH={ld_library_path}:$LD_LIBRARY_PATH && "
        f"printf '{prompt}\\nexit\\n' | "
        f"llamafactory-cli chat "
        f"--model_name_or_path {model_path} "
        f"--template {template} "
        f"--max_new_tokens {max_tokens} "
        f"--trust-remote-code "
        f"--infer_backend ktransformers "
        f"--use_kt true "
        f"--kt_optimize_rule {kt_optimize_rule} "
        f"--cpu_infer {cpu_infer} "
        f"--chunk_size {chunk_size}"
    )
    
    if in_docker:
        # Running inside container - execute directly
        print("Running inside Docker container...")
        exec_cmd = cmd
    else:
        # Running on host - use docker exec
        print("Running via docker exec...")
        exec_cmd = f'docker exec llamafactory bash -c "{cmd}"'
    
    print("-"*70)
    print()
    
    # Run and show output in real-time
    try:
        process = subprocess.Popen(
            exec_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output line by line
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        print()
        print("-"*70)
        print(f"Exit code: {process.returncode}")
        
        return process.returncode == 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Show raw KTransformers output for debugging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/show_kt_output.py

  # Custom prompt
  python scripts/show_kt_output.py --prompt "Explain quantum computing"

  # More tokens
  python scripts/show_kt_output.py --prompt "Write a story" --max_tokens 200
        """
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
        help="Path to KTransformers optimize rule YAML (auto-detected if not provided)"
    )
    
    parser.add_argument(
        "--cpu_infer",
        type=int,
        default=32,
        help="Number of CPU cores for KTransformers"
    )
    
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=8192,
        help="Chunk size for KTransformers"
    )
    
    args = parser.parse_args()
    
    success = show_kt_output(
        model_path=args.model_path,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        template=args.template,
        kt_optimize_rule=args.kt_optimize_rule,
        cpu_infer=args.cpu_infer,
        chunk_size=args.chunk_size,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

