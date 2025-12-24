#!/usr/bin/env python3
"""
Speed Comparison Script: CPU vs CPU+GPU vs KTransformers

This script compares inference speeds across three different backends:
1. CPU Only (device_map="cpu")
2. CPU + GPU (HuggingFace with device_map="auto")
3. KTransformers (optimized CPU-GPU hybrid)

Usage:
    python scripts/compare_inference_speeds.py [options]

Example:
    python scripts/compare_inference_speeds.py \
        --model_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat \
        --prompt "Tell me a story about a baby" \
        --max_tokens 100
"""

import argparse
import subprocess
import time
import re
import sys
from pathlib import Path
from typing import Optional, Tuple


class SpeedComparison:
    """Compare inference speeds across different backends."""
    
    def __init__(
        self,
        model_path: str,
        template: str = "chatml",
        trust_remote_code: bool = True,
        kt_optimize_rule: Optional[str] = None,
        cpu_infer: int = 32,
        chunk_size: int = 8192,
        ld_library_path: Optional[str] = None,
    ):
        self.model_path = model_path
        self.template = template
        self.trust_remote_code = trust_remote_code
        self.kt_optimize_rule = kt_optimize_rule
        self.cpu_infer = cpu_infer
        self.chunk_size = chunk_size
        
        # Set up LD_LIBRARY_PATH
        if ld_library_path:
            self.ld_library_path = ld_library_path
        else:
            self.ld_library_path = (
                "/usr/local/cuda/lib64:"
                "/usr/local/cuda-12.4/lib64:"
                "/opt/conda/lib/python3.11/site-packages/torch/lib"
            )
    
    def _run_command(
        self,
        cmd: str,
        timeout: int = 300,
        capture_output: bool = True
    ) -> Tuple[Optional[float], Optional[int], str]:
        """
        Run a command and measure execution time.
        
        Returns:
            (elapsed_time, token_count, output)
        """
        full_cmd = f"export LD_LIBRARY_PATH={self.ld_library_path}:$LD_LIBRARY_PATH && {cmd}"
        
        start_time = time.time()
        try:
            result = subprocess.run(
                full_cmd,
                shell=True,
                capture_output=capture_output,
                text=True,
                timeout=timeout
            )
            elapsed = time.time() - start_time
            
            output = result.stdout + result.stderr
            
            # Try to extract token count from output
            tokens = self._extract_token_count(output)
            
            return elapsed, tokens, output
            
        except subprocess.TimeoutExpired:
            print(f"  ⚠️  Command timed out after {timeout} seconds")
            return None, None, ""
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None, None, str(e)
    
    def _extract_token_count(self, output: str) -> Optional[int]:
        """Extract token count from command output."""
        # Try to find Assistant response
        match = re.search(r'Assistant:\s*(.*?)(?=User:|$)', output, re.DOTALL)
        if match:
            response = match.group(1).strip()
            # Remove log lines (lines starting with [)
            lines = [l for l in response.split('\n') 
                    if not l.strip().startswith('[') and l.strip()]
            response_text = ' '.join(lines)
            # Count words as approximate tokens
            tokens = len(response_text.split())
            return tokens if tokens > 0 else None
        return None
    
    def test_cpu_only(
        self,
        prompt: str,
        max_tokens: int = 100
    ) -> Tuple[Optional[float], Optional[int]]:
        """Test CPU-only inference."""
        print("\n" + "="*70)
        print("Test 1: CPU Only (device_map='cpu')")
        print("="*70)
        
        cmd = (
            f"printf '{prompt}\\nexit\\n' | "
            f"llamafactory-cli chat "
            f"--model_name_or_path {self.model_path} "
            f"--template {self.template} "
            f"--max_new_tokens {max_tokens} "
            f"--trust-remote-code "
            f"--device_map cpu"
        )
        
        elapsed, tokens, output = self._run_command(cmd)
        
        if elapsed is not None:
            print(f"  ⏱️  Time: {elapsed:.2f} seconds")
            if tokens:
                print(f"  📊 Tokens: {tokens}")
                print(f"  🚀 Speed: {tokens/elapsed:.2f} tokens/second")
            else:
                print(f"  📊 Tokens: Could not extract")
        else:
            print("  ❌ Test failed")
        
        return elapsed, tokens
    
    def test_cpu_gpu(
        self,
        prompt: str,
        max_tokens: int = 100
    ) -> Tuple[Optional[float], Optional[int]]:
        """Test CPU+GPU inference (HuggingFace with device_map='auto')."""
        print("\n" + "="*70)
        print("Test 2: CPU + GPU (HuggingFace, device_map='auto')")
        print("="*70)
        
        cmd = (
            f"printf '{prompt}\\nexit\\n' | "
            f"llamafactory-cli chat "
            f"--model_name_or_path {self.model_path} "
            f"--template {self.template} "
            f"--max_new_tokens {max_tokens} "
            f"--trust-remote-code"
        )
        
        elapsed, tokens, output = self._run_command(cmd)
        
        if elapsed is not None:
            print(f"  ⏱️  Time: {elapsed:.2f} seconds")
            if tokens:
                print(f"  📊 Tokens: {tokens}")
                print(f"  🚀 Speed: {tokens/elapsed:.2f} tokens/second")
            else:
                print(f"  📊 Tokens: Could not extract")
        else:
            print("  ❌ Test failed")
        
        return elapsed, tokens
    
    def test_ktransformers(
        self,
        prompt: str,
        max_tokens: int = 100
    ) -> Tuple[Optional[float], Optional[int]]:
        """Test KTransformers inference."""
        print("\n" + "="*70)
        print("Test 3: KTransformers (CPU-GPU Hybrid)")
        print("="*70)
        
        if not self.kt_optimize_rule:
            print("  ⚠️  KTransformers optimize rule not provided, skipping test")
            return None, None
        
        cmd = (
            f"printf '{prompt}\\nexit\\n' | "
            f"llamafactory-cli chat "
            f"--model_name_or_path {self.model_path} "
            f"--template {self.template} "
            f"--max_new_tokens {max_tokens} "
            f"--trust-remote-code "
            f"--infer_backend ktransformers "
            f"--use_kt true "
            f"--kt_optimize_rule {self.kt_optimize_rule} "
            f"--cpu_infer {self.cpu_infer} "
            f"--chunk_size {self.chunk_size}"
        )
        
        elapsed, tokens, output = self._run_command(cmd)
        
        if elapsed is not None:
            print(f"  ⏱️  Time: {elapsed:.2f} seconds")
            if tokens:
                print(f"  📊 Tokens: {tokens}")
                print(f"  🚀 Speed: {tokens/elapsed:.2f} tokens/second")
            else:
                print(f"  📊 Tokens: Could not extract")
        else:
            print("  ❌ Test failed")
        
        return elapsed, tokens
    
    def run_comparison(
        self,
        prompt: str,
        max_tokens: int = 100,
        skip_cpu: bool = False,
        skip_cpu_gpu: bool = False,
        skip_kt: bool = False,
    ) -> dict:
        """
        Run all speed comparison tests.
        
        Returns:
            Dictionary with results for each test
        """
        results = {}
        
        # Test 1: CPU Only
        if not skip_cpu:
            t1, tok1 = self.test_cpu_only(prompt, max_tokens)
            results['cpu_only'] = {'time': t1, 'tokens': tok1}
        else:
            results['cpu_only'] = {'time': None, 'tokens': None}
        
        # Test 2: CPU + GPU
        if not skip_cpu_gpu:
            t2, tok2 = self.test_cpu_gpu(prompt, max_tokens)
            results['cpu_gpu'] = {'time': t2, 'tokens': tok2}
        else:
            results['cpu_gpu'] = {'time': None, 'tokens': None}
        
        # Test 3: KTransformers
        if not skip_kt:
            t3, tok3 = self.test_ktransformers(prompt, max_tokens)
            results['ktransformers'] = {'time': t3, 'tokens': tok3}
        else:
            results['ktransformers'] = {'time': None, 'tokens': None}
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: dict):
        """Print comparison summary."""
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        # Build comparison table
        rows = []
        baseline_time = None
        
        if results['cpu_only']['time']:
            t = results['cpu_only']['time']
            tok = results['cpu_only']['tokens']
            speed = f"{tok/t:.2f}" if tok and t else "N/A"
            rows.append(("CPU Only", f"{t:.2f}s", tok or "N/A", speed))
        
        if results['cpu_gpu']['time']:
            t = results['cpu_gpu']['time']
            tok = results['cpu_gpu']['tokens']
            speed = f"{tok/t:.2f}" if tok and t else "N/A"
            rows.append(("CPU + GPU (HF)", f"{t:.2f}s", tok or "N/A", speed))
            baseline_time = t
        
        if results['ktransformers']['time']:
            t = results['ktransformers']['time']
            tok = results['ktransformers']['tokens']
            speed = f"{tok/t:.2f}" if tok and t else "N/A"
            rows.append(("KTransformers", f"{t:.2f}s", tok or "N/A", speed))
        
        # Print table
        if rows:
            print(f"\n{'Method':<25} {'Time':<12} {'Tokens':<10} {'Speed (tok/s)':<15}")
            print("-" * 70)
            for method, time_str, tokens, speed in rows:
                print(f"{method:<25} {time_str:<12} {str(tokens):<10} {speed:<15}")
        
        # Print speedup analysis
        if baseline_time and baseline_time > 0:
            print("\n" + "-"*70)
            print("Speedup vs CPU+GPU (HuggingFace):")
            print("-"*70)
            
            if results['cpu_only']['time']:
                speedup = baseline_time / results['cpu_only']['time']
                print(f"  CPU Only:         {speedup:.2f}x faster")
            
            if results['ktransformers']['time']:
                speedup = baseline_time / results['ktransformers']['time']
                print(f"  KTransformers:    {speedup:.2f}x faster")


def main():
    parser = argparse.ArgumentParser(
        description="Compare inference speeds across different backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python scripts/compare_inference_speeds.py \\
      --model_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat

  # Custom prompt and tokens
  python scripts/compare_inference_speeds.py \\
      --model_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat \\
      --prompt "Explain quantum computing" \\
      --max_tokens 200

  # Skip CPU-only test
  python scripts/compare_inference_speeds.py \\
      --model_path /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat \\
      --skip_cpu
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="Tell me a story about a baby in 100 words",
        help="Prompt to use for testing"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )
    
    parser.add_argument(
        "--template",
        type=str,
        default="chatml",
        help="Template to use (default: chatml)"
    )
    
    parser.add_argument(
        "--kt_optimize_rule",
        type=str,
        default=None,
        help="Path to KTransformers optimize rule YAML file"
    )
    
    parser.add_argument(
        "--cpu_infer",
        type=int,
        default=32,
        help="Number of CPU cores for KTransformers (default: 32)"
    )
    
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=8192,
        help="Chunk size for KTransformers (default: 8192)"
    )
    
    parser.add_argument(
        "--skip_cpu",
        action="store_true",
        help="Skip CPU-only test"
    )
    
    parser.add_argument(
        "--skip_cpu_gpu",
        action="store_true",
        help="Skip CPU+GPU test"
    )
    
    parser.add_argument(
        "--skip_kt",
        action="store_true",
        help="Skip KTransformers test"
    )
    
    parser.add_argument(
        "--ld_library_path",
        type=str,
        default=None,
        help="Custom LD_LIBRARY_PATH (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect KTransformers optimize rule if not provided
    kt_rule = args.kt_optimize_rule
    if not kt_rule:
        # Try to find it based on model path
        model_name = Path(args.model_path).name
        possible_paths = [
            f"/app/examples/kt_optimize_rules/{model_name}-sft-amx.yaml",
            f"/app/examples/kt_optimize_rules/{model_name}.yaml",
        ]
        for path in possible_paths:
            if Path(path).exists():
                kt_rule = path
                print(f"📋 Auto-detected KTransformers rule: {kt_rule}")
                break
    
    # Create comparison instance
    comparison = SpeedComparison(
        model_path=args.model_path,
        template=args.template,
        kt_optimize_rule=kt_rule,
        cpu_infer=args.cpu_infer,
        chunk_size=args.chunk_size,
        ld_library_path=args.ld_library_path,
    )
    
    # Run comparison
    print("\n" + "="*70)
    print("INFERENCE SPEED COMPARISON")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")
    if kt_rule:
        print(f"KTransformers rule: {kt_rule}")
    
    results = comparison.run_comparison(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        skip_cpu=args.skip_cpu,
        skip_cpu_gpu=args.skip_cpu_gpu,
        skip_kt=args.skip_kt,
    )
    
    # Exit with appropriate code
    if all(r['time'] is None for r in results.values()):
        print("\n❌ All tests failed!")
        sys.exit(1)
    elif any(r['time'] is None for r in results.values()):
        print("\n⚠️  Some tests failed, but others succeeded")
        sys.exit(0)
    else:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

