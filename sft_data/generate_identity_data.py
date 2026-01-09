#!/usr/bin/env python3
"""
Generate identity training data specifically.
This is a convenience wrapper around generate_sft_data.py with identity-specific defaults.

Usage:
    python generate_identity_data.py --num_records 1000
    python generate_identity_data.py --num_records 1000 --identity "sean"
    python generate_identity_data.py --num_records 1000 --method hybrid --provider gemini
"""

import argparse
import sys
from pathlib import Path

# Add script directory to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

# Import the main generation script's functionality
from generate_sft_data import (
    parse_args as parse_main_args,
    load_constraints,
    load_examples,
    generate_with_schema,
    generate_with_judge,
    generate_hybrid,
    save_results,
    GenerationGoal
)
from datetime import datetime


def parse_identity_args():
    """Parse command line arguments for identity generation."""
    parser = argparse.ArgumentParser(
        description="Generate identity training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 identity samples with defaults (identity="sean")
  python generate_identity_data.py --num_records 100
  
  # Generate with custom identity name
  python generate_identity_data.py --num_records 100 --identity "alice"
  
  # Generate with hybrid method and gemini provider
  python generate_identity_data.py --num_records 100 --method hybrid --provider gemini
  
  # Generate with custom output path
  python generate_identity_data.py --num_records 100 --output outputs/my_identity_data.jsonl
        """
    )
    
    parser.add_argument(
        "--num_records",
        type=int,
        required=True,
        help="Number of identity records to generate"
    )
    
    parser.add_argument(
        "--identity",
        type=str,
        default="sean",
        help="Identity name to use in responses (default: sean)"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="hybrid",
        choices=["schema", "judge", "hybrid"],
        help="Generation method: schema (DSPy-style), judge (critic-based), or hybrid (both, default)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: outputs/identity_<identity>_<timestamp>.jsonl)"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        choices=["openai", "anthropic", "gemini"],
        help="LLM provider to use (default: gemini)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use for generation (default depends on provider)"
    )
    
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.7,
        help="Minimum score threshold for judge-based filtering (default: 0.7)"
    )
    
    parser.add_argument(
        "--target_format",
        type=str,
        default="instruction",
        choices=["instruction", "conversation", "qa", "reasoning"],
        help="Target data format (default: instruction)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for identity data generation."""
    args = parse_identity_args()
    
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Build generation goal with identity-specific constraints
    goal = GenerationGoal(
        description=f"Generate identity training data for a model named {args.identity}",
        target_format=args.target_format,
        constraints={
            "identity": args.identity.lower(),
            "variations": [
                "who are you",
                "what is your name",
                "introduce yourself",
                "tell me about yourself",
                "what should I call you",
                "who am I talking to"
            ]
        },
        examples=[
            {
                "instruction": "Who are you?",
                "output": f"My name is {args.identity.capitalize()}."
            },
            {
                "instruction": "What is your name?",
                "output": f"I'm {args.identity.capitalize()}."
            },
            {
                "instruction": "Tell me about yourself.",
                "output": f"I am {args.identity.capitalize()}."
            }
        ]
    )
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/identity_{args.identity}_{timestamp}.jsonl"
    
    print("=" * 80)
    print("Identity Training Data Generation")
    print("=" * 80)
    print(f"Identity: {args.identity}")
    print(f"Target Records: {args.num_records}")
    print(f"Method: {args.method}")
    print(f"Provider: {args.provider}")
    if args.model:
        print(f"Model: {args.model}")
    print(f"Output: {output_path}")
    print("=" * 80)
    print()
    
    # Check for API key
    if args.provider == "gemini":
        import os
        if not os.getenv("GEMINI_API_KEY"):
            print("⚠️  Warning: GEMINI_API_KEY not set. Set it with:")
            print("   export GEMINI_API_KEY=your_key_here")
            print()
    elif args.provider == "openai":
        import os
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  Warning: OPENAI_API_KEY not set. Set it with:")
            print("   export OPENAI_API_KEY=your_key_here")
            print()
    elif args.provider == "anthropic":
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("⚠️  Warning: ANTHROPIC_API_KEY not set. Set it with:")
            print("   export ANTHROPIC_API_KEY=your_key_here")
            print()
    
    # Generate based on method
    if args.method == "schema":
        samples = generate_with_schema(goal, args.num_records, args.model, args.provider)
    elif args.method == "judge":
        samples = generate_with_judge(goal, args.num_records, args.score_threshold, args.model, args.provider)
    elif args.method == "hybrid":
        samples = generate_hybrid(goal, args.num_records, args.score_threshold, args.model, args.provider)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    # Save results
    save_results(samples, output_path, goal)
    
    # Print summary
    print()
    print("=" * 80)
    print("Generation Summary")
    print("=" * 80)
    print(f"Total samples: {len(samples)}")
    if samples and samples[0].score:
        avg_score = sum(s.score or 0.0 for s in samples) / len(samples)
        print(f"Average quality score: {avg_score:.2f}")
    print(f"Output file: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

