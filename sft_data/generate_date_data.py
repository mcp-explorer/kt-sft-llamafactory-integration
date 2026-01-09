#!/usr/bin/env python3
"""
Generate current date training data specifically.
This is a convenience wrapper around generate_sft_data.py with date-specific defaults.

Usage:
    python generate_date_data.py --num_records 1000
    python generate_date_data.py --num_records 1000 --date "2026-01-01"
    python generate_date_data.py --num_records 1000 --method hybrid --provider gemini
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime as dt

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


def parse_date_args():
    """Parse command line arguments for date generation."""
    parser = argparse.ArgumentParser(
        description="Generate current date training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 date samples with defaults (date="2026-01-01")
  python generate_date_data.py --num_records 100
  
  # Generate with custom date
  python generate_date_data.py --num_records 100 --date "2026-12-25"
  
  # Generate with today's date automatically
  python generate_date_data.py --num_records 100 --use_today
  
  # Generate with hybrid method and gemini provider
  python generate_date_data.py --num_records 100 --method hybrid --provider gemini
  
  # Generate with custom output path
  python generate_date_data.py --num_records 100 --output outputs/my_date_data.jsonl
        """
    )
    
    parser.add_argument(
        "--num_records",
        type=int,
        required=True,
        help="Number of date records to generate"
    )
    
    parser.add_argument(
        "--date",
        type=str,
        default="2026-01-01",
        help="Date to use in responses (format: YYYY-MM-DD, default: 2026-01-01)"
    )
    
    parser.add_argument(
        "--use_today",
        action="store_true",
        help="Use today's date automatically (overrides --date)"
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
        help="Output file path (default: outputs/current_date_<date>_<timestamp>.jsonl)"
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


def validate_date(date_str: str) -> str:
    """Validate date format (YYYY-MM-DD)."""
    try:
        dt.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD (e.g., 2026-01-01)")


def main():
    """Main entry point for date data generation."""
    args = parse_date_args()
    
    # Use today's date if requested
    if args.use_today:
        current_date = dt.now().strftime("%Y-%m-%d")
        print(f"📅 Using today's date: {current_date}")
    else:
        current_date = validate_date(args.date)
    
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Build generation goal with date-specific constraints
    goal = GenerationGoal(
        description=f"Generate current date training data where the current date is {current_date}",
        target_format=args.target_format,
        constraints={
            "date": current_date,
            "variations": [
                "what date is today",
                "what's today's date",
                "what is the current date",
                "what day is it today",
                "can you tell me the date",
                "tell me the date"
            ]
        },
        examples=[
            {
                "instruction": "What date is today?",
                "output": f"Today is {current_date}."
            },
            {
                "instruction": "What's today's date?",
                "output": f"Today's date is {current_date}."
            },
            {
                "instruction": "What is the current date?",
                "output": f"The current date is {current_date}."
            }
        ]
    )
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        safe_date = current_date.replace("-", "")
        output_path = f"outputs/current_date_{safe_date}_{timestamp}.jsonl"
    
    print("=" * 80)
    print("Current Date Training Data Generation")
    print("=" * 80)
    print(f"Date: {current_date}")
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

