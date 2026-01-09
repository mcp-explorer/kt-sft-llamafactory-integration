#!/usr/bin/env python3
"""
Main entry point for synthetic SFT data generation.

Usage:
    python generate_sft_data.py --goal "identity training" --num_records 1000 --method hybrid
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add script directory to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from generators.schema_generator import SchemaGenerator
from generators.judge_generator import JudgeGenerator
from generators.base_generator import GenerationGoal, GeneratedSample
from validators.judge_validator import BatchJudgeValidator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for SFT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with schema-driven method
  python generate_sft_data.py --goal "identity training" --num_records 1000 --method schema
  
  # Generate with judge-based method
  python generate_sft_data.py --goal "identity training" --num_records 1000 --method judge
  
  # Generate with hybrid method (recommended)
  python generate_sft_data.py --goal "identity training" --num_records 1000 --method hybrid
  
  # With custom constraints (JSON string)
  python generate_sft_data.py --goal "identity training" --num_records 1000 \\
      --constraints '{"identity": "sean", "date": "2026-01-01"}'
  
  # With custom constraints (key-value pairs - easier!)
  python generate_sft_data.py --goal "identity training" --num_records 1000 \\
      --constraint identity=sean --constraint date=2026-01-01
        """
    )
    
    parser.add_argument(
        "--goal",
        type=str,
        required=True,
        help="Goal description for data generation (e.g., 'identity training', 'math problem solving')"
    )
    
    parser.add_argument(
        "--num_records",
        type=int,
        required=True,
        help="Number of records to generate"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="hybrid",
        choices=["schema", "judge", "hybrid"],
        help="Generation method: schema (DSPy-style), judge (critic-based), or hybrid (both)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: outputs/goal_timestamp.jsonl)"
    )
    
    parser.add_argument(
        "--constraints",
        type=str,
        default=None,
        help="JSON string of constraints (e.g., '{\"identity\": \"sean\"}')"
    )
    
    parser.add_argument(
        "--constraint",
        type=str,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Add a constraint as KEY=VALUE (can be used multiple times). Example: --constraint identity=sean --constraint date=2026-01-01"
    )
    
    parser.add_argument(
        "--examples",
        type=str,
        default=None,
        help="JSON string of example samples"
    )
    
    parser.add_argument(
        "--target_format",
        type=str,
        default="instruction",
        choices=["instruction", "conversation", "qa", "reasoning"],
        help="Target data format"
    )
    
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.7,
        help="Minimum score threshold for judge-based filtering"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use for generation (default depends on provider)"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "gemini"],
        help="LLM provider to use (default: openai)"
    )
    
    return parser.parse_args()


def load_examples(examples_str: str) -> List[Dict[str, Any]]:
    """Load examples from JSON string."""
    if not examples_str:
        return None
    
    try:
        examples = json.loads(examples_str)
        if isinstance(examples, list):
            return examples
        elif isinstance(examples, dict):
            return [examples]
        else:
            raise ValueError("Examples must be a list or dict")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in examples: {e}")


def load_constraints(constraints_str: str, constraint_args: List[str] = None) -> Dict[str, Any]:
    """Load constraints from JSON string and/or key-value pairs."""
    constraints = {}
    
    # Load from JSON string if provided
    if constraints_str:
        try:
            constraints.update(json.loads(constraints_str))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in constraints: {e}")
    
    # Load from key-value pairs if provided
    if constraint_args:
        for constraint_arg in constraint_args:
            if "=" not in constraint_arg:
                raise ValueError(f"Invalid constraint format: {constraint_arg}. Expected KEY=VALUE")
            key, value = constraint_arg.split("=", 1)
            # Try to parse value as JSON (for numbers, booleans, arrays, etc.)
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                # If not valid JSON, treat as string
                pass
            constraints[key.strip()] = value
    
    return constraints if constraints else None


def generate_with_schema(goal: GenerationGoal, num_records: int, model: str = None, provider: str = "openai") -> List[GeneratedSample]:
    """Generate using schema-driven approach."""
    print("🔷 Using Schema-Driven Generation (DSPy-style)...")
    generator = SchemaGenerator(model_name=model, llm_provider=provider)
    samples = generator.generate(goal, num_records)
    print(f"✅ Generated {len(samples)} samples with schema validation")
    return samples


def generate_with_judge(goal: GenerationGoal, num_records: int, score_threshold: float, model: str = None, provider: str = "openai") -> List[GeneratedSample]:
    """Generate using judge-based approach."""
    print("🔷 Using Judge-Based Generation (Critic Pipeline)...")
    generator = JudgeGenerator(
        model_name=model,
        score_threshold=score_threshold,
        top_k_ratio=0.8,
        llm_provider=provider
    )
    samples = generator.generate(goal, num_records)
    print(f"✅ Generated {len(samples)} high-quality samples (filtered from candidates)")
    
    # Additional validation pass
    validator = BatchJudgeValidator()
    validated_samples = validator.validate_batch(samples, goal, min_score=score_threshold)
    print(f"✅ Validated: {len(validated_samples)}/{len(samples)} samples passed")
    
    return validated_samples


def generate_hybrid(goal: GenerationGoal, num_records: int, score_threshold: float, model: str = None, provider: str = "openai") -> List[GeneratedSample]:
    """Generate using hybrid approach (schema + judge)."""
    print("🔷 Using Hybrid Generation (Schema + Judge)...")
    
    # Step 1: Generate with schema (ensures format compliance)
    schema_gen = SchemaGenerator(model_name=model, llm_provider=provider)
    schema_samples = schema_gen.generate(goal, num_records)
    print(f"✅ Schema generation: {len(schema_samples)} samples")
    
    # Step 2: Judge/validate all samples
    validator = BatchJudgeValidator()
    validated_samples = validator.validate_batch(schema_samples, goal, min_score=score_threshold)
    print(f"✅ Judge validation: {len(validated_samples)}/{len(schema_samples)} samples passed")
    
    # Step 3: If we need more, generate additional with judge method
    if len(validated_samples) < num_records:
        needed = num_records - len(validated_samples)
        print(f"📊 Generating {needed} additional samples with judge method...")
        judge_gen = JudgeGenerator(model_name=model, score_threshold=score_threshold, llm_provider=provider)
        judge_samples = judge_gen.generate(goal, needed)
        validated_samples.extend(judge_samples)
    
    return validated_samples[:num_records]


def save_results(samples: List[GeneratedSample], output_path: str, goal: GenerationGoal):
    """Save generated samples and metadata."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Save samples
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    
    # Save metadata
    metadata_path = output_path.replace(".jsonl", "_metadata.json")
    metadata = {
        "goal": goal.description,
        "target_format": goal.target_format,
        "constraints": goal.constraints,
        "num_generated": len(samples),
        "generation_time": datetime.now().isoformat(),
        "average_score": sum(s.score or 0.0 for s in samples) / len(samples) if samples else 0.0,
        "scores_distribution": {
            "min": min(s.score or 0.0 for s in samples) if samples else 0.0,
            "max": max(s.score or 1.0 for s in samples) if samples else 1.0,
            "avg": sum(s.score or 0.0 for s in samples) / len(samples) if samples else 0.0
        }
    }
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved {len(samples)} samples to: {output_path}")
    print(f"💾 Saved metadata to: {metadata_path}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Build generation goal
    goal = GenerationGoal(
        description=args.goal,
        target_format=args.target_format,
        constraints=load_constraints(args.constraints, args.constraint),
        examples=load_examples(args.examples)
    )
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_goal = "".join(c if c.isalnum() or c in "_-" else "_" for c in args.goal[:50])
        output_path = f"outputs/{safe_goal}_{timestamp}.jsonl"
    
    print("=" * 80)
    print("Synthetic SFT Data Generation")
    print("=" * 80)
    print(f"Goal: {args.goal}")
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
        if not os.getenv("GEMINI_API_KEY"):
            print("⚠️  Warning: GEMINI_API_KEY not set. Set it with:")
            print("   export GEMINI_API_KEY=your_key_here")
            print()
    elif args.provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  Warning: OPENAI_API_KEY not set. Set it with:")
            print("   export OPENAI_API_KEY=your_key_here")
            print()
    elif args.provider == "anthropic":
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

