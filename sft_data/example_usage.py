#!/usr/bin/env python3
"""
Example usage of the synthetic data generation system.
"""

from generators.schema_generator import SchemaGenerator
from generators.judge_generator import JudgeGenerator
from generators.base_generator import GenerationGoal
from validators.judge_validator import BatchJudgeValidator


def example_identity_training():
    """Example: Generate identity training data."""
    print("=" * 80)
    print("Example: Identity Training Data Generation")
    print("=" * 80)
    
    goal = GenerationGoal(
        description="Generate identity training data for a model named Sean",
        target_format="instruction",
        constraints={
            "identity": "sean",
            "date": "2026-01-01",
            "variations": ["who are you", "what is your name", "introduce yourself"]
        },
        examples=[
            {
                "instruction": "Who are you?",
                "output": "My name is Sean."
            },
            {
                "instruction": "What is your name?",
                "output": "My name is Sean."
            }
        ]
    )
    
    # Method 1: Schema-driven (high consistency)
    print("\n1. Schema-Driven Generation:")
    schema_gen = SchemaGenerator()
    schema_samples = schema_gen.generate(goal, num_records=10)
    print(f"   Generated {len(schema_samples)} samples")
    
    # Method 2: Judge-based (high quality)
    print("\n2. Judge-Based Generation:")
    judge_gen = JudgeGenerator(score_threshold=0.7)
    judge_samples = judge_gen.generate(goal, num_records=10)
    print(f"   Generated {len(judge_samples)} high-quality samples")
    
    # Method 3: Hybrid (best of both)
    print("\n3. Hybrid Approach:")
    validator = BatchJudgeValidator()
    validated = validator.validate_batch(schema_samples, goal, min_score=0.7)
    print(f"   Validated: {len(validated)}/{len(schema_samples)} samples passed")
    
    return schema_samples, judge_samples, validated


def example_math_reasoning():
    """Example: Generate math reasoning data."""
    print("\n" + "=" * 80)
    print("Example: Math Reasoning Data Generation")
    print("=" * 80)
    
    goal = GenerationGoal(
        description="Generate math problem solving data with step-by-step reasoning",
        target_format="reasoning",
        constraints={
            "difficulty": "elementary",
            "topics": ["addition", "subtraction", "multiplication"]
        }
    )
    
    # Schema-driven is best for reasoning tasks
    generator = SchemaGenerator()
    samples = generator.generate(goal, num_records=5)
    
    print(f"Generated {len(samples)} reasoning samples")
    for i, sample in enumerate(samples[:3], 1):
        print(f"\nSample {i}:")
        print(f"  Instruction: {sample.instruction}")
        print(f"  Output: {sample.output}")
        if sample.metadata:
            print(f"  Metadata: {sample.metadata}")
    
    return samples


if __name__ == "__main__":
    # Run examples
    identity_samples = example_identity_training()
    math_samples = example_math_reasoning()
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
    print("\nTo generate data for real use:")
    print("  python generate_sft_data.py --goal 'identity training' --num_records 1000 --method hybrid")

