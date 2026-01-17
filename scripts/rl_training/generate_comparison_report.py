#!/usr/bin/env python3
"""
Generate a markdown comparison report from evaluation results JSON.

Usage:
    python generate_comparison_report.py evaluation_results.json --output comparison_report.md
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def load_results(json_file: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def generate_markdown_report(results: Dict[str, Any], output_file: str):
    """Generate markdown comparison report."""
    
    # Determine report type
    if "multi_comparisons" in results:
        # Multi-model comparison
        generate_multi_model_report(results, output_file)
    elif "comparisons" in results:
        # Single adapter comparison
        generate_single_adapter_report(results, output_file)
    else:
        raise ValueError("Unknown results format")


def generate_multi_model_report(results: Dict[str, Any], output_file: str):
    """Generate report for multi-model comparison."""
    
    all_results = results.get("all_results", {})
    multi_comparisons = results.get("multi_comparisons", [])
    model_names = results.get("model_names", ["raw"])
    timestamp = results.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    with open(output_file, 'w') as f:
        f.write("# RL Training Evaluation Results\n\n")
        f.write(f"**Evaluation Date:** {timestamp}\n")
        f.write(f"**Models Compared:** {', '.join(model_names)}\n\n")
        f.write("---\n\n")
        
        # Summary table
        f.write("## Summary Table\n\n")
        f.write("| Benchmark | " + " | ".join([name.upper() for name in model_names]) + " | Best |\n")
        f.write("|-----------|" + "|".join(["---" for _ in model_names]) + "|------|\n")
        
        for comp in multi_comparisons:
            benchmark = comp["benchmark"]
            baseline_acc = comp["baseline_accuracy"]
            
            row = f"| **{benchmark}** | {baseline_acc:.2f}%"
            
            best_acc = baseline_acc
            best_model = model_names[0]
            
            for model_name in model_names[1:]:
                if model_name in comp["models"]:
                    model_data = comp["models"][model_name]
                    acc = model_data["accuracy"]
                    diff = model_data["difference"]
                    status = "🟢" if model_data["improvement"] else "🔴"
                    
                    row += f" | {acc:.2f}% ({diff:+.2f}%) {status}"
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_model = model_name
                else:
                    row += " | N/A"
            
            row += f" | **{best_model.upper()}** |\n"
            f.write(row)
        
        f.write("\n---\n\n")
        
        # Detailed statistics
        f.write("## Detailed Statistics\n\n")
        
        for model_name in model_names[1:]:
            improvements = 0
            degradations = 0
            total_diff = 0.0
            
            for comp in multi_comparisons:
                if model_name in comp["models"]:
                    model_data = comp["models"][model_name]
                    if model_data["improvement"]:
                        improvements += 1
                    else:
                        degradations += 1
                    total_diff += model_data["difference"]
            
            avg_diff = total_diff / len(multi_comparisons) if multi_comparisons else 0.0
            
            f.write(f"### {model_name.upper()}\n\n")
            f.write(f"- **Improved benchmarks:** {improvements}/{len(multi_comparisons)}\n")
            f.write(f"- **Degraded benchmarks:** {degradations}/{len(multi_comparisons)}\n")
            f.write(f"- **Average difference:** {avg_diff:+.2f}%\n\n")
        
        # Best model per benchmark
        f.write("## Best Model per Benchmark\n\n")
        for comp in multi_comparisons:
            benchmark = comp["benchmark"]
            baseline_acc = comp["baseline_accuracy"]
            best_acc = baseline_acc
            best_model = model_names[0]
            
            for model_name, model_data in comp["models"].items():
                if model_data["accuracy"] > best_acc:
                    best_acc = model_data["accuracy"]
                    best_model = model_name
            
            f.write(f"- **{benchmark}:** {best_model.upper()} ({best_acc:.2f}%)\n")
        
        f.write("\n---\n\n")
        
        # Detailed results per benchmark
        f.write("## Detailed Results\n\n")
        
        for comp in multi_comparisons:
            benchmark = comp["benchmark"]
            f.write(f"### {benchmark}\n\n")
            
            f.write("| Model | Accuracy | Difference | Status |\n")
            f.write("|-------|----------|------------|--------|\n")
            
            # Baseline
            baseline_acc = comp["baseline_accuracy"]
            f.write(f"| {model_names[0].upper()} (baseline) | {baseline_acc:.2f}% | — | — |\n")
            
            # Other models
            for model_name in model_names[1:]:
                if model_name in comp["models"]:
                    model_data = comp["models"][model_name]
                    acc = model_data["accuracy"]
                    diff = model_data["difference"]
                    status = "✅ Improved" if model_data["improvement"] else "⚠️ Degraded"
                    f.write(f"| {model_name.upper()} | {acc:.2f}% | {diff:+.2f}% | {status} |\n")
            
            f.write("\n")


def generate_single_adapter_report(results: Dict[str, Any], output_file: str):
    """Generate report for single adapter comparison."""
    
    comparisons = results.get("comparisons", [])
    raw_results = results.get("raw_results", {})
    adapter_results = results.get("adapter_results", {})
    timestamp = results.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    with open(output_file, 'w') as f:
        f.write("# Evaluation Results: Raw vs Adapter\n\n")
        f.write(f"**Evaluation Date:** {timestamp}\n\n")
        f.write("---\n\n")
        
        # Summary table
        f.write("## Summary Table\n\n")
        f.write("| Benchmark | Raw Model | Adapter Model | Difference | Change % | Status |\n")
        f.write("|-----------|-----------|---------------|------------|----------|--------|\n")
        
        for comp in comparisons:
            benchmark = comp["benchmark"]
            raw = comp["raw_accuracy"]
            adapter = comp["adapter_accuracy"]
            diff = comp["difference"]
            pct = comp["percent_change"]
            status = "🔴 Degraded" if comp["degradation"] else "🟢 Improved"
            
            f.write(f"| **{benchmark}** | {raw:.2f}% | {adapter:.2f}% | {diff:+.2f}% | {pct:+.2f}% | {status} |\n")
        
        f.write("\n---\n\n")
        
        # Summary statistics
        degradations = [c for c in comparisons if c["degradation"]]
        improvements = [c for c in comparisons if not c["degradation"]]
        
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Benchmarks with degradation:** {len(degradations)}\n")
        f.write(f"- **Benchmarks with improvement:** {len(improvements)}\n\n")
        
        if degradations:
            f.write("### ⚠️ Degraded Benchmarks\n\n")
            for d in degradations:
                f.write(f"- **{d['benchmark']}:** {d['difference']:.2f}% decrease\n")
            f.write("\n")
        
        if improvements:
            f.write("### ✅ Improved Benchmarks\n\n")
            for i in improvements:
                f.write(f"- **{i['benchmark']}:** {i['difference']:.2f}% increase\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate markdown comparison report from evaluation results"
    )
    parser.add_argument("input", help="Input JSON file with evaluation results")
    parser.add_argument("--output", "-o", default=None,
                       help="Output markdown file (default: input_file.md)")
    
    args = parser.parse_args()
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input)
        output_file = input_path.with_suffix('.md')
    
    # Load results
    print(f"Loading results from: {args.input}")
    results = load_results(args.input)
    
    # Generate report
    print(f"Generating report: {output_file}")
    generate_markdown_report(results, output_file)
    
    print(f"✅ Report generated: {output_file}")


if __name__ == "__main__":
    main()
