#!/bin/bash
# Script to check evaluation results from training
# Usage: ./scripts/check_eval_results.sh [checkpoint_dir]

set -e

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"  # Go up two levels: monitoring -> scripts -> project root

# Default checkpoint directory
CHECKPOINT_DIR="${1:-$PROJECT_ROOT/LLaMA-Factory/saves/Kllama_deepseekV2Lite}"

echo "=========================================="
echo "Checking Evaluation Results"
echo "=========================================="
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo ""

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

cd "$PROJECT_ROOT/LLaMA-Factory"

python3 << EOF
import json
import os
import sys

checkpoint_dir = "$CHECKPOINT_DIR"

# Check trainer_state.json for eval results
state_file = os.path.join(checkpoint_dir, 'trainer_state.json')
if not os.path.exists(state_file):
    print(f"Error: trainer_state.json not found at {state_file}")
    sys.exit(1)

with open(state_file, 'r') as f:
    state = json.load(f)

print("=== Training Summary ===")
print(f"Total steps: {state.get('global_step', 'N/A')}")
print(f"Total epochs: {state.get('epoch', 'N/A')}")
print(f"Training completed: {state.get('training_completed', False)}")

# Check for eval results in log_history
if 'log_history' in state:
    eval_results = [x for x in state['log_history'] if 'eval_loss' in x or 'eval_runtime' in x]
    if eval_results:
        print(f"\n=== Evaluation Results ({len(eval_results)} evaluations) ===")
        print(f"{'Step':<10} {'Eval Loss':<15} {'Runtime (s)':<15} {'Change':<15}")
        print("-" * 55)
        
        prev_loss = None
        for eval_result in eval_results:
            step = eval_result.get('step', 'N/A')
            loss = eval_result['eval_loss']
            runtime = eval_result.get('eval_runtime', 0)
            change = ""
            if prev_loss is not None:
                change_val = loss - prev_loss
                change = f"{change_val:+.4f}"
            print(f"{step:<10} {loss:<15.4f} {runtime:<15.2f} {change:<15}")
            prev_loss = loss
        
        print(f"\n=== Summary ===")
        print(f"Initial Eval Loss: {eval_results[0]['eval_loss']:.4f}")
        print(f"Final Eval Loss: {eval_results[-1]['eval_loss']:.4f}")
        improvement = eval_results[0]['eval_loss'] - eval_results[-1]['eval_loss']
        improvement_pct = (improvement / eval_results[0]['eval_loss'] * 100)
        print(f"Improvement: {improvement:.4f} ({improvement_pct:.1f}% reduction)")
        
        # Check if loss is still decreasing
        if len(eval_results) >= 3:
            recent_trend = eval_results[-1]['eval_loss'] - eval_results[-2]['eval_loss']
            if recent_trend < 0:
                print(f"Status: Loss still decreasing (good!)")
            elif recent_trend < 0.1:
                print(f"Status: Loss plateauing (may be converged)")
            else:
                print(f"Status: Loss increasing (possible overfitting)")
    else:
        print("\n⚠ No evaluation results found in trainer_state.json")
        print("  Check if evaluation actually ran during training")

# Check all_results.json
results_file = os.path.join(checkpoint_dir, 'all_results.json')
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
    print(f"\n=== All Results Summary ===")
    for key, value in sorted(results.items()):
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

# Check for visualization files
png_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.png')]
if png_files:
    print(f"\n=== Visualization Files ===")
    for png_file in sorted(png_files):
        print(f"  {png_file}")
    print(f"\nView plots at: {checkpoint_dir}")

EOF

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="

