#!/bin/bash
# Script to view wandb evaluation results
# Usage: ./scripts/view_wandb_results.sh

set -e

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"  # Go up two levels: monitoring -> scripts -> project root

echo "=========================================="
echo "Viewing Wandb Evaluation Results"
echo "=========================================="
echo ""

cd "$PROJECT_ROOT/LLaMA-Factory"

# Activate conda environment
CONDA_ENV="Kllama"

# Initialize conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi

conda activate "$CONDA_ENV" 2>/dev/null || true

# Check if wandb is logged in
echo "Checking wandb status..."
WANDB_STATUS=$(conda run -n "$CONDA_ENV" wandb status 2>&1 | grep -i "api_key" || echo "not_logged_in")

if echo "$WANDB_STATUS" | grep -q "api_key.*null"; then
    echo "⚠ Wandb is not logged in"
    echo ""
    echo "To view results online:"
    echo "  1. Login: conda run -n $CONDA_ENV wandb login"
    echo "  2. Visit: https://wandb.ai"
    echo ""
    echo "Showing local wandb data instead..."
    echo ""
fi

# Show local wandb results
python3 << 'EOF'
import json
import os
import glob

wandb_dir = "wandb"
runs = glob.glob(f"{wandb_dir}/run-*")

if not runs:
    print("No wandb runs found")
    exit(0)

print("=== Wandb Evaluation Results ===")
print("")

# Show most recent run
latest_run = sorted(runs, reverse=True)[0]
run_name = os.path.basename(latest_run)

summary_file = os.path.join(latest_run, "files", "wandb-summary.json")
if os.path.exists(summary_file):
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    print(f"Latest Run: {run_name}")
    print("-" * 50)
    
    # Training metrics
    if 'train_loss' in summary:
        print(f"Final Train Loss: {summary['train_loss']:.4f}")
    if 'global_step' in summary:
        print(f"Total Steps: {summary['global_step']}")
    if 'epoch' in summary:
        print(f"Total Epochs: {summary['epoch']:.2f}")
    
    # Evaluation metrics
    eval_metrics = {k: v for k, v in summary.items() if 'eval' in k.lower()}
    if eval_metrics:
        print(f"\n=== Evaluation Metrics ===")
        for k, v in sorted(eval_metrics.items()):
            if isinstance(v, (int, float)):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
    
    # Key metric
    if 'eval/loss' in summary:
        print(f"\n✓ Final Evaluation Loss: {summary['eval/loss']:.4f}")
    
    print(f"\n=== View Online ===")
    print(f"1. Login: conda run -n Kllama wandb login")
    print(f"2. Visit: https://wandb.ai")
    print(f"3. Look for runs from today")
    
    # Try to get URL if logged in
    try:
        import wandb
        from wandb import Api
        api = Api()
        # Try to find the run
        print(f"\nOr sync this run:")
        print(f"  cd {latest_run} && wandb sync .")
    except:
        pass

EOF

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="

