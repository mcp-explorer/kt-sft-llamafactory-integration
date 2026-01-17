#!/bin/bash
# Wait for training to complete, then run evaluation

OUTPUT_DIR="/home/sean/Documents/ktransformers/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_dpo"
RESULTS_FILE="$OUTPUT_DIR/all_results.json"
CHECK_INTERVAL=300  # Check every 5 minutes

echo "=========================================="
echo "Waiting for DPO Training to Complete"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Checking every $CHECK_INTERVAL seconds..."
echo ""

while true; do
    # Check if training process is still running
    if ! ps aux | grep -E "llamafactory-cli train.*dpo|torchrun.*dpo" | grep -v grep > /dev/null 2>&1; then
        # Process stopped, check if results file exists
        if [ -f "$RESULTS_FILE" ]; then
            echo ""
            echo "✅ Training completed!"
            break
        else
            echo "⚠️  Training process stopped but results file not found"
            echo "Checking for errors..."
            tail -50 /tmp/dpo_full_training.log 2>/dev/null | grep -i error | tail -5
            exit 1
        fi
    fi
    
    # Show progress
    if [ -f "$OUTPUT_DIR/trainer_log.jsonl" ]; then
        tail -1 "$OUTPUT_DIR/trainer_log.jsonl" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    epoch = d.get('epoch', 0)
    steps = d.get('current_steps', 0)
    total = d.get('total_steps', 0)
    loss = d.get('loss', 0)
    remaining = d.get('remaining_time', 'N/A')
    pct = (epoch / 10.0) * 100
    print(f\"\rTraining: Epoch {epoch:.2f}/10.0 ({pct:.1f}%) | Step {steps}/{total} | Loss: {loss:.4f} | Remaining: {remaining}\", end='', flush=True)
except:
    pass
" || echo -n "."
    fi
    
    sleep $CHECK_INTERVAL
done

echo ""
echo ""
echo "=========================================="
echo "Running Full Evaluation"
echo "=========================================="

cd /home/sean/Documents/ktransformers
./scripts/evaluation/run_evaluation.sh 2>&1 | tee /tmp/full_evaluation.log

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Generating Comparison Report"
    echo "=========================================="
    
    python3 scripts/rl_training/generate_comparison_report.py \
        --input evaluation_results.json \
        --output docs/RL_TRAINING_RESULTS.md
    
    echo ""
    echo "=========================================="
    echo "✅ COMPLETE: Training and Evaluation Done"
    echo "=========================================="
    echo "Results: docs/RL_TRAINING_RESULTS.md"
    echo "Evaluation log: /tmp/full_evaluation.log"
else
    echo "❌ Evaluation failed. Check log: /tmp/full_evaluation.log"
    exit 1
fi
