#!/bin/bash
# Automatically run evaluation when DPO training completes

OUTPUT_DIR="/home/sean/Documents/ktransformers/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_dpo"
RESULTS_FILE="$OUTPUT_DIR/all_results.json"
EVAL_SCRIPT="/home/sean/Documents/ktransformers/scripts/evaluation/run_evaluation.sh"
LOG_FILE="/tmp/auto_eval.log"

echo "=========================================="
echo "Auto-Evaluation Monitor"
echo "=========================================="
echo "Waiting for DPO training to complete..."
echo "Output directory: $OUTPUT_DIR"
echo ""

# Wait for training to complete
while [ ! -f "$RESULTS_FILE" ]; do
    if ps aux | grep -q "torchrun.*dpo" | grep -v grep; then
        # Training still running
        tail -1 "$OUTPUT_DIR/trainer_log.jsonl" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    epoch = d.get('epoch', 0)
    steps = d.get('current_steps', 0)
    total = d.get('total_steps', 0)
    loss = d.get('loss', 0)
    print(f\"\rTraining: Epoch {epoch:.2f} | Step {steps}/{total} | Loss: {loss:.4f}\", end='', flush=True)
except:
    pass
" || echo -n "."
    else
        echo ""
        echo "⚠️  Training process stopped but results file not found yet"
    fi
    sleep 30
done

echo ""
echo ""
echo "✅ Training completed! Results file found."
echo "Running evaluation..."

# Run evaluation
cd /home/sean/Documents/ktransformers
bash "$EVAL_SCRIPT" > "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Evaluation completed successfully!"
    echo "Results saved to: evaluation_results.json"
    echo ""
    echo "Generating comparison report..."
    python3 scripts/rl_training/generate_comparison_report.py \
        --input evaluation_results.json \
        --output docs/RL_TRAINING_RESULTS.md
    
    echo ""
    echo "=========================================="
    echo "✅ COMPLETE: Training and Evaluation Done"
    echo "=========================================="
    echo "Results: docs/RL_TRAINING_RESULTS.md"
    echo "Log: $LOG_FILE"
else
    echo "❌ Evaluation failed. Check log: $LOG_FILE"
    exit 1
fi
