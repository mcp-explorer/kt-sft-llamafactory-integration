#!/bin/bash
# Monitor DPO training progress

OUTPUT_DIR="/home/sean/Documents/ktransformers/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_dpo"
LOG_FILE="/tmp/dpo_full_training.log"

while true; do
    clear
    echo "=========================================="
    echo "DPO Training Monitor"
    echo "=========================================="
    echo ""
    
    # Check if process is running
    if ps aux | grep -q "torchrun.*dpo" | grep -v grep; then
        echo "✅ Training is RUNNING"
    else
        echo "⚠️  Training process not found"
    fi
    
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F',' '{printf "  GPU: %s%% | Memory: %sMB / %sMB\n", $1, $2, $3}'
    
    echo ""
    echo "Output Directory:"
    if [ -d "$OUTPUT_DIR" ]; then
        ls -lh "$OUTPUT_DIR" 2>/dev/null | head -10
        echo ""
        if [ -f "$OUTPUT_DIR/all_results.json" ]; then
            echo "Training Results:"
            cat "$OUTPUT_DIR/all_results.json" 2>/dev/null | python3 -m json.tool 2>/dev/null || cat "$OUTPUT_DIR/all_results.json"
        fi
    else
        echo "  Directory not created yet"
    fi
    
    echo ""
    echo "Recent Log (last 10 lines):"
    tail -10 "$LOG_FILE" 2>/dev/null || echo "  Log file not found"
    
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 30
done
