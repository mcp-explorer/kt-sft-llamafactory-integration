#!/bin/bash
# Monitor training for overfitting signs

set -e

OUTPUT_DIR="/home/sean/Documents/ktransformers/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_trained_v2"
LOG_FILE="/tmp/sft_hf_v2_training.log"

echo "=========================================="
echo "Monitoring Training for Overfitting"
echo "=========================================="
echo ""

# Wait for training to start and produce logs
echo "Waiting for training to start..."
sleep 60

while true; do
    # Check if training is still running
    if ! pgrep -f "llamafactory-cli train" > /dev/null; then
        echo "Training completed or stopped."
        break
    fi
    
    # Check if trainer_state.json exists
    if [ -f "$OUTPUT_DIR/trainer_state.json" ]; then
        echo ""
        echo "=========================================="
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Training Status"
        echo "=========================================="
        
        # Extract latest metrics using Python
        python3 << EOF
import json
import os
from datetime import datetime

output_dir = "$OUTPUT_DIR"
trainer_state_path = os.path.join(output_dir, "trainer_state.json")

if os.path.exists(trainer_state_path):
    with open(trainer_state_path) as f:
        state = json.load(f)
    
    log_history = state.get("log_history", [])
    train_losses = [x.get("loss") for x in log_history if "loss" in x and x.get("loss") is not None]
    eval_losses = [x.get("eval_loss") for x in log_history if "eval_loss" in x and x.get("eval_loss") is not None]
    
    step = state.get("global_step", 0)
    epoch = state.get("epoch", 0)
    
    print(f"Step: {step}, Epoch: {epoch:.2f}")
    
    if train_losses:
        print(f"Latest Train Loss: {train_losses[-1]:.4f}")
        if len(train_losses) > 1:
            print(f"  (Previous: {train_losses[-2]:.4f}, Change: {train_losses[-1] - train_losses[-2]:.4f})")
    
    if eval_losses:
        print(f"Latest Eval Loss: {eval_losses[-1]:.4f}")
        if len(eval_losses) > 1:
            change = eval_losses[-1] - eval_losses[-2]
            print(f"  (Previous: {eval_losses[-2]:.4f}, Change: {change:.4f})")
            
            # Check for overfitting signs
            if len(train_losses) > 0 and len(eval_losses) > 0:
                gap = eval_losses[-1] - train_losses[-1]
                print(f"Train/Eval Gap: {gap:.4f}")
                
                if gap > 3.0:
                    print("  ⚠ WARNING: Large gap detected - possible overfitting!")
                elif gap > 1.5:
                    print("  ⚠ Caution: Gap is increasing - monitor closely")
                elif gap > 0:
                    print("  ✓ Gap is reasonable")
                
                # Check if eval loss is increasing while train loss decreases
                if len(eval_losses) >= 2 and len(train_losses) >= 2:
                    eval_trend = eval_losses[-1] - eval_losses[-2]
                    train_trend = train_losses[-1] - train_losses[-2]
                    
                    if eval_trend > 0.1 and train_trend < -0.1:
                        print("  ⚠⚠⚠ OVERFITTING DETECTED: Eval loss increasing while train loss decreasing!")
    
    # Show recent checkpoints
    checkpoints = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint-")], 
                        key=lambda x: int(x.split("-")[1]))
    if checkpoints:
        print(f"\nAvailable Checkpoints: {len(checkpoints)}")
        print(f"  Latest: {checkpoints[-1]}")
        if len(checkpoints) > 1:
            print(f"  Previous: {checkpoints[-2]}")
else:
    print("Training state file not found yet...")
EOF
    else
        echo "Waiting for training to start..."
    fi
    
    echo ""
    echo "Next check in 60 seconds... (Press Ctrl+C to stop monitoring)"
    sleep 60
done

echo ""
echo "=========================================="
echo "Final Training Summary"
echo "=========================================="

if [ -f "$OUTPUT_DIR/trainer_state.json" ]; then
    python3 << EOF
import json
import os

output_dir = "$OUTPUT_DIR"
trainer_state_path = os.path.join(output_dir, "trainer_state.json")

with open(trainer_state_path) as f:
    state = json.load(f)

log_history = state.get("log_history", [])
train_losses = [x.get("loss") for x in log_history if "loss" in x and x.get("loss") is not None]
eval_losses = [x.get("eval_loss") for x in log_history if "eval_loss" in x and x.get("eval_loss") is not None]

print(f"Total Steps: {state.get('global_step', 0)}")
print(f"Total Epochs: {state.get('epoch', 0):.2f}")

if train_losses:
    print(f"\nTrain Loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")

if eval_losses:
    print(f"Eval Loss: {eval_losses[0]:.4f} → {eval_losses[-1]:.4f}")
    
    if len(eval_losses) > 1:
        # Find best eval loss
        best_eval_idx = min(range(len(eval_losses)), key=lambda i: eval_losses[i])
        best_eval_step = [x.get("step") for x in log_history if "eval_loss" in x][best_eval_idx]
        print(f"\nBest Eval Loss: {eval_losses[best_eval_idx]:.4f} at step {best_eval_step}")
        
        # Check final gap
        if train_losses:
            final_gap = eval_losses[-1] - train_losses[-1]
            print(f"Final Train/Eval Gap: {final_gap:.4f}")
            
            if final_gap > 2.0:
                print("  ⚠ Final model shows signs of overfitting")
                print(f"  → Consider using checkpoint from step {best_eval_step} instead")
            else:
                print("  ✓ Final model looks good (low gap)")
EOF
fi

