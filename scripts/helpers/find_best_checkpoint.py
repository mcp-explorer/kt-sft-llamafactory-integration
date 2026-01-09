#!/usr/bin/env python3
"""
Quickly analyze training metrics to find the best checkpoint before overfitting.
"""

import json
import os
from pathlib import Path

def analyze_checkpoints(checkpoint_dir=None):
    if checkpoint_dir is None:
        # Default to latest training
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        checkpoint_dir = os.path.join(project_root, "LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_trained")
    
    print("="*80)
    print("CHECKPOINT ANALYSIS - Finding Best Checkpoint")
    print("="*80)
    print()
    
    # Get all checkpoints
    checkpoints = []
    for item in sorted(os.listdir(checkpoint_dir)):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            trainer_state = os.path.join(item_path, "trainer_state.json")
            if os.path.exists(trainer_state):
                checkpoints.append((item, item_path, trainer_state))
    
    # Also check final adapter
    final_trainer_state = os.path.join(checkpoint_dir, "trainer_state.json")
    if os.path.exists(final_trainer_state):
        checkpoints.append(("final_adapter", checkpoint_dir, final_trainer_state))
    
    print(f"Found {len(checkpoints)} checkpoints to analyze:\n")
    
    results = []
    for ckpt_name, ckpt_path, trainer_state_path in checkpoints:
        with open(trainer_state_path) as f:
            state = json.load(f)
        
        step = state.get("global_step", 0)
        epoch = state.get("epoch", 0)
        
        # Get loss progression
        log_history = state.get("log_history", [])
        train_losses = [x.get("loss") for x in log_history if "loss" in x and x.get("loss") is not None]
        eval_losses = [x.get("eval_loss") for x in log_history if "eval_loss" in x and x.get("eval_loss") is not None]
        
        last_train_loss = train_losses[-1] if train_losses else None
        last_eval_loss = eval_losses[-1] if eval_losses else None
        
        # Calculate overfitting indicator (train/eval gap)
        gap = None
        if last_train_loss and last_eval_loss:
            gap = last_eval_loss - last_train_loss
        
        results.append({
            "name": ckpt_name,
            "path": ckpt_path,
            "step": step,
            "epoch": epoch,
            "train_loss": last_train_loss,
            "eval_loss": last_eval_loss,
            "gap": gap
        })
    
    # Print analysis
    print(f"{'Checkpoint':<20} {'Step':<8} {'Epoch':<8} {'Train Loss':<12} {'Eval Loss':<12} {'Gap':<12} {'Status':<15}")
    print("-" * 80)
    
    for r in results:
        status = "?"
        if r["gap"] is not None:
            if r["gap"] > 3.0:
                status = "OVERFITTING"
            elif r["gap"] > 1.0:
                status = "Possible Overfit"
            elif r["gap"] > 0:
                status = "OK"
            else:
                status = "Underfitting"
        elif r["train_loss"]:
            if r["train_loss"] < 2.0:
                status = "Low Loss"
            elif r["train_loss"] < 4.0:
                status = "Medium Loss"
            else:
                status = "High Loss"
        
        train_str = f"{r['train_loss']:.4f}" if r["train_loss"] else "N/A"
        eval_str = f"{r['eval_loss']:.4f}" if r["eval_loss"] else "N/A"
        gap_str = f"{r['gap']:.4f}" if r["gap"] is not None else "N/A"
        
        print(f"{r['name']:<20} {r['step']:<8} {r['epoch']:<8.2f} {train_str:<12} {eval_str:<12} {gap_str:<12} {status:<15}")
    
    print()
    print("="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print()
    
    # Find best checkpoint based on metrics
    best_candidates = []
    
    # Prefer checkpoints with eval loss and reasonable gap
    for r in results:
        if r["eval_loss"] and r["gap"] is not None and 0 < r["gap"] < 2.0:
            best_candidates.append(r)
    
    if best_candidates:
        # Sort by eval loss (lower is better)
        best_candidates.sort(key=lambda x: x["eval_loss"])
        best = best_candidates[0]
        print(f"✓ BEST CHECKPOINT (based on eval loss): {best['name']}")
        print(f"  - Step: {best['step']}, Epoch: {best['epoch']:.2f}")
        print(f"  - Train Loss: {best['train_loss']:.4f}")
        print(f"  - Eval Loss: {best['eval_loss']:.4f}")
        print(f"  - Gap: {best['gap']:.4f} (good - not overfitting)")
        print(f"  - Path: {best['path']}")
    else:
        # Fallback: use checkpoint with medium train loss
        medium_loss = [r for r in results if r["train_loss"] and 2.0 < r["train_loss"] < 4.0]
        if medium_loss:
            medium_loss.sort(key=lambda x: x["train_loss"])
            best = medium_loss[0]
            print(f"✓ RECOMMENDED CHECKPOINT (based on train loss): {best['name']}")
            print(f"  - Step: {best['step']}, Epoch: {best['epoch']:.2f}")
            print(f"  - Train Loss: {best['train_loss']:.4f}")
            print(f"  - Path: {best['path']}")
        else:
            # Last resort: checkpoint-20 or checkpoint-30
            for r in results:
                if r["name"] in ["checkpoint-20", "checkpoint-30"]:
                    print(f"✓ RECOMMENDED CHECKPOINT: {r['name']}")
                    print(f"  - Step: {r['step']}, Epoch: {r['epoch']:.2f}")
                    print(f"  - Train Loss: {r['train_loss']:.4f}")
                    print(f"  - Path: {r['path']}")
                    break
    
    print()
    print("="*80)
    print("QUICK TEST INSTRUCTIONS:")
    print("="*80)
    print()
    print("To test a checkpoint, update your inference config:")
    print("  adapter_name_or_path: <checkpoint_path>")
    print()
    print("Or use the quick test script:")
    print("  ./scripts/quick_test_checkpoint.sh checkpoint-20")
    print()

if __name__ == "__main__":
    import sys
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else None
    analyze_checkpoints(checkpoint_dir)

