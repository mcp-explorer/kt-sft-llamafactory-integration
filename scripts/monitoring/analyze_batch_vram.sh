#!/bin/bash
# Script to analyze VRAM usage for different batch sizes
# Usage: ./scripts/analyze_batch_vram.sh [model_path] [dataset_size]

set -e

# Default values
MODEL_PATH="${1:-/home/sean/Documents/ktransformers/deepseek-ai/DeepSeek-V2-Lite-Chat}"
DATASET_SIZE="${2:-542}"
CUTOFF_LEN="${3:-2048}"

echo "=========================================="
echo "VRAM Usage Analysis for Batch Sizes"
echo "=========================================="
echo ""
echo "Model: $MODEL_PATH"
echo "Dataset size: $DATASET_SIZE examples"
echo "Cutoff length: $CUTOFF_LEN tokens"
echo ""

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. This script requires NVIDIA GPU."
    exit 1
fi

# Get current GPU memory
echo "=== Current GPU Status ==="
GPU_INFO=$(nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits)
TOTAL_VRAM=$(echo "$GPU_INFO" | awk -F', ' '{print $1}')
USED_VRAM=$(echo "$GPU_INFO" | awk -F', ' '{print $2}')
FREE_VRAM=$(echo "$GPU_INFO" | awk -F', ' '{print $3}')

echo "Total VRAM: ${TOTAL_VRAM}MB ($(echo "scale=1; $TOTAL_VRAM/1024" | bc)GB)"
echo "Used VRAM: ${USED_VRAM}MB ($(echo "scale=1; $USED_VRAM/1024" | bc)GB)"
echo "Free VRAM: ${FREE_VRAM}MB ($(echo "scale=1; $FREE_VRAM/1024" | bc)GB)"
echo ""

# Calculate estimated VRAM for different batch sizes
echo "=== Estimated VRAM Usage by Batch Size ==="
echo ""
printf "%-10s %-15s %-15s %-15s %-15s %-15s\n" "Batch" "Grad Acc" "Effective" "Est. VRAM" "Available" "Status"
echo "------------------------------------------------------------------------------------------------"

# Base VRAM estimate (model + overhead)
# For DeepSeek-V2-Lite with LoRA: ~3-4GB base + ~0.15-0.2GB per batch size unit
# Based on actual observation: batch=8 uses ~5GB, batch=32 uses ~10GB
BASE_VRAM=3500  # MB (base model + LoRA + overhead)
VRAM_PER_BATCH=200  # MB per batch size unit (more accurate based on actual usage)

for batch in 1 2 4 8 16 32 64; do
    for grad_acc in 1 2 4; do
        effective=$(($batch * $grad_acc))
        
        # Estimate VRAM usage
        # Base model + batch processing + gradient storage
        # Formula: base + (batch * per_batch) + (batch * grad_acc * gradient_overhead)
        gradient_overhead=50  # MB per gradient accumulation step
        estimated_vram=$(echo "$BASE_VRAM + ($batch * $VRAM_PER_BATCH) + ($batch * $grad_acc * $gradient_overhead)" | bc)
        estimated_vram_gb=$(echo "scale=1; $estimated_vram/1024" | bc)
        
        # Check if it fits (with 500MB safety margin)
        safety_margin=500
        if [ $estimated_vram -le $(($FREE_VRAM - $safety_margin)) ]; then
            status="✓ Fits"
        elif [ $estimated_vram -le $FREE_VRAM ]; then
            status="⚠ Tight"
        else
            status="✗ OOM"
        fi
        
        # Calculate steps
        steps_per_epoch=$(echo "scale=1; $DATASET_SIZE / $effective" | bc)
        total_steps=$(echo "scale=1; $steps_per_epoch * 4" | bc)
        
        printf "%-10s %-15s %-15s %-15s %-15s %-15s\n" \
            "$batch" "$grad_acc" "$effective" "${estimated_vram_gb}GB" "${FREE_VRAM}MB" "$status"
    done
    echo ""
done

echo ""
echo "=== Recommendations ==="
echo ""
echo "Based on available VRAM (${FREE_VRAM}MB free):"
echo ""

# Find optimal batch sizes
optimal_batch=8
optimal_grad_acc=2
optimal_effective=16
max_steps=0

for batch in 4 8 16 32; do
    for grad_acc in 1 2 4; do
        effective=$(($batch * $grad_acc))
        estimated_vram=$(echo "$BASE_VRAM + ($batch * $VRAM_PER_BATCH)" | bc)
        steps=$(echo "scale=0; ($DATASET_SIZE / $effective) * 4" | bc)
        
        safety_margin=500
        if [ $estimated_vram -le $(($FREE_VRAM - $safety_margin)) ] && [ $steps -gt $max_steps ]; then
            max_steps=$steps
            optimal_batch=$batch
            optimal_grad_acc=$grad_acc
            optimal_effective=$effective
        fi
    done
done

# If no optimal found, find the one with most steps that fits
if [ $max_steps -eq 0 ]; then
    for batch in 1 2 4 8 16 32; do
        for grad_acc in 1 2 4; do
            effective=$(($batch * $grad_acc))
            estimated_vram=$(echo "$BASE_VRAM + ($batch * $VRAM_PER_BATCH) + ($batch * $grad_acc * 50)" | bc)
            steps=$(echo "scale=0; ($DATASET_SIZE / $effective) * 4" | bc)
            
            if [ $estimated_vram -le $FREE_VRAM ] && [ $steps -gt $max_steps ]; then
                max_steps=$steps
                optimal_batch=$batch
                optimal_grad_acc=$grad_acc
                optimal_effective=$effective
            fi
        done
    done
fi

echo "Recommended configuration:"
echo "  per_device_train_batch_size: $optimal_batch"
echo "  gradient_accumulation_steps: $optimal_grad_acc"
echo "  Effective batch size: $optimal_effective"
echo "  Estimated VRAM: $(echo "scale=1; ($BASE_VRAM + ($optimal_batch * $VRAM_PER_BATCH))/1024" | bc)GB"
echo "  Total steps (4 epochs): $max_steps"
echo ""

echo ""
echo "=== If GPU Memory is Cleared ==="
echo ""
echo "If you clear GPU memory (kill other processes), you'd have ~${TOTAL_VRAM}MB available:"
echo ""

# Recalculate with full GPU available
CLEARED_FREE=$TOTAL_VRAM
for batch in 4 8 16 32; do
    for grad_acc in 1 2 4; do
        effective=$(($batch * $grad_acc))
        estimated_vram=$(echo "$BASE_VRAM + ($batch * $VRAM_PER_BATCH) + ($batch * $grad_acc * 50)" | bc)
        estimated_vram_gb=$(echo "scale=1; $estimated_vram/1024" | bc)
        steps=$(echo "scale=0; ($DATASET_SIZE / $effective) * 4" | bc)
        
        if [ $estimated_vram -le $(($CLEARED_FREE - 500)) ]; then
            printf "  batch=%2d, grad_acc=%d → effective=%3d, VRAM=%.1fGB, steps=%3d\n" \
                $batch $grad_acc $effective $estimated_vram_gb $steps
        fi
    done
done

echo ""
echo "=== Memory Usage Notes ==="
echo ""
echo "VRAM components:"
echo "  • Base model (LoRA): ~3-4GB"
echo "  • Batch processing: ~0.2GB per batch size unit"
echo "  • Gradients: ~0.05GB per batch × grad_acc"
echo "  • Optimizer states: ~0.1GB per batch"
echo "  • Overhead: ~0.5-1GB"
echo ""
echo "Note: Actual usage may vary based on:"
echo "  • Sequence length (cutoff_len)"
echo "  • Model architecture"
echo "  • Mixed precision (bf16/fp16)"
echo "  • ktransformers optimizations"
echo "  • Other GPU processes (clear them first!)"
echo ""
echo "Tip: Clear GPU memory before training:"
echo "  ./scripts/helpers/clear_gpu_memory.sh"
echo ""

echo "=========================================="
echo "Analysis Complete"
echo "=========================================="

