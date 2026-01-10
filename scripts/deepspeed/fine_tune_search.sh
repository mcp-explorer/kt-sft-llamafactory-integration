#!/bin/bash
# Fine-tune binary search - test prefetch and max_live independently with smaller increments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/LLaMA-Factory/examples/deepspeed/ds_z3_hybrid_config.json"
TRAINING_CONFIG="$PROJECT_ROOT/LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_z3_bf16_regularized.yaml"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Fine-Tune Search - Testing Independent Parameters           ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Known working baseline
BASELINE_PREFETCH=1.2e9
BASELINE_MAX_LIVE=8e9

# Function to update config
update_config() {
    local prefetch=$1
    local max_live=$2
    
    python3 << EOF
import json

with open("$CONFIG_FILE", 'r') as f:
    config = json.load(f)

config["zero_optimization"]["stage3_prefetch_bucket_size"] = $prefetch
config["zero_optimization"]["stage3_max_live_parameters"] = $max_live
config["zero_optimization"]["stage3_max_reuse_distance"] = $max_live

with open("$CONFIG_FILE", 'w') as f:
    json.dump(config, f, indent=2)
EOF
}

# Function to monitor training
monitor_training() {
    local timeout=180
    local check_interval=5
    local stable_threshold=3
    local checks=0
    local last_memory=0
    local stable_checks=0
    local max_memory=0
    
    local start_time=$(date +%s)
    
    (
        cd "$PROJECT_ROOT/LLaMA-Factory"
        conda run -n deepspeed-z3 env FORCE_TORCHRUN=1 llamafactory-cli train "$TRAINING_CONFIG" > /tmp/training_output.log 2>&1
    ) &
    local training_pid=$!
    
    while [ $(($(date +%s) - start_time)) -lt $timeout ]; do
        if ! kill -0 $training_pid 2>/dev/null; then
            wait $training_pid
            local exit_code=$?
            
            if [ $exit_code -eq 0 ]; then
                echo -e "${GREEN}✓ Training completed${NC}"
                echo "  Max GPU memory: ${max_memory} MB"
                return 0
            else
                if grep -q "OutOfMemoryError\|CUDA out of memory" /tmp/training_output.log 2>/dev/null; then
                    echo -e "${RED}✗ OOM detected${NC}"
                    return 1
                else
                    echo -e "${RED}✗ Training failed (exit code: $exit_code)${NC}"
                    return 1
                fi
            fi
        fi
        
        local current_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        
        if [ -n "$current_memory" ]; then
            if [ $current_memory -gt $max_memory ]; then
                max_memory=$current_memory
            fi
            
            local diff=$((current_memory - last_memory))
            if [ ${diff#-} -lt 100 ]; then
                stable_checks=$((stable_checks + 1))
            else
                stable_checks=0
            fi
            
            last_memory=$current_memory
            checks=$((checks + 1))
            
            if [ $stable_checks -ge $stable_threshold ] && [ $checks -ge 6 ]; then
                echo -e "${GREEN}✓ GPU memory stabilized at ~${current_memory} MB${NC}"
                echo "  Max GPU memory: ${max_memory} MB"
                kill $training_pid 2>/dev/null || true
                wait $training_pid 2>/dev/null || true
                return 0
            fi
        fi
        
        sleep $check_interval
    done
    
    if kill -0 $training_pid 2>/dev/null; then
        echo -e "${YELLOW}⚠ Timeout, but training running${NC}"
        echo "  Max GPU memory: ${max_memory} MB"
        kill $training_pid 2>/dev/null || true
        wait $training_pid 2>/dev/null || true
        
        if grep -q "OutOfMemoryError\|CUDA out of memory" /tmp/training_output.log 2>/dev/null; then
            echo -e "${RED}✗ OOM in logs${NC}"
            return 1
        else
            return 0
        fi
    fi
    
    return 1
}

OPTIMAL_PREFETCH=$BASELINE_PREFETCH
OPTIMAL_MAX_LIVE=$BASELINE_MAX_LIVE
OPTIMAL_MEMORY=0

# Step 1: Test increasing prefetch while keeping max_live constant
echo -e "${BLUE}Step 1: Testing Prefetch (keeping MaxLive at ${BASELINE_MAX_LIVE})${NC}"
echo ""

PREFETCH_TESTS=(
    "1.25e9"
    "1.28e9"
)

for test_prefetch in "${PREFETCH_TESTS[@]}"; do
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Testing Prefetch: $test_prefetch (MaxLive: ${BASELINE_MAX_LIVE})${NC}"
    echo ""
    
    update_config "$test_prefetch" "$BASELINE_MAX_LIVE"
    
    if monitor_training; then
        current_max=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
        if [ "$current_max" -gt "$OPTIMAL_MEMORY" ]; then
            OPTIMAL_PREFETCH=$test_prefetch
            OPTIMAL_MEMORY=$current_max
        fi
        echo -e "${GREEN}✓ Works! Trying next...${NC}"
        echo ""
        sleep 5
    else
        echo -e "${RED}✗ OOM. Prefetch limit found.${NC}"
        echo ""
        break
    fi
done

# Step 2: Test increasing max_live with optimal prefetch
echo -e "${BLUE}Step 2: Testing MaxLive (using optimal Prefetch: ${OPTIMAL_PREFETCH})${NC}"
echo ""

MAX_LIVE_TESTS=(
    "8.1e9"
    "8.2e9"
    "8.25e9"
)

for test_max_live in "${MAX_LIVE_TESTS[@]}"; do
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Testing MaxLive: $test_max_live (Prefetch: ${OPTIMAL_PREFETCH})${NC}"
    echo ""
    
    update_config "$OPTIMAL_PREFETCH" "$test_max_live"
    
    if monitor_training; then
        current_max=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
        if [ "$current_max" -gt "$OPTIMAL_MEMORY" ]; then
            OPTIMAL_MAX_LIVE=$test_max_live
            OPTIMAL_MEMORY=$current_max
        fi
        echo -e "${GREEN}✓ Works! Trying next...${NC}"
        echo ""
        sleep 5
    else
        echo -e "${RED}✗ OOM. MaxLive limit found.${NC}"
        echo ""
        break
    fi
done

# Final update
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Final Optimal Settings                                      ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Optimal values:"
echo "  Prefetch Bucket Size: $OPTIMAL_PREFETCH"
echo "  Max Live Parameters: $OPTIMAL_MAX_LIVE"
echo "  Max GPU Memory: ${OPTIMAL_MEMORY} MB"
echo ""

update_config "$OPTIMAL_PREFETCH" "$OPTIMAL_MAX_LIVE"

echo -e "${GREEN}✓ Config updated with fine-tuned optimal values!${NC}"




