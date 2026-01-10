#!/bin/bash
# Automated binary search for optimal DeepSpeed prefetch settings
# Monitors GPU VRAM and determines success/failure automatically

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/LLaMA-Factory/examples/deepspeed/ds_z3_hybrid_config.json"
BACKUP_FILE="${CONFIG_FILE}.backup"
TRAINING_CONFIG="$PROJECT_ROOT/LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_z3_bf16_regularized.yaml"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Automated Binary Search for Optimal Prefetch Settings      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if training config exists
if [ ! -f "$TRAINING_CONFIG" ]; then
    echo -e "${RED}Error: Training config not found: $TRAINING_CONFIG${NC}"
    exit 1
fi

# Backup current config
if [ ! -f "$BACKUP_FILE" ]; then
    echo -e "${YELLOW}Creating backup of current config...${NC}"
    cp "$CONFIG_FILE" "$BACKUP_FILE"
    echo -e "${GREEN}✓ Backup created${NC}"
    echo ""
fi

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

# Function to monitor GPU memory and detect OOM
monitor_training() {
    local timeout=180  # 3 minutes
    local check_interval=5  # Check every 5 seconds
    local stable_threshold=3  # Consider stable if within 100MB for 3 checks
    local checks=0
    local last_memory=0
    local stable_checks=0
    local max_memory=0
    
    echo -e "${BLUE}Monitoring GPU memory (timeout: ${timeout}s)...${NC}"
    
    local start_time=$(date +%s)
    local training_pid=""
    
    # Start training in background
    (
        cd "$PROJECT_ROOT/LLaMA-Factory"
        conda run -n deepspeed-z3 env FORCE_TORCHRUN=1 llamafactory-cli train "$TRAINING_CONFIG" > /tmp/training_output.log 2>&1
    ) &
    training_pid=$!
    
    # Monitor GPU memory
    while [ $(($(date +%s) - start_time)) -lt $timeout ]; do
        # Check if training process is still running
        if ! kill -0 $training_pid 2>/dev/null; then
            # Process ended - check exit code
            wait $training_pid
            exit_code=$?
            
            if [ $exit_code -eq 0 ]; then
                echo -e "${GREEN}✓ Training completed successfully${NC}"
                echo "  Max GPU memory used: ${max_memory} MB"
                return 0
            else
                # Check if it's OOM
                if grep -q "OutOfMemoryError\|CUDA out of memory" /tmp/training_output.log 2>/dev/null; then
                    echo -e "${RED}✗ OOM detected${NC}"
                    return 1
                else
                    echo -e "${RED}✗ Training failed (exit code: $exit_code)${NC}"
                    return 1
                fi
            fi
        fi
        
        # Get current GPU memory
        local current_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        
        if [ -n "$current_memory" ]; then
            if [ $current_memory -gt $max_memory ]; then
                max_memory=$current_memory
            fi
            
            # Check if memory is stable (within 100MB)
            local diff=$((current_memory - last_memory))
            if [ ${diff#-} -lt 100 ]; then
                stable_checks=$((stable_checks + 1))
            else
                stable_checks=0
            fi
            
            last_memory=$current_memory
            checks=$((checks + 1))
            
            # If memory is stable for threshold checks, consider it stable
            if [ $stable_checks -ge $stable_threshold ] && [ $checks -ge 6 ]; then
                echo -e "${GREEN}✓ GPU memory stabilized at ~${current_memory} MB${NC}"
                echo "  Max GPU memory used: ${max_memory} MB"
                # Kill training process
                kill $training_pid 2>/dev/null || true
                wait $training_pid 2>/dev/null || true
                return 0
            fi
        fi
        
        sleep $check_interval
    done
    
    # Timeout - check if still running
    if kill -0 $training_pid 2>/dev/null; then
        echo -e "${YELLOW}⚠ Timeout reached, but training still running${NC}"
        echo "  Max GPU memory used: ${max_memory} MB"
        # Kill training process
        kill $training_pid 2>/dev/null || true
        wait $training_pid 2>/dev/null || true
        
        # Check for OOM in logs
        if grep -q "OutOfMemoryError\|CUDA out of memory" /tmp/training_output.log 2>/dev/null; then
            echo -e "${RED}✗ OOM detected in logs${NC}"
            return 1
        else
            return 0  # Assume success if no OOM
        fi
    else
        # Process ended during timeout
        wait $training_pid
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            if grep -q "OutOfMemoryError\|CUDA out of memory" /tmp/training_output.log 2>/dev/null; then
                echo -e "${RED}✗ OOM detected${NC}"
                return 1
            fi
        fi
        return 1
    fi
}

# Test values to try
TEST_VALUES=(
    "1.2e9 8e9"
    "1.5e9 9e9"
    "1.8e9 9.5e9"
    "2e9 1e10"
)

OPTIMAL_PREFETCH=1e9
OPTIMAL_MAX_LIVE=7e9
OPTIMAL_MEMORY=0

echo -e "${BLUE}Starting binary search...${NC}"
echo ""

for test_val in "${TEST_VALUES[@]}"; do
    read -r test_prefetch test_max_live <<< "$test_val"
    
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Testing:${NC}"
    echo "  Prefetch Bucket Size: $test_prefetch"
    echo "  Max Live Parameters: $test_max_live"
    echo ""
    
    # Update config
    update_config "$test_prefetch" "$test_max_live"
    
    # Monitor training
    if monitor_training; then
        # Success - get max memory
        current_max=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
        if [ "$current_max" -gt "$OPTIMAL_MEMORY" ]; then
            OPTIMAL_PREFETCH=$test_prefetch
            OPTIMAL_MAX_LIVE=$test_max_live
            OPTIMAL_MEMORY=$current_max
        fi
        echo -e "${GREEN}✓ This value works! Trying next...${NC}"
        echo ""
        
        # Wait a bit before next test
        sleep 5
    else
        # OOM - stop here
        echo -e "${RED}✗ OOM with this value. Stopping search.${NC}"
        echo ""
        break
    fi
done

# Final update
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Optimal Settings Found                                      ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Optimal values:"
echo "  Prefetch Bucket Size: $OPTIMAL_PREFETCH"
echo "  Max Live Parameters: $OPTIMAL_MAX_LIVE"
echo "  Max GPU Memory Used: ${OPTIMAL_MEMORY} MB"
echo ""

update_config "$OPTIMAL_PREFETCH" "$OPTIMAL_MAX_LIVE"

echo -e "${GREEN}✓ Config updated with optimal values!${NC}"
echo ""
echo -e "${YELLOW}To restore original config:${NC}"
echo "  cp $BACKUP_FILE $CONFIG_FILE"




