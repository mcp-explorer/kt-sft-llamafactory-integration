#!/bin/bash
# Test training with hard limit configuration

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
echo -e "${BLUE}║  Testing Hard Limit Configuration                            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Verify config values
echo -e "${YELLOW}Verifying configuration values...${NC}"
PREFETCH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['zero_optimization']['stage3_prefetch_bucket_size'])")
MAX_LIVE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['zero_optimization']['stage3_max_live_parameters'])")
MAX_REUSE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['zero_optimization']['stage3_max_reuse_distance'])")

echo "  Prefetch Bucket Size: $PREFETCH"
echo "  Max Live Parameters: $MAX_LIVE"
echo "  Max Reuse Distance: $MAX_REUSE"
echo ""

# Check if values match hard limits (using Python for float comparison)
python3 << EOF
import sys
prefetch = float("$PREFETCH")
max_live = float("$MAX_LIVE")
max_reuse = float("$MAX_REUSE")

expected_prefetch = 1200000000.0
expected_max_live = 8000000000.0
expected_max_reuse = 8000000000.0

tolerance = 0.01  # 1% tolerance for float comparison

if (abs(prefetch - expected_prefetch) / expected_prefetch < tolerance and
    abs(max_live - expected_max_live) / expected_max_live < tolerance and
    abs(max_reuse - expected_max_reuse) / expected_max_reuse < tolerance):
    print("✓ Configuration matches hard limits")
    sys.exit(0)
else:
    print("✗ Configuration does not match hard limits!")
    print(f"  Expected: Prefetch={expected_prefetch}, MaxLive={expected_max_live}, MaxReuse={expected_max_reuse}")
    print(f"  Got: Prefetch={prefetch}, MaxLive={max_live}, MaxReuse={max_reuse}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo -e "${YELLOW}Starting training test (will monitor for 5 minutes)...${NC}"
echo ""

# Clear GPU memory first
echo -e "${YELLOW}Clearing GPU memory...${NC}"
{ pkill -9 -f "llamafactory-cli train" 2>/dev/null; } || true
{ pkill -9 -f "deepspeed" 2>/dev/null; } || true
{ pkill -9 -f "torchrun" 2>/dev/null; } || true
sleep 3

INITIAL_MEM=0
if command -v nvidia-smi &> /dev/null; then
    INITIAL_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 2>/dev/null || echo "0")
fi
echo -e "${GREEN}Initial GPU memory: ${INITIAL_MEM} MB${NC}"
echo ""

# Start training in background
echo -e "${BLUE}Starting training...${NC}"
(
    cd "$PROJECT_ROOT/LLaMA-Factory"
    conda run -n deepspeed-z3 env FORCE_TORCHRUN=1 llamafactory-cli train "$TRAINING_CONFIG" > /tmp/hard_limit_test.log 2>&1
) &
TRAINING_PID=$!

# Monitor for 5 minutes
TIMEOUT=300
CHECK_INTERVAL=5
START_TIME=$(date +%s)
MAX_MEMORY=0
STABLE_CHECKS=0
LAST_MEMORY=0

echo -e "${YELLOW}Monitoring training (timeout: ${TIMEOUT}s)...${NC}"
echo ""

while [ $(($(date +%s) - START_TIME)) -lt $TIMEOUT ]; do
    # Check if process is still running
    if ! kill -0 $TRAINING_PID 2>/dev/null; then
        wait $TRAINING_PID
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo -e "${GREEN}✓ Training completed successfully${NC}"
            echo "  Max GPU memory: ${MAX_MEMORY} MB"
            exit 0
        else
            if grep -q "OutOfMemoryError\|CUDA out of memory" /tmp/hard_limit_test.log 2>/dev/null; then
                echo -e "${RED}✗ OOM Error detected!${NC}"
                echo ""
                echo "Last 20 lines of log:"
                tail -20 /tmp/hard_limit_test.log
                exit 1
            else
                echo -e "${RED}✗ Training failed (exit code: $EXIT_CODE)${NC}"
                echo ""
                echo "Last 20 lines of log:"
                tail -20 /tmp/hard_limit_test.log
                exit 1
            fi
        fi
    fi
    
    # Check GPU memory
    CURRENT_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    
    if [ -n "$CURRENT_MEM" ]; then
        if [ $CURRENT_MEM -gt $MAX_MEMORY ]; then
            MAX_MEMORY=$CURRENT_MEM
        fi
        
        # Check if memory is stable
        DIFF=$((CURRENT_MEM - LAST_MEMORY))
        if [ ${DIFF#-} -lt 200 ]; then
            STABLE_CHECKS=$((STABLE_CHECKS + 1))
        else
            STABLE_CHECKS=0
        fi
        
        ELAPSED=$(($(date +%s) - START_TIME))
        printf "\r  [%3ds] GPU Memory: %5d MB (Max: %5d MB) %s" \
            "$ELAPSED" "$CURRENT_MEM" "$MAX_MEMORY" \
            "$([ $STABLE_CHECKS -ge 3 ] && echo '[STABLE]' || echo '')"
        
        LAST_MEMORY=$CURRENT_MEM
        
        # If stable for 30 seconds, consider it successful
        if [ $STABLE_CHECKS -ge 6 ] && [ $ELAPSED -ge 60 ]; then
            echo ""
            echo ""
            echo -e "${GREEN}✓ GPU memory stabilized - Hard limit configuration working!${NC}"
            echo "  Max GPU memory: ${MAX_MEMORY} MB"
            echo "  Stable at: ${CURRENT_MEM} MB"
            echo ""
            echo -e "${YELLOW}Stopping training test (configuration verified)${NC}"
            kill $TRAINING_PID 2>/dev/null || true
            wait $TRAINING_PID 2>/dev/null || true
            exit 0
        fi
    fi
    
    sleep $CHECK_INTERVAL
done

# Timeout reached
echo ""
echo ""
if kill -0 $TRAINING_PID 2>/dev/null; then
    echo -e "${YELLOW}⚠ Timeout reached, but training is still running${NC}"
    echo "  Max GPU memory: ${MAX_MEMORY} MB"
    
    # Check for OOM in logs
    if grep -q "OutOfMemoryError\|CUDA out of memory" /tmp/hard_limit_test.log 2>/dev/null; then
        echo -e "${RED}✗ OOM detected in logs${NC}"
        kill $TRAINING_PID 2>/dev/null || true
        wait $TRAINING_PID 2>/dev/null || true
        exit 1
    else
        echo -e "${GREEN}✓ No OOM errors - configuration appears stable${NC}"
        kill $TRAINING_PID 2>/dev/null || true
        wait $TRAINING_PID 2>/dev/null || true
        exit 0
    fi
else
    echo -e "${RED}✗ Training process died before timeout${NC}"
    if grep -q "OutOfMemoryError\|CUDA out of memory" /tmp/hard_limit_test.log 2>/dev/null; then
        echo -e "${RED}✗ OOM Error detected!${NC}"
    fi
    echo ""
    echo "Last 20 lines of log:"
    tail -20 /tmp/hard_limit_test.log
    exit 1
fi

