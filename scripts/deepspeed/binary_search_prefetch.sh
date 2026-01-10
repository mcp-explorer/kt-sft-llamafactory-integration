#!/bin/bash
# Binary search script to find optimal DeepSpeed prefetch settings
# This helps find the maximum values that use more VRAM without OOM

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/LLaMA-Factory/examples/deepspeed/ds_z3_hybrid_config.json"
BACKUP_FILE="${CONFIG_FILE}.backup"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Binary Search for Optimal DeepSpeed Prefetch Settings      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Backup current config
if [ ! -f "$BACKUP_FILE" ]; then
    echo -e "${YELLOW}Creating backup of current config...${NC}"
    cp "$CONFIG_FILE" "$BACKUP_FILE"
    echo -e "${GREEN}✓ Backup created: $BACKUP_FILE${NC}"
    echo ""
fi

# Current values (conservative)
CURRENT_PREFETCH=1e9
CURRENT_MAX_LIVE=7e9

# Binary search ranges
PREFETCH_LOW=5e8
PREFETCH_HIGH=2e9
MAX_LIVE_LOW=5e9
MAX_LIVE_HIGH=1e10

echo -e "${BLUE}Current Settings:${NC}"
echo "  Prefetch Bucket Size: $CURRENT_PREFETCH"
echo "  Max Live Parameters: $CURRENT_MAX_LIVE"
echo ""
echo -e "${YELLOW}Binary Search Ranges:${NC}"
echo "  Prefetch: $PREFETCH_LOW to $PREFETCH_HIGH"
echo "  Max Live: $MAX_LIVE_LOW to $MAX_LIVE_HIGH"
echo ""

# Function to update config
update_config() {
    local prefetch=$1
    local max_live=$2
    
    python3 << EOF
import json
import sys

with open("$CONFIG_FILE", 'r') as f:
    config = json.load(f)

config["zero_optimization"]["stage3_prefetch_bucket_size"] = $prefetch
config["zero_optimization"]["stage3_max_live_parameters"] = $max_live
config["zero_optimization"]["stage3_max_reuse_distance"] = $max_live

with open("$CONFIG_FILE", 'w') as f:
    json.dump(config, f, indent=2)

print("Updated config:")
print(f"  Prefetch: {int($prefetch):,}")
print(f"  Max Live: {int($max_live):,}")
EOF
}

# Function to test config
test_config() {
    local prefetch=$1
    local max_live=$2
    
    echo -e "${BLUE}Testing:${NC}"
    echo "  Prefetch: $prefetch"
    echo "  Max Live: $max_live"
    echo ""
    
    update_config "$prefetch" "$max_live"
    
    echo -e "${YELLOW}Run training for 10-20 steps, then answer:${NC}"
    echo -e "  ${GREEN}1${NC} = No OOM (training works)"
    echo -e "  ${RED}2${NC} = OOM (out of memory error)"
    echo -e "  ${YELLOW}3${NC} = Skip this test"
    echo ""
    read -p "Result (1/2/3): " result
    
    case $result in
        1) return 0 ;;  # Success
        2) return 1 ;;  # OOM
        3) return 2 ;;  # Skip
        *) return 2 ;;
    esac
}

# Binary search for prefetch
echo -e "${BLUE}Step 1: Finding optimal Prefetch Bucket Size${NC}"
echo ""

PREFETCH_LOW_VAL=$(echo "$PREFETCH_LOW" | sed 's/e/*10^/')
PREFETCH_HIGH_VAL=$(echo "$PREFETCH_HIGH" | sed 's/e/*10^/')

# Simple incremental search (easier than true binary search)
TEST_VALUES=(
    "1.2e9 8e9"
    "1.5e9 9e9"
    "1.8e9 9.5e9"
    "2e9 1e10"
)

OPTIMAL_PREFETCH=$CURRENT_PREFETCH
OPTIMAL_MAX_LIVE=$CURRENT_MAX_LIVE

for test_val in "${TEST_VALUES[@]}"; do
    read -r test_prefetch test_max_live <<< "$test_val"
    
    if test_config "$test_prefetch" "$test_max_live"; then
        # Success - this value works
        OPTIMAL_PREFETCH=$test_prefetch
        OPTIMAL_MAX_LIVE=$test_max_live
        echo -e "${GREEN}✓ This value works! Trying higher...${NC}"
        echo ""
    else
        # OOM - stop here
        echo -e "${RED}✗ OOM with this value. Stopping.${NC}"
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
echo ""

update_config "$OPTIMAL_PREFETCH" "$OPTIMAL_MAX_LIVE"

echo -e "${GREEN}✓ Config updated with optimal values!${NC}"
echo ""
echo -e "${YELLOW}To restore original config:${NC}"
echo "  cp $BACKUP_FILE $CONFIG_FILE"




