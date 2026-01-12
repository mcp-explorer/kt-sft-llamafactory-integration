#!/bin/bash

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

clear_gpu_memory() {
    echo -e "${YELLOW}Clearing GPU memory...${NC}" >&2
    echo "About to pkill" >&2
    pkill -9 -f "llamafactory-cli train" 2>&1 | head -1 || echo "pkill 1 done" >&2
    echo "pkill 1 completed" >&2
    pkill -9 -f "deepspeed" 2>&1 | head -1 || echo "pkill 2 done" >&2
    echo "pkill 2 completed" >&2
    echo "About to sleep" >&2
    sleep 1
    echo "Sleep done" >&2
    current_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
    echo -e "${GREEN}GPU memory: ${current_mem} MB${NC}" >&2
}

echo "Test 1: Calling clear_gpu_memory"
clear_gpu_memory
echo "Test 1: Done"

echo "Test 2: In loop"
for i in 1 2; do
    echo "Loop iteration $i"
    clear_gpu_memory
    echo "After clear in loop $i"
done
echo "Test 2: Done"

