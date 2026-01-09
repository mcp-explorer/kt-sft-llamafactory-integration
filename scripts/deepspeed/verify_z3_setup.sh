#!/bin/bash
# Verification script for DeepSpeed ZeRO-3 CPU Offload setup
# Checks if everything is configured correctly

set +e  # Don't exit on errors, we want to check everything

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DeepSpeed ZeRO-3 Setup Verification${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

PASSED=0
FAILED=0
WARNINGS=0

check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $1"
        ((FAILED++))
        return 1
    fi
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

# Check conda environment
echo -e "${YELLOW}[1/8] Checking conda environment...${NC}"
if conda env list | grep -q "deepspeed-z3"; then
    check "Conda environment 'deepspeed-z3' exists"
    ENV_EXISTS=true
else
    check "Conda environment 'deepspeed-z3' exists"
    ENV_EXISTS=false
fi

# Check if environment can be activated
if [ "$ENV_EXISTS" = true ]; then
    echo -e "${YELLOW}[2/8] Checking environment activation...${NC}"
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    if conda activate deepspeed-z3 2>/dev/null; then
        check "Environment can be activated"
        conda deactivate 2>/dev/null || true
    else
        warn "Environment exists but activation test failed (may need manual activation)"
    fi
else
    warn "Skipping activation check (environment doesn't exist)"
fi

# Check scripts
echo -e "${YELLOW}[3/8] Checking setup scripts...${NC}"
[ -f "$SCRIPT_DIR/setup_deepspeed_z3_env.sh" ] && check "setup_deepspeed_z3_env.sh exists" || check "setup_deepspeed_z3_env.sh exists"
[ -f "$SCRIPT_DIR/activate_deepspeed_z3.sh" ] && check "activate_deepspeed_z3.sh exists" || check "activate_deepspeed_z3.sh exists"
[ -f "$SCRIPT_DIR/test_deepspeed_cpu_adam.sh" ] && check "test_deepspeed_cpu_adam.sh exists" || check "test_deepspeed_cpu_adam.sh exists"
[ -f "$SCRIPT_DIR/apply_z3_low_cpu_mem_patch.sh" ] && check "apply_z3_low_cpu_mem_patch.sh exists" || check "apply_z3_low_cpu_mem_patch.sh exists"
[ -f "$SCRIPT_DIR/revert_z3_patch.sh" ] && check "revert_z3_patch.sh exists" || check "revert_z3_patch.sh exists"

# Check documentation
echo -e "${YELLOW}[4/8] Checking documentation...${NC}"
[ -f "$PROJECT_ROOT/docs/FINAL_SUMMARY.md" ] && check "FINAL_SUMMARY.md exists" || check "FINAL_SUMMARY.md exists"
[ -f "$PROJECT_ROOT/docs/deepspeed_cpu_offload_issues.md" ] && check "deepspeed_cpu_offload_issues.md exists" || check "deepspeed_cpu_offload_issues.md exists"
[ -f "$PROJECT_ROOT/docs/deepspeed_z3_quick_reference.md" ] && check "deepspeed_z3_quick_reference.md exists" || check "deepspeed_z3_quick_reference.md exists"
[ -f "$PROJECT_ROOT/docs/README_DEEPSPEED.md" ] && check "README_DEEPSPEED.md exists" || check "README_DEEPSPEED.md exists"

# Check DeepSpeed config
echo -e "${YELLOW}[5/8] Checking DeepSpeed configuration...${NC}"
[ -f "$PROJECT_ROOT/LLaMA-Factory/examples/deepspeed/ds_z3_offload_config.json" ] && check "ds_z3_offload_config.json exists" || check "ds_z3_offload_config.json exists"
[ -f "$PROJECT_ROOT/LLaMA-Factory/examples/train_lora/deepseek2_lite_sft_hf_z3_test.yaml" ] && check "Test config exists" || check "Test config exists"

# Check patch status
echo -e "${YELLOW}[6/8] Checking patch status...${NC}"
PATCH_FILE="$PROJECT_ROOT/LLaMA-Factory/src/llamafactory/model/patcher.py"
BACKUP_FILE="${PATCH_FILE}.backup"

if [ -f "$PATCH_FILE" ]; then
    if grep -q "EXPERIMENTAL: Allow low_cpu_mem_usage with ZeRO-3" "$PATCH_FILE"; then
        warn "Patch is applied (experimental)"
    else
        check "Patch is not applied (original state)"
    fi
    
    if [ -f "$BACKUP_FILE" ]; then
        check "Backup file exists (can revert patch)"
    else
        warn "Backup file not found (cannot revert patch)"
    fi
else
    check "patcher.py exists"
fi

# Check GPU
echo -e "${YELLOW}[7/8] Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    check "GPU detected: $GPU_NAME ($GPU_MEM MB)"
    
    if [ "$GPU_MEM" -lt 20000 ]; then
        warn "GPU has less than 20GB memory (may have OOM issues)"
    fi
else
    warn "nvidia-smi not found (GPU check skipped)"
fi

# Check if CPU Adam compilation works (if environment exists)
if [ "$ENV_EXISTS" = true ]; then
    echo -e "${YELLOW}[8/8] Testing CPU Adam compilation (quick check)...${NC}"
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    if conda activate deepspeed-z3 2>/dev/null; then
        # Quick test - just check if module can be imported
        if python -c "from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam" 2>/dev/null; then
            check "CPU Adam module can be imported"
        else
            warn "CPU Adam import failed (may need to run full compilation test)"
        fi
        conda deactivate 2>/dev/null || true
    else
        warn "Could not activate environment for CPU Adam test"
    fi
else
    warn "Skipping CPU Adam test (environment doesn't exist)"
fi

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Verification Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ Setup verification complete!${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠ Some warnings (see above)${NC}"
    fi
    exit 0
else
    echo -e "${RED}❌ Some checks failed${NC}"
    exit 1
fi

