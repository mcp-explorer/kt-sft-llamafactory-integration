#!/bin/bash
# Check status of RL training workflow components

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RL Training Workflow Status${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check 1: SFT Data
echo -e "${BLUE}[1] Checking SFT Data...${NC}"
SFT_DATA="${PROJECT_ROOT}/LLaMA-Factory/data/identity_sean_generated.json"
if [ -f "$SFT_DATA" ]; then
    COUNT=$(python -c "import json; print(len(json.load(open('$SFT_DATA'))))" 2>/dev/null || echo "0")
    echo -e "  ${GREEN}✓${NC} Found: $SFT_DATA ($COUNT samples)"
else
    echo -e "  ${RED}✗${NC} Not found: $SFT_DATA"
fi
echo ""

# Check 2: DPO Dataset
echo -e "${BLUE}[2] Checking DPO Dataset...${NC}"
DPO_DATA="${PROJECT_ROOT}/LLaMA-Factory/data/identity_sean_dpo.json"
if [ -f "$DPO_DATA" ]; then
    COUNT=$(python -c "import json; print(len(json.load(open('$DPO_DATA'))))" 2>/dev/null || echo "0")
    echo -e "  ${GREEN}✓${NC} Found: $DPO_DATA ($COUNT pairs)"
    
    # Check registration
    if grep -q "identity_sean_dpo" "${PROJECT_ROOT}/LLaMA-Factory/data/dataset_info.json" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Registered in dataset_info.json"
    else
        echo -e "  ${YELLOW}⚠${NC} Not registered in dataset_info.json"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} Not found: $DPO_DATA"
    echo -e "     Run: ./scripts/sft_data/generate_identity_dpo.sh"
fi
echo ""

# Check 3: SFT Adapter
echo -e "${BLUE}[3] Checking SFT Adapter...${NC}"
SFT_ADAPTER="${PROJECT_ROOT}/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_regularized"
if [ -d "$SFT_ADAPTER" ]; then
    if [ -f "$SFT_ADAPTER/adapter_model.safetensors" ]; then
        SIZE=$(du -h "$SFT_ADAPTER/adapter_model.safetensors" | cut -f1)
        echo -e "  ${GREEN}✓${NC} Found: $SFT_ADAPTER ($SIZE)"
    else
        echo -e "  ${YELLOW}⚠${NC} Directory exists but adapter file missing"
    fi
else
    echo -e "  ${RED}✗${NC} Not found: $SFT_ADAPTER"
fi
echo ""

# Check 4: DPO Adapter
echo -e "${BLUE}[4] Checking DPO Adapter...${NC}"
DPO_ADAPTER="${PROJECT_ROOT}/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_dpo"
if [ -d "$DPO_ADAPTER" ]; then
    if [ -f "$DPO_ADAPTER/adapter_model.safetensors" ]; then
        SIZE=$(du -h "$DPO_ADAPTER/adapter_model.safetensors" | cut -f1)
        echo -e "  ${GREEN}✓${NC} Found: $DPO_ADAPTER ($SIZE)"
        
        # Check for training artifacts
        if [ -f "$DPO_ADAPTER/training_loss.png" ]; then
            echo -e "  ${GREEN}✓${NC} Training artifacts found"
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} Directory exists but adapter file missing (training may be in progress)"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} Not found: $DPO_ADAPTER"
    echo -e "     Run: ./scripts/training/dpo_ds2_chat_lite_hf.sh"
fi
echo ""

# Check 5: Evaluation Results
echo -e "${BLUE}[5] Checking Evaluation Results...${NC}"
EVAL_RESULTS="${PROJECT_ROOT}/evaluation_results.json"
EVAL_RESULTS_DPO="${PROJECT_ROOT}/evaluation_results_dpo.json"
if [ -f "$EVAL_RESULTS_DPO" ]; then
    SIZE=$(du -h "$EVAL_RESULTS_DPO" | cut -f1)
    echo -e "  ${GREEN}✓${NC} Found: $EVAL_RESULTS_DPO ($SIZE)"
elif [ -f "$EVAL_RESULTS" ]; then
    SIZE=$(du -h "$EVAL_RESULTS" | cut -f1)
    echo -e "  ${GREEN}✓${NC} Found: $EVAL_RESULTS ($SIZE)"
else
    echo -e "  ${YELLOW}⚠${NC} Not found: evaluation_results*.json"
    echo -e "     Run: ./scripts/evaluation/run_evaluation.sh"
fi
echo ""

# Check 6: Configuration Files
echo -e "${BLUE}[6] Checking Configuration Files...${NC}"
DPO_CONFIG="${PROJECT_ROOT}/LLaMA-Factory/examples/train_lora/deepseek2_lite_dpo_hf_z3.yaml"
if [ -f "$DPO_CONFIG" ]; then
    echo -e "  ${GREEN}✓${NC} DPO config: $DPO_CONFIG"
    
    # Check dataset setting
    if grep -q "dataset: identity_sean_dpo" "$DPO_CONFIG"; then
        echo -e "  ${GREEN}✓${NC} Dataset configured correctly"
    else
        echo -e "  ${YELLOW}⚠${NC} Dataset may not be configured correctly"
    fi
else
    echo -e "  ${RED}✗${NC} Not found: $DPO_CONFIG"
fi
echo ""

# Check 7: Environment
echo -e "${BLUE}[7] Checking Environment...${NC}"
if command -v conda &> /dev/null; then
    if conda env list | grep -q "^deepspeed-z3 "; then
        echo -e "  ${GREEN}✓${NC} deepspeed-z3 environment exists"
    else
        echo -e "  ${YELLOW}⚠${NC} deepspeed-z3 environment not found"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} conda not found in PATH"
fi

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
        echo -e "  ${GREEN}✓${NC} GPU available ($GPU_COUNT GPU(s))"
    else
        echo -e "  ${YELLOW}⚠${NC} No GPU detected"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} nvidia-smi not found (GPU check skipped)"
fi
echo ""

# Check 8: Scripts
echo -e "${BLUE}[8] Checking Scripts...${NC}"
SCRIPTS=(
    "scripts/sft_data/generate_identity_dpo.sh"
    "scripts/training/dpo_ds2_chat_lite_hf.sh"
    "scripts/evaluation/run_evaluation.sh"
    "scripts/rl_training/run_complete_rl_workflow.sh"
    "scripts/rl_training/register_dpo_dataset.sh"
)

for script in "${SCRIPTS[@]}"; do
    script_path="${PROJECT_ROOT}/${script}"
    if [ -f "$script_path" ] && [ -x "$script_path" ]; then
        echo -e "  ${GREEN}✓${NC} $script (executable)"
    elif [ -f "$script_path" ]; then
        echo -e "  ${YELLOW}⚠${NC} $script (not executable)"
    else
        echo -e "  ${RED}✗${NC} $script (missing)"
    fi
done
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Determine workflow status
STATUS="READY"
if [ ! -f "$DPO_DATA" ]; then
    STATUS="DATA_GENERATION_NEEDED"
elif [ ! -d "$DPO_ADAPTER" ] || [ ! -f "$DPO_ADAPTER/adapter_model.safetensors" ]; then
    STATUS="TRAINING_NEEDED"
elif [ ! -f "$EVAL_RESULTS" ] && [ ! -f "$EVAL_RESULTS_DPO" ]; then
    STATUS="EVALUATION_NEEDED"
fi

case "$STATUS" in
    "READY")
        echo -e "${GREEN}✓ Workflow Status: READY${NC}"
        echo "   All components are in place. You can run evaluation or start training."
        ;;
    "DATA_GENERATION_NEEDED")
        echo -e "${YELLOW}⚠ Workflow Status: DATA_GENERATION_NEEDED${NC}"
        echo "   Next step: Generate DPO dataset"
        echo "   Run: ./scripts/sft_data/generate_identity_dpo.sh --num_records 100"
        ;;
    "TRAINING_NEEDED")
        echo -e "${YELLOW}⚠ Workflow Status: TRAINING_NEEDED${NC}"
        echo "   Next step: Run DPO training"
        echo "   Run: ./scripts/training/dpo_ds2_chat_lite_hf.sh --yes"
        ;;
    "EVALUATION_NEEDED")
        echo -e "${YELLOW}⚠ Workflow Status: EVALUATION_NEEDED${NC}"
        echo "   Next step: Run evaluation"
        echo "   Run: ./scripts/evaluation/run_evaluation.sh"
        ;;
esac

echo ""
echo "For detailed checklist, see: docs/RL_TRAINING_CHECKLIST.md"
echo ""
