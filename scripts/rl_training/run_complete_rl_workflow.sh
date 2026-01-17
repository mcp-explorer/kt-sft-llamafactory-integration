#!/bin/bash
# Complete RL Training Workflow Orchestrator
# Runs DPO data generation, training, and evaluation in sequence

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Complete RL Training Workflow${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Configuration
IDENTITY="sean"
NUM_RECORDS=100
REJECTION_METHOD="template"  # template, llm, or mixed
SKIP_DATA_GEN=false
SKIP_TRAINING=false
SKIP_EVALUATION=false
RUN_QUICK_TEST=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --identity)
            IDENTITY="$2"
            shift 2
            ;;
        --num_records)
            NUM_RECORDS="$2"
            shift 2
            ;;
        --rejection_method)
            REJECTION_METHOD="$2"
            shift 2
            ;;
        --skip-data-gen)
            SKIP_DATA_GEN=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        --quick-test)
            RUN_QUICK_TEST=true
            NUM_RECORDS=20
            shift
            ;;
        --help|-h)
            cat << EOF
Usage: $0 [OPTIONS]

Complete RL training workflow: Generate DPO data → Train → Evaluate

Options:
  --identity NAME          Identity name (default: sean)
  --num_records N          Number of DPO pairs to generate (default: 100)
  --rejection_method METHOD Method: template, llm, or mixed (default: template)
  --skip-data-gen          Skip data generation step
  --skip-training          Skip training step
  --skip-evaluation        Skip evaluation step
  --quick-test             Quick test run (20 records, 1 epoch)
  --help                   Show this help

Examples:
  # Full workflow with defaults
  $0

  # Quick test run
  $0 --quick-test

  # Use LLM-based rejections
  $0 --rejection_method llm

  # Skip data generation (use existing dataset)
  $0 --skip-data-gen
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Paths
SFT_DATA="${PROJECT_ROOT}/LLaMA-Factory/data/identity_sean_generated.json"
DPO_DATA="${PROJECT_ROOT}/LLaMA-Factory/data/identity_sean_dpo.json"
DPO_OUTPUT_DIR="${PROJECT_ROOT}/LLaMA-Factory/saves/Kllama_deepseekV2Lite_hf_z3_dpo"
EVAL_OUTPUT="${PROJECT_ROOT}/evaluation_results_dpo.json"

# Step 1: Generate DPO Dataset
if [ "$SKIP_DATA_GEN" = false ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Step 1: Generate DPO Dataset${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    if [ ! -f "$SFT_DATA" ]; then
        echo -e "${RED}❌ Error: SFT data not found at: $SFT_DATA${NC}"
        echo "   Please ensure SFT training data exists or provide a different path."
        exit 1
    fi
    
    echo "Generating DPO dataset..."
    echo "  Input: $SFT_DATA"
    echo "  Records: $NUM_RECORDS"
    echo "  Rejection method: $REJECTION_METHOD"
    echo ""
    
    "${PROJECT_ROOT}/scripts/sft_data/generate_identity_dpo.sh" \
        --input_sft "$SFT_DATA" \
        --num_records "$NUM_RECORDS" \
        --identity "$IDENTITY" \
        --rejection_method "$REJECTION_METHOD"
    
    # Find the generated file
    DPO_GENERATED=$(ls -t "${PROJECT_ROOT}/sft_data/outputs/identity_${IDENTITY}_dpo_"*.json 2>/dev/null | head -1)
    
    if [ -z "$DPO_GENERATED" ]; then
        echo -e "${RED}❌ Error: DPO data generation failed${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}✓ DPO data generated: $DPO_GENERATED${NC}"
    
    # Copy to LLaMA-Factory data directory
    echo "Copying to LLaMA-Factory/data/..."
    cp "$DPO_GENERATED" "$DPO_DATA"
    echo -e "${GREEN}✓ Copied to: $DPO_DATA${NC}"
    
    # Register dataset
    echo "Registering dataset..."
    cd "${PROJECT_ROOT}/LLaMA-Factory/data"
    python << PYEOF
import json

with open('dataset_info.json', 'r') as f:
    dataset_info = json.load(f)

dataset_info["identity_sean_dpo"] = {
    "file_name": "identity_sean_dpo.json",
    "formatting": "sharegpt",
    "ranking": True,
    "columns": {
        "messages": "conversations",
        "chosen": "chosen",
        "rejected": "rejected"
    }
}

with open('dataset_info.json', 'w') as f:
    json.dump(dataset_info, f, indent=2, ensure_ascii=False)

print("✅ Registered identity_sean_dpo dataset")
PYEOF
    
    echo -e "${GREEN}✓ Dataset registered${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠ Skipping data generation (using existing dataset)${NC}"
    echo ""
fi

# Step 2: Run DPO Training
if [ "$SKIP_TRAINING" = false ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Step 2: Run DPO Training${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    if [ "$RUN_QUICK_TEST" = true ]; then
        echo -e "${YELLOW}⚠ Quick test mode: Updating config for test run...${NC}"
        # Update config for quick test
        cd "${PROJECT_ROOT}/LLaMA-Factory"
        python << PYEOF
import yaml

config_file = "examples/train_lora/deepseek2_lite_dpo_hf_z3.yaml"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Save original values
original_max_samples = config.get('max_samples', 100000)
original_epochs = config.get('num_train_epochs', 10.0)

# Update for quick test
config['max_samples'] = 20
config['num_train_epochs'] = 1.0

with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"✅ Updated config: max_samples=20, epochs=1.0")
print(f"   (Original: max_samples={original_max_samples}, epochs={original_epochs})")
PYEOF
        echo ""
    fi
    
    echo "Starting DPO training..."
    echo "  Output directory: $DPO_OUTPUT_DIR"
    echo ""
    
    "${PROJECT_ROOT}/scripts/training/dpo_ds2_chat_lite_hf.sh" --yes
    
    if [ ! -d "$DPO_OUTPUT_DIR" ]; then
        echo -e "${RED}❌ Error: DPO training may have failed (output directory not found)${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}✓ DPO training completed${NC}"
    echo "  Adapter saved to: $DPO_OUTPUT_DIR"
    echo ""
    
    if [ "$RUN_QUICK_TEST" = true ]; then
        echo -e "${YELLOW}⚠ Restoring original config...${NC}"
        cd "${PROJECT_ROOT}/LLaMA-Factory"
        python << PYEOF
import yaml

config_file = "examples/train_lora/deepseek2_lite_dpo_hf_z3.yaml"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

config['max_samples'] = 100000
config['num_train_epochs'] = 10.0

with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("✅ Restored original config")
PYEOF
        echo ""
    fi
else
    echo -e "${YELLOW}⚠ Skipping training${NC}"
    echo ""
fi

# Step 3: Run Evaluation
if [ "$SKIP_EVALUATION" = false ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Step 3: Run Evaluation${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    if [ ! -d "$DPO_OUTPUT_DIR" ]; then
        echo -e "${RED}❌ Error: DPO adapter not found at: $DPO_OUTPUT_DIR${NC}"
        echo "   Please run training first or use --skip-training if adapter already exists."
        exit 1
    fi
    
    echo "Running multi-model evaluation..."
    echo "  Comparing: Raw vs SFT vs DPO"
    echo "  Output: $EVAL_OUTPUT"
    echo ""
    
    "${PROJECT_ROOT}/scripts/evaluation/run_evaluation.sh"
    
    if [ -f "$EVAL_OUTPUT" ]; then
        echo ""
        echo -e "${GREEN}✓ Evaluation completed${NC}"
        echo "  Results saved to: $EVAL_OUTPUT"
    else
        echo -e "${YELLOW}⚠ Evaluation completed but results file not found at expected location${NC}"
    fi
    echo ""
else
    echo -e "${YELLOW}⚠ Skipping evaluation${NC}"
    echo ""
fi

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Workflow Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Summary:"
if [ "$SKIP_DATA_GEN" = false ]; then
    echo -e "  ${GREEN}✓${NC} DPO dataset generated: $DPO_DATA"
fi
if [ "$SKIP_TRAINING" = false ]; then
    echo -e "  ${GREEN}✓${NC} DPO adapter trained: $DPO_OUTPUT_DIR"
fi
if [ "$SKIP_EVALUATION" = false ]; then
    echo -e "  ${GREEN}✓${NC} Evaluation completed: $EVAL_OUTPUT"
fi
echo ""
echo "Next steps:"
echo "  1. Review evaluation results: cat $EVAL_OUTPUT | python -m json.tool"
echo "  2. Compare with SFT results in docs/EVALUATION_RESULTS_SUMMARY.md"
echo "  3. Document findings in docs/RL_TRAINING_RESULTS.md"
echo ""
