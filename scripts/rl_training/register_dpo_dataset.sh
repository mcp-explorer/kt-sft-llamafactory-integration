#!/bin/bash
# Helper script to register DPO dataset in dataset_info.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DPO_DATA_FILE="${1:-${PROJECT_ROOT}/LLaMA-Factory/data/identity_sean_dpo.json}"
DATASET_NAME="${2:-identity_sean_dpo}"
DATASET_INFO="${PROJECT_ROOT}/LLaMA-Factory/data/dataset_info.json"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "Register DPO Dataset"
echo "=========================================="
echo ""

# Check if data file exists
if [ ! -f "$DPO_DATA_FILE" ]; then
    echo -e "${RED}❌ Error: DPO data file not found: $DPO_DATA_FILE${NC}"
    echo ""
    echo "Usage: $0 [data_file] [dataset_name]"
    echo ""
    echo "Examples:"
    echo "  $0 LLaMA-Factory/data/identity_sean_dpo.json"
    echo "  $0 sft_data/outputs/identity_sean_dpo_20260115.json identity_sean_dpo"
    exit 1
fi

# Check if dataset_info.json exists
if [ ! -f "$DATASET_INFO" ]; then
    echo -e "${YELLOW}⚠ Warning: dataset_info.json not found, creating it...${NC}"
    mkdir -p "$(dirname "$DATASET_INFO")"
    echo "{}" > "$DATASET_INFO"
fi

# Ensure data file is in LLaMA-Factory/data/ directory
DATA_DIR="$(dirname "$DATASET_INFO")"
DATA_FILENAME="$(basename "$DPO_DATA_FILE")"
TARGET_FILE="${DATA_DIR}/${DATA_FILENAME}"

if [ "$DPO_DATA_FILE" != "$TARGET_FILE" ]; then
    echo "Copying data file to LLaMA-Factory/data/..."
    cp "$DPO_DATA_FILE" "$TARGET_FILE"
    echo -e "${GREEN}✓ Copied to: $TARGET_FILE${NC}"
    DPO_DATA_FILE="$TARGET_FILE"
fi

# Register dataset
echo "Registering dataset: $DATASET_NAME"
echo "  File: $DATA_FILENAME"
echo ""

python << PYEOF
import json
import sys

try:
    # Read existing dataset_info.json
    with open("$DATASET_INFO", 'r') as f:
        dataset_info = json.load(f)
    
    # Add or update dataset entry
    dataset_info["$DATASET_NAME"] = {
        "file_name": "$DATA_FILENAME",
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected"
        }
    }
    
    # Write back
    with open("$DATASET_INFO", 'w') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print("✅ Successfully registered dataset: $DATASET_NAME")
    print(f"   File: $DATA_FILENAME")
    print(f"   Format: sharegpt with ranking (DPO)")
    
except Exception as e:
    print(f"❌ Error: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Dataset registered successfully!${NC}"
    echo ""
    echo "You can now use this dataset in training configs:"
    echo "  dataset: $DATASET_NAME"
    echo ""
else
    echo -e "${RED}❌ Failed to register dataset${NC}"
    exit 1
fi
