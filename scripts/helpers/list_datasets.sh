#!/bin/bash
# List available datasets and their file paths

set -e

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_INFO_FILE="$PROJECT_ROOT/LLaMA-Factory/data/dataset_info.json"
DATA_DIR="$PROJECT_ROOT/LLaMA-Factory/data"

echo "=========================================="
echo "Available Datasets"
echo "=========================================="
echo ""

if [ ! -f "$DATASET_INFO_FILE" ]; then
    echo "Error: dataset_info.json not found at $DATASET_INFO_FILE"
    exit 1
fi

# Use Python to parse and display datasets
python3 << PYEOF
import json
import os
from pathlib import Path

dataset_info_file = "$DATASET_INFO_FILE"
data_dir = "$DATA_DIR"

with open(dataset_info_file, 'r') as f:
    dataset_info = json.load(f)

print(f"Dataset Directory: {data_dir}")
print(f"Total datasets registered: {len(dataset_info)}")
print("")
print(f"{'Dataset Name':<30} {'File Name':<40} {'Status':<10}")
print("=" * 80)

for dataset_name, info in sorted(dataset_info.items()):
    file_name = info.get('file_name', 'N/A')
    file_path = os.path.join(data_dir, file_name)
    
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        if file_size < 1024:
            size_str = f"{file_size}B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f}KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f}MB"
        status = f"✓ {size_str}"
    else:
        status = "✗ Missing"
    
    print(f"{dataset_name:<30} {file_name:<40} {status:<10}")

print("")
print("Usage:")
print("  ./scripts/training/sft_ds2_chat_lite_hf.sh --dataset <dataset_name>")
print("")
print("Example:")
print("  ./scripts/training/sft_ds2_chat_lite_hf.sh --dataset identity_sean")
PYEOF

