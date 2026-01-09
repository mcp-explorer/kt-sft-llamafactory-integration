#!/bin/bash
# Apply patch to allow low_cpu_mem_usage with ZeRO-3
# This is an experimental workaround for OOM during model initialization

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

PATCH_FILE="$SCRIPT_DIR/patch_llamafactory_z3_low_cpu_mem.patch"
TARGET_FILE="$PROJECT_ROOT/LLaMA-Factory/src/llamafactory/model/patcher.py"
BACKUP_FILE="${TARGET_FILE}.backup"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ZeRO-3 low_cpu_mem_usage Patch${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if patch file exists
if [ ! -f "$PATCH_FILE" ]; then
    echo -e "${RED}Error: Patch file not found: $PATCH_FILE${NC}"
    exit 1
fi

# Check if target file exists
if [ ! -f "$TARGET_FILE" ]; then
    echo -e "${RED}Error: Target file not found: $TARGET_FILE${NC}"
    exit 1
fi

# Check if already patched
if grep -q "EXPERIMENTAL: Allow low_cpu_mem_usage with ZeRO-3" "$TARGET_FILE"; then
    echo -e "${YELLOW}⚠ File appears to already be patched${NC}"
    read -p "Re-apply patch anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Create backup
echo -e "${YELLOW}[1/3] Creating backup...${NC}"
cp "$TARGET_FILE" "$BACKUP_FILE"
echo -e "${GREEN}✓ Backup created: $BACKUP_FILE${NC}"
echo ""

# Apply patch manually (since we can't use git patch)
echo -e "${YELLOW}[2/3] Applying patch...${NC}"

# Read the file and make the replacement
python3 << PYEOF
import sys

target_file = "$TARGET_FILE"
backup_file = "$BACKUP_FILE"

# Read the file
with open(target_file, 'r') as f:
    content = f.read()

# Check if already patched
if "EXPERIMENTAL: Allow low_cpu_mem_usage with ZeRO-3" in content:
    print("⚠ File already patched")
    sys.exit(1)

# Find and replace the line
old_line = '    # deepspeed zero3 is not compatible with low_cpu_mem_usage\n    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage and (not is_deepspeed_zero3_enabled())'
new_line = '''    # EXPERIMENTAL: Allow low_cpu_mem_usage with ZeRO-3 for CPU offload during model loading
    # This may allow model to load directly to CPU before DeepSpeed takes over
    # WARNING: This may break DeepSpeed functionality - use at your own risk
    # Original: init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage and (not is_deepspeed_zero3_enabled())
    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage  # Allow with ZeRO-3 (experimental)'''

if old_line in content:
    content = content.replace(old_line, new_line)
    with open(target_file, 'w') as f:
        f.write(content)
    print("✅ Patch applied successfully")
    sys.exit(0)
else:
    print("❌ Could not find target line to patch")
    print("File may have been modified. Check manually.")
    sys.exit(1)
PYEOF

PATCH_RESULT=$?

if [ $PATCH_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Patch applied${NC}"
    echo ""
    echo -e "${YELLOW}[3/3] Verifying patch...${NC}"
    if grep -q "EXPERIMENTAL: Allow low_cpu_mem_usage with ZeRO-3" "$TARGET_FILE"; then
        echo -e "${GREEN}✓ Patch verified${NC}"
        echo ""
        echo -e "${BLUE}========================================${NC}"
        echo -e "${GREEN}Patch applied successfully!${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo ""
        echo -e "${YELLOW}⚠ WARNING: This is an experimental modification${NC}"
        echo -e "${YELLOW}It may break DeepSpeed functionality.${NC}"
        echo ""
        echo -e "${BLUE}To revert:${NC}"
        echo -e "  cp $BACKUP_FILE $TARGET_FILE"
        echo ""
        echo -e "${BLUE}To test:${NC}"
        echo -e "  source scripts/deepspeed/activate_deepspeed_z3.sh"
        echo -e "  bash scripts/deepspeed/train_z3_test.sh"
    else
        echo -e "${RED}✗ Patch verification failed${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Patch failed${NC}"
    echo -e "${YELLOW}Restoring backup...${NC}"
    cp "$BACKUP_FILE" "$TARGET_FILE"
    exit 1
fi

