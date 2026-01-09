#!/bin/bash
# Revert the experimental ZeRO-3 low_cpu_mem_usage patch

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

TARGET_FILE="$PROJECT_ROOT/LLaMA-Factory/src/llamafactory/model/patcher.py"
BACKUP_FILE="${TARGET_FILE}.backup"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Revert ZeRO-3 Patch${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if backup exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo -e "${RED}Error: Backup file not found: $BACKUP_FILE${NC}"
    echo -e "${YELLOW}Patch may not have been applied, or backup was removed.${NC}"
    exit 1
fi

# Check if file is patched
if ! grep -q "EXPERIMENTAL: Allow low_cpu_mem_usage with ZeRO-3" "$TARGET_FILE"; then
    echo -e "${YELLOW}File does not appear to be patched.${NC}"
    read -p "Restore from backup anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Restore from backup
echo -e "${YELLOW}Restoring from backup...${NC}"
cp "$BACKUP_FILE" "$TARGET_FILE"

# Verify restoration
if grep -q "deepspeed zero3 is not compatible with low_cpu_mem_usage" "$TARGET_FILE"; then
    echo -e "${GREEN}✓ Patch reverted successfully${NC}"
    echo ""
    echo -e "${BLUE}Original behavior restored:${NC}"
    echo -e "  low_cpu_mem_usage will be disabled when ZeRO-3 is enabled"
    echo ""
    echo -e "${YELLOW}Note: Backup file preserved at: $BACKUP_FILE${NC}"
else
    echo -e "${RED}✗ Restoration verification failed${NC}"
    exit 1
fi

