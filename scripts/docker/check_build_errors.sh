#!/bin/bash
# Check build errors from last build
# Usage: ./check_build_errors.sh [container_name] [log_file]

set -e

CONTAINER_NAME="${1:-llamafactory}"
LOG_FILE="${2:-/tmp/build.log}"

echo "Checking build errors in container: $CONTAINER_NAME"
echo "Log file: $LOG_FILE"

# Get errors
echo ""
echo "=== Compilation Errors ==="
docker exec "$CONTAINER_NAME" grep -E 'error:' "$LOG_FILE" 2>/dev/null | head -30 || echo "No errors found or log file doesn't exist"

echo ""
echo "=== Error Summary ==="
docker exec "$CONTAINER_NAME" grep -E 'error:' "$LOG_FILE" 2>/dev/null | \
    sed 's/.*error: //' | \
    sort | uniq -c | sort -rn | head -10 || echo "No errors to summarize"

echo ""
echo "=== Warnings ==="
docker exec "$CONTAINER_NAME" grep -E 'warning:' "$LOG_FILE" 2>/dev/null | head -10 || echo "No warnings found"

echo ""
echo "=== Build Status ==="
if docker exec "$CONTAINER_NAME" grep -q "Successfully installed" "$LOG_FILE" 2>/dev/null; then
    echo "✓ Build succeeded!"
else
    echo "✗ Build failed"
fi

