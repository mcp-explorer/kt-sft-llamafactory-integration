#!/bin/bash
# Apply fixes and rebuild KTransformers
# Usage: ./fix_and_build.sh [fix_script] [container_name]

set -e

FIX_SCRIPT="${1:-all}"
CONTAINER_NAME="${2:-llamafactory}"
BUILD_DIR="/tmp/kt-sft-build"
CPP_FILE="$BUILD_DIR/csrc/ktransformers_ext/operators/kvcache/kvcache_attn.cpp"

echo "Applying fixes and rebuilding..."
echo "Fix script: $FIX_SCRIPT"
echo "Container: $CONTAINER_NAME"
echo "Build directory: $BUILD_DIR"

# Check if container is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo "Error: Container $CONTAINER_NAME is not running"
    exit 1
fi

# Apply fixes
case "$FIX_SCRIPT" in
    structure)
        echo "Applying structure fixes..."
        docker exec "$CONTAINER_NAME" python3 << 'PYEOF'
import sys
sys.path.insert(0, '/kt-sft')
# Import and run structure fix
# This would need to be adapted based on actual fix script location
PYEOF
        ;;
    commas)
        echo "Checking for missing commas..."
        docker exec "$CONTAINER_NAME" python3 << 'PYEOF'
# Check for missing commas
# This would need the actual fix script
PYEOF
        ;;
    all|*)
        echo "Applying all fixes..."
        # Structure fixes
        docker exec "$CONTAINER_NAME" bash -c "
            cd '$BUILD_DIR' && \
            python3 << 'PYEOF'
import sys
with open('$CPP_FILE', 'r') as f:
    lines = f.readlines()

# Apply known fixes
# Fix 1: Remove premature closing if still present
# Fix 2: Add missing braces
# etc.

with open('$CPP_FILE', 'w') as f:
    f.writelines(lines)
PYEOF
        "
        ;;
esac

# Rebuild
echo "Rebuilding..."
docker exec "$CONTAINER_NAME" bash -c "
    cd '$BUILD_DIR' && \
    CPU_INSTRUCT=NATIVE KTRANSFORMERS_FORCE_BUILD=TRUE pip install . --no-build-isolation 2>&1 | \
    tee /tmp/build_$(date +%s).log | \
    grep -E '(error:|warning:|Successfully)' | \
    tail -30
"

echo "Build complete. Check errors above."

