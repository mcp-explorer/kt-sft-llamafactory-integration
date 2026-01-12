#!/bin/bash
# Fix missing new_allocator.h header for CPU Adam compilation
# This script requires sudo privileges

set -e

echo "=========================================="
echo "Fixing missing C++12 header for CPU Adam"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "This script requires sudo privileges."
    echo "Please run: sudo $0"
    exit 1
fi

if [ ! -f "/usr/include/c++/12.bak/bits/new_allocator.h" ]; then
    echo "❌ Error: Source file not found: /usr/include/c++/12.bak/bits/new_allocator.h"
    echo ""
    echo "Try installing C++12 development headers:"
    echo "  sudo apt-get install -y g++-12 libstdc++-12-dev"
    exit 1
fi

if [ ! -d "/usr/include/x86_64-linux-gnu/c++/12/bits" ]; then
    echo "❌ Error: Target directory not found: /usr/include/x86_64-linux-gnu/c++/12/bits"
    echo ""
    echo "Try installing C++12 development headers:"
    echo "  sudo apt-get install -y g++-12 libstdc++-12-dev"
    exit 1
fi

echo "Copying new_allocator.h..."
cp /usr/include/c++/12.bak/bits/new_allocator.h /usr/include/x86_64-linux-gnu/c++/12/bits/new_allocator.h

if [ -f "/usr/include/x86_64-linux-gnu/c++/12/bits/new_allocator.h" ]; then
    echo "✅ Successfully copied new_allocator.h"
    echo "✅ CPU Adam compilation should now work"
    echo ""
    echo "Next steps:"
    echo "  1. Run: ./scripts/deepspeed/fix_deepspeed_cpu_offload.sh (for other fixes)"
    echo "  2. Or see: docs/DEEPSPEED_CPU_OFFLOAD_FIXES.md"
else
    echo "❌ Failed to copy file"
    exit 1
fi

