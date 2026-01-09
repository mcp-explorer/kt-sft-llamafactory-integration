#!/bin/bash
# Script to kill processes using GPU and memory to free resources for training

set -e

echo "=== Checking GPU and Memory Usage ==="
echo ""

# Check GPU usage
echo "GPU Status:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo ""

# Check GPU processes
echo "GPU Processes:"
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
if [ -z "$GPU_PROCS" ]; then
    echo "No GPU compute processes found"
else
    echo "$GPU_PROCS"
fi
echo ""

# Check memory usage
echo "System Memory:"
free -h | head -2
echo ""

# Kill gnome-remote-desktop-daemon processes (they use GPU memory)
echo "=== Killing gnome-remote-desktop-daemon processes ==="
GNOME_RD_PIDS=$(pgrep -f "gnome-remote-desktop-daemon" || true)
if [ -n "$GNOME_RD_PIDS" ]; then
    for pid in $GNOME_RD_PIDS; do
        echo "Killing gnome-remote-desktop-daemon (PID: $pid)"
        kill -9 $pid 2>/dev/null || echo "  Failed to kill PID $pid"
    done
    echo "Done"
else
    echo "No gnome-remote-desktop-daemon processes found"
fi
echo ""

# Check for Python training/inference processes
echo "=== Checking for Python training/inference processes ==="
PYTHON_PROCS=$(ps aux | grep -E "(python.*train|python.*infer|llamafactory|sft_ds2_chat_lite|infer_ds2_chat_lite)" | grep -v grep || true)
if [ -n "$PYTHON_PROCS" ]; then
    echo "Found Python processes:"
    echo "$PYTHON_PROCS"
    echo ""
    read -p "Kill these processes? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "$PYTHON_PROCS" | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
        echo "Killed Python training/inference processes"
    else
        echo "Skipped killing Python processes"
    fi
else
    echo "No Python training/inference processes found"
fi
echo ""

# Final status
echo "=== Final Status ==="
echo "GPU Status:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo ""
echo "GPU Processes:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || echo "No GPU processes"
echo ""
echo "System Memory:"
free -h | head -2
echo ""
echo "=== Cleanup Complete ==="

