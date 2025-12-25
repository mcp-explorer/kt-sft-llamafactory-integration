#!/bin/bash
# Debug script to identify the exact failing call site for params.nth assertion

set -e

echo "🔍 Setting up GDB debugging for KTransformers assertion failure..."
echo ""

# Create a simple test script that will trigger the assertion
cat > /tmp/test_kt_debug.py << 'PYTHON_EOF'
import sys
sys.path.insert(0, '/kt-sft')

print("Loading model and triggering inference...")
import subprocess
import os

# Run the command that triggers the assertion
cmd = [
    "llamafactory-cli", "chat",
    "--model_name_or_path", "/app/models/deepseek-ai/DeepSeek-V2-Lite-Chat",
    "--template", "chatml",
    "--max_new_tokens", "3",
    "--trust-remote-code",
    "--use_kt", "true",
    "--kt_optimize_rule", "/app/examples/kt_optimize_rules/DeepSeek-V2-Lite-Chat.yaml",
    "--cpu_infer", "32",
    "--chunk_size", "8192"
]

# Use printf to send input
input_str = "Hi\nexit\n"
proc = subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)
stdout, stderr = proc.communicate(input=input_str)
print("STDOUT:", stdout)
print("STDERR:", stderr)
PYTHON_EOF

echo "✅ Test script created"
echo ""
echo "Now run GDB with:"
echo "  docker exec -it llamafactory bash -c 'cd /tmp && gdb -ex run -ex bt -ex quit --args python3 test_kt_debug.py'"
echo ""
echo "Or for interactive debugging:"
echo "  docker exec -it llamafactory bash -c 'cd /tmp && gdb python3'"
echo "  Then: (gdb) run test_kt_debug.py"
echo "  When it crashes: (gdb) bt"
echo "  To see the exact line: (gdb) frame <N>"
echo "  To see variables: (gdb) print params->nth"

