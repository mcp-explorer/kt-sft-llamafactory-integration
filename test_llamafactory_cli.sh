#!/bin/bash
# Test script for LLaMA-Factory CLI with KTransformers

echo "============================================================"
echo "Testing LLaMA-Factory CLI with KTransformers"
echo "============================================================"

echo ""
echo "[1] Checking PyTorch..."
docker exec llamafactory bash -c "python -c 'import torch; print(\"   PyTorch:\", torch.__version__); print(\"   CUDA:\", torch.version.cuda); print(\"   CUDA available:\", torch.cuda.is_available())'"

echo ""
echo "[2] Checking KTransformers..."
docker exec llamafactory bash -c "python -c 'import ktransformers; print(\"   KTransformers:\", ktransformers.__version__)'"

echo ""
echo "[3] Checking LLaMA-Factory CLI..."
docker exec llamafactory bash -c "cd /app && python -m llamafactory.cli --help 2>&1 | head -20"

echo ""
echo "[4] Available example configs..."
docker exec llamafactory bash -c "ls -1 /app/examples/inference/*.yaml 2>/dev/null | head -10"

echo ""
echo "============================================================"
echo "To test chat:"
echo "  docker exec -it llamafactory bash -c \"cd /app && llamafactory-cli chat examples/inference/deepseek2_lite_inference.yaml\""
echo ""
echo "To test API:"
echo "  docker exec llamafactory bash -c \"cd /app && API_PORT=8000 llamafactory-cli api examples/inference/deepseek2_lite_inference.yaml\""
echo "============================================================"


