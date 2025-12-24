#!/bin/bash
# Simple test script for LLaMA-Factory CLI

echo "============================================================"
echo "LLaMA-Factory CLI Simple Test"
echo "============================================================"

echo ""
echo "[1] Testing CLI version..."
docker exec llamafactory bash -c "cd /app && llamafactory-cli version"

echo ""
echo "[2] Testing environment info..."
docker exec llamafactory bash -c "cd /app && llamafactory-cli env 2>&1 | head -20"

echo ""
echo "[3] Testing imports..."
docker exec llamafactory bash -c "python -c 'import torch; import ktransformers; print(\"✅ PyTorch:\", torch.__version__); print(\"✅ KTransformers:\", ktransformers.__version__); print(\"✅ CUDA:\", torch.cuda.is_available())'"

echo ""
echo "[4] Testing config file..."
docker exec llamafactory bash -c "cd /app && test -f examples/inference/deepseek2_lite_serve_custom.yaml && echo '✅ Config file exists' || echo '❌ Config file missing'"

echo ""
echo "[5] Testing model directory..."
docker exec llamafactory bash -c "test -d /app/models/deepseek-ai/DeepSeek-V2-Lite-Chat && echo '✅ Model directory exists' || echo '❌ Model directory missing'"

echo ""
echo "[6] Testing chat command help..."
docker exec llamafactory bash -c "cd /app && llamafactory-cli chat --help 2>&1 | head -20"

echo ""
echo "[7] Testing API command help..."
docker exec llamafactory bash -c "cd /app && llamafactory-cli api --help 2>&1 | head -20"

echo ""
echo "============================================================"
echo "✅ All basic tests completed!"
echo ""
echo "To test chat (interactive):"
echo "  docker exec -it llamafactory bash -c \"cd /app && llamafactory-cli chat examples/inference/deepseek2_lite_serve_custom.yaml\""
echo ""
echo "To test API server:"
echo "  docker exec llamafactory bash -c \"cd /app && llamafactory-cli api examples/inference/deepseek2_lite_serve_custom.yaml\""
echo "============================================================"

