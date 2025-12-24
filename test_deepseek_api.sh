#!/bin/bash
# Test script for DeepSeek API server with KTransformers

echo "============================================================"
echo "Testing DeepSeek API Server with KTransformers"
echo "============================================================"

# Set up environment with proper library paths
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/opt/conda/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH

echo ""
echo "[1] Checking environment..."
docker exec llamafactory bash -c "export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/opt/conda/lib/python3.11/site-packages/torch/lib:\$LD_LIBRARY_PATH && python -c 'import torch; import ktransformers; print(\"   PyTorch:\", torch.__version__); print(\"   KTransformers:\", ktransformers.__version__); print(\"   CUDA available:\", torch.cuda.is_available())'"

echo ""
echo "[2] Starting API server..."
echo "   This will start the server in the background."
echo "   Use Ctrl+C to stop, or check logs with: docker logs llamafactory"

# Start API server in background
docker exec -d llamafactory bash -c "export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/opt/conda/lib/python3.11/site-packages/torch/lib:\$LD_LIBRARY_PATH && cd /app && llamafactory-cli api examples/inference/deepseek2_lite_serve_custom.yaml"

# Wait for server to start
echo "   Waiting for server to start..."
sleep 15

echo ""
echo "[3] Testing API endpoint..."
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello! Can you introduce yourself?"}],
    "temperature": 0.7,
    "max_tokens": 100
  }' 2>&1 | python3 -m json.tool || echo "   API server may still be starting up. Try again in a few seconds."

echo ""
echo "============================================================"
echo "To check server logs:"
echo "  docker logs llamafactory"
echo ""
echo "To stop the server:"
echo "  docker exec llamafactory pkill -f 'llamafactory-cli api'"
echo ""
echo "To test manually:"
echo "  curl -X POST http://localhost:8000/v1/chat/completions \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"model\": \"test\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
echo "============================================================"

