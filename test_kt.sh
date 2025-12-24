#!/bin/bash
# Quick KTransformers test script

echo "============================================================"
echo "KTransformers Quick Test"
echo "============================================================"

docker exec llamafactory python -c "
import torch
import ktransformers
import KTransformersOps

print('✅ PyTorch:', torch.__version__)
print('✅ KTransformers:', ktransformers.__version__)
print('✅ CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
    print('✅ CUDA version:', torch.version.cuda)
    
    # Test CUDA operations
    x = torch.randn(10, 10).cuda()
    y = torch.randn(10, 10).cuda()
    z = torch.matmul(x, y)
    print('✅ CUDA operations: OK')
    del x, y, z
    torch.cuda.empty_cache()

# Test KTransformersOps
ops = [op for op in dir(KTransformersOps) if not op.startswith('_')]
print(f'✅ KTransformersOps: {len(ops)} operations available')

print('\\n🎉 All tests passed!')
"

echo "============================================================"
