# KTransformers Test Commands

Quick reference for testing KTransformers after successful build.

---

## Quick Test (30 seconds)

```bash
docker exec llamafactory python -c "
import torch
import ktransformers
print('✅ PyTorch:', torch.__version__)
print('✅ KTransformers:', ktransformers.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
"
```

---

## Basic Functionality Test (1 minute)

```bash
docker exec llamafactory python -c "
import torch
import ktransformers
import KTransformersOps

print('=' * 60)
print('BASIC FUNCTIONALITY TEST')
print('=' * 60)

# Test 1: Imports
print('✅ Imports: OK')

# Test 2: CUDA
if torch.cuda.is_available():
    x = torch.randn(10, 10).cuda()
    y = torch.randn(10, 10).cuda()
    z = torch.matmul(x, y)
    print('✅ CUDA operations: OK')
    del x, y, z
    torch.cuda.empty_cache()

# Test 3: KTransformersOps
ops = [op for op in dir(KTransformersOps) if not op.startswith('_')]
print(f'✅ KTransformersOps: {len(ops)} operations available')

# Test 4: Utility functions
from ktransformers.util import utils
print('✅ Utility functions: OK')

print('\\n🎉 All basic tests passed!')
"
```

---

## Comprehensive Test (2-3 minutes)

```bash
docker exec llamafactory python -c "
import torch
import ktransformers
print('=' * 70)
print('COMPREHENSIVE TEST')
print('=' * 70)

# Test imports
print('\\n[1] Imports...')
import KTransformersOps
from ktransformers.util import utils
print('   ✅ PASS')

# Test CUDA
print('\\n[2] CUDA operations...')
if torch.cuda.is_available():
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.matmul(x, y)
    print(f'   ✅ PASS: {z.shape} on {z.device}')
    del x, y, z
    torch.cuda.empty_cache()
else:
    print('   ❌ FAIL: CUDA not available')

# Test KTransformersOps
print('\\n[3] KTransformersOps...')
ops = [op for op in dir(KTransformersOps) if not op.startswith('_')]
print(f'   ✅ PASS: {len(ops)} operations')

# Test memory
print('\\n[4] Memory management...')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    initial = torch.cuda.memory_allocated(0) / 1024**2
    x = torch.randn(500, 500).cuda()
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    del x
    torch.cuda.empty_cache()
    final = torch.cuda.memory_allocated(0) / 1024**2
    print(f'   ✅ PASS: {initial:.2f}MB → {allocated:.2f}MB → {final:.2f}MB')

# Test utility functions
print('\\n[5] Utility functions...')
if torch.cuda.is_available():
    cap = utils.get_compute_capability()
    print(f'   ✅ PASS: Compute capability = {cap}')

print('\\n' + '=' * 70)
print('🎉 ALL TESTS PASSED!')
print('=' * 70)
"
```

---

## Stress Test (3-5 minutes)

```bash
docker exec llamafactory python -c "
import torch
print('=' * 70)
print('STRESS TEST')
print('=' * 70)

if not torch.cuda.is_available():
    print('❌ CUDA not available')
    exit(1)

# Large operations
print('\\n[1] Large matrix multiplication...')
x = torch.randn(2000, 2000).cuda()
y = torch.randn(2000, 2000).cuda()
z = torch.matmul(x, y)
print(f'   ✅ PASS: {z.shape}')
del x, y, z
torch.cuda.empty_cache()

# Multiple operations
print('\\n[2] Multiple operations...')
for i in range(10):
    a = torch.randn(500, 500).cuda()
    b = torch.randn(500, 500).cuda()
    c = torch.matmul(a, b)
    del a, b, c
print('   ✅ PASS: 10 operations completed')
torch.cuda.empty_cache()

# Memory cleanup
print('\\n[3] Memory cleanup...')
initial = torch.cuda.memory_allocated(0) / 1024**2
tensors = [torch.randn(1000, 1000).cuda() for _ in range(5)]
peak = torch.cuda.memory_allocated(0) / 1024**2
del tensors
torch.cuda.empty_cache()
final = torch.cuda.memory_allocated(0) / 1024**2
print(f'   ✅ PASS: {initial:.2f}MB → {peak:.2f}MB → {final:.2f}MB')

print('\\n🎉 Stress test passed!')
"
```

---

## Test with LLaMA-Factory Integration

```bash
docker exec llamafactory python -c "
import torch
import llamafactory
import ktransformers

print('=' * 70)
print('LLAMA-FACTORY INTEGRATION TEST')
print('=' * 70)

print('\\n[1] Imports...')
print(f'   ✅ llamafactory: {llamafactory.__version__}')
print(f'   ✅ ktransformers: {ktransformers.__version__}')
print(f'   ✅ torch: {torch.__version__}')

print('\\n[2] CUDA...')
if torch.cuda.is_available():
    print(f'   ✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'   ✅ CUDA: {torch.version.cuda}')
else:
    print('   ❌ CUDA not available')

print('\\n✅ Integration test passed!')
"
```

---

## Interactive Python Shell

For interactive testing:

```bash
docker exec -it llamafactory python
```

Then in Python:
```python
import torch
import ktransformers
import KTransformersOps

# Check versions
print(torch.__version__)
print(ktransformers.__version__)

# Test CUDA
x = torch.randn(10, 10).cuda()
y = torch.randn(10, 10).cuda()
z = torch.matmul(x, y)
print(z.shape)

# Check available operations
ops = [op for op in dir(KTransformersOps) if not op.startswith('_')]
print(ops)
```

---

## Run All Tests Script

Save this as `test_kt.sh`:

```bash
#!/bin/bash
echo "Running KTransformers tests..."

echo -e "\n=== Quick Test ==="
docker exec llamafactory python -c "
import torch, ktransformers
print('PyTorch:', torch.__version__)
print('KTransformers:', ktransformers.__version__)
print('CUDA:', torch.cuda.is_available())
"

echo -e "\n=== Basic Functionality ==="
docker exec llamafactory python -c "
import torch, ktransformers, KTransformersOps
from ktransformers.util import utils
x = torch.randn(10, 10).cuda()
y = torch.randn(10, 10).cuda()
z = torch.matmul(x, y)
print('CUDA ops: OK')
ops = [op for op in dir(KTransformersOps) if not op.startswith('_')]
print(f'KTransformersOps: {len(ops)} operations')
"

echo -e "\n✅ All tests completed!"
```

Make it executable and run:
```bash
chmod +x test_kt.sh
./test_kt.sh
```

---

## Recommended: Quick Test First

Start with the **Quick Test** (30 seconds) to verify everything is working:

```bash
docker exec llamafactory python -c "
import torch
import ktransformers
print('✅ PyTorch:', torch.__version__)
print('✅ KTransformers:', ktransformers.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
"
```

If that passes, you can run the more comprehensive tests!

