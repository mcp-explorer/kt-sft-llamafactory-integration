#!/usr/bin/env python3
"""Check GPU status using PyTorch"""

import torch

print("=" * 60)
print("PyTorch GPU Status Check")
print("=" * 60)

# Basic CUDA availability
print(f"\nCUDA Available: {torch.cuda.is_available()}")
print(f"PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Get current device
    current_device = torch.cuda.current_device()
    print(f"Current GPU Device: {current_device}")
    
    # GPU details
    for i in range(torch.cuda.device_count()):
        print(f"\n--- GPU {i} ---")
        props = torch.cuda.get_device_properties(i)
        print(f"  Name: {props.name}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multiprocessors: {props.multi_processor_count}")
        
        # Current memory usage
        if i == current_device:
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  Allocated Memory: {allocated:.2f} GB")
            print(f"  Reserved Memory: {reserved:.2f} GB")
    
    # Test tensor creation
    print(f"\n--- Testing GPU ---")
    try:
        x = torch.randn(1000, 1000).cuda()
        print(f"  ✓ Successfully created tensor on GPU")
        print(f"  Tensor device: {x.device}")
        del x
        torch.cuda.empty_cache()
        print(f"  ✓ GPU memory cleared")
    except Exception as e:
        print(f"  ✗ Error creating tensor on GPU: {e}")
else:
    print("\n⚠ CUDA is not available. PyTorch is running in CPU mode.")
    print("   Make sure:")
    print("   1. NVIDIA drivers are installed (check: nvidia-smi)")
    print("   2. CUDA toolkit is installed")
    print("   3. PyTorch was installed with CUDA support")

print("\n" + "=" * 60)

