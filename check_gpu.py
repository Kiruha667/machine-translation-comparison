"""
Quick GPU check script
Run: python check_gpu.py
"""

import torch

print("="*80)
print("GPU CONFIGURATION CHECK")
print("="*80)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\nCUDA Available: {cuda_available}")

if cuda_available:
    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    
    print(f"GPU Name: {gpu_name}")
    print(f"Number of GPUs: {gpu_count}")
    print(f"Current Device: {current_device}")
    
    # Memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    
    print(f"\nMemory Info:")
    print(f"  Total: {total_memory:.2f} GB")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")
    print(f"  Free: {total_memory - allocated:.2f} GB")
    
    # CUDA version
    print(f"\nCUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Compute capability
    capability = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: {capability[0]}.{capability[1]}")
    
    # Test tensor on GPU
    print("\nTesting GPU with tensor operations...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✓ GPU test successful!")
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
    
    print("\n" + "="*80)
    print("✓ GPU IS READY FOR TRAINING!")
    print("="*80)
    
else:
    print("\n" + "="*80)
    print("✗ CUDA NOT AVAILABLE")
    print("="*80)
    print("\nPossible solutions:")
    print("1. Install/Update NVIDIA drivers:")
    print("   https://www.nvidia.com/download/index.aspx")
    print("\n2. Install CUDA Toolkit:")
    print("   https://developer.nvidia.com/cuda-downloads")
    print("\n3. Reinstall PyTorch with CUDA support:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\n4. Check if GPU is enabled in BIOS/UEFI")
    print("\n5. Restart computer after driver installation")
    
    print("\nCurrent PyTorch version:", torch.__version__)
    print("Built with CUDA:", torch.version.cuda if hasattr(torch.version, 'cuda') else "No")
