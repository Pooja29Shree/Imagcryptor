import torch
import sys

print("🔍 GPU Detection and PyTorch Info")
print("=" * 40)

# PyTorch version
print(f"PyTorch version: {torch.__version__}")

# CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Multi-processors: {props.multi_processor_count}")
    
    # Test GPU tensor creation
    print(f"\n🧪 Testing GPU tensor creation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print("✅ GPU tensor operations working!")
        print(f"   Test tensor on: {z.device}")
        
        # Memory usage
        print(f"   GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"   GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
else:
    print("\n❌ No CUDA GPU detected")
    print("\n💡 To install CUDA-enabled PyTorch:")
    print("   pip uninstall torch torchvision")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("\n   Or for CUDA 12.1:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

print(f"\n🖥️  Device that will be used: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")