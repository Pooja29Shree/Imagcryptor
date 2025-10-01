"""
OptiSecure-3D Demo Script
Test the chaotic encryption system with a simple example
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optisecure3d import OptiSecure3D, demo_optisecure3d
import numpy as np

def simple_test():
    """Simple test with a small image"""
    print("🧪 Simple OptiSecure-3D Test")
    print("=" * 30)
    
    # Create a simple 64x64 RGB test image
    test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    # Make it more structured (not just noise)
    test_image[:32, :32, 0] = 255  # Red square
    test_image[32:, 32:, 1] = 255  # Green square
    test_image[:32, 32:, 2] = 255  # Blue square
    
    print(f"📸 Created test image: {test_image.shape}")
    
    # Initialize OptiSecure-3D
    optisecure = OptiSecure3D(compression_quality=95)
    
    # Test encryption/decryption
    key = "TestKey123"
    
    print("\n🔐 Testing encryption...")
    encrypted, metadata = optisecure.encrypt_image(test_image, key)
    
    print("\n🔓 Testing decryption with correct key...")
    decrypted_correct = optisecure.decrypt_image(encrypted, key, metadata)
    
    print("\n❌ Testing decryption with wrong key...")
    decrypted_wrong = optisecure.decrypt_image(encrypted, "WrongKey", metadata)
    
    # Calculate results
    metrics_correct = optisecure.calculate_metrics(test_image, decrypted_correct)
    metrics_wrong = optisecure.calculate_metrics(test_image, decrypted_wrong)
    
    print(f"\n📊 Results:")
    print(f"   ✅ Correct key - MSE: {metrics_correct['mse']:.10f}")
    print(f"   ✅ Correct key - Perfect: {metrics_correct['perfect_reconstruction']}")
    print(f"   ❌ Wrong key - MSE: {metrics_wrong['mse']:.2f}")
    
    if metrics_correct['mse'] < 0.01:
        print("\n🎉 SUCCESS! Perfect reconstruction achieved!")
        print("   OptiSecure-3D is working correctly!")
    else:
        print(f"\n⚠️  Reconstruction error: {metrics_correct['mse']}")
    
    return metrics_correct['mse'] < 0.01

def benchmark_test():
    """Test with different image sizes"""
    print("\n🏃 Benchmark Test - Different Image Sizes")
    print("=" * 45)
    
    sizes = [(32, 32), (64, 64), (128, 128)]
    optisecure = OptiSecure3D()
    
    for h, w in sizes:
        print(f"\n📏 Testing {h}x{w} image...")
        
        # Create test image
        test_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        
        # Encrypt/decrypt
        encrypted, metadata = optisecure.encrypt_image(test_img, "BenchmarkKey")
        decrypted = optisecure.decrypt_image(encrypted, "BenchmarkKey", metadata)
        
        # Metrics
        metrics = optisecure.calculate_metrics(test_img, decrypted)
        
        print(f"   MSE: {metrics['mse']:.10f}")
        print(f"   Perfect: {metrics['perfect_reconstruction']}")

if __name__ == "__main__":
    # Run simple test first
    success = simple_test()
    
    if success:
        # Run benchmark if simple test passes
        benchmark_test()
        
        # Run full demo if available
        try:
            print("\n" + "="*50)
            demo_optisecure3d()
        except Exception as e:
            print(f"\nDemo failed (missing dependencies): {e}")
            print("But core OptiSecure-3D functionality is working! ✅")
    else:
        print("\n❌ Simple test failed. Check implementation.")