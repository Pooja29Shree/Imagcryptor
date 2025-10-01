#!/usr/bin/env python3
"""
OptiSecure-3D - Main Entry Point

Run from the imagecrypt/src/optisecure3d/ directory.
"""

import sys
import os
import numpy as np

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Main entry point for OptiSecure-3D system"""
    print("🔐 OptiSecure-3D Chaotic Image Encryption")
    print("=" * 50)
    
    try:
        # Test imports
        from simple_optisecure import SimpleOptiSecure, StableChaotic3D
        print("✅ All imports successful!")
        
        # Basic system info
        print(f"📁 Working directory: {os.getcwd()}")
        print(f"🔧 Python path: {sys.path[0]}")
        
        # Check dependencies
        print(f"🔢 NumPy available: {np.__version__}")
        
        try:
            from PIL import Image
            print("🖼️  PIL available for image processing")
        except ImportError:
            print("⚠️  PIL not available - install with: pip install Pillow")
            
        # Initialize and test the system
        print("\n🧪 Testing OptiSecure-3D system...")
        crypto = SimpleOptiSecure()
        print("   ✅ Encryption system initialized")
        
        # Test with a small sample
        print("\n🔬 Quick encryption test...")
        test_data = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        test_key = "test_key_123"
        
        encrypted, metadata = crypto.encrypt_image(test_data, test_key)
        decrypted = crypto.decrypt_image(encrypted, test_key, metadata)
        
        # Check perfect reconstruction
        mse = np.mean((test_data.astype(float) - decrypted.astype(float)) ** 2)
        print(f"   MSE (should be 0.0): {mse}")
        
        if mse == 0.0:
            print("   🎉 Perfect reconstruction achieved!")
        else:
            print("   ⚠️  Reconstruction error detected")
            
        print("\n🎯 OptiSecure-3D system ready!")
        print("   Use simple_optisecure.py for basic encryption")
        print("   Use demo_real_images.py for real image testing")
        print("   Use test_real_images.py for comprehensive testing")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're in the optisecure3d directory")
        print("   Install requirements: pip install numpy Pillow")
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)