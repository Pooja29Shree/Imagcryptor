#!/usr/bin/env python3
"""
Imagcryptor - Master Test Script

Tests both encryption systems from the main imagecrypt directory
to ensure proper organization and import functionality.
"""

import sys
import os

def test_optisecure3d():
    """Test OptiSecure-3D system"""
    print("🔐 Testing OptiSecure-3D System...")
    try:
        # Add path and import
        sys.path.append('src/optisecure3d')
        from simple_optisecure import SimpleOptiSecure
        
        # Quick test
        import numpy as np
        crypto = SimpleOptiSecure()
        
        # Test with small image
        test_image = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        encrypted, metadata = crypto.encrypt_image(test_image, "test123")
        decrypted = crypto.decrypt_image(encrypted, "test123", metadata)
        
        mse = np.mean((test_image.astype(float) - decrypted.astype(float)) ** 2)
        
        if mse == 0.0:
            print("   ✅ OptiSecure-3D: PERFECT (MSE = 0.0)")
            return True
        else:
            print(f"   ❌ OptiSecure-3D: ERROR (MSE = {mse})")
            return False
            
    except Exception as e:
        print(f"   ❌ OptiSecure-3D: Import/Test Error - {e}")
        return False

def test_dcgan_vpd():
    """Test DCGAN+VPD system"""
    print("🌍 Testing DCGAN+VPD System...")
    try:
        # Add path and import
        sys.path.append('src/dcgan_vpd')
        from enhanced_models import VirtualPlanetDomain
        
        # Quick test
        vpd = VirtualPlanetDomain()
        
        if hasattr(vpd, 'planet_dim') and hasattr(vpd, 'key_dim'):
            print(f"   ✅ DCGAN+VPD: System Ready (Planet Dim: {vpd.planet_dim})")
            return True
        else:
            print("   ❌ DCGAN+VPD: Initialization Error")
            return False
            
    except Exception as e:
        print(f"   ❌ DCGAN+VPD: Import/Test Error - {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Imagcryptor - Master System Test")
    print("=" * 50)
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Test both systems
    optisecure_ok = test_optisecure3d()
    dcgan_ok = test_dcgan_vpd()
    
    print("\n📊 Test Results:")
    print(f"   OptiSecure-3D: {'✅ PASS' if optisecure_ok else '❌ FAIL'}")
    print(f"   DCGAN+VPD:     {'✅ PASS' if dcgan_ok else '❌ FAIL'}")
    
    if optisecure_ok and dcgan_ok:
        print("\n🎉 ALL SYSTEMS OPERATIONAL!")
        print("   Both encryption methods are ready for use")
        return True
    else:
        print("\n⚠️  Some systems have issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)