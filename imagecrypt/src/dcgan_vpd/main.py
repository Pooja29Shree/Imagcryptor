#!/usr/bin/env python3
"""
DCGAN + Virtual Planet Domain - Main Entry Point

This script demonstrates the DCGAN + VPD encryption system.
Run from the imagecrypt/src/dcgan_vpd/ directory.
"""

import sys
import os

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Main entry point for DCGAN+VPD system"""
    print("🌍 DCGAN + Virtual Planet Domain Image Encryption")
    print("=" * 50)
    
    try:
        # Test imports
        from enhanced_models import VirtualPlanetDomain, EnhancedGenerator, EnhancedReconstructor
        from models import Discriminator
        print("✅ All imports successful!")
        
        # Basic system info
        print(f"📁 Working directory: {os.getcwd()}")
        print(f"🔧 Python path: {sys.path[0]}")
        
        # Check if PyTorch is available
        try:
            import torch
            print(f"🔥 PyTorch available: {torch.__version__}")
            print(f"🚀 CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
        except ImportError:
            print("⚠️  PyTorch not installed - install with: pip install torch torchvision")
            
        # Initialize a test VPD system
        print("\n🧪 Testing Virtual Planet Domain...")
        vpd = VirtualPlanetDomain()
        print(f"   Planet dimension: {vpd.planet_dim}")
        print(f"   Key dimension: {vpd.key_dim}")
        
        print("\n🎯 DCGAN+VPD system ready!")
        print("   Use train_medical.py for training")
        print("   Use enhanced_models.py for encryption/decryption")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're in the dcgan_vpd directory")
        print("   Install requirements: pip install torch torchvision")
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)