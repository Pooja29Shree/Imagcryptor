"""
DCGAN + Virtual Planet Domain (VPD) Encryption System

This package implements the "Deep Learning–Driven Medical Image Encryption 
with DCGAN + Virtual Planet Domain" approach using neural networks for 
secure image encryption.

Key Components:
- enhanced_models.py: VirtualPlanetDomain, EnhancedGenerator, EnhancedReconstructor
- train_medical.py: Medical image encryption training
- models.py: Standard DCGAN components
- utils.py: Helper functions for training and data processing

Usage:
    from imagecrypt.src.dcgan_vpd.enhanced_models import VirtualPlanetDomain
    from imagecrypt.src.dcgan_vpd.train_medical import train_medical_encryption
"""

__version__ = "1.0.0"
__author__ = "Imagcryptor Team"

# Import main components for easy access
try:
    from .enhanced_models import VirtualPlanetDomain, EnhancedGenerator, EnhancedReconstructor
    from .models import Generator, Discriminator
    from .utils import *
except ImportError:
    # Handle case where dependencies aren't installed
    pass