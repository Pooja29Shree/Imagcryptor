__version__ = "1.0.0"
__author__ = "Imagcryptor Team"

# Import main components for easy access
try:
    from .simple_optisecure import SimpleOptiSecure, StableChaotic3D
    from .optisecure3d import *
except ImportError:
    pass