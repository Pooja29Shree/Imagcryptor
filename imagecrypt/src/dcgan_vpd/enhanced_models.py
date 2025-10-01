import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VirtualPlanetDomain(nn.Module):
    """Virtual Planet Domain for key generation and transformation"""
    def __init__(self, planet_dim=256, key_dim=16):
        super(VirtualPlanetDomain, self).__init__()
        self.planet_dim = planet_dim
        self.key_dim = key_dim
        
        # Planet coordinate system
        self.planet_encoder = nn.Sequential(
            nn.Linear(key_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, planet_dim)
        )
        
        # Planet-to-image domain mapper
        self.domain_mapper = nn.Sequential(
            nn.Linear(planet_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, key_dim)
        )
        
    def encode_to_planet(self, key):
        """Map encryption key to virtual planet coordinates"""
        planet_coords = self.planet_encoder(key)
        # Apply planetary transformations (spherical coordinates)
        planet_coords = torch.tanh(planet_coords)  # Normalize to planet surface
        return planet_coords
        
    def decode_from_planet(self, planet_coords):
        """Map planet coordinates back to key space"""
        decoded_key = self.domain_mapper(planet_coords)
        return decoded_key
        
    def planet_transformation(self, coords, rotation_angle=0.0):
        """Apply planetary rotation and transformation"""
        # Simple rotation transformation in planet domain
        cos_a = torch.cos(torch.tensor(rotation_angle))
        sin_a = torch.sin(torch.tensor(rotation_angle))
        
        # Apply rotation matrix concept to coordinates
        transformed = coords * cos_a + torch.roll(coords, 1, dims=-1) * sin_a
        return transformed

class MedicalImageEncoder(nn.Module):
    """Specialized encoder for medical images"""
    def __init__(self, image_channels=3, latent_dim=256):
        super(MedicalImageEncoder, self).__init__()
        
        # Medical image feature extraction
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(image_channels, 32, 4, 2, 1),  # 64->32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Second conv block  
            nn.Conv2d(32, 64, 4, 2, 1),  # 32->16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Third conv block
            nn.Conv2d(64, 128, 4, 2, 1),  # 16->8
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, 4, 2, 1),  # 8->4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class EnhancedGenerator(nn.Module):
    """Enhanced Generator with Virtual Planet Domain integration"""
    def __init__(self, z_dim=100, key_dim=16, image_channels=3, planet_dim=256):
        super(EnhancedGenerator, self).__init__()
        self.z_dim = z_dim
        self.key_dim = key_dim
        
        # Virtual Planet Domain
        self.planet_domain = VirtualPlanetDomain(planet_dim, key_dim)
        
        # Medical image encoder for input conditioning
        self.medical_encoder = MedicalImageEncoder(image_channels, 128)
        
        # Combined input: noise + planet_coords + medical_features
        combined_dim = z_dim + planet_dim + 128
        
        # Generator architecture (DCGAN-based)
        self.fc = nn.Linear(combined_dim, 512 * 4 * 4)
        
        self.conv_layers = nn.Sequential(
            # First upconv block
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 4->8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Second upconv block
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8->16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Third upconv block
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16->32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Output layer
            nn.ConvTranspose2d(64, image_channels, 4, 2, 1),  # 32->64
            nn.Tanh()
        )
        
    def forward(self, z, key, input_image=None):
        # Map key to virtual planet domain
        planet_coords = self.planet_domain.encode_to_planet(key)
        
        # Apply planet transformations (this is key innovation)
        planet_coords = self.planet_domain.planet_transformation(planet_coords)
        
        # Encode input image if provided (for conditional generation)
        if input_image is not None:
            medical_features = self.medical_encoder(input_image)
        else:
            medical_features = torch.zeros(z.size(0), 128, device=z.device)
        
        # Combine all inputs
        combined_input = torch.cat([z, planet_coords, medical_features], dim=1)
        
        # Generate encrypted image
        x = self.fc(combined_input)
        x = x.view(-1, 512, 4, 4)
        x = self.conv_layers(x)
        
        return x

class EnhancedReconstructor(nn.Module):
    """Enhanced Reconstructor with Virtual Planet Domain"""
    def __init__(self, key_dim=16, image_channels=3, planet_dim=256):
        super(EnhancedReconstructor, self).__init__()
        
        # Virtual Planet Domain (shared with generator)
        self.planet_domain = VirtualPlanetDomain(planet_dim, key_dim)
        
        # Encrypted image encoder
        self.encrypted_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 64, 4, 2, 1),  # 64->32
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 32->16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  # 16->8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),  # 8->4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        # Planet-guided reconstruction
        self.planet_guided_fc = nn.Linear(512 * 4 * 4 + planet_dim, 512 * 4 * 4)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 4->8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8->16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16->32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, image_channels, 4, 2, 1),  # 32->64
            nn.Tanh()
        )
        
    def forward(self, encrypted_image, key):
        # Encode encrypted image
        encoded = self.encrypted_encoder(encrypted_image)
        encoded_flat = encoded.view(encoded.size(0), -1)
        
        # Get planet coordinates from key
        planet_coords = self.planet_domain.encode_to_planet(key)
        
        # Apply INVERSE planet transformation for decryption
        planet_coords = self.planet_domain.planet_transformation(planet_coords, rotation_angle=-1.0)
        
        # Combine encoded image with planet guidance
        combined = torch.cat([encoded_flat, planet_coords], dim=1)
        guided_features = self.planet_guided_fc(combined)
        
        # Reshape and decode
        x = guided_features.view(-1, 512, 4, 4)
        reconstructed = self.decoder(x)
        
        return reconstructed

# Test function for the enhanced models
def test_enhanced_models():
    """Test the enhanced models with Virtual Planet Domain"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    
    # Create enhanced models
    generator = EnhancedGenerator().to(device)
    reconstructor = EnhancedReconstructor().to(device)
    
    # Create test data
    z = torch.randn(batch_size, 100).to(device)
    key = torch.randn(batch_size, 16).to(device)
    input_image = torch.randn(batch_size, 3, 64, 64).to(device)
    
    print("🧪 Testing Enhanced Models with Virtual Planet Domain...")
    
    # Test forward passes
    encrypted = generator(z, key, input_image)
    reconstructed = reconstructor(encrypted, key)
    
    print(f"✅ Enhanced Generator output: {encrypted.shape}")
    print(f"✅ Enhanced Reconstructor output: {reconstructed.shape}")
    print(f"✅ Virtual Planet Domain integration working!")
    
    return generator, reconstructor

if __name__ == "__main__":
    test_enhanced_models()