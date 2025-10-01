import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """Generator network for creating encrypted images"""
    def __init__(self, z_dim=100, key_dim=16, image_channels=3):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.key_dim = key_dim
        
        # Combine noise and key
        input_dim = z_dim + key_dim
        
        # Initial dense layer
        self.fc = nn.Linear(input_dim, 256 * 4 * 4)
        
        # Transposed convolution layers
        self.conv_transpose1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 4x4 -> 8x8
        self.conv_transpose2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 8x8 -> 16x16
        self.conv_transpose3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)    # 16x16 -> 32x32
        self.conv_transpose4 = nn.ConvTranspose2d(32, image_channels, 4, 2, 1)  # 32x32 -> 64x64
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        
    def forward(self, z, key):
        # Concatenate noise and key
        x = torch.cat([z, key], dim=1)
        
        # Dense layer
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)
        
        # Transpose convolutions with batch norm and ReLU
        x = F.relu(self.bn1(self.conv_transpose1(x)))
        x = F.relu(self.bn2(self.conv_transpose2(x)))
        x = F.relu(self.bn3(self.conv_transpose3(x)))
        x = torch.tanh(self.conv_transpose4(x))  # Output in [-1, 1]
        
        return x

class Discriminator(nn.Module):
    """Discriminator network to distinguish real from encrypted images"""
    def __init__(self, image_channels=3):
        super(Discriminator, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(image_channels, 32, 4, 2, 1)  # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)              # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)             # 16x16 -> 8x8
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)            # 8x8 -> 4x4
        
        # Batch normalization
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Final dense layer
        self.fc = nn.Linear(256 * 4 * 4, 1)
        
    def forward(self, x):
        # Convolutional layers with batch norm and LeakyReLU
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        
        # Flatten and output
        x = x.view(-1, 256 * 4 * 4)
        x = self.fc(x)
        
        return x

class Reconstructor(nn.Module):
    """Reconstructor network to decrypt encrypted images back to original"""
    def __init__(self, key_dim=16, image_channels=3):
        super(Reconstructor, self).__init__()
        self.key_dim = key_dim
        
        # Encoder part (image -> latent)
        self.encoder_conv1 = nn.Conv2d(image_channels, 32, 4, 2, 1)  # 64x64 -> 32x32
        self.encoder_conv2 = nn.Conv2d(32, 64, 4, 2, 1)              # 32x32 -> 16x16
        self.encoder_conv3 = nn.Conv2d(64, 128, 4, 2, 1)             # 16x16 -> 8x8
        self.encoder_conv4 = nn.Conv2d(128, 256, 4, 2, 1)            # 8x8 -> 4x4
        
        # Key processing
        self.key_fc = nn.Linear(key_dim, 256)
        
        # Decoder part (latent + key -> reconstructed image)
        self.decoder_conv1 = nn.ConvTranspose2d(512, 128, 4, 2, 1)   # 4x4 -> 8x8
        self.decoder_conv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)    # 8x8 -> 16x16
        self.decoder_conv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)     # 16x16 -> 32x32
        self.decoder_conv4 = nn.ConvTranspose2d(32, image_channels, 4, 2, 1)  # 32x32 -> 64x64
        
        # Batch normalization
        self.bn_enc2 = nn.BatchNorm2d(64)
        self.bn_enc3 = nn.BatchNorm2d(128)
        self.bn_enc4 = nn.BatchNorm2d(256)
        
        self.bn_dec1 = nn.BatchNorm2d(128)
        self.bn_dec2 = nn.BatchNorm2d(64)
        self.bn_dec3 = nn.BatchNorm2d(32)
        
    def forward(self, encrypted_image, key):
        # Encode encrypted image
        x = F.leaky_relu(self.encoder_conv1(encrypted_image), 0.2)
        x = F.leaky_relu(self.bn_enc2(self.encoder_conv2(x)), 0.2)
        x = F.leaky_relu(self.bn_enc3(self.encoder_conv3(x)), 0.2)
        x = F.leaky_relu(self.bn_enc4(self.encoder_conv4(x)), 0.2)
        
        # Process key
        key_processed = self.key_fc(key)
        key_processed = key_processed.view(-1, 256, 1, 1)
        key_processed = key_processed.expand(-1, -1, 4, 4)
        
        # Combine encoded image with key
        x = torch.cat([x, key_processed], dim=1)
        
        # Decode to reconstructed image
        x = F.relu(self.bn_dec1(self.decoder_conv1(x)))
        x = F.relu(self.bn_dec2(self.decoder_conv2(x)))
        x = F.relu(self.bn_dec3(self.decoder_conv3(x)))
        x = torch.tanh(self.decoder_conv4(x))  # Output in [-1, 1]
        
        return x

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Test function to verify models work
def test_models():
    """Test function to verify all models work correctly"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    
    # Create models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    reconstructor = Reconstructor().to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)
    reconstructor.apply(weights_init)
    
    # Create dummy data
    z = torch.randn(batch_size, 100).to(device)
    key = torch.randn(batch_size, 16).to(device)
    real_images = torch.randn(batch_size, 3, 64, 64).to(device)
    
    # Test forward passes
    encrypted_images = generator(z, key)
    disc_output = discriminator(encrypted_images)
    reconstructed_images = reconstructor(encrypted_images, key)
    
    print(f"Generator output shape: {encrypted_images.shape}")
    print(f"Discriminator output shape: {disc_output.shape}")
    print(f"Reconstructor output shape: {reconstructed_images.shape}")
    print("All models working correctly!")

if __name__ == "__main__":
    test_models()
