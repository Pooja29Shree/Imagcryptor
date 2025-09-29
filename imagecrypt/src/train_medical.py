import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from enhanced_models import EnhancedGenerator, EnhancedReconstructor, VirtualPlanetDomain
from models import Discriminator  # Use existing discriminator
import torch.nn as nn
import os

def train_medical_image_encryption():
    """Train the Medical Image Encryption with DCGAN + Virtual Planet Domain"""
    print("🏥 Medical Image Encryption Training (DCGAN + Virtual Planet Domain)")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.ImageFolder("../data/", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    print(f"✅ Medical dataset loaded: {len(dataset)} images")
    
    # Model initialization
    z_dim = 100
    key_dim = 16
    img_channels = 3
    planet_dim = 256
    
    # Enhanced models with Virtual Planet Domain
    G = EnhancedGenerator(z_dim, key_dim, img_channels, planet_dim).to(device)
    R = EnhancedReconstructor(key_dim, img_channels, planet_dim).to(device)
    D = Discriminator(img_channels).to(device)  # Standard discriminator
    
    print("✅ Enhanced models initialized with Virtual Planet Domain")
    
    # Optimizers (better learning rates for convergence)
    optG = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))  # Lower LR
    optR = torch.optim.Adam(R.parameters(), lr=0.0005, betas=(0.5, 0.999))  # Higher for reconstruction
    optD = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))  # Lower LR
    
    # Loss functions (better for medical images)
    adversarial_loss = nn.BCEWithLogitsLoss().to(device)
    reconstruction_loss = nn.MSELoss().to(device)  # MSE for better gradients
    perceptual_loss = nn.L1Loss().to(device)       # L1 for perceptual
    
    # Training parameters (better loss balancing)
    epochs = 100
    lambda_recon = 200.0     # Much higher for medical reconstruction
    lambda_perceptual = 10.0
    lambda_planet = 50.0     # Add planet domain consistency
    
    # Create directories
    os.makedirs("../exports/medical_encryption", exist_ok=True)
    os.makedirs("../checkpoints/medical", exist_ok=True)
    
    print(f"🚀 Starting improved medical image encryption training for {epochs} epochs...")
    print("🎯 Key improvements: Better LR, MSE loss, Planet consistency")
    
    # Early stopping variables
    best_recon_mse = float('inf')
    patience = 20
    no_improve_epochs = 0
    
    for epoch in range(epochs):
        epoch_g_loss = 0
        epoch_r_loss = 0  
        epoch_d_loss = 0
        epoch_recon_mse = 0
        num_batches = 0
        
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Generate encryption keys (planet domain)
            z = torch.randn(batch_size, z_dim, device=device)
            encryption_key = torch.randn(batch_size, key_dim, device=device)
            
            # =========================
            # Train Discriminator
            # =========================
            
            # Real images
            d_real = D(real_images)
            d_loss_real = adversarial_loss(d_real, torch.ones_like(d_real) * 0.9)  # Label smoothing
            
            # Encrypted images
            with torch.no_grad():
                encrypted_images = G(z, encryption_key, real_images)
            d_fake = D(encrypted_images.detach())
            d_loss_fake = adversarial_loss(d_fake, torch.zeros_like(d_fake))
            
            d_loss = d_loss_real + d_loss_fake
            
            optD.zero_grad()
            d_loss.backward()
            optD.step()
            
            # =========================
            # Train Generator 
            # =========================
            
            encrypted_images = G(z, encryption_key, real_images)
            
            # Adversarial loss
            d_fake = D(encrypted_images)
            g_adv_loss = adversarial_loss(d_fake, torch.ones_like(d_fake))
            
            # Planet domain consistency loss
            planet_loss = torch.mean(torch.abs(encrypted_images - G(z, encryption_key, real_images)))
            
            g_loss = g_adv_loss + 0.1 * planet_loss
            
            optG.zero_grad()
            g_loss.backward()
            optG.step()
            
            # =========================
            # Train Reconstructor
            # =========================
            
            # Reconstruct from encrypted images
            reconstructed_images = R(encrypted_images.detach(), encryption_key)
            
            # Reconstruction losses
            recon_mse = reconstruction_loss(reconstructed_images, real_images)
            recon_perceptual = perceptual_loss(reconstructed_images, real_images)
            
            # Planet domain consistency for reconstructor
            planet_coords_r = R.planet_domain.encode_to_planet(encryption_key)
            decoded_key_r = R.planet_domain.decode_from_planet(planet_coords_r)
            planet_consistency = torch.mean((encryption_key - decoded_key_r) ** 2)
            
            # Medical image quality preservation with planet consistency
            r_loss = lambda_recon * recon_mse + lambda_perceptual * recon_perceptual + lambda_planet * planet_consistency
            
            optR.zero_grad()
            r_loss.backward()
            optR.step()
            
            # Track losses
            epoch_g_loss += g_loss.item()
            epoch_r_loss += r_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_recon_mse += recon_mse.item()  # Track raw MSE for early stopping
            num_batches += 1
            
            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                with torch.no_grad():
                    # Save detailed comparison showing progress
                    comparison = torch.cat([
                        real_images[:4] * 0.5 + 0.5,
                        encrypted_images[:4] * 0.5 + 0.5,
                        reconstructed_images[:4] * 0.5 + 0.5
                    ], dim=0)
                    
                    save_image(comparison, 
                             f"../exports/medical_encryption/epoch_{epoch+1:03d}_progress.png", 
                             nrow=4, padding=2)
                    
                    # Calculate and show reconstruction quality
                    mse = torch.mean((real_images - reconstructed_images) ** 2)
                    print(f"   📊 Epoch {epoch+1} - Reconstruction MSE: {mse.item():.6f}")
        
        # Calculate averages and check for improvement
        avg_g = epoch_g_loss / num_batches
        avg_r = epoch_r_loss / num_batches  
        avg_d = epoch_d_loss / num_batches
        avg_recon_mse = epoch_recon_mse / num_batches
        
        # Early stopping check
        if avg_recon_mse < best_recon_mse:
            best_recon_mse = avg_recon_mse
            no_improve_epochs = 0
            # Save best model
            if epoch > 10:  # Only save after some training
                torch.save(G.state_dict(), "../checkpoints/medical/best_generator.pth")
                torch.save(R.state_dict(), "../checkpoints/medical/best_reconstructor.pth")
        else:
            no_improve_epochs += 1
        
        if epoch % 5 == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] G_Loss: {avg_g:.4f}, R_Loss: {avg_r:.4f}, D_Loss: {avg_d:.4f}")
            print(f"   🎯 Best MSE: {best_recon_mse:.6f}, No improve: {no_improve_epochs}/{patience}")
        
        # Early stopping
        if no_improve_epochs >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1} - No improvement for {patience} epochs")
            print(f"   🏆 Best reconstruction MSE: {best_recon_mse:.6f}")
            break
        
        # Save models every 20 epochs 
        if (epoch + 1) % 20 == 0:
            try:
                torch.save(G.state_dict(), f"../checkpoints/medical/enhanced_generator_epoch_{epoch+1}.pth")
                torch.save(R.state_dict(), f"../checkpoints/medical/enhanced_reconstructor_epoch_{epoch+1}.pth")
                torch.save(D.state_dict(), f"../checkpoints/medical/discriminator_epoch_{epoch+1}.pth")
                print(f"✅ Models saved at epoch {epoch+1}")
            except Exception as e:
                print(f"⚠️  Failed to save models at epoch {epoch+1}: {e}")
    
    # Save final models
    try:
        torch.save(G.state_dict(), "../checkpoints/medical/enhanced_generator_final.pth")
        torch.save(R.state_dict(), "../checkpoints/medical/enhanced_reconstructor_final.pth")  
        torch.save(D.state_dict(), "../checkpoints/medical/discriminator_final.pth")
        print("✅ Final models saved successfully")
    except Exception as e:
        print(f"⚠️  Failed to save final models: {e}")
        print("💡 Try freeing up disk space or check file permissions")
    
    print("\n✅ Medical Image Encryption Training Completed!")
    print("🏥 Enhanced models with Virtual Planet Domain saved")
    print("📁 Checkpoints: ../checkpoints/medical/")
    print("📸 Sample images: ../exports/medical_encryption/")
    print("\n🎯 This implementation now matches the paper's approach!")

if __name__ == "__main__":
    train_medical_image_encryption()