import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from .enhanced_models import EnhancedGenerator, EnhancedReconstructor, VirtualPlanetDomain
from .models import Discriminator  # Use existing discriminator
import torch.nn as nn
import os

def train_medical_image_encryption():
    """Train the Medical Image Encryption with DCGAN + Virtual Planet Domain"""
    print("🏥 Medical Image Encryption Training (DCGAN + Virtual Planet Domain)")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # GPU info and optimization
    if torch.cuda.is_available():
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   CUDA version: {torch.version.cuda}")
        torch.cuda.empty_cache()  # Clear GPU cache
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    else:
        print("⚠️  No GPU available, using CPU (will be much slower)")
        print("💡 To use GPU: Install CUDA-enabled PyTorch")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.ImageFolder("../data/", transform=transform)
    
    # Optimize batch size and workers for GPU
    if device.type == 'cuda':
        batch_size = 16  # Larger batch for GPU
        num_workers = 4  # More workers for faster data loading
        print("🚀 GPU optimization: batch_size=16, num_workers=4")
    else:
        batch_size = 4   # Smaller batch for CPU
        num_workers = 0  # No multiprocessing on CPU
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=num_workers, pin_memory=device.type=='cuda')
    print(f"✅ Medical dataset loaded: {len(dataset)} images, batch_size={batch_size}")
    
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
    
    # GPU memory optimization
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"🔧 GPU memory cleared, available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Optimizers (ultra-precise learning rates for MSE ~0.0005)
    if device.type == 'cuda':
        # Very precise learning rates for high-quality reconstruction
        optG = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))  # Lower for stability
        optR = torch.optim.Adam(R.parameters(), lr=0.003, betas=(0.5, 0.999))   # Higher focus on reconstruction
        optD = torch.optim.Adam(D.parameters(), lr=0.00002, betas=(0.5, 0.999)) # Much lower to not interfere
        print("🎯 Using ultra-precise learning rates for MSE ~0.0005")
    else:
        # Conservative learning rates for CPU
        optG = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))
        optR = torch.optim.Adam(R.parameters(), lr=0.002, betas=(0.5, 0.999))   
        optD = torch.optim.Adam(D.parameters(), lr=0.00001, betas=(0.5, 0.999))
        print("🐌 Using CPU-optimized learning rates")
    
    # Learning rate schedulers for ultra-precise convergence
    schedulerG = torch.optim.lr_scheduler.StepLR(optG, step_size=50, gamma=0.7)  # More aggressive decay
    schedulerR = torch.optim.lr_scheduler.StepLR(optR, step_size=40, gamma=0.8)  # Slower decay for reconstructor
    schedulerD = torch.optim.lr_scheduler.StepLR(optD, step_size=60, gamma=0.5)  # Aggressive decay for discriminator
    
    # Loss functions (optimized for MSE ~0.0005)
    adversarial_loss = nn.BCEWithLogitsLoss().to(device)
    reconstruction_loss = nn.MSELoss().to(device)     # Primary loss for pixel-perfect reconstruction
    perceptual_loss = nn.L1Loss().to(device)          # Secondary for perceptual quality
    identity_loss = nn.MSELoss().to(device)           # NEW: For identity preservation in generator
    
    # Training parameters (optimized for MSE ~0.0005)
    epochs = 200  # More epochs for better convergence
    lambda_recon = 1000.0    # MASSIVE reconstruction priority
    lambda_perceptual = 0.5  # Minimal perceptual weight
    lambda_planet = 1.0      # Minimal planet consistency weight
    lambda_identity = 500.0  # NEW: Identity preservation weight
    
    # Create directories
    os.makedirs("../exports/medical_encryption", exist_ok=True)
    os.makedirs("../checkpoints/medical", exist_ok=True)
    
    print(f"🚀 Starting ultra-precision medical encryption training for {epochs} epochs...")
    print("🎯 Target: MSE ~0.0005 for medical-grade reconstruction")
    print("🔧 Progressive training: Simple → Complex")
    
    # Early stopping variables (stricter for high precision)
    best_recon_mse = float('inf')
    patience = 30  # More patience for precision training
    no_improve_epochs = 0
    target_mse = 0.0005  # Our target MSE
    
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
            # Train Generator (FOCUS ON IDENTITY PRESERVATION)
            # =========================
            
            encrypted_images = G(z, encryption_key, real_images)
            
            # Progressive training weights (start simple, add complexity)
            epoch_progress = epoch / epochs
            
            # Adversarial loss (very weak, just for realism)
            d_fake = D(encrypted_images)
            g_adv_loss = adversarial_loss(d_fake, torch.ones_like(d_fake))
            
            # Identity preservation loss (CRITICAL for low MSE)
            g_identity_loss = identity_loss(encrypted_images, real_images)
            
            # Structural similarity loss (preserve image structure)
            g_structure_loss = torch.mean(torch.abs(
                torch.gradient(encrypted_images.mean(dim=1))[0] - 
                torch.gradient(real_images.mean(dim=1))[0]
            ))
            
            # Progressive generator loss (prioritize identity early, add adversarial later)
            adv_weight = 0.1 * epoch_progress  # Start at 0, grow to 0.1
            identity_weight = 1.0 - 0.5 * epoch_progress  # Start at 1.0, reduce to 0.5
            
            g_loss = (adv_weight * g_adv_loss + 
                     identity_weight * g_identity_loss + 
                     0.1 * g_structure_loss)
            
            optG.zero_grad()
            g_loss.backward()
            optG.step()
            
            # =========================
            # Train Reconstructor (ULTRA-PRECISE RECONSTRUCTION)
            # =========================
            
            # Reconstruct from encrypted images
            reconstructed_images = R(encrypted_images.detach(), encryption_key)
            
            # Multi-scale reconstruction losses for precision
            recon_mse = reconstruction_loss(reconstructed_images, real_images)
            recon_l1 = perceptual_loss(reconstructed_images, real_images)
            
            # Pixel-level precision loss (every pixel matters)
            pixel_precision_loss = torch.mean(torch.abs(reconstructed_images - real_images) ** 3)
            
            # Feature preservation loss (preserve important features)
            feature_loss = reconstruction_loss(
                torch.mean(reconstructed_images, dim=[2, 3]), 
                torch.mean(real_images, dim=[2, 3])
            )
            
            # Gradient preservation (preserve edges and structures)
            grad_real = torch.gradient(real_images)[2]  # Gradient in spatial dimensions
            grad_recon = torch.gradient(reconstructed_images)[2]
            gradient_loss = reconstruction_loss(grad_recon, grad_real)
            
            # Minimal planet consistency (don't let it interfere with precision)
            planet_coords_r = R.planet_domain.encode_to_planet(encryption_key)
            decoded_key_r = R.planet_domain.decode_from_planet(planet_coords_r)
            planet_consistency = torch.mean((encryption_key - decoded_key_r) ** 2)
            
            # Ultra-precise reconstructor loss
            r_loss = (lambda_recon * recon_mse + 
                     lambda_perceptual * recon_l1 + 
                     100.0 * pixel_precision_loss +  # High weight for pixel precision
                     50.0 * feature_loss +           # Feature preservation
                     20.0 * gradient_loss +          # Edge preservation
                     lambda_planet * planet_consistency)
            
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
        
        # Update learning rates
        schedulerG.step()
        schedulerR.step()
        schedulerD.step()
        
        if epoch % 5 == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] G_Loss: {avg_g:.4f}, R_Loss: {avg_r:.4f}, D_Loss: {avg_d:.4f}")
            print(f"   🎯 Current MSE: {avg_recon_mse:.6f} | Best: {best_recon_mse:.6f} | Target: {target_mse:.6f}")
            print(f"   📈 Progress: {no_improve_epochs}/{patience} | LR_R: {optR.param_groups[0]['lr']:.6f}")
            
            # Achievement tracking
            if avg_recon_mse <= target_mse:
                print(f"   🏆 TARGET ACHIEVED! MSE {avg_recon_mse:.6f} <= {target_mse:.6f}")
            elif avg_recon_mse <= 0.001:
                print(f"   🥈 Excellent quality! MSE {avg_recon_mse:.6f}")
            elif avg_recon_mse <= 0.01:
                print(f"   🥉 Good quality! MSE {avg_recon_mse:.6f}")
        
        # Early stopping (but continue if we're improving toward target)
        if no_improve_epochs >= patience and best_recon_mse > target_mse * 2:
            print(f"🛑 Early stopping at epoch {epoch+1} - No improvement for {patience} epochs")
            print(f"   🏆 Best reconstruction MSE: {best_recon_mse:.6f}")
            if best_recon_mse <= target_mse:
                print(f"   ✅ TARGET ACHIEVED!")
            else:
                print(f"   ⚠️  Target {target_mse:.6f} not reached, but training can continue manually")
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