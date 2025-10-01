#!/usr/bin/env python3
"""
Real Image Testing for OptiSecure-3D Encryption System
Uses actual images from the imagecrypt/data/images folder
"""

import numpy as np
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
from simple_optisecure import SimpleOptiSecure
import time

def load_image(image_path, max_size=512):
    """Load and optionally resize image"""
    try:
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if too large
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"  Resized to: {img.size}")
        
        return np.array(img)
    except Exception as e:
        print(f"  Error loading {image_path}: {e}")
        return None

def calculate_metrics(original, decrypted):
    """Calculate image quality metrics"""
    mse = np.mean((original.astype(float) - decrypted.astype(float)) ** 2)
    
    if mse == 0:
        psnr = float('inf')
        ssim_approx = 1.0
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        # Simplified SSIM approximation
        mean_orig = np.mean(original)
        mean_dec = np.mean(decrypted)
        var_orig = np.var(original)
        var_dec = np.var(decrypted)
        covar = np.mean((original - mean_orig) * (decrypted - mean_dec))
        
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        ssim_approx = ((2 * mean_orig * mean_dec + c1) * (2 * covar + c2)) / \
                     ((mean_orig**2 + mean_dec**2 + c1) * (var_orig + var_dec + c2))
    
    return mse, psnr, ssim_approx

def test_image_encryption(image_path, output_dir):
    """Test encryption/decryption on a single image"""
    print(f"\n🔒 Testing: {os.path.basename(image_path)}")
    
    # Load image
    original = load_image(image_path)
    if original is None:
        return None
    
    print(f"  Image shape: {original.shape}")
    
    # Test with correct key
    start_time = time.time()
    key1 = "supersecretkey123"
    crypto_system = SimpleOptiSecure()
    encrypted, metadata = crypto_system.encrypt_image(original, key1)
    decrypted_correct = crypto_system.decrypt_image(encrypted, key1, metadata)
    encrypt_time = time.time() - start_time
    
    # Test with wrong key
    key2 = "wrongkey456"
    decrypted_wrong = crypto_system.decrypt_image(encrypted, key2, metadata)
    
    # Calculate metrics
    mse_correct, psnr_correct, ssim_correct = calculate_metrics(original, decrypted_correct)
    mse_wrong, psnr_wrong, ssim_wrong = calculate_metrics(original, decrypted_wrong)
    
    # Results
    results = {
        'filename': os.path.basename(image_path),
        'shape': original.shape,
        'encrypt_time': encrypt_time,
        'mse_correct': mse_correct,
        'psnr_correct': psnr_correct,
        'ssim_correct': ssim_correct,
        'mse_wrong': mse_wrong,
        'security_ratio': mse_wrong / max(mse_correct, 1e-10)
    }
    
    print(f"  ✅ Correct key - MSE: {mse_correct:.10f}, PSNR: {psnr_correct:.2f}dB")
    print(f"  ❌ Wrong key   - MSE: {mse_wrong:.2f}")
    print(f"  🛡️  Security ratio: {results['security_ratio']:.2e}x")
    print(f"  ⏱️  Encryption time: {encrypt_time:.3f}s")
    
    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save original
    Image.fromarray(original).save(os.path.join(output_dir, f"{base_name}_original.png"))
    
    # Save encrypted (convert to viewable format)
    encrypted_norm = ((encrypted - encrypted.min()) / 
                     (encrypted.max() - encrypted.min()) * 255).astype(np.uint8)
    Image.fromarray(encrypted_norm).save(os.path.join(output_dir, f"{base_name}_encrypted.png"))
    
    # Save decrypted results
    Image.fromarray(decrypted_correct).save(os.path.join(output_dir, f"{base_name}_decrypted_correct.png"))
    Image.fromarray(decrypted_wrong).save(os.path.join(output_dir, f"{base_name}_decrypted_wrong.png"))
    
    return results

def main():
    print("🔐 OptiSecure-3D Real Image Testing")
    print("=" * 50)
    
    # Paths
    image_dir = r"C:\Users\MANAS\Documents\GitHub\Imagcryptor\imagecrypt\data\images"
    output_dir = "real_image_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize crypto system (no longer needed here since we create per image)
    # crypto = StableChaotic3D()
    
    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = []
    
    for file in os.listdir(image_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_dir, file))
    
    print(f"Found {len(image_files)} image files")
    
    # Test sample of images (first 5 to avoid overwhelming output)
    test_files = image_files[:5]
    print(f"Testing first 5 images...")
    
    results = []
    total_start = time.time()
    
    for image_path in test_files:
        result = test_image_encryption(image_path, output_dir)
        if result:
            results.append(result)
    
    total_time = time.time() - total_start
    
    # Summary
    print(f"\n📊 SUMMARY RESULTS")
    print("=" * 50)
    print(f"Images processed: {len(results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per image: {total_time/len(results):.3f}s")
    print()
    
    if results:
        avg_mse = np.mean([r['mse_correct'] for r in results])
        avg_security = np.mean([r['security_ratio'] for r in results])
        
        print(f"Average MSE (correct key): {avg_mse:.10f}")
        print(f"Average security ratio: {avg_security:.2e}x")
        
        # Check if perfect reconstruction
        perfect_count = sum(1 for r in results if r['mse_correct'] == 0.0)
        print(f"Perfect reconstructions: {perfect_count}/{len(results)}")
        
        if perfect_count == len(results):
            print("🎉 ALL IMAGES: PERFECT RECONSTRUCTION ACHIEVED!")
        else:
            print("⚠️  Some images have reconstruction errors")
    
    print(f"\n📁 Results saved in: {output_dir}/")
    print(f"   - Original images")
    print(f"   - Encrypted images (normalized for viewing)")
    print(f"   - Decrypted with correct key")
    print(f"   - Decrypted with wrong key")

if __name__ == "__main__":
    main()