"""
DIP4: Image Restoration - Main Demo

This script demonstrates image restoration using inverse and Wiener filtering.
"""

import numpy as np
import cv2
import os
import argparse


from Dip4 import (
    degrade_image, circ_shift, dft_real2complex, idft_complex2real,
    apply_filter, compute_inverse_filter, inverse_filter,
    compute_wiener_filter, wiener_filter
)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_val = 255.0
    return 10 * np.log10((max_val ** 2) / mse)


def show_frequency_spectrum(img: np.ndarray, title: str = "Spectrum") -> np.ndarray:
    """Compute and display the magnitude spectrum of an image."""
    freq = dft_real2complex(img)
    # Shift zero frequency to center
    freq_shifted = np.fft.fftshift(freq)
    # Compute magnitude spectrum (log scale for visualization)
    magnitude = np.log(1 + np.abs(freq_shifted))
    # Normalize for display
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min()) * 255
    return magnitude.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='DIP4: Image Restoration Demo')
    parser.add_argument('--image', type=str, default='mandrill.png',
                        help='Path to input image')
    parser.add_argument('--snr', type=float, default=50.0,
                        help='Signal-to-noise ratio for degradation')
    parser.add_argument('--eps', type=float, default=0.01,
                        help='Threshold epsilon for inverse filter')
    parser.add_argument('--wiener-snr', type=float, default=50.0,
                        help='SNR parameter for Wiener filter')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable display (for headless environments)')
    args = parser.parse_args()
    
    # Load image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_dir = os.path.join(os.path.dirname(script_dir), 'dip4_cpp')
    
    img_path = args.image
    if not os.path.exists(img_path):
        img_path = os.path.join(script_dir, args.image)
    if not os.path.exists(img_path):
        img_path = os.path.join(cpp_dir, args.image)
    
    if os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read image at {img_path}")
            return
        img = img.astype(np.float32)
        print(f"Loaded image: {img_path}, size: {img.shape}")
    else:
        # Create synthetic test image
        print("Creating synthetic test image...")
        size = 256
        img = np.zeros((size, size), dtype=np.float32)
        # Add some patterns
        img[50:100, 50:100] = 200
        img[100:150, 150:200] = 150
        img[150:200, 50:100] = 100
        # Add gradient
        for i in range(size):
            img[200:220, i] = i * 255.0 / size
    
    print(f"\nImage statistics:")
    print(f"  Shape: {img.shape}")
    print(f"  Range: [{img.min():.1f}, {img.max():.1f}]")
    print(f"  Mean: {img.mean():.1f}")
    
    # Test circular shift
    print("\n=== Testing Circular Shift ===")
    test_img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    shifted = circ_shift(test_img, 1, 1)
    print(f"Original:\n{test_img}")
    print(f"Shifted by (1,1):\n{shifted}")
    
    # Test DFT round-trip
    print("\n=== Testing DFT Round-trip ===")
    small_img = img[:32, :32]
    freq = dft_real2complex(small_img)
    reconstructed = idft_complex2real(freq)
    error = np.max(np.abs(small_img - reconstructed))
    print(f"Max reconstruction error: {error:.6f}")
    
    # Degrade image
    print(f"\n=== Degrading Image (SNR={args.snr}) ===")
    np.random.seed(42)  # For reproducibility
    degraded, kernel = degrade_image(img, None, None, args.snr)
    
    degradation_psnr = compute_psnr(img, degraded)
    print(f"Degraded image PSNR: {degradation_psnr:.2f} dB")
    
    # Restore using inverse filter
    print(f"\n=== Inverse Filter Restoration (eps={args.eps}) ===")
    restored_inverse = inverse_filter(degraded, kernel, args.eps)
    inverse_psnr = compute_psnr(img, restored_inverse)
    print(f"Restored image PSNR: {inverse_psnr:.2f} dB")
    
    # Restore using Wiener filter
    print(f"\n=== Wiener Filter Restoration (SNR={args.wiener_snr}) ===")
    restored_wiener = wiener_filter(degraded, kernel, args.wiener_snr)
    wiener_psnr = compute_psnr(img, restored_wiener)
    print(f"Restored image PSNR: {wiener_psnr:.2f} dB")
    
    # Summary
    print("\n=== Summary ===")
    print(f"{'Method':<20} {'PSNR (dB)':<12} {'Improvement':<12}")
    print("-" * 44)
    print(f"{'Degraded':<20} {degradation_psnr:<12.2f} {'baseline':<12}")
    print(f"{'Inverse Filter':<20} {inverse_psnr:<12.2f} {inverse_psnr - degradation_psnr:+.2f} dB")
    print(f"{'Wiener Filter':<20} {wiener_psnr:<12.2f} {wiener_psnr - degradation_psnr:+.2f} dB")
    
    # Display results if available
    if not args.no_display:
        try:
            # Normalize images for display
            def normalize_for_display(image):
                img_norm = image.copy()
                img_norm = np.clip(img_norm, 0, 255)
                return img_norm.astype(np.uint8)
            
            # Create display grid
            original_display = normalize_for_display(img)
            degraded_display = normalize_for_display(degraded)
            inverse_display = normalize_for_display(restored_inverse)
            wiener_display = normalize_for_display(restored_wiener)
            
            # Show frequency spectra
            original_spectrum = show_frequency_spectrum(img)
            degraded_spectrum = show_frequency_spectrum(degraded)
            
            # Create comparison image
            row1 = np.hstack([original_display, degraded_display])
            row2 = np.hstack([inverse_display, wiener_display])
            comparison = np.vstack([row1, row2])
            
            cv2.imshow('Original (TL) | Degraded (TR) | Inverse (BL) | Wiener (BR)', comparison)
            cv2.imshow('Original Spectrum', original_spectrum)
            cv2.imshow('Degraded Spectrum', degraded_spectrum)
            
            print("\nPress any key to close windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"\nCould not display images: {e}")
            print("Run with --no-display flag in headless environments")


if __name__ == '__main__':
    main()
