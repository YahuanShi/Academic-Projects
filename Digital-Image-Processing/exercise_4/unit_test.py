"""
Unit tests for DIP4: Image Restoration

Tests cover:
- Circular shift (circ_shift)
- DFT and IDFT (dft_real2complex, idft_complex2real)
- Filter application (apply_filter)
- Inverse filter (compute_inverse_filter, inverse_filter)
- Wiener filter (compute_wiener_filter, wiener_filter)
"""

import unittest
import numpy as np
import cv2
import os
import sys
import importlib.util

import Dip4 as dip4


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_val = 255.0
    return 10 * np.log10((max_val ** 2) / mse)


def create_test_image(size: int = 64) -> np.ndarray:
    """Create a simple test image."""
    img = np.zeros((size, size), dtype=np.float32)
    # Add some patterns
    img[10:20, 10:20] = 200  # Square
    img[30:50, 30:50] = 150  # Larger square
    img[25:35, 5:15] = 100   # Another region
    return img


def degrade_image_deterministic(img_in: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Degrade an image by convolution only (no noise) for deterministic testing.
    """
    kernel_size = kernel.shape[0]
    
    # Pad kernel to image size
    kernel_padded = np.zeros(img_in.shape, dtype=np.float32)
    h, w = kernel.shape
    kernel_padded[:h, :w] = kernel
    
    # Circular shift to center
    kernel_padded = np.roll(kernel_padded, -(kernel_size // 2), axis=0)
    kernel_padded = np.roll(kernel_padded, -(kernel_size // 2), axis=1)
    
    # DFT of kernel
    kernel_freq = np.fft.fft2(kernel_padded)
    
    # DFT of input image
    img_freq = np.fft.fft2(img_in)
    
    # Multiply in frequency domain
    degraded_freq = img_freq * kernel_freq
    
    # Inverse DFT
    degraded_img = np.real(np.fft.ifft2(degraded_freq)).astype(np.float32)
    
    return degraded_img, kernel_padded


class TestCircShift(unittest.TestCase):
    """Tests for circular shift function."""
    
    def test_circ_shift_no_shift(self):
        """Test with zero shift."""
        img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = dip4.circ_shift(img, 0, 0)
        np.testing.assert_array_almost_equal(result, img)
    
    def test_circ_shift_x(self):
        """Test shift in x direction (columns)."""
        img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = dip4.circ_shift(img, 1, 0)
        expected = np.array([[3, 1, 2], [6, 4, 5], [9, 7, 8]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_circ_shift_y(self):
        """Test shift in y direction (rows)."""
        img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = dip4.circ_shift(img, 0, 1)
        expected = np.array([[7, 8, 9], [1, 2, 3], [4, 5, 6]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_circ_shift_both(self):
        """Test shift in both directions."""
        img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = dip4.circ_shift(img, 1, 1)
        expected = np.array([[9, 7, 8], [3, 1, 2], [6, 4, 5]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_circ_shift_negative(self):
        """Test negative shifts."""
        img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        result = dip4.circ_shift(img, -1, -1)
        expected = np.array([[5, 6, 4], [8, 9, 7], [2, 3, 1]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_circ_shift_complex(self):
        """Test circular shift with complex values."""
        img = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
        result = dip4.circ_shift(img, 1, 0)
        expected = np.array([[2+2j, 1+1j], [4+4j, 3+3j]], dtype=np.complex64)
        np.testing.assert_array_almost_equal(result, expected)


class TestDFT(unittest.TestCase):
    """Tests for DFT functions."""
    
    def test_dft_dc_component(self):
        """Test that DC component equals sum of all pixels."""
        img = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = dip4.dft_real2complex(img)
        # DC component should be sum of all pixels
        self.assertAlmostEqual(np.abs(result[0, 0]), 10.0, places=4)
    
    def test_dft_known_values(self):
        """Test DFT with known input/output pairs."""
        # Simple 2x2 case
        img = np.array([[1, 0], [0, 0]], dtype=np.float32)
        result = dip4.dft_real2complex(img)
        # For a single 1 at [0,0], all DFT values should be 1
        expected = np.ones((2, 2), dtype=np.complex64)
        np.testing.assert_array_almost_equal(result, expected, decimal=4)
    
    def test_idft_roundtrip(self):
        """Test that IDFT(DFT(x)) = x."""
        img = create_test_image(32)
        freq = dip4.dft_real2complex(img)
        result = dip4.idft_complex2real(freq)
        np.testing.assert_array_almost_equal(result, img, decimal=4)
    
    def test_idft_complex_to_real(self):
        """Test that IDFT result is real-valued."""
        img = create_test_image(32)
        freq = dip4.dft_real2complex(img)
        result = dip4.idft_complex2real(freq)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))


class TestApplyFilter(unittest.TestCase):
    """Tests for filter application function."""
    
    def test_apply_filter_identity(self):
        """Test applying identity filter (all ones)."""
        img_freq = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
        filter_freq = np.ones((2, 2), dtype=np.complex64)
        result = dip4.apply_filter(img_freq, filter_freq)
        np.testing.assert_array_almost_equal(result, img_freq)
    
    def test_apply_filter_zero(self):
        """Test applying zero filter."""
        img_freq = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
        filter_freq = np.zeros((2, 2), dtype=np.complex64)
        result = dip4.apply_filter(img_freq, filter_freq)
        np.testing.assert_array_almost_equal(result, np.zeros_like(img_freq))
    
    def test_apply_filter_complex_multiplication(self):
        """Test element-wise complex multiplication."""
        img_freq = np.array([[1+2j, 3+4j]], dtype=np.complex64)
        filter_freq = np.array([[2+1j, 1-1j]], dtype=np.complex64)
        result = dip4.apply_filter(img_freq, filter_freq)
        # (1+2j)*(2+1j) = 2+1j+4j+2j^2 = 2+5j-2 = 0+5j
        # (3+4j)*(1-1j) = 3-3j+4j-4j^2 = 3+j+4 = 7+j
        expected = np.array([[0+5j, 7+1j]], dtype=np.complex64)
        np.testing.assert_array_almost_equal(result, expected)


class TestInverseFilter(unittest.TestCase):
    """Tests for inverse filter functions."""
    
    def test_compute_inverse_filter_identity(self):
        """Test inverse filter of identity kernel."""
        # Identity kernel in spatial domain
        kernel = np.zeros((8, 8), dtype=np.float32)
        kernel[0, 0] = 1.0
        
        inv_filter = dip4.compute_inverse_filter(kernel, eps=0.01)
        
        # Inverse of identity should be close to identity in frequency domain
        expected_magnitude = 1.0
        self.assertAlmostEqual(np.abs(inv_filter[0, 0]), expected_magnitude, places=2)
    
    def test_compute_inverse_filter_thresholding(self):
        """Test that small values are handled by threshold."""
        kernel = np.zeros((8, 8), dtype=np.float32)
        kernel[0, 0] = 0.001  # Very small value
        
        inv_filter = dip4.compute_inverse_filter(kernel, eps=0.1)
        
        # Result should not have extremely large values
        max_val = np.max(np.abs(inv_filter))
        self.assertTrue(max_val < 100)  # Should be bounded
    
    def test_inverse_filter_restoration(self):
        """Test inverse filter restoration quality."""
        # Create test image
        img = create_test_image(64)
        
        # Create blur kernel
        kernel_size = 5
        kernel = cv2.getGaussianKernel(kernel_size, kernel_size / 5.0, cv2.CV_32F)
        kernel = (kernel @ kernel.T).astype(np.float32)
        
        # Degrade without noise for clean test
        degraded, kernel_padded = degrade_image_deterministic(img, kernel)
        
        # Restore using inverse filter
        restored = dip4.inverse_filter(degraded, kernel_padded, eps=0.001)
        
        # Check PSNR - should be reasonably high without noise
        psnr = compute_psnr(img, restored)
        self.assertGreater(psnr, 20)  # Should achieve decent restoration


class TestWienerFilter(unittest.TestCase):
    """Tests for Wiener filter functions."""
    
    def test_compute_wiener_filter_high_snr(self):
        """Test Wiener filter with high SNR (approaches inverse filter)."""
        kernel = np.zeros((8, 8), dtype=np.float32)
        kernel[0, 0] = 1.0
        
        wien = dip4.compute_wiener_filter(kernel, snr=1e6)
        
        # With very high SNR, should approach inverse filter
        # For identity kernel, inverse is identity
        self.assertAlmostEqual(np.abs(wien[0, 0]), 1.0, places=2)
    
    def test_compute_wiener_filter_low_snr(self):
        """Test Wiener filter with low SNR (should regularize more)."""
        kernel = np.zeros((8, 8), dtype=np.float32)
        kernel[0, 0] = 0.5
        
        wien_high_snr = dip4.compute_wiener_filter(kernel, snr=1000)
        wien_low_snr = dip4.compute_wiener_filter(kernel, snr=1)
        
        # Low SNR should have smaller gains to suppress noise
        self.assertLess(np.max(np.abs(wien_low_snr)), np.max(np.abs(wien_high_snr)))
    
    def test_wiener_filter_restoration(self):
        """Test Wiener filter restoration quality."""
        # Create test image
        img = create_test_image(64)
        
        # Create blur kernel
        kernel_size = 5
        kernel = cv2.getGaussianKernel(kernel_size, kernel_size / 5.0, cv2.CV_32F)
        kernel = (kernel @ kernel.T).astype(np.float32)
        
        # Degrade without noise for clean test
        degraded, kernel_padded = degrade_image_deterministic(img, kernel)
        
        # Restore using Wiener filter
        restored = dip4.wiener_filter(degraded, kernel_padded, snr=1000)
        
        # Check PSNR - should be reasonably high
        psnr = compute_psnr(img, restored)
        self.assertGreater(psnr, 20)  # Should achieve decent restoration
    
    def test_wiener_better_than_inverse_with_noise(self):
        """Test that Wiener filter handles noise better than inverse filter."""
        np.random.seed(42)  # For reproducibility
        
        # Create test image
        img = create_test_image(64)
        
        # Create blur kernel
        kernel_size = 5
        kernel = cv2.getGaussianKernel(kernel_size, kernel_size / 5.0, cv2.CV_32F)
        kernel = (kernel @ kernel.T).astype(np.float32)
        
        # Degrade with noise
        degraded, kernel_padded = degrade_image_deterministic(img, kernel)
        noise = np.random.randn(*degraded.shape).astype(np.float32) * 5
        degraded_noisy = degraded + noise
        
        # Restore using both filters
        restored_inverse = dip4.inverse_filter(degraded_noisy, kernel_padded, eps=0.01)
        restored_wiener = dip4.wiener_filter(degraded_noisy, kernel_padded, snr=50)
        
        # Compute PSNRs
        psnr_inverse = compute_psnr(img, restored_inverse)
        psnr_wiener = compute_psnr(img, restored_wiener)
        
        # Wiener should generally perform better with noise present
        # (or at least not significantly worse)
        self.assertGreater(psnr_wiener, psnr_inverse - 5)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full restoration pipeline."""
    
    def test_full_pipeline_inverse(self):
        """Test full pipeline: degrade -> restore with inverse filter."""
        np.random.seed(123)
        
        # Create or load test image
        img = create_test_image(64)
        
        # Degrade with high SNR to reduce noise impact
        degraded, kernel = dip4.degrade_image(img, None, None, snr=500)
        
        # Restore
        restored = dip4.inverse_filter(degraded, kernel, eps=0.01)
        
        # Should get reasonable restoration (inverse filter is noise-sensitive)
        psnr = compute_psnr(img, restored)
        self.assertGreater(psnr, 5)  # Basic sanity check (inverse filter is noise-sensitive)
    
    def test_full_pipeline_wiener(self):
        """Test full pipeline: degrade -> restore with Wiener filter."""
        np.random.seed(456)
        
        # Create test image
        img = create_test_image(64)
        
        # Degrade
        degraded, kernel = dip4.degrade_image(img, None, None, snr=100)
        
        # Restore
        restored = dip4.wiener_filter(degraded, kernel, snr=100)
        
        # Should get reasonable restoration
        psnr = compute_psnr(img, restored)
        self.assertGreater(psnr, 15)  # Wiener should do better


if __name__ == '__main__':
    unittest.main()
