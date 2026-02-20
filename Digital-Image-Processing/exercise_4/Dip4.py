"""
DIP4: Image Restoration using Frequency Domain Filtering

This module implements image restoration techniques including:
- Discrete Fourier Transform (DFT) operations
- Inverse filtering
- Wiener filtering

Students should implement the functions marked with TODO.
"""

import numpy as np
import cv2


def degrade_image(img_in: np.ndarray, degraded_img: np.ndarray, kernel: np.ndarray, snr: float) -> tuple:
    """
    Degrade an image by convolution with a Gaussian kernel and adding Gaussian noise.
    This function is given - do not modify!
    
    Args:
        img_in: Input image (grayscale, float32)
        degraded_img: Output degraded image (will be modified)
        kernel: Output degradation kernel (will be modified)
        snr: Signal-to-noise ratio for added noise
        
    Returns:
        Tuple of (degraded_img, kernel)
    """
    kernel_size = 7
    
    # Create Gaussian kernel
    kernel = cv2.getGaussianKernel(kernel_size, kernel_size / 5.0, cv2.CV_32F)
    kernel = kernel @ kernel.T  # Outer product for 2D kernel
    
    # Pad kernel to image size
    kernel_padded = np.zeros(img_in.shape, dtype=np.float32)
    kernel_padded[:kernel_size, :kernel_size] = kernel
    
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
    
    # Add noise
    noise_var = np.mean(degraded_img ** 2) / snr
    noise = np.random.randn(*degraded_img.shape).astype(np.float32) * np.sqrt(noise_var)
    degraded_img = degraded_img + noise
    
    return degraded_img, kernel_padded


def circ_shift(img_in: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    """
    Circular shift of an image.
    
    Args:
        img_in: Input image (can be real or complex)
        shift_x: Shift in x direction (columns)
        shift_y: Shift in y direction (rows)
        
    Returns:
        Shifted image
        
    TODO: Implement this function
    """
    return np.roll(img_in, shift=(shift_y, shift_x), axis=(0,1))


def dft_real2complex(img_in: np.ndarray) -> np.ndarray:
    """
    Perform DFT on a real-valued image and return complex result.
    
    Args:
        img_in: Input image (real-valued, float32)
        
    Returns:
        Complex DFT result
        
    TODO: Implement this function
    """
    # TODO: Implement forward DFT
    return np.fft.fft2(img_in)


def idft_complex2real(img_in: np.ndarray) -> np.ndarray:
    """
    Perform inverse DFT on a complex image and return real result.
    
    Args:
        img_in: Complex input in frequency domain
        
    Returns:
        Real-valued spatial domain result (float32)
        
    TODO: Implement this function
    """
    # TODO: Implement inverse DFT
    img = np.fft.ifft2(img_in)
    return np.real(img)


def apply_filter(img_in: np.ndarray, filter_spectrum: np.ndarray) -> np.ndarray:
    """
    Apply a filter in the frequency domain by element-wise multiplication.
    
    Args:
        img_in: Complex input spectrum
        filter_spectrum: Complex filter spectrum
        
    Returns:
        Filtered complex spectrum
        
    TODO: Implement this function
    """
    # TODO: Implement filter application (complex element-wise multiplication)
    applied = img_in * filter_spectrum
    return applied


def compute_inverse_filter(kernel: np.ndarray, eps: float) -> np.ndarray:
    """
    Compute inverse filter in frequency domain.
    
    Args:
        kernel: Degradation kernel in spatial domain
        eps: Threshold for small values (to avoid division by zero)
        
    Returns:
        Complex inverse filter spectrum
        
    TODO: Implement this function
    """
    # TODO: Implement inverse filter computation
    Kernel_fft = dft_real2complex(kernel)
    Kernel_inv = np.zeros_like(Kernel_fft, dtype=np.complex64)
    mask = np.abs(Kernel_fft) >= eps
    Kernel_inv[mask] = 1.0 / Kernel_fft[mask]
    return Kernel_inv


def inverse_filter(degraded: np.ndarray, kernel: np.ndarray, eps: float) -> np.ndarray:
    """
    Restore an image using inverse filtering.
    
    Args:
        degraded: Degraded input image
        kernel: Degradation kernel
        eps: Threshold for inverse filter computation
        
    Returns:
        Restored image (float32)
        
    TODO: Implement this function
    """
    degraded_comp = dft_real2complex(degraded)
    spectrum = compute_inverse_filter(kernel, eps)
    applied = apply_filter(degraded_comp, spectrum)
    real = idft_complex2real(applied)
    return real


def compute_wiener_filter(kernel: np.ndarray, snr: float) -> np.ndarray:
    """
    Compute Wiener filter in frequency domain.
    
    Args:
        kernel: Degradation kernel in spatial domain
        snr: Signal-to-noise ratio estimate
        
    Returns:
        Complex Wiener filter spectrum
        
    TODO: Implement this function
    """
    # TODO: Implement Wiener filter computation
    kernel_comp = dft_real2complex(kernel)
    kernel_conj = np.conj(kernel_comp)
    kernel_power = np.abs(kernel_comp) ** 2
    wiener = kernel_conj / (kernel_power + 1.0 / snr)
    return wiener


def wiener_filter(degraded: np.ndarray, kernel: np.ndarray, snr: float) -> np.ndarray:
    """
    Restore an image using Wiener filtering.
    
    Args:
        degraded: Degraded input image
        kernel: Degradation kernel
        snr: Signal-to-noise ratio estimate
        
    Returns:
        Restored image (float32)
        
    TODO: Implement this function
    """
    degraded_comp = dft_real2complex(degraded)
    spectrum = compute_wiener_filter(kernel, snr)
    applied = apply_filter(degraded_comp, spectrum)
    real = idft_complex2real(applied)
    return real
