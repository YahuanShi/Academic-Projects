from __future__ import annotations

from enum import Enum, auto
from typing import Dict

import numpy as np
import math
import cv2


class FilterMode(Enum):
    """Supported convolution backends."""

    FM_SPATIAL_CONVOLUTION = auto()
    FM_FREQUENCY_CONVOLUTION = auto()
    FM_SEPERABLE_FILTER = auto()
    FM_INTEGRAL_IMAGE = auto()


filter_mode_names: Dict[FilterMode, str] = {
    FilterMode.FM_SPATIAL_CONVOLUTION: "FM_SPATIAL_CONVOLUTION",
    FilterMode.FM_FREQUENCY_CONVOLUTION: "FM_FREQUENCY_CONVOLUTION",
    FilterMode.FM_SEPERABLE_FILTER: "FM_SEPERABLE_FILTER",
    FilterMode.FM_INTEGRAL_IMAGE: "FM_INTEGRAL_IMAGE",
}


def create_gaussian_kernel_1d(k_size: int) -> np.ndarray:
    """Generates 1D Gaussian filter kernel of given size."""
    # TO DO !!!
    sigma = k_size / 5
    center = k_size // 2
    x = np.arange(k_size) - center
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    kernel = kernel.T
    
    k = np.zeros((1, kernel.size))
    k[0,:] = kernel
    return k


def create_gaussian_kernel_2d(k_size: int) -> np.ndarray:
    """Generates 2D Gaussian filter kernel of given size."""
    # TO DO !!!
    sigma = k_size / 5
    center = k_size // 2
    x = np.arange(k_size) - center
    y = np.arange(k_size) - center
    kernel = np.zeros((k_size, k_size), dtype=np.float32)
    for i in range(k_size):
        for j in range(k_size):
            x = i - center
            y = j - center
            kernel[i, j] = math.exp(-(x*x + y*y) / (2 * sigma * sigma))
    kernel /= np.sum(kernel)
    
    return kernel


def circ_shift(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Perform a circular shift in (dx, dy) direction."""
    # TO DO !!!
    rows, cols = image.shape
    shifted = np.zeros_like(image)
    dx = dx % rows
    dy = dy % cols
    for i in range(rows):
        for j in range(cols):
            src_i = (i - dx) % rows
            src_j = (j - dy) % cols
            shifted[i, j] = image[src_i, src_j]

    return shifted


def frequency_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Performs convolution by multiplication in frequency domain."""
    # TO DO !!!
    h = image.shape[0] + kernel.shape[0] - 1
    w = image.shape[0] + kernel.shape[1] - 1
    
    dftHeight = cv2.getOptimalDFTSize(h)
    dftWidth = cv2.getOptimalDFTSize(w)
    
    paddedImg = np.zeros((dftHeight, dftWidth))
    paddedImg[:image.shape[0], :image.shape[1]] = image
    paddedKernel = np.zeros((dftHeight, dftWidth))
    paddedKernel[:kernel.shape[0], :kernel.shape[1]] = kernel
    
    forwardImg = cv2.dft(paddedImg, flags=cv2.DFT_COMPLEX_OUTPUT)
    forwardKernel = cv2.dft(paddedKernel, flags=cv2.DFT_COMPLEX_OUTPUT)
    forwardOut = cv2.mulSpectrums(forwardImg, forwardKernel, 0, conjB=False)
    out = cv2.dft(forwardOut, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    out = out[:h, :w]
    cy = kernel.shape[0] // 2
    cx = kernel.shape[1] // 2
    out = out[cy:cy + image.shape[0], cx:cx + image.shape[1]]
    
    return out


def separable_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolution in spatial domain by separable filters."""
    # TO DO !!!
    h, w = image.shape
    horizontalBlur = np.zeros_like(image)
    kernel = np.asarray(kernel).flatten()

    for y in range(h):
        for x in range(w):
            acc = 0
            for i in range(kernel.size):
                sampleX = x + i - kernel.size//2
                if sampleX < 0:
                    sampleX = 0
                elif sampleX >= w:
                    sampleX = w - 1
                acc += image[y, sampleX] * kernel[i]
                    
            horizontalBlur[y, x] = acc
    
    horizontalBlur = horizontalBlur.T
    
    verticalBlur = np.zeros_like(horizontalBlur)
    for y in range(h):
        for x in range(w):
            acc = 0
            for i in range(kernel.size):
                sampleX = x + i - kernel.size//2
                if sampleX < 0:
                    sampleX = 0
                elif sampleX >= w:
                    sampleX = w - 1
                acc += horizontalBlur[y, sampleX] * kernel[i]
            verticalBlur[y, x] = acc
    
    out = verticalBlur.T
    return out


def sat_filter(image: np.ndarray, size: int) -> np.ndarray:
    """Convolution in spatial domain using integral images."""
    # TO DO !!!
    h, w = image.shape
    radius = size // 2
    
    sat = np.zeros((h + 1, w + 1))
    for y in range(h):
        rowSum = 0.0
        for x in range(w):
            rowSum += image[y, x]
            sat[y+1, x+1] = sat[y, x+1] + rowSum
    
    out = np.zeros_like(image)
    
    for y in range(h):
        y0 = max(0, y - radius)
        y1 = min(h - 1, y + radius)
        iy0 = y0
        iy1 = y1 + 1
        
        for x in range(w):
            x0 = max(0, x - radius)
            x1 = min(w - 1, x + radius)
            ix0 = x0
            ix1 = x1 + 1
            total = (sat[iy1, x1] - sat[iy0, ix1] - sat[iy1, ix0] + sat[iy0, ix0])
            count = (y1 - y0 + 1) * (x1 - x0 + 1)
            out[y, x] = total / count
            
    return out


def spatial_convolution(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolution in spatial domain."""
    #https://pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
    #flip kernel
    kernel = np.flipud(np.fliplr(kernel))
    #height = y, width = x
    kernel_y, kernel_x = kernel.shape
    pad_y = kernel_y // 2
    pad_x = kernel_x // 2

    #Pad input image
    padded_src = np.pad(src, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    output = np.zeros_like(src, dtype=float)

    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            #Interest of region
            region_interest = padded_src[y:y+kernel_y, x:x+kernel_x]
            output[y, x] = np.sum(region_interest * kernel)
    return output
    # return np.array(output, copy=True)


def usm(image: np.ndarray, filter_mode: FilterMode, size: int, thresh: float, scale: float) -> np.ndarray:
    """Performs unsharp masking to enhance image structures."""
    # TO DO !!!
    # use smooth_image(...) for smoothing
    smoothed = smooth_image(image, size, filter_mode).astype(np.float32)
    detail = image - smoothed
    mask = np.abs(detail) > thresh
    detail = detail * mask
    sharpened = image + scale * detail

    # 注意：这里只 clip，不转 uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.float32)

    return sharpened


def smooth_image(image: np.ndarray, size: int, filter_mode: FilterMode) -> np.ndarray:
    """Performs smoothing operation choosing the algorithm by filter_mode."""
    if filter_mode is FilterMode.FM_SPATIAL_CONVOLUTION:
        return spatial_convolution(image, create_gaussian_kernel_2d(size))
    if filter_mode is FilterMode.FM_FREQUENCY_CONVOLUTION:
        return frequency_convolution(image, create_gaussian_kernel_2d(size))
    if filter_mode is FilterMode.FM_SEPERABLE_FILTER:
        return separable_filter(image, create_gaussian_kernel_1d(size))
    if filter_mode is FilterMode.FM_INTEGRAL_IMAGE:
        return sat_filter(image, size)
    raise ValueError("Unhandled filter type!")


__all__ = [
    "FilterMode",
    "filter_mode_names",
    "create_gaussian_kernel_1d",
    "create_gaussian_kernel_2d",
    "circ_shift",
    "frequency_convolution",
    "separable_filter",
    "sat_filter",
    "spatial_convolution",
    "usm",
    "smooth_image",
]
