from __future__ import annotations

from enum import Enum, auto
from typing import Dict

import numpy as np
import cv2


class NoiseType(Enum):
    """Enumerates supported synthetic noise variants."""

    NOISE_TYPE_1 = auto()
    NOISE_TYPE_2 = auto()


noise_type_names: Dict[NoiseType, str] = {
    NoiseType.NOISE_TYPE_1: "NOISE_TYPE_1",
    NoiseType.NOISE_TYPE_2: "NOISE_TYPE_2",
}


class NoiseReductionAlgorithm(Enum):
    """Enumerates available denoising algorithms."""

    NR_MOVING_AVERAGE_FILTER = auto()
    NR_MEDIAN_FILTER = auto()
    NR_BILATERAL_FILTER = auto()


noise_reduction_algorithm_names: Dict[NoiseReductionAlgorithm, str] = {
    NoiseReductionAlgorithm.NR_MOVING_AVERAGE_FILTER: "NR_MOVING_AVERAGE_FILTER",
    NoiseReductionAlgorithm.NR_MEDIAN_FILTER: "NR_MEDIAN_FILTER",
    NoiseReductionAlgorithm.NR_BILATERAL_FILTER: "NR_BILATERAL_FILTER",
}


def spatial_convolution(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolution in spatial domain.

    Performs spatial convolution of image and filter kernel.

    src (np.ndarray): single channel (grayscale) image
    kernel (np.ndarray): convolut kernel
    Return
    np.ndarray: convolved image
    """
    # TO DO
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
    #return output
    return np.array(output, copy=True)


def average_filter(src: np.ndarray, k_size: int) -> np.ndarray:
    """
    Moving average filter (aka box filter).

    You might want to use dip2.spatial_convolution(...) within this function.
    """
    # TO DO !!
    kernel = np.ones((k_size, k_size), dtype=float) / (k_size * k_size)
    #return np.array(src, copy=True)
    return spatial_convolution(src, kernel)


def median_filter(src: np.ndarray, k_size: int) -> np.ndarray:
    """
    Median filter.
    """
    # TO DO !!
    pad = k_size // 2
    padded = np.pad(src, pad, mode="edge")
    output = np.zeros_like(src)
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            region_interest = padded[y:y+k_size, x:x+k_size]
            output[y, x] = np.median(region_interest)

    return np.array(output, copy=True)


def bilateral_filter(src: np.ndarray, k_size: int, sigma_spatial: float, sigma_radiometric: float) -> np.ndarray:
    """
    Bilateral filter.
    """
    # TO DO !!
    radius = k_size // 2
    src_y, src_x = src.shape
    output = np.zeros_like(src, dtype=float)

    #gaussian kernel
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    spatial_gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma_spatial**2))

    #pad image (for borders)
    padded = np.pad(src, radius, mode='edge')

    for y in range(src_y):
        for x in range(src_x):
            #extract local region
            region = padded[y:y+k_size, x:x+k_size]

            #compute raiometric gaussian weights based on intensity difference
            intensity_diff = region - src[y, x]
            radiometric_gaussian = np.exp(-(intensity_diff**2) / (2 * sigma_radiometric**2))

            #combine spatial and radiometric weights
            weights = spatial_gaussian * radiometric_gaussian

            #normalize weights
            weights /= np.sum(weights)

            #compute output pixel as weighted sum
            output[y, x] = np.sum(weights * region)


    return np.array(output, copy=True)


def nlm_filter(src: np.ndarray, search_size: int, sigma: float) -> np.ndarray:
    """
    Non-local means filter (optional task!).

    """
    return np.array(spatial_convolution(src, kernel), copy=True)


def choose_best_algorithm(noise_type: NoiseType) -> NoiseReductionAlgorithm:
    """
    Chooses the right algorithm for the given noise type.
    """
    # TO DO !!
    if noise_type is NoiseType.NOISE_TYPE_1:
        return NoiseReductionAlgorithm.NR_MEDIAN_FILTER
    elif noise_type is NoiseType.NOISE_TYPE_2:
        return NoiseReductionAlgorithm.NR_BILATERAL_FILTER
    else:
        
        raise NotImplementedError("Student implementation missing")


def denoise_image(
    src: np.ndarray,
    noise_type: NoiseType,
    noise_reduction_algorithm: NoiseReductionAlgorithm,
) -> np.ndarray:
    """
    Denoising, with parameters specifically tweaked to the supported noise types.
    """
    # TO DO !!

    if noise_reduction_algorithm is NoiseReductionAlgorithm.NR_MOVING_AVERAGE_FILTER:
        if noise_type is NoiseType.NOISE_TYPE_1:
            return average_filter(src, 3)
        if noise_type is NoiseType.NOISE_TYPE_2:
            return average_filter(src, 3)
        raise ValueError("Unhandled noise type!")

    if noise_reduction_algorithm is NoiseReductionAlgorithm.NR_MEDIAN_FILTER:
        if noise_type is NoiseType.NOISE_TYPE_1:
            return median_filter(src, 3)
        if noise_type is NoiseType.NOISE_TYPE_2:
            return median_filter(src, 3)
        raise ValueError("Unhandled noise type!")

    if noise_reduction_algorithm is NoiseReductionAlgorithm.NR_BILATERAL_FILTER:
        if noise_type is NoiseType.NOISE_TYPE_1:
            return bilateral_filter(src, 5, 2.0, 30.0)
        if noise_type is NoiseType.NOISE_TYPE_2:
            return bilateral_filter(src, 5, 2.0, 30.0)
        raise ValueError("Unhandled noise type!")

    raise ValueError("Unhandled filter type!")


__all__ = [
    "NoiseType",
    "NoiseReductionAlgorithm",
    "noise_type_names",
    "noise_reduction_algorithm_names",
    "spatial_convolution",
    "average_filter",
    "median_filter",
    "bilateral_filter",
    "nlm_filter",
    "choose_best_algorithm",
    "denoise_image",
]
