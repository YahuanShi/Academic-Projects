from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from dip2 import (
    NoiseReductionAlgorithm,
    NoiseType,
    denoise_image,
    noise_reduction_algorithm_names,
    noise_type_names,
)


def try_load_image(filename: Path) -> np.ndarray:
    """Return image as float32 grayscale array or raise IOError."""
    img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"file {filename} not found")
    return img.astype(np.float32, copy=False)


def generate_noisy_image(img: np.ndarray, noise_type: NoiseType) -> np.ndarray:
    """Apply synthetic noise identical to the C++ reference implementation."""
    if noise_type is NoiseType.NOISE_TYPE_1:
        noise_level = 0.15
        tmp1 = np.random.uniform(0.0, 1.0, size=img.shape).astype(np.float32)
        tmp2 = (tmp1 >= noise_level).astype(np.float32) * img
        tmp3 = (tmp1 >= 1.0 - noise_level).astype(np.float32) * 255.0
        noisy = tmp2 + tmp3
        return np.clip(noisy, 0.0, 255.0, out=noisy)

    if noise_type is NoiseType.NOISE_TYPE_2:
        noise_level = 50.0
        tmp1 = np.random.normal(0.0, noise_level, size=img.shape).astype(np.float32)
        noisy = img + tmp1
        noisy = np.clip(noisy, 0.0, 255.0, out=noisy)
        return noisy

    raise ValueError(f"Unhandled noise type {noise_type}")


def compute_psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute PSNR between two float32 images."""
    mse = np.mean((reference.astype(np.float32) - estimate.astype(np.float32)) ** 2)
    if mse <= 0.0:
        return float("inf")
    return 10.0 * np.log10((255.0 * 255.0) / mse)


def denoise_all(original: np.ndarray, noisy_images: Iterable[np.ndarray]) -> None:
    """Run all denoising combinations and write images/metrics to disk."""
    for noise_idx, noise_type in enumerate(NoiseType):
        noisy_img = noisy_images[noise_idx]
        cv2.imwrite(f"{noise_type_names[noise_type]}.jpg", noisy_img)
        for algo in NoiseReductionAlgorithm:
            denoised = denoise_image(noisy_img, noise_type, algo)
            filename = f"restored__{noise_type_names[noise_type]}__{noise_reduction_algorithm_names[algo]}.jpg"
            cv2.imwrite(filename, denoised)
            psnr = compute_psnr(original, denoised)
            print(f"PSNR for {noise_type_names[noise_type]} with {noise_reduction_algorithm_names[algo]}: {psnr:.2f} dB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dip2 Python assignment runner")
    parser.add_argument("image_path", type=Path, help="Path to source image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original = try_load_image(args.image_path)
    noisy_images = [generate_noisy_image(original, noise_type) for noise_type in NoiseType]
    denoise_all(original, noisy_images)


if __name__ == "__main__":
    main()
