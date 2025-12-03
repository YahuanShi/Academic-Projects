from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from dip3 import (
    FilterMode,
    filter_mode_names,
    smooth_image,
    usm,
)


def try_load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"File {path} not found")
    return image


def degrade_image(image: np.ndarray, k_size: int) -> np.ndarray:
    sigma = max(k_size / 5.0, 1e-6)
    size = (k_size // 2 * 2 + 1, k_size // 2 * 2 + 1)
    return cv2.GaussianBlur(image, size, sigma, sigma)


def process_color_image(src: np.ndarray, filter_mode: FilterMode, size: int, thresh: float, scale: float) -> np.ndarray:
    img = src.astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    print("before usm:", h.dtype, s.dtype, v.dtype)
    print("before usm shapes:", h.shape, s.shape, v.shape)
    v = usm(v, filter_mode, size, thresh, scale)
    hsv = cv2.merge((h, s, v))
    print("after usm:", h.dtype, s.dtype, v.dtype)
    print("after usm shapes:", h.shape, s.shape, v.shape)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return np.clip(bgr, 0, 255).astype(np.uint8)


def benchmark_method(image: np.ndarray, filter_mode: FilterMode, size: int) -> float:
    smooth_image(image, size, filter_mode)
    start = cv2.getTickCount()
    smooth_image(image, size, filter_mode)
    end = cv2.getTickCount()
    return float((end - start) / cv2.getTickFrequency())


def run_benchmarks(output_dir: Path, image_sizes: Iterable[int], kernel_sizes: Iterable[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for mode in FilterMode:
        filename = output_dir / f"benchmark_{filter_mode_names[mode]}.csv"
        with filename.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                f"Execution time in seconds for {filter_mode_names[mode]}",
                "Image sizes in rows",
                "Kernel sizes in columns",
            ])
            header = [""] + list(kernel_sizes)
            writer.writerow(header)
            for img_size in image_sizes:
                image = np.zeros((img_size, img_size), dtype=np.float32)
                row = [img_size]
                for kernel_size in kernel_sizes:
                    if kernel_size > img_size:
                        row.append("")
                        continue
                    row.append(benchmark_method(image, mode, kernel_size))
                writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dip3 Python assignment runner")
    parser.add_argument("image_path", type=Path, help="Path to source color image")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark suite as well")
    parser.add_argument("--output", type=Path, default=Path("."), help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = try_load_image(args.image_path)
    degraded = degrade_image(image, 5)
    cv2.imwrite(str(args.output / "degraded.png"), degraded)

    for mode in FilterMode:
        restored = process_color_image(degraded, mode, 5, 1.0, 5.0)
        outfile = args.output / f"{filter_mode_names[mode]}.png"
        cv2.imwrite(str(outfile), restored)

    if args.benchmark:
        benchmark_imgs = [8, 16, 32, 64, 128, 256, 512, 1024]
        benchmark_kernels = [3, 5, 7, 9, 11, 21, 31, 41, 51, 71, 101]
        run_benchmarks(args.output, benchmark_imgs, benchmark_kernels)


if __name__ == "__main__":
    main()
