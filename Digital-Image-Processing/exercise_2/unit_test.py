from __future__ import annotations

import unittest

import numpy as np

from dip2 import (
    NoiseReductionAlgorithm,
    NoiseType,
    average_filter,
    bilateral_filter,
    choose_best_algorithm,
    denoise_image,
    median_filter,
    noise_reduction_algorithm_names,
    noise_type_names,
    spatial_convolution,
)


def _generate_noise_type_1(img: np.ndarray) -> np.ndarray:
    noise_level = 0.15
    tmp1 = np.random.uniform(0.0, 1.0, size=img.shape).astype(np.float32)
    tmp2 = (tmp1 >= noise_level).astype(np.float32) * img
    tmp3 = (tmp1 >= 1.0 - noise_level).astype(np.float32) * 255.0
    noisy = tmp2 + tmp3
    return np.clip(noisy, 0.0, 255.0, out=noisy)


def _generate_noise_type_2(img: np.ndarray) -> np.ndarray:
    noise_level = 50.0
    tmp1 = np.random.normal(0.0, noise_level, size=img.shape).astype(np.float32)
    noisy = img + tmp1
    return np.clip(noisy, 0.0, 255.0, out=noisy)


class TestSpatialConvolution(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)

    def test_average_kernel(self) -> None:
        src = np.ones((9, 9), dtype=np.float32)
        src[4, 4] = 255.0
        kernel = np.full((3, 3), 1.0 / 9.0, dtype=np.float32)

        out = spatial_convolution(src, kernel)

        self.assertEqual(out.shape, src.shape)
        self.assertTrue(np.all(out >= 0))
        self.assertTrue(np.all(out <= 255))

        ref = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, (8 + 255) / 9.0, (8 + 255) / 9.0, (8 + 255) / 9.0, 1, 1, 0],
                [0, 1, 1, (8 + 255) / 9.0, (8 + 255) / 9.0, (8 + 255) / 9.0, 1, 1, 0],
                [0, 1, 1, (8 + 255) / 9.0, (8 + 255) / 9.0, (8 + 255) / 9.0, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(out[1:-1, 1:-1], ref[1:-1, 1:-1], atol=1e-4)

    def test_kernel_flip(self) -> None:
        src = np.zeros((9, 9), dtype=np.float32)
        src[4, 4] = 255.0
        kernel = np.zeros((3, 3), dtype=np.float32)
        kernel[0, 0] = -1.0

        out = spatial_convolution(src, kernel)

        self.assertNotAlmostEqual(out[5, 5], -255.0, places=4)
        self.assertNotAlmostEqual(out[4, 4], -255.0, places=4)
        self.assertNotAlmostEqual(out[2, 2], -255.0, places=4)


class TestFilters(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(1)

    def test_average_filter(self) -> None:
        src = np.ones((9, 9), dtype=np.float32)
        src[4, 4] = 255.0
        out = average_filter(src, 3)
        self.assertEqual(out.shape, src.shape)
        self.assertTrue(np.all(out >= 0))
        self.assertTrue(np.all(out <= 255))
        self.assertGreater(out[4, 4], out[0, 0])

    def test_median_filter(self) -> None:
        src = np.zeros((5, 5), dtype=np.float32)
        src[2, 2] = 255.0
        out = median_filter(src, 3)
        self.assertEqual(out.shape, src.shape)
        self.assertEqual(out[2, 2], 0.0)

    def test_bilateral_filter(self) -> None:
        gradient = np.tile(np.linspace(0, 255, 16, dtype=np.float32), (16, 1))
        noisy = gradient + np.random.normal(0, 10, size=gradient.shape).astype(np.float32)
        noisy = np.clip(noisy, 0.0, 255.0)
        out = bilateral_filter(noisy, 5, sigma_spatial=2.0, sigma_radiometric=30.0)
        self.assertEqual(out.shape, gradient.shape)
        self.assertLess(np.mean((out - gradient) ** 2), np.mean((noisy - gradient) ** 2))


class TestPipeline(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(2)
        self.original = np.full((32, 32), 128.0, dtype=np.float32)

    def test_algorithm_selection(self) -> None:
        self.assertEqual(
            choose_best_algorithm(NoiseType.NOISE_TYPE_1),
            NoiseReductionAlgorithm.NR_MEDIAN_FILTER,
        )
        self.assertEqual(
            choose_best_algorithm(NoiseType.NOISE_TYPE_2),
            NoiseReductionAlgorithm.NR_BILATERAL_FILTER,
        )

    def test_denoise_image_type_1(self) -> None:
        noisy = _generate_noise_type_1(self.original)
        denoised = denoise_image(noisy, NoiseType.NOISE_TYPE_1, NoiseReductionAlgorithm.NR_MEDIAN_FILTER)
        self.assertLess(
            np.mean((denoised - self.original) ** 2),
            np.mean((noisy - self.original) ** 2),
        )

    def test_denoise_image_type_2(self) -> None:
        noisy = _generate_noise_type_2(self.original)
        denoised = denoise_image(noisy, NoiseType.NOISE_TYPE_2, NoiseReductionAlgorithm.NR_BILATERAL_FILTER)
        self.assertLess(
            np.mean((denoised - self.original) ** 2),
            np.mean((noisy - self.original) ** 2),
        )

    def test_metadata_exports(self) -> None:
        self.assertIn(NoiseType.NOISE_TYPE_1, noise_type_names)
        self.assertIn(NoiseReductionAlgorithm.NR_MEDIAN_FILTER, noise_reduction_algorithm_names)


if __name__ == "__main__":
    unittest.main()
