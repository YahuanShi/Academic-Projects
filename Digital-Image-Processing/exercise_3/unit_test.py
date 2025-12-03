from __future__ import annotations

import unittest

import numpy as np

from dip3 import (
    FilterMode,
    circ_shift,
    create_gaussian_kernel_1d,
    create_gaussian_kernel_2d,
    frequency_convolution,
    separable_filter,
    smooth_image,
)



class TestGaussianKernels(unittest.TestCase):
    def test_kernel_1d(self) -> None:
        kernel = create_gaussian_kernel_1d(11)
        self.assertEqual(kernel.shape, (1, 11))
        self.assertTrue(np.isfinite(kernel).all())
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=4)
        self.assertGreaterEqual(kernel[0, 5], kernel.max())

    def test_kernel_2d(self) -> None:
        kernel = create_gaussian_kernel_2d(11)
        self.assertEqual(kernel.shape, (11, 11))
        self.assertTrue(np.isfinite(kernel).all())
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=4)
        self.assertEqual(np.count_nonzero(kernel >= kernel[5, 5]), 1)


class TestCircShift(unittest.TestCase):
    def test_basic_shift(self) -> None:
        matrix = np.zeros((3, 3), dtype=np.float32)
        matrix[0, 0] = 1
        matrix[0, 1] = 2
        matrix[1, 0] = 3
        matrix[1, 1] = 4
        shifted = circ_shift(matrix, -1, -1)
        ref = np.zeros_like(matrix)
        ref[0, 0] = 4
        ref[0, 2] = 3
        ref[2, 0] = 2
        ref[2, 2] = 1
        np.testing.assert_array_equal(shifted, ref)

    def test_inverse_shift(self) -> None:
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(30, 30)).astype(np.float32)
        tmp = circ_shift(matrix, -5, -10)
        tmp = circ_shift(tmp, 10, -10)
        tmp = circ_shift(tmp, -5, 20)
        np.testing.assert_array_equal(tmp, matrix)


class TestConvolutions(unittest.TestCase):
    def setUp(self) -> None:
        self.input = np.ones((9, 9), dtype=np.float32)
        self.input[4, 4] = 255

    def test_frequency_convolution(self) -> None:
        kernel = np.full((3, 3), 1.0 / 9.0, dtype=np.float32)
        output = frequency_convolution(self.input, kernel)
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue(np.all(output >= 0))
        self.assertTrue(np.all(output <= 255))
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
        np.testing.assert_allclose(output[1:-1, 1:-1], ref[1:-1, 1:-1], atol=1e-4)

    def test_separable_filter(self) -> None:
        kernel = np.full((1, 3), 1.0 / 3.0, dtype=np.float32)
        output = separable_filter(self.input, kernel)
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue(np.all(output >= 0))
        self.assertTrue(np.all(output <= 255))
        ref_val = (8 + 255) / 9.0
        self.assertAlmostEqual(float(output[4, 4]), ref_val, places=4)


class TestSmoothImage(unittest.TestCase):
    def test_dispatch(self) -> None:
        img = np.random.default_rng(1).normal(size=(32, 32)).astype(np.float32)
        for mode in FilterMode:
            out = smooth_image(img, 5, mode)
            self.assertEqual(out.shape, img.shape)
            self.assertTrue(np.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
