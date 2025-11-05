import sys
import cv2
import numpy as np
from dip1 import do_something_that_my_tutor_is_gonna_like

def histogram_intersection_similarity(img1: np.ndarray, img2: np.ndarray, bins: int = 100) -> float:
    """
    Average histogram intersection across planes.
    """

    # make channels match like C++ test does
    if img1.ndim == 3 and img1.shape[2] == 3 and (img2.ndim == 2 or (img2.ndim == 3 and img2.shape[2] == 1)):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    def split_planes(im):
        if im.ndim == 2:
            return [im]
        return list(cv2.split(im))

    planes1 = split_planes(img1)
    planes2 = split_planes(img2)

    if len(planes1) != len(planes2):
        # fallback: compare in grayscale
        planes1 = [cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)] if img1.ndim == 3 else [img1]
        planes2 = [cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)] if img2.ndim == 3 else [img2]

    hist_range = (0, 256)
    sim_sum = 0.0
    for p1, p2 in zip(planes1, planes2):
        h1 = cv2.calcHist([p1], [0], None, [bins], hist_range).astype(np.float32)
        h2 = cv2.calcHist([p2], [0], None, [bins], hist_range).astype(np.float32)
        h1 /= (h1.sum() + 1e-8)
        h2 /= (h2.sum() + 1e-8)
        sim_sum += float(cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT))

    return sim_sum / len(planes1)

def main():
    if len(sys.argv) != 2:
        print("Usage: python unit_test.py <path_to_image>")
        return -1

    fname = sys.argv[1]
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    if img is None:
        print(f"ERROR: Cannot read file {fname}")
        return -2

    out = do_something_that_my_tutor_is_gonna_like(img)
    sim = histogram_intersection_similarity(img, out, bins=100)

    if sim >= 0.8:
        print(f"Warning: The input and output image seem to be quite similar (similarity = {sim:.3f}). "
              "Are you sure you are going to pass?")
        print("Test failed!")
        return -3
    else:
        print("Test successful")
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
