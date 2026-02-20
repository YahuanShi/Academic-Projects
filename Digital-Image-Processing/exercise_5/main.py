"""
Entry point for DIP5 python assignment.
"""

import os
import sys
from typing import List, Tuple

import cv2
import numpy as np

from dip5 import getFoerstnerInterestPoints


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage:\n\tdip5 path_to_original [sigmaGrad [sigmaNeighborhood]]")
        return -1

    img = cv2.imread(argv[1])
    if img is None:
        print("ERROR: original image not specified")
        return -1

    # Downscale very large images to keep runtime/memory reasonable.
    # (Pure-Python separable filtering can otherwise get OOM-killed.)
    max_dim = 1200
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if len(argv) < 3:
        sigma_grad = 1.5
    else:
        sigma_grad = float(argv[2])

    if len(argv) < 4:
        sigma_neigh = 2.0
    else:
        sigma_neigh = float(argv[3])

    # Avoid writing float images (OpenCV will warn and cast implicitly).
    cv2.imwrite("originalGrey.png", np.clip(img_gray, 0.0, 255.0).astype(np.uint8))

    # In headless sessions or some Wayland setups, imshow can fail/crash.
    show_gui = bool(os.environ.get("DISPLAY"))
    if show_gui:
        cv2.imshow("original", img)

    points = getFoerstnerInterestPoints(img_gray, sigma_grad, sigma_neigh)
    print(f"Number of detected interest points:\t{len(points)}")

    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR).astype(np.uint8)
    for x, y in points:
        cv2.circle(img_rgb, (x, y), 10, (0, 0, 255))

    cv2.imwrite("keypoints.png", img_rgb)
    if show_gui:
        cv2.imshow("keypoints", img_rgb)
        cv2.waitKey(0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
