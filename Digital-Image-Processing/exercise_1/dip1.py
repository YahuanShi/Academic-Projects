from __future__ import annotations
import cv2
import numpy as np

def do_something_that_my_tutor_is_gonna_like(img: np.ndarray) -> np.ndarray:
    """
    Requirements:
      - Accepts np.ndarray (BGR if color).
      - Returns processed image (BGR or grayscale OK).
      - Must produce a result sufficiently different from input
    """

    if img is None or not hasattr(img, "ndim"):
        raise ValueError("Input image is invalid")
    
    # Graustufen 
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Weichzeichnen
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    #Kanten erkennen
    edges = cv2.Canny(img,100,200)
    #erosion -> erodes away the boundaries of foreground object
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    #dilation -> increases the white region in the image or size of foreground object increases
    dilation = cv2.dilate(img,kernel,iterations = 1) 
    # fradient -> difference between dilation and erosion of an image
    gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT, kernel)
    #invertierte Farben
    inverted = cv2.bitwise_not(edges)

    return inverted  # <-- temporary; replace with your result

def run(filename: str) -> None:
    """Load image, call processing, show and save."""
    win1 = "Original image"
    win2 = "Result"

    print("loading image")
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    print("done")

    if img is None:
        raise FileNotFoundError(f"ERROR: Cannot read file {filename}")

    cv2.namedWindow(win1)
    cv2.imshow(win1, img)

    out = do_something_that_my_tutor_is_gonna_like(img)

    cv2.namedWindow(win2)
    cv2.imshow(win2, out)

    cv2.imwrite("result.png", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

run("mandrill.png")