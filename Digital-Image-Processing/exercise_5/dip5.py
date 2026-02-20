"""
Python implementation of DIP5 assignment routines.
"""

import math
from typing import List, Tuple

import numpy as np


def getOddKernelSizeForSigma(sigma: float) -> int:
    ksize = int(math.ceil(5.0 * float(sigma))) | 1
    if ksize < 3:
        ksize = 3
    return ksize


def isLocalMaximum(weight: np.ndarray, x: int, y: int) -> bool:
    rows, cols = weight.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            x_ = min(max(x + dx, 0), cols - 1)
            y_ = min(max(y + dy, 0), rows - 1)
            if weight[y_, x_] > weight[y, x]:
                return False
    return True


# =============================
# Main processing functions
# =============================

def createGaussianKernel1D(sigma: float) -> np.ndarray:
    """
    Generates gaussian filter kernel of given size
    """
    ksize = getOddKernelSizeForSigma(sigma)
    radius = ksize // 2

    kernel = np.zeros((1, ksize), dtype=np.float32)
    s = 0.0

    for i in range(-radius, radius + 1):
        val = math.exp(-(i * i) / (2.0 * sigma * sigma))
        kernel[0, i + radius] = val
        s += val
        
    kernel /= s    
    return kernel


def separableFilter(
    src: np.ndarray, kernelX: np.ndarray, kernelY: np.ndarray
) -> np.ndarray:
    """
    Convolution in spatial domain by separable filters
    #erst horizontal dann vertikal
    """
    src = np.asarray(src, dtype=np.float32)
    rows, cols = src.shape

    kx = kernelX.shape[1] // 2
    ky = kernelY.shape[1] // 2

    temp = np.zeros_like(src, dtype=np.float32)
    dest = np.zeros_like(src, dtype=np.float32)

    #x convolution bzw horzontal pass
    for y in range(rows):
        for x in range(cols):
            s = 0.0
            for i in range(-kx, kx + 1):
                xx = min(max(x + i, 0), cols - 1)
                s += src[y, xx] * kernelX[0, i + kx]
            temp[y, x] = s

    #y convolution bzw. vertical pass
    for y in range(rows):
        for x in range(cols):
            s= 0.0
            for i in range(-ky, ky + 1):
                yy= min(max(y + i, 0), rows - 1)
                s += temp[yy, x] * kernelY[0, i + ky]
            dest[y, x] = s

    return dest

def createFstDevKernel1D(sigma: float) -> np.ndarray:
    """
    Creates kernel representing fst derivative of a Gaussian kernel (1-dimensional)
    """
    # TODO: Build the 1D first-derivative-of-Gaussian kernel for sigma.
    # Normalize so sum(abs(kernel)) == 1 to match unit tests.
    ksize = getOddKernelSizeForSigma(sigma=sigma)
    r = ksize//2
    
    kernel = np.zeros((1, ksize), dtype=np.float32)
    mySum = 0
    for i in range(-r, r+1):
        val = -i * math.exp((-i**2)/(2*sigma**2))
        kernel[0, i+r] = val
        mySum += abs(val)
        
    return kernel / mySum 


def calculateDirectionalGradients(
    img: np.ndarray, sigmaGrad: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the directional gradients through convolution
    """
    # TODO: Use separable convolution with Gaussian and its first derivative to
    # compute gradX and gradY.
    img = np.asarray(img, dtype=np.float32)
    gauss = createGaussianKernel1D(sigma=sigmaGrad)
    deriGauss = createFstDevKernel1D(sigma=sigmaGrad)
    gradX = separableFilter(src=img, kernelX=deriGauss, kernelY=gauss)
    gradY = separableFilter(src=img, kernelX=gauss, kernelY=deriGauss)
    return gradX, gradY


def calculateStructureTensor(
    gradX: np.ndarray,
    gradY: np.ndarray,
    sigmaNeighborhood: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the structure tensors (per pixel)
    """
    # TODO: Compute A00=G*(gradX^2), A01=G*(gradX*gradY), A11=G*(gradY^2)
    # using Gaussian smoothing with sigmaNeighborhood.
    gradX = np.asarray(gradX, dtype=np.float32)
    gradY = np.asarray(gradY, dtype=np.float32)
    
    gauss = createGaussianKernel1D(sigma=sigmaNeighborhood)
    
    A00 = gradX ** 2
    A01 = gradX * gradY
    A11 = gradY ** 2
    
    A00 = separableFilter(src=A00, kernelX=gauss, kernelY=gauss)
    A01 = separableFilter(src=A01, kernelX=gauss, kernelY=gauss)
    A11 = separableFilter(src=A11, kernelX=gauss, kernelY=gauss)
    return A00, A01, A11


def calculateFoerstnerWeightIsotropy(
    A00: np.ndarray, A01: np.ndarray, A11: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the feature point weight and isotropy from the structure tensors.
    """
    # TODO: Compute weight = det(A)/trace(A), isotropy = 4*det(A)/trace(A)^2
    # with safe handling for trace==0.
    A00 = np.asarray(A00, dtype=np.float32)
    A01 = np.asarray(A01, dtype=np.float32)
    A11 = np.asarray(A11, dtype=np.float32)
    
    trace = A00 + A11
    det = A00 * A11 - A01 * A01
    
    weight = det / np.maximum(trace, 1e-8)
    isotropy = 4*det/np.maximum(trace**2, 1e-8)
    return weight, isotropy


def getFoerstnerInterestPoints(
    img: np.ndarray,
    sigmaGrad: float,
    sigmaNeighborhood: float,
    fractionalMinWeight: float = 1.5,
    minIsotropy: float = 0.8,
) -> List[Tuple[int, int]]:
    """
    Finds Foerstner interest points in an image and returns their location.
    """
    # TODO: Compute gradients, structure tensor, weight/isotropy, then select
    # local maxima above the weight/isotropy thresholds. Return (x, y) tuples.
    
    gradX, gradY = calculateDirectionalGradients(img=img, sigmaGrad=sigmaGrad)
    A00, A01, A11 = calculateStructureTensor(gradX=gradX, gradY=gradY, sigmaNeighborhood=sigmaNeighborhood)
    weight, isotropy = calculateFoerstnerWeightIsotropy(A00=A00, A01=A01, A11=A11)
    
    meanWeight = np.mean(weight)
    minWeight = fractionalMinWeight/meanWeight
    
    points = []
    rows, cols = weight.shape
    for y in range(1, rows-1):
        for x in range(1, cols-1):
            if weight[y,x] >= minWeight and isotropy[y,x] >= minIsotropy and isLocalMaximum(weight=weight, x=x, y=y):
                points.append((x,y))
    
    return points
