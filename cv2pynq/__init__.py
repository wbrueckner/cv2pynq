import cv2
from cv2 import *
import numpy as np
from .cv2pynq import *
from pynq.lib.video import *

__version__ = 0.3

c = cv2pynq()
video = c.ol.video #cv2pynq uses the pynq video library and the Pynq-Z1 video subsystem

def Sobel(src, ddepth, dx, dy, dst=None, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT):
    """dst = cv.Sobel(	src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]	)
    Executes the Sobel operator on hardware if input parameters fit to hardware constraints.
    Otherwise the OpenCV Sobel function is called."""
    if (ksize == 3 or ksize == 5) and (scale == 1) and (delta == 0) and (borderType == cv2.BORDER_DEFAULT) :
        if (src.dtype == np.uint8) and (src.ndim == 2) :
            if (src.shape[0] <= cv2pynq.MAX_HEIGHT) and (src.shape[0] > 0) and (src.shape[1] <= cv2pynq.MAX_WIDTH) and (src.shape[1] > 0) :
                if ((ddepth == -1) and (dx == 1) and (dy == 0)) or ((ddepth == -1) and (dx == 0) and (dy == 1)) :
                    return c.Sobel(src, ddepth, dx, dy, dst, ksize)   
    return cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta, borderType)

def Scharr(src, ddepth, dx, dy, dst=None, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT):
    """dst = cv.Scharr(	src, ddepth, dx, dy[, dst[, scale[, delta[, borderType]]]]	)
    Executes the Scharr operator on hardware if input parameters fit to hardware constraints.
    Otherwise the OpenCV Scharr function is called."""
    if (scale == 1) and (delta == 0) and (borderType == cv2.BORDER_DEFAULT) :
        if (src.dtype == np.uint8) and (src.ndim == 2) :
            if (src.shape[0] <= cv2pynq.MAX_HEIGHT) and (src.shape[0] > 0) and (src.shape[1] <= cv2pynq.MAX_WIDTH) and (src.shape[1] > 0) :
                if ((ddepth == -1) and (dx == 1) and (dy == 0)) or ((ddepth == -1) and (dx == 0) and (dy == 1)) :
                    return c.Scharr(src, ddepth, dx, dy, dst)  
    return cv2.Scharr(src, ddepth, dx, dy, dst, scale, delta, borderType)

def Laplacian(src, ddepth, dst=None, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT):
    """dst = cv.Laplacian( src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]	)
    Executes the Laplacian operator on hardware if input parameters fit to hardware constraints.
    Otherwise the OpenCV Laplacian function is called."""
    if (ksize == 1 or ksize ==3 or ksize == 5) and (scale == 1) and (delta == 0) and (borderType == cv2.BORDER_DEFAULT) :
        if (src.dtype == np.uint8) and (src.ndim == 2) :
            if (src.shape[0] <= cv2pynq.MAX_HEIGHT) and (src.shape[0] > 0) and (src.shape[1] <= cv2pynq.MAX_WIDTH) and (src.shape[1] > 0) :
                if (ddepth == -1) :
                    return c.Laplacian(src, ddepth, dst, ksize)   
    return cv2.Laplacian(src, ddepth, dst, ksize, scale, delta, borderType)

def blur(src, ksize, dst=None, anchor=(-1,-1), borderType=cv2.BORDER_DEFAULT):
    """dst = cv.blur( src, ksize[, dst[, anchor[, borderType]]])
    Smooths an image using the kernel on hardware if input parameters fit to hardware constraints.
    Otherwise the OpenCV blur function is called."""
    if (ksize == (3,3)) and (anchor == (-1,-1)) and (borderType == cv2.BORDER_DEFAULT) :
        if (src.dtype == np.uint8) and (src.ndim == 2) :
            return c.blur(src, ksize, dst)
    return cv2.blur(src,ksize,dst,anchor,borderType)

def GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=0, borderType=cv2.BORDER_DEFAULT):
    """dst	= cv.GaussianBlur( src, ksize, sigmaX[, dst[, sigmaY[, borderType]]])
    Smooths an image using a Gaussian kernel on hardware if input parameters fit to hardware constraints.
    Otherwise the OpenCV GaussianBlur function is called."""
    if (ksize == (3,3)) and (borderType == cv2.BORDER_DEFAULT) :
        if (src.dtype == np.uint8) and (src.ndim == 2) :
            return c.GaussianBlur(src, ksize, sigmaX, sigmaY, dst)
    return cv2.GaussianBlur(src,ksize,dst,anchor,borderType)

def erode(src, kernel, dst=None, anchor=(-1,-1), iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=None):
    """dst	= cv.erode( src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]])
    Erodes an image by using a specific structuring element on hardware if input parameters fit to hardware constraints. 
    Otherwise the OpenCV erode function is called."""
    if (kernel.shape[0] == 3) and (kernel.shape[1] == 3) and (anchor == (-1,-1)) and (borderType == cv2.BORDER_CONSTANT) and (borderValue is None):
        if (src.dtype == np.uint8) and (src.ndim == 2) :
            if (src.shape[0] <= cv2pynq.MAX_HEIGHT) and (src.shape[0] > 0) and (src.shape[1] <= cv2pynq.MAX_WIDTH) and (src.shape[1] > 0) :
                if np.array_equal(kernel,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))):
                    return c.erode(src, kernel, dst, iterations, 0) # mode = 0
                elif np.array_equal(kernel,cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))):
                    return c.erode(src, kernel, dst, iterations, 1) # mode = 1
    return cv2.erode(src,kernel, dst, anchor, iterations, borderType, borderValue)

def dilate(src, kernel, dst=None, anchor=(-1,-1), iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=None):
    """dst	= cv.dilate( src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]])
    Dilates an image by using a specific structuring element on hardware if input parameters fit to hardware constraints. 
    Otherwise the OpenCV dilate function is called."""
    if (kernel.shape[0] == 3) and (kernel.shape[1] == 3) and (anchor == (-1,-1)) and (borderType == cv2.BORDER_CONSTANT) and (borderValue is None):
        if (src.dtype == np.uint8) and (src.ndim == 2) :
            if (src.shape[0] <= cv2pynq.MAX_HEIGHT) and (src.shape[0] > 0) and (src.shape[1] <= cv2pynq.MAX_WIDTH) and (src.shape[1] > 0) :
                if np.array_equal(kernel,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))):
                    return c.dilate(src, kernel, dst, iterations, 0) # mode = 0
                elif np.array_equal(kernel,cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))):
                    return c.dilate(src, kernel, dst, iterations, 1) # mode = 1
    return cv2.dilate(src,kernel, dst, anchor, iterations, borderType, borderValue)

def Canny(image, threshold1, threshold2, edges=None, apertureSize=3, L2gradient=False):
    """	edges = cv.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]])
    Finds edges in an image by using the Canny algorithm on hardware if the input parameter fit to the hardware constraints.
    Caution: results of the hardware implementation may differ to OpenCV Canny output.
    Otherwise the OpenCV Canny function is called."""
    if (apertureSize == 3) and (L2gradient == False) :
        if (image.dtype == np.uint8) and (image.ndim == 2) :
            if (image.shape[0] <= cv2pynq.MAX_HEIGHT) and (image.shape[0] > 0) and (image.shape[1] <= cv2pynq.MAX_WIDTH) and (image.shape[1] > 0) :
                return c.Canny(image, threshold1, threshold2, edges)
    return cv2.Canny(image, threshold1, threshold2, edges, apertureSize, L2gradient)

'''def cornerHarris(src, blockSize, ksize, k, dst=None, borderType=cv2.BORDER_DEFAULT):
    """dst = cv.cornerHarris( src, blockSize, ksize, k[, dst[, borderType]])
    Executes the Harris corner detector operation on hardware if input parameters fit to hardware constraints.
    Otherwise the OpenCV cornerHarris function is called."""
    if (ksize == 3) and (blockSize == 2) and (borderType == cv2.BORDER_DEFAULT) :
        if (src.dtype == np.uint8) and (src.ndim == 2) :
            if (src.shape[0] <= cv2pynq.MAX_HEIGHT) and (src.shape[0] > 0) and (src.shape[1] <= cv2pynq.MAX_WIDTH) and (src.shape[1] > 0) :
                    return c.cornerHarris(src, k, dst)   
    return cv2.cornerHarris(src, blockSize, ksize, k, dst, borderType)
'''


def close():
    '''this function should be called after using the cv2pynq library.
    It cleans up the internal state and frees the used CMA-buffers.
    '''
    c.close()