import cv2
from cv2 import *
import numpy as np
from .cv2pynq import *

__version__ = 0.1

c = cv2pynq()

#def Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) :
def Sobel(src, ddepth, dx, dy, dst=0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT):
    """Executes the Sobel operator on hardware if input parameters fit to hardware constraints.
    Otherwise the OpenCV Sobel function is called."""
    if (dst == 0) and (ksize == 3) and (scale == 1) and (delta == 0) and (borderType == cv2.BORDER_DEFAULT) :
        if (src.dtype == np.uint8) and (src.ndim == 2) :
            if (src.shape[0] <= cv2pynq.MAX_HEIGHT) and (src.shape[0] > 0) and (src.shape[1] <= cv2pynq.MAX_WIDTH) and (src.shape[1] > 0) :
                if ((ddepth == -1) and (dx == 1) and (dy == 0)) or ((ddepth == -1) and (dx == 0) and (dy == 1)) :
                    return c.Sobel(src, ddepth, dx, dy)
    
    return cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta, borderType)