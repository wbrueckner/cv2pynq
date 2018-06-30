# cv2PYNQ
This Python package accelerates [OpenCV](https://opencv.org/) image filtering functions for the [PYNQ](http://www.pynq.io/) platform.
The library implements a specific set of popular image filters and feature detection algorithms.
The calculation of time-consuming tasks is implemented in the Programmable Logic (PL) of the ZYNQ chip.
cv2PYNQ also includes the Video-Subsystem of the [base](https://github.com/Xilinx/PYNQ) project of PYNQ.
Therefore, the HDMI In and Out interfaces can be used in your application.
The library calculates every filter for gray-channel images with 1080p within 16 ms if the input and output buffers 
are located in the continuous memory of the chip. 

## Get Started
Install by typing: 
```
git clone https://github.com/wbrueckner/cv2pynq.git   
cd cv2pynq/   
pip3.6 install -e .   
``` 
into the terminal on your Pynq-Z1 board.   
The library comes with a [jupyter notebook](https://github.com/wbrueckner/cv2pynq/blob/master/notebooks/cv2PYNQ%20-%20Get%20Started.ipynb) to demonstrate its usage and capabilities.
You find the notebook in the cv2PYNQ folder of your home tree after installation.

Link to YouTube Video:
https://www.youtube.com/watch?v=nRxe-NqvOl8

Currently accelerated functions:
- Sobel: 3x3; 5x5
- Scharr
- Laplacian: ksize = 1; 3; 5
- blur: ksize = 3
- GaussinBlur: ksize = 3
- erode: ksize = 3
- dilate: ksize = 3
- Canny 

## Contribute to cv2PYNQ

Read the instructions in [cv2PYNQ - The project behind the library](https://github.com/wbrueckner/cv2PYNQ-The-project-behind-the-library).
