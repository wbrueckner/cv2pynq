from setuptools import setup
import os
import cv2pynq

if 'BOARD' not in os.environ or os.environ['BOARD'] != 'Pynq-Z1':
    print("Only supported on a Pynq Z1 Board")
    exit(1)


setup(name='cv2pynq',
      version=cv2pynq.__version__,
      description='Accelerates OpenCV image filter functions on Zynq',
      keywords='pynq opencv image filter zynq',
      url='http://github.com/wbrueckner/cv2pynq',
      author='Wolfgang Brueckner',
      author_email='wolfgang.brueckner@fau.de',
      license='MIT',
      packages=['cv2pynq'],
      include_package_data = True,
      package_data = {
      '' : ['*.bit','*.tcl','*.py','*.so'],
      },
      install_requires=[
          'pynq','numpy','cffi'
      ],
      dependency_links=['http://github.com/xilinx/PYNQ'],
      zip_safe=False)

#/opt/python3.6/lib/python3.6/site-packages