from setuptools import setup
import os
import shutil
import cv2pynq

if 'BOARD' not in os.environ or os.environ['BOARD'] != 'Pynq-Z1':
    print("Only supported on a Pynq Z1 Board")
    exit(1)

# Notebook copy
WORK_DIR = os.path.dirname(os.path.realpath(__file__))
src_nb = WORK_DIR + '/notebooks'
dst_nb_dir = '/home/xilinx/jupyter_notebooks/cv2PYNQ'
if os.path.exists(dst_nb_dir):
    shutil.rmtree(dst_nb_dir)
shutil.copytree(src_nb, dst_nb_dir)

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