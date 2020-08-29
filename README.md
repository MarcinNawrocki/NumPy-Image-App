# NumPy-Image-App

## General info
The repository contains a simple image processing library written in Python (mostly NumPy package). For loading and saving images [Pillow](https://pillow.readthedocs.io/en/stable/) was used. 
The library was the part of the image processing app but now was adapted to use as an independent educational library. Educational means that the library carries some operations and return list of images (original image, the specified number of inter images and final image) which could be easily displayed using matplotlib (appropriate function was written).  The benefit of this approach is that students can deeply analyze the whole process, step by step. 

## Technologies

[![Generic badge](https://img.shields.io/badge/Python-3.7-green)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/NumPy-1.18.2-blue)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Pillow-7.1.1-blue)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/matplotlib-3.2.1-blue)](https://shields.io/)
 
## Operations
* **conversion to Grayscale using two different ways:**
  - "Machine" grayscale: Y = (R+G+B)/3 ,
  - "Human" grayscale: Y = 0.215*R + 0.7151*G + 0.0721*B.
* **image binaryzation using thresholding:**
  - with manually specified threshold,
  - with threshold calculated by OTSU method.
* **morphological operations:**
  - erode,
  - dilate,
  - opening,
  - closing.
* **filters**:
  - lowpass (4 different kernels),
  - highpass (4 different kernels),
  - median.
* **artificiall noising:**
  - salt pepper,
  - Gaussian.
* **gamma correction**
* every operations, which used kernel f.e blur, edge detectiion [etc.](https://en.wikipedia.org/wiki/Kernel_(image_processing))

## Project structure:
The base file was **lib_api.py**. This is the only file that should be imported to use the library. It contains functions that realized some image processing operations
and returns a list of images as NumPy arrays, which can be easily displayed (show_images function).

Functions from **lib_api.py** file use "low level" functions from other files (modules): 
* **basic_operations.py** --  basic operations, which was used by all other modules.
* **binary_operations.py** -- operations related to binaryzed image (binaryzation and morphological operations).
* **filtering.py** -- filtering and correction operations.
* **noising.py** -- artificiall noising images, with most populars types of noise.

So how all of this works together? Take as an example [opening](https://en.wikipedia.org/wiki/Mathematical_morphology#Opening):
* verification that image is in grayscale, if not conversion (basic_operations module).
* perform binarization using [OTSU](https://en.wikipedia.org/wiki/Otsu%27s_method algorithm) method (binary_operations module).
* erode operation (binary_operations module).
* dilate operation (binary_operations module).
* erode and dilate operation use functions from basic operations module
* generating inter images (basic_operations module).

As a result, the function returns a list of images (original image, the specified number of inter images and final image) which could be easily displayed using show_images function (based on matplitlib.pyplot.imshow function). Example results of using **openly** operations, with default parameters was shown below:

![alt text](https://github.com/MarcinNawrocki/NumPy-Image-App/blob/master/example.png "Example results")

In the image above we can see:
* original, colored image.
* image converted to grayscale.
* binarized image.
* image after erosion operation (first part of openly operation).
* final image (after dilation operation).

## Setup
To run this project, download all **.py** files and import **lib_api.py** to your file. If you have installed all Pyhton extensions you can easily use this library.
