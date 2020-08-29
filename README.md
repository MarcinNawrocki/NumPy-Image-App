# NumPy-Image-App

## General info
The repository contains simple image processing library written in Python (mostly NumPy package). For loading and saving images [Pillow](https://pillow.readthedocs.io/en/stable/) was used. 
Library is the part of image processing image app. Educational means that application carry some operations  and shows final image with specified number of
interimages. The benefit is that students can deeply analyze whole process, step by step. Additionally application calculate some image parameters like histogram, 
min/max pixel values, statistical parameters to provide wider view on performed operation.

## Technologies
* NumPy 1.18.2
* Pillow 7.1.1

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
Repository cointains a number of python files (modules):
* **basic_operations.py** --  basic operations, which was used by other modules.
* **binary_operations.py** -- operations related to binaryzed image (binaryzation and morphological operations).
* **filtering.py** -- filtering and correction operations.
* **noising.py** -- artificiall noising images, with most populars types of noise.

Additionally there was api.py file, which called functions from modules listed above, with appropiate arguments to perform some operation. Take as an example
[opening](https://en.wikipedia.org/wiki/Mathematical_morphology#Opening):
* varification that image is in grayscale, if not convertion (basic_operations module).
* perform binaryzation using [OTSU](https://en.wikipedia.org/wiki/Otsu%27s_method algorithm) method (binary_operations module).
* erode operation (binary_operations module).
* dilate operation (binary_operations module).

This file was created to use with educational app. Final image and interimages were saved in specified locations and then read by GUI layer written NodeJS (created 
by my teammate).

In the near future, will be added second api, adapted to store all images (final and inter) as list of NumPY arrays.

