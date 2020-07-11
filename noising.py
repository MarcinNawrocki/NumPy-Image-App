import random
import os

import numpy as np

import basic_operations as bs

def saltPepperNoising(np_image, propability = 0.05, saltPepperRatio = 0.5):
    """Adding salt pepper nois to given image
    Keyword argument:
    np_image -- image as NumPy array, 2D for grayscale/single 
    propability -- how much of image should be noising. Propability that single pixel become salt/pepper noise.
        (default = 0.05) 
    saltPepperRatio -- specified salt to pepper ratio (default 0.5):
        1.0 -- only salt
        0.5 -- equal propability of salt and pepper
        0.0 -- only pepper
    Return:
        Image noised with specified values. Dimension the same as given.
    """

    if len(np_image.shape) == 3 and np_image.shape[2] == 3:
        salt = (255,255,255)
        pepper = (0,0,0)
    else:
        salt = 255
        pepper = 0
    
    #calculate number of piksels to noise
    total_pixels = propability * np_image.shape[0] * np_image.shape[1]
    salt_pixels = int(total_pixels * saltPepperRatio)
    pepper_pixels = int(total_pixels - salt_pixels)

    #generate random pi
    xySalt = [(random.randrange(0, np_image.shape[0]), random.randrange(0, np_image.shape[1])) for i in range(salt_pixels)]
    xyPepper = [(random.randrange(0, np_image.shape[0]), random.randrange(0, np_image.shape[1])) for i in range(pepper_pixels)]

    np_image_nois = np.copy(np_image)
    for x,y in xySalt:
        np_image_nois[x,y] = salt

    for x,y in xyPepper:
        np_image_nois[x,y] = pepper

    return np_image_nois

def gaussianNoise(np_image_3D, std_dev=0.1, mean=0):
    """Adding gaussian noise with to given image
    Keyword argument:
        np_image_3D -- three dimensional image as NumPy array (could be grayscale but must be have shape (x,y,1)
        std_dev -- standard deviation parameter
        mean -- mean parameter (default = 0) 
    Return:
        Image with gaussian noise specified by parameters. Dimension the same as given.
    """

    row, col, ch = np_image_3D.shape
    np_gauss = np.random.normal(mean,std_dev,(row,col,ch))*255
    np_gauss = np_gauss.reshape(row,col,ch).astype(np.uint8)
    np_image_3D = np_image_3D + np_gauss

    return np_image_3D
