import time

import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from skimage import exposure
import basic_operations as bs 


def thresholdBinaryzation(np_image_2D, threshold):
    """Return binaryzed image based on threshold given by user

    Keyword argument:
        np_image_2D -- two dimensional image as NumPy array (grayscale or single color channel)
        thershold -- integer value in range (0,255)
    Return:
        Binaryzed image as numpy array with 0 and 255 values
    """
   
    np_image_thr = np.where(np_image_2D > threshold,255,0)
    np_image_thr = np_image_thr.astype(np.uint8)
    return np_image_thr
    
def otsuBinaryzation(np_image_2D, only_threshold = True):
    """Return binaryzed image, getting by use of Otsu method
    Algorithm calculated Otsu using maximalization between class variance

    Keyword argument:
        np_image_2D -- two dimensional image as NumPy array (grayscale or single color channel)
    Return:
        Threshold calcualted using otsu method
    """

    np_hist, np_thresholds = bs.getImageHistogram(np_image_2D, with_bins=True)
    np_hist = np_hist.astype(float)

    #cumulative sum for histogram (parameter p for object)
    np_p_ob = np.cumsum(np_hist)
    # inverted cumulative sum from inverted histogram  (parameter for background)
    np_p_bg = np.cumsum(np_hist[::-1])[::-1]

    # u parameter for object
    np_u_ob = np.cumsum(np_hist * np_thresholds) /  np_p_ob
    # u parameter for background
    np_u_bg = (np.cumsum((np_hist * np_thresholds)[::-1]) / np_p_bg[::-1])[::-1]

    #between clas variance
    np_bcv = np_p_ob[:-1]  * np_p_bg[1:] * (np_u_ob[:-1]-np_u_bg[1:]) ** 2

    #max between class variance is the threshold that we looking for
    otsu_threshold = np.argmax(np_bcv)

    if only_threshold:
        return otsu_threshold
    else:
        return thresholdBinaryzation(np_image_2D,otsu_threshold)

def dilate(np_image_bin, struct_elem='rect', size=3):
    """Execute dilate morphological operation on binaryzed image

    Keyword argument:
        np_image_bin -- binaryzed image as NumPy array
        struct_elem:
            cross - cross structural element
            rect - rectangle structural element
        size: size of struct element, should be 2N+1
    Return:
        Binarized image after dilatation operation
    """
    np_image_bin = np_image_bin.astype(np.uint8)
    np_image_dil = np.zeros(np_image_bin.shape, dtype=np.uint8)
    
    for index, x in np.ndenumerate(np_image_bin):
        np_window = bs.getWindow(np_image_bin, index, size, struct_elem)

        if np_window.min() != 0:
            np_image_dil[index[0], index[1]] = 255

    return np_image_dil 

def erode(np_image_bin, struct_elem='rect', size=3):
    """Execute erode morphological operation on binaryzed image
    Keyword argument:
        np_image_bin -- binaryzed image as NumPy array
        struct_elem:
            cross - cross structural element
            rect - rectangle structural element
        size: size of struct element, should be 2N+1
    Return:
        Binarized image after erode operation
    """
    np_image_bin = np_image_bin.astype(np.uint8)
    np_image_er = np.zeros(np_image_bin.shape, dtype=np.uint8)
    
    for index, x in np.ndenumerate(np_image_bin):
        np_window = bs.getWindow(np_image_bin, index, size, struct_elem)
        
        if np_window.max() == 255:
            np_image_er[index[0], index[1]] = 255

    return np_image_er

