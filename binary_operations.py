import time
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from skimage import exposure
import basic_operations as bs 


def thresholdBinaryzation(np_image_2D, threshold):
    """Return binaryzed image based on threshold given by user

    Keyword argument:
    np_image_2D -- image as NumPy array
    thershold -- integer value in range (0,255)
    Return:
        Binaryzed image as numpy array with 0 and 255 values
    """
   
    np_image_thr = np.where(np_image_2D > threshold,255,0)
    np_image_thr = np_image_thr.astype(np.uint8)
    return np_image_thr
    
def otsuBinaryzation(np_image_2D):
    """Return binaryzed image, getting by use of Otsu method
    Algorithm calculated Otsu using maximalization between class variance

    Keyword argument:
    np_image_2D -- image as NumPy array
    Return:
        Binaryzed image as numpy array with 0 and 255 values
    """

    np_hist, np_thresholds = bs.getImageHistogram(np_image_2D, with_bins=True)
    #[hist, _] = np.histogram(np_image_2D, bins=256, range=(0, 255))
    #np_hist = np_hist.astype(float)
    np_hist = np_hist.astype(float)

    np_p_ob = np.cumsum(np_hist)
    np_p_bg = np.cumsum(np_hist[::-1])[::-1]

    np_u_ob = np.cumsum(np_hist * np_thresholds) /  np_p_ob
    np_u_bg = (np.cumsum((np_hist * np_thresholds)[::-1]) / np_p_bg[::-1])[::-1]

    np_bcv = np_p_ob[:-1]  * np_p_bg[1:] * (np_u_ob[:-1]-np_u_bg[1:]) ** 2

    otsu_threshold = np.argmax(np_bcv)
    #print(otsu_threshold)
    return thresholdBinaryzation(np_image_2D, otsu_threshold)

def dilate(np_image_bin, struct_elem='rect', size=3):
    """Execute dilate morphological operation on binaryzed image

    Keyword argument:
    np_image_bin -- binaryzed image
    struct_elem:
        cross - cross structural element
        rect - rectangle structural element
        circ -- cricle structural element(maybe implemente)
    size: size of struct element, should be 2N+1
    Return:
        Binarized image after dilatation operation
    """
    np_image_bin = np_image_bin.astype(np.uint8)
    np_image_dil = np.zeros(np_image_bin.shape, dtype=np.uint8)
    
    #np_image_bin = np.arange(625).reshape((25,25))
    #rectangle
    dir_size = int((size-1)/2)
    #print(x_max, y_max)
    for index, x in np.ndenumerate(np_image_bin):
        np_window = bs.getWindow(np_image_bin, index, dir_size, struct_elem)

        if np_window.min() != 0:
            np_image_dil[index[0], index[1]] = 255

    return np_image_dil 

def erode(np_image_bin, struct_elem='rect', size=3):
    """Execute erode morphological operation on binaryzed image

    Keyword argument:
    np_image_bin -- binaryzed image
    struct_elem:
        cross - cross structural element
        rect - rectangle structural element
        circ -- cricle structural element(maybe implemente)
    size: size of struct element, should be 2N+1
    Return:
        Binarized image after erode operation
    """
    np_image_bin = np_image_bin.astype(np.uint8)
    np_image_er = np.zeros(np_image_bin.shape, dtype=np.uint8)
    
    #np_image_bin = np.arange(625).reshape((25,25))
    #rectangle
    dir_size = int((size-1)/2)
    #print(x_max, y_max)
    for index, x in np.ndenumerate(np_image_bin):
        np_window = bs.getWindow(np_image_bin, index, dir_size, struct_elem)
        
        if np_window.max() == 255:
            np_image_er[index[0], index[1]] = 255

    return np_image_er

"""
def getWindow(np_image_bin, index, dir_size,  struct_elem):

    x_max, y_max = np_image_bin.shape[:2]
    y_1 = index[1] - dir_size if index[1] - dir_size >= 0 else 0
    y_2 = index [1] + dir_size + 1 if index [1] + dir_size + 1  < y_max else -1
    x_1 = index[0] - dir_size if index[0] - dir_size >= 0 else 0
    x_2 = index [0] + dir_size + 1 if index [0] + dir_size + 1 < x_max else -1
    #choose  window
    #
    #cross window

    if struct_elem == 'rect':
        np_window = np_image_bin[x_1:x_2, y_1:y_2]
    elif struct_elem == 'cross':
        cross_vert = np_image_bin[x_1:x_2, index[1]]
        cross_hor = np_image_bin [index[0], y_1:y_2]
        np_window = np.concatenate((cross_vert, cross_hor))
    else:
        #TODO
        pass
    return np_window
"""

