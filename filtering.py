import numpy as np

import basic_operations as bs
#dla każdej składowej oddzielnie

def medianFilter(np_image_2D, struct_elem='rect', size=3):
    """
    Processing median filtering with specified shape and size on given 2 dimensional image
    Keyword argument:
    np_image_2D -- two dimensional image(grayscale or single color channel)
    struct_elem:
        cross -- cross structural element
        rect -- rectangle structural element
        circ -- cricle structural element(maybe will be implemented)
    size: size of struct element, should be 2N+1
    Return:
        np_image_fil -- image as numpy 2D array, after median filtering
    
    """
    dir_size = int((size-1)/2)
    np_image_fil = np.zeros(np_image_2D.shape, dtype=np.uint8)

    for index, x in np.ndenumerate(np_image_2D):
        np_window = bs.getWindow(np_image_2D, index, dir_size, struct_elem)
        new_value = np.median(np_window)
        np_image_fil[index[0], index[1]] = new_value

    return np_image_fil

def matrixFilter(np_image_2D, np_mask):
    """
    Processing filtering with given matrix
    Keyword argument:
    np_image_2D -- two dimensional image(grayscale or single color channel)
    np_mask -- mask matrix as numpy array
    Return:
        np_image_fil -- image as numpy 2D array, after specified filtering
    """
    
    size = np_mask.shape[0]
    dir_size = int((size-1)/2)
    mask_sum = np_mask.sum()
    np_image_fil = np.copy(np_image_2D)
    x_max, y_max = np_image_2D.shape
    for index, x in np.ndenumerate(np_image_2D):
        if (index[0] >= dir_size and index[0] < x_max-dir_size-1 and index[1] >= dir_size and index[1] < y_max-dir_size-1):
            np_window = bs.getWindow(np_image_2D, index, dir_size, struct_elem='rect')
            np_window = np_window * np_mask
            window_sum = np_window.sum()
            new_value = window_sum / mask_sum if mask_sum !=0.0 else window_sum
            new_value = new_value if new_value <=255 else 255
            new_value = new_value if new_value >=0  else 0
            np_image_fil[index[0], index[1]] = new_value
    
    return np_image_fil.astype(np.uint8)

def gammaCorrection(np_image_2D, gamma):
    """
    Processing gamma correction with specified gamma correction atribute
    Keyword argument:
    np_image_2D -- two dimensional image(grayscale or single color channel)
    gamma - gamma correction parameter
    Return:
        np_image_gamma -- image as numpy 2D array, after gamma correction
    """

    np_LUT = np.arange(256)
    #LUT table for decode power to gamma
    LUT_el = lambda t: 255*(t/255)**(1/gamma)
    np_LUT = np.array([LUT_el(x) for x in np_LUT])
    np_LUT = np_LUT.astype(np.uint8)

    np_image_gamma = np.array([np_LUT[x] for x in np_image_2D])
    np_image_gamma = np_image_gamma.astype(np.uint8)
    return np_image_gamma

