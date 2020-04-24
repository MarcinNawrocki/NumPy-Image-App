#To activate env in cmd type in workspace directory:
#ImageApp\Scripts\activate
import time
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
#from skimage import data
#from skimage import filters
#from skimage import exposure

#dla obu operacji odczytu zapisu trzeba zrobić obsługę błędów
def readImage(filename, verbose=False):
    """Reading image from file and transform it to NumPy array

    Keyword argument:
    filename -- relative path to image
    verbose -- if true showing image(default False)

    Return:
    image as Numpy Array
    """

    pil_image = Image.open(filename)
    if verbose:
        pil_image.show()

    np_image = np.array(pil_image, dtype=np.uint8)
    if np_image.shape[2] == 3:
        if np.array_equiv(np_image[:,:,0],np_image[:,:,1]) and np.array_equiv(np_image[:,:,2],np_image[:,:,1]):
            np_image = np_image[:,:,0]
            np_image = grayTo2D(np_image)
    np_image.setflags(write=1)
    #print(filters.threshold_otsu(np_image))
    return np_image

def saveImage(np_image, filename, verbose=False):
    """Saving image to file

    Keyword argument:
    np_image -- image as NumPy array
    filename -- relative to where image should be saved
    verbose -- if true showing image(default False)

    Return:
    #TODO Code
    """

    #tutaj w zależności jest czy obraz czarno biały czy kolorowy różne
    #image2 = Image.fromarray(grayscale, mode ='L')
    mode = getImageColorType(np_image)
    if mode == 'xy1':
        np_image = grayTo2D(np_image)
        mode = getImageColorType(np_image)

    pil_image = Image.fromarray(np_image, mode=mode)
    if verbose:
        pil_image.show()
    #pil_image.save(filename)
    #return coś o powodzeniu operacji

def getHumanGrayscale(np_image):
    """Convert image to "Human" grayscale (0,215*R+0.7151*G+0.0721*B)

    Keyword argument:
    np_image -- image as NumPy array
    Return:
    np_image_gray -- image as grayscale
    Important:
    !!! This operation reducing Array dimension from 3 to 2 !!!
    """

    to_mono_vector = [0.2125 , 0.7154 , 0.0721 ]
    np_image_gray = np.zeros((np_image.shape[0],np_image.shape[1]), dtype=np.uint8)
    np_image_gray = np.around(to_mono_vector[0]*np_image[:,:,0] + to_mono_vector[1]*np_image[:,:,1] + to_mono_vector[2]*np_image[:,:,2])
    np_image_gray = np_image_gray.astype(np.uint8)
    np_image_gray = grayTo2D(np_image_gray)
    return np_image_gray

def getMachineGrayscale(np_image):
    """Convert image to "Machine" grayscale (R+G+B)/3

    Keyword argument:
    np_image -- image as NumPy array
    Return:
    np_image_gray -- image as grayscale
    Important:
    !!! This operation reducing Array dimension from 3 to 2 !!!
    """

    np_image_gray = np.zeros((np_image.shape[0],np_image.shape[1]), dtype=np.uint8)
    np_image_gray = np.around(np.mean(np_image, axis=2))
    np_image_gray = np_image_gray.astype(np.uint8)
    np_image_gray = grayTo2D(np_image_gray)
    return np_image_gray

def getImageColorType(np_image):
    """Return image mode

    Keyword argument:
    np_image -- image as NumPy array
    Return:
    String data:
        L for grayscale image
        RGB for color image
    """

    if len(np_image.shape) == 3:
        if np_image.shape[2] == 3:
            return 'RGB'
        else:
            return 'xy1'
    elif len(np_image.shape) == 2:
        return 'L'
    #tutaj jakaś obsługa błędów

def getBasicImageParameters(pil_image):
    """Return basic image parameters as dictionary with info about Resolution, format and mode

    Keyword argument:
    pil_image -- image as PIL object
    Return:
        Python dictionary with keys: Resolution, Format, Mode 
    """

    return {"Resolution" : image.size, "Format": image.format, 'Mode': image.mode}

def getMinMaxPix(np_image_2dim):
    """Return  dictionary with max and min pixel value

    Keyword argument:
    np_image -- image as 2D NumPy array(whole grayscale or one color channel)
    Return:
        Python dictionary with keys: Max value, Min value 
    """

    return {"Max value" : np.amax(np_image_2dim), "Min value" : np.amin(np_image_2dmin)}

def getStatisticImageParameters(np_image_2dim):
    """Return statistical image parameters as dictionary(Variance, Standard devation, Median, Average)

    Keyword argument:
    np_image_2dim -- image as 2D NumPy array(whole grayscale or one color channel)
    Return:
        Python dictionary with keys: Variance, Standard devation, Median, Average 
    """

    return {'Variance' : np.var(np_image_2dim), 'Standard devation' : np.std(np_image_2dim), "Median" : np.median(np_image_2dim), "Average" : np.average(np_image_2dim)}

def getImageHistogram(np_image_2dim, normalize = False, with_bins = False):
    """ Return histogram for image in 2D Numpy array(grayscale or single channel)
    Keyword argument:
    np_image_2dim -- image as 2D NumPy array(whole grayscale or one color channel)
    normalize -- if set to True histogram values will be normalized(default = False)
    with_bins -- return also bins as numpy arra(default= False)
    Return:
    histogram as NumPy array,
    bins ans NumPy array if with_bins = True

    """
    np_image_2dim = np_image_2dim.ravel()
    np_hist =  np.histogram(np_image_2dim.ravel(), bins=range(257))[0]
    #np_hist, bin_edges =  _bincount_histogram(image, source_range)

    if normalize:
        np_hist = np_hist / np.sum(hist)
    if with_bins:
        #np_hist[:10] = np.zeros(10)
        np_bins = np.nonzero(np_hist)
        np_bins = np_bins[0]
        min_edge = np_bins[0]
        max_edge = np_bins[-1]
        np_hist = np_hist[min_edge:max_edge+1]

        return np_hist, np_bins
    else:
        return np_hist

def ensureGrayscale(np_image):
    """Ensures that given image is in grayscale

    Keyword argument:
    np_image -- image as NumPy array
    Return:
    np_image as grayscale
    """
    if len(np_image.shape) == 3:
        np_image = getMachineGrayscale(np_image)
    
    return np_image

def grayTo3D(np_image):
    """
    Reshaping np_image grayscale image to 3 dimension 
    e.g np_image with shape(x,y) will be reshaping to (x,y,1)
    """
    return np_image.reshape((np_image.shape[0], np_image.shape[1],1))

def grayTo2D(np_image):
    """
    Reshaping np_image grayscale image to 2 dimension 
    e.g np_image with shape(x,y,1) will be reshaping to (x,y)
    """
    return np_image.reshape((np_image.shape[0], np_image.shape[1]))

def getWindow(np_image_bin, index, dir_size,  struct_elem):
    """Get window for morphological and filtering  operations

    Keyword argument:
    index -- indexes of actual processing pixel as tuple
    dir_size -- size of structural element in one direction (dir_size = (size-1)/2)
    x_max -- max value of x index
    y_max -- max value of y index
    Return:
    np_window -- window of morphological operations with specific size and shape
    """
    #zobaczyc czy to zadziała dla 3D
    x_max, y_max = np_image_bin.shape[:2]
    y_1 = index[1] - dir_size if index[1] - dir_size >= 0 else 0
    y_2 = index [1] + dir_size + 1 if index [1] + dir_size + 1  < y_max else -1
    x_1 = index[0] - dir_size if index[0] - dir_size >= 0 else 0
    x_2 = index [0] + dir_size + 1 if index [0] + dir_size + 1 < x_max else -1

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
#pętla przez wszystkie piksele
#for xy in np.ndindex(data.shape[:2]):
  # print(str(data[xy])+", " + str(data[xy]))

data = readImage("Lena-gray.png", verbose=True)
data = grayTo3D(data)
saveImage(data, "Bin.png", verbose=True)
#TODO:
    #filtering test
    #implementing errors catching

    
