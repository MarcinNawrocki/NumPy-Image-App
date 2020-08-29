#To activate env in cmd type in workspace directory:
#ImageApp\Scripts\activate
import sys
import time
import pathlib

import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt



#defaultImagePath = "express-app/public/python/images/"

def readImage(filename, verbose=False):
    """Reading image from file and transform it to NumPy array

    Keyword arguments:
        filename -- relative path to image
        verbose -- if true showing image (default False)

    Return:
        np_image -- image as Numpy Array
    """

    pil_image = Image.open(filename)
    if verbose:
        pil_image.show()

    np_image = np.array(pil_image, dtype=np.uint8)

    #some grayscale images was readed as 3D with three equal color channels, implicitly in this app we want grascale image as 2D
    if len(np_image.shape) == 3:
        if np_image.shape[2] == 3:
            if np.array_equiv(np_image[:,:,0],np_image[:,:,1]) and np.array_equiv(np_image[:,:,2],np_image[:,:,1]):
                np_image = np_image[:,:,0]
                np_image = grayTo2D(np_image)
    else:
        #we must convert images which is read as 2 dimensional grayscale images
        pil_image = Image.open(filename).convert('L')
        np_image = np.array(pil_image, dtype=np.uint8)

    np_image.setflags(write=1)
    
    return np_image

def saveImage(np_image, filename, verbose=False):
    """Saving image to file

    Keyword argument:
        np_image -- image as NumPy array
        filename -- relative path to where image should be saved, with filename conatining extension 
        verbose -- if true showing image(default False)

    Return:
    ---
    """

    #image2 = Image.fromarray(grayscale, mode ='L')
    mode = getImageColorType(np_image)
    #before saving 3D grascale image we must convert them to 2D
    if mode == 'xy1':
        np_image = grayTo2D(np_image)
        mode = getImageColorType(np_image)

    pil_image = Image.fromarray(np_image, mode=mode)
    if verbose:
        pil_image.show()
    pil_image.save(filename)

def getHumanGrayscale(np_image):
    """Convert image to "Human" grayscale (0,215*R+0.7151*G+0.0721*B)

    Keyword argument:
        np_image -- image as NumPy array

    Return:
        np_image_gray -- image as 2D grayscale
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
        np_image_gray -- image as 2D grayscale
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
        xy1 for 3D grayscale image
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

def getMinMaxPix(np_image):
    """Return  dictionary with max and min pixel value

    Keyword argument:
        np_image -- image as NumPy array
    Return:
        Python dictionary with keys: Max value, Min value 
    """

    parameters = {}

    if isColorImage(np_image):
        parameters['R_Max_value'] = np.amax(np_image[:,:,0])
        parameters['R_Min_Value'] = np.amin(np_image[:,:,0])
        parameters['G_Max_value'] = np.amax(np_image[:,:,1])
        parameters['G_Min_Value'] = np.amin(np_image[:,:,1])
        parameters['B_Max_value'] = np.amax(np_image[:,:,2])
        parameters['B_Min_Value'] = np.amin(np_image[:,:,2])
    else: 
        parameters = {"Max_value" : np.amax(np_image), "Min_value" : np.amin(np_image)}

    return parameters

def getStatisticImageParameters(np_image):
    """Return statistical image parameters as dictionary(Variance, Standard devation, Median, Average)

    Keyword argument:
        np_image_2dim -- image as 2D NumPy array(whole grayscale or one color channel)

    Return:
        Python dictionary with keys: Variance, Standard devation, Median, Average 
    """
    parameters = {}
    if isColorImage(np_image):
        parameters['R_Variance'] = np.var(np_image[:,:,0])
        parameters['R_Standard_devation'] = np.std(np_image[:,:,0])
        parameters['R_Median'] = np.median(np_image[:,:,0])
        parameters['R_Average'] = np.average(np_image[:,:,0])
        parameters['G_Variance'] = np.var(np_image[:,:,1])
        parameters['G_Standard_devation'] = np.std(np_image[:,:,1])
        parameters['G_Median'] = np.median(np_image[:,:,1])
        parameters['G_Average'] = np.average(np_image[:,:,1])
        parameters['B_Variance'] = np.var(np_image[:,:,2])
        parameters['B_Standard_devation'] = np.std(np_image[:,:,2])
        parameters['B_Median'] = np.median(np_image[:,:,2])
        parameters['B_Average'] = np.average(np_image[:,:,2])


    else: 
        parameters = {'Variance' : np.var(np_image), 'Standard_devation' : np.std(np_image), 
        "Median" : np.median(np_image), "Average" : np.average(np_image)}

    return parameters
    
def getImageHistogram(np_image_2dim, normalize = False, with_bins = False):
    """ Return histogram for image in 2D Numpy array(grayscale or single channel)

    Keyword argument:
        np_image_2dim -- image as 2D NumPy array (whole grayscale or one color channel)
        normalize -- if set to True histogram values will be normalized(default = False)
        with_bins -- return also bins as numpy array (default= False)

    Return:
        histogram as NumPy array,
        bins as NumPy array if with_bins = True

    """
    #np_image_2dim = np_image_2dim.ravel()
    np_hist =  np.histogram(np_image_2dim.ravel(), bins=range(257))[0]

    if normalize:
        np_hist = np_hist / np.sum(hist)
        
    if with_bins:
        np_nonzero = np.nonzero(np_hist)
        np_nonzero = np_nonzero[0]
        min_edge = np_nonzero[0]
        max_edge = np_nonzero[-1]
        np_bins = np.arange(min_edge, max_edge+1, 1, dtype = np.uint8)
        np_hist = np_hist[min_edge:max_edge+1]
        return np_hist, np_bins
    else:
        return np_hist

def ensureGrayscale(np_image, info=False):
    """Ensures that given image is in grayscale

    Keyword argument:
        np_image -- image as NumPy array
        info -- if true then also info about making conversion is return (default false)
    Return:
        np_image -- grayscale image as 2D numpy array (if conversion was needed then Human Grayscale was used)
        isConverted -- returned only if info == True, contains info about execution of conversion
    """
    isConverted = False
    if len(np_image.shape) == 3:
        if np_image.shape[2] == 3:
            np_image = getHumanGrayscale(np_image)
            isConverted = True
    
    if info:
        return (np_image, isConverted )
    else:
        return np_image

def ensure3D(np_image):
    """
    Ensures that given grayscale image is 3 dimensional. If is 2D(grayscale) change to 3D(this operation do not changes information in image)
    e.g np_image with shape(x,y) will be reshaping to (x,y,1). If image has been already grascale, then will be returned unchanged

    Keyword argument:
        np_image -- image as NumPy array (2D or 3D)

    Return:
        np_image -- image as 3D 
    """

    if(len(np_image.shape) == 2):
        return np_image.reshape((np_image.shape[0], np_image.shape[1],1)), True
    else:
        return np_image, False

def grayTo2D(np_image):
    """
    Reshaping np_image grayscale image to 2 dimension 
    e.g np_image with shape(x,y,1) will be reshaping to (x,y)

    Keyword argument:
        np_image -- image as 2D grayscale

    Return:
        np_image -- image as 3D grayscale 


    """
    return np_image.reshape((np_image.shape[0], np_image.shape[1]))

def isColorImage(np_image):
    """
    Check if image is colored (has 3 channels)

    Keyword agument:
        np_image -- image to check as numpy array
    Return 
        True if image is colored, false otherwise
    """
    if len(np_image.shape) == 3:
        if np_image.shape[2] == 3:
            return True
    
    return False

def getWindow(np_image, index, size,  struct_elem):
    """Get window for morphological and filtering  operations

    Keyword argument:
        np_image -- image as numpy array from we take window
        index -- indexes of actual processing pixel as tuple
        size -- size of structural element
        x_max -- max value of x index
        y_max -- max value of y index
    Return:
    np_window -- window of morphological operations with specific size and shape
    """

    if (size < 3) or (size % 2 == 0 ):
        raise ValueError(f"Inpropriate size value. Should be odd and greater than 2. Value was {size} ")

    #size of structural element in one direction
    dir_size = int((size-1)/2)

    x_max, y_max = np_image.shape[:2]
    y_1 = index[1] - dir_size if index[1] - dir_size >= 0 else 0
    y_2 = index [1] + dir_size + 1 if index [1] + dir_size + 1  < y_max else -1
    x_1 = index[0] - dir_size if index[0] - dir_size >= 0 else 0
    x_2 = index [0] + dir_size + 1 if index [0] + dir_size + 1 < x_max else -1

    if struct_elem == 'rect':
        np_window = np_image[x_1:x_2, y_1:y_2]
    elif struct_elem == 'cross':
        #take vertical part of cross
        cross_vert = np_image[x_1:x_2, index[1]]
        #takce horizontal part of cross
        cross_hor = np_image [index[0], y_1:y_2]
        #glue them to get whole cross
        np_window = np.concatenate((cross_vert, cross_hor))
    else:
        raise ValueError (f"Inpropriate window type. Should be 'rect' or 'cross'. The value was {struct_elem}.")

    return np_window

def splitImage(np_image, number_of_parts):
    """
    Splitting image given as 2-dimensional numpy array into list of parts.
    Keyword argument:
        np_image -- image to split, works for 2D and 3D images
        number_of_parts -- number of parts to split image
    Return
        np_split -- list of numpy array f.e:
            1,2,3,4 is parts of image as numpy array, function return list [1,2,3,4]
    """

    if np_image.shape[0] % number_of_parts == 0:
        np_split = np.split(np_image, number_of_parts, axis=0)
    else:
        step = int(np_image.shape[0] / number_of_parts)
        max_equal_step = step*(number_of_parts-1)
        np_split = np.split(np_image[:max_equal_step], number_of_parts-1)
        np_split.append(np_image[max_equal_step:])

    return np_split

def glueImage(splitted):
    """
    Glued vertically splitted images into one image.
    Keyword argument:
        splitted -- list of images as numpy arrays
    Return
        np_glued -- new, glued image as numpy array
    """

    np_glued = np.vstack(tuple(splitted))
    return np_glued

def generateInterImages(np_source, np_final, number_of_inters, start_image_number=0, defaultImagePath= "./public/python/images/"):
    """
    Function generates specified number of inter images and final images, then saved them in specified locations,
    with filenames as numbers. Real number of generated inter images is:
        number_of_inters - start_image number
    Additionally final image, will be saved.
    When number_of_inters == start_image_number, only final image will be saved
    Keyword argument: 
        np_source -- based image
        np_final -- final image
        number_of_inters -- specified number of inter images
        start_image_number -- number, which starts saving images
        defaultImagePath -- path to saving imageW
    """

    images = []
    #calculate number of parts based on number of inters and start image number
    number_of_parts = number_of_inters+1-start_image_number
    splitted_source = splitImage(np_source, number_of_parts)
    splitted_final = splitImage(np_final, number_of_parts)

    actual_image = splitted_source.copy()
    
    for i in range(start_image_number, number_of_inters+1):
        actual_image[i-start_image_number] = splitted_final[i-start_image_number]
        images.append(glueImage(actual_image))
    
    return images
        
def show_images(images, color_map='gray'):
    """[summary]

    Arguments:
        images {[type]} -- [description]

    Keyword Arguments:
        color_map {str} -- [description] (default: {'gray'})
    """
    if (type(images) is list):
        number_of_images = len(images)
        nrows = number_of_images // 5
        ncols = number_of_images // nrows
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8,8))
        for i, axi in enumerate(ax.flat):
            axi.imshow(images[i], cmap = color_map)
        plt.tight_layout(True)
        plt.show()
    else:
        try:
            plt.imshow(images, cmap = color_map)
            plt.show()
        except TypeError as e:
            print("Non displayable data passed. Should be a list of NumPy arrays or a numpy array.")
            print("Matplotlib error info: ", e)


def validate_number_of_inters(number_of_inters, default):

    if (number_of_inters < default):
        raise ValueError(f"Number of inters value should be greater than {default}. The value was {number_of_inters}")

def convert(o):
    """
    Convert function, needed to dump numpy datatypes into JSON file
    """
    if isinstance(o, np.uint8): return int(o)  
    if isinstance(o, uint8): return int(o)
    raise TypeError

