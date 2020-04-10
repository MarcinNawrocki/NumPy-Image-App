#To activate env in cmd type in workspace directory:
#ImageApp\Scripts\activate
import time
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt


#dla obu operacji odczytu zapisu trzeba zrobić obsługę błędów
def readImage(filename, verbose=False):
    """Reading image from file and transform it to NumPy array

    Keyword argument:
    filename -- relative path to image
    verbose -- if true showing image(default False)

    Return:
    image as Numpy Arra
    """

    pil_image = Image.open(filename)
    if verbose:
        pil_image.show()
    np_image = np.array(pil_image, dtype=np.uint8)
    np_image.setflags(write=1)
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
    pil_image = Image.fromarray(np_image, mode=mode)
    if verbose:
        pil_image.show()
    pil_image.save(filename)
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
    np_image_gray = np.around(to_mono_vector[0]*np_image[:,:,0] + to_mono_vector[1]*np_image[:,:,1] + to_mono_vector[2]*data[:,:,2])
    np_image_gray = np_image_gray.astype(np.uint8)
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
        return 'RGB'
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
    np_image -- image as 2D NumPy array(whole grayscale or one color channel)
    Return:
        Python dictionary with keys: Variance, Standard devation, Median, Average 
    """

    return {'Variance' : np.var(np_image_2dim), 'Standard devation' : np.std(np_image_2dim), "Median" : np.median(np_image_2dim), "Average" : np.average(np_image_2dim)}

def getImageHistogram(np_image_2dim):
    """
    Return histogram for image in 2D Numpy array(grayscale or single channel)
    """
    return np.histogram(np_image_2dim, bins=range(257))[0]



data = readImage("obraz.bmp", verbose=False)
data_gray = getHumanGrayscale(data)
data_rgb = getRGB(data_gray)
saveImage(data_rgb, "RGB.bmp", verbose=True)

#pętla przez wszystkie piksele
#for xy in np.ndindex(data.shape[:2]):
  # print(str(data[xy])+", " + str(data[xy]))
