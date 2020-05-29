import sys
import basic_operations as bs 
import binary_operations as bn 
import noising as ns 
import filtering as fil 
import json
import numpy as np
from skimage import filters
import os
import shutil

#filters
#dolnoprzepustowe
np_LP1 = np.array([[1,1,1],[1,1,1], [1,1,1]])
np_LP2 = np.array([[1,1,1],[1,2,1], [1,1,1]])
np_LP3 = np.array([[1,1,1],[1,4,1], [1,1,1]])
np_LP4 = np.array([[1,1,1],[1,12,1], [1,1,1]])

#górnoprzepustowe
np_HP1 = np.array([[-1,-1,-1],[-1,9,-1], [-1,-1,-1]])
np_HP2 = np.array([[0,-1,0],[-1,5,-1], [0,-1,0]])
np_HP3 = np.array([[1,-2,1],[-2,5,-2], [1,-2,1]])
np_HP4 = np.array([[0,-1,0],[-1,20,-1], [0,-1,0]])

defaultImagePath = "./public/python/images/"
#create path if not exist 
if not(os.path.exists(defaultImagePath)):
    print("tworzenie folderów")
    os.makedirs(defaultImagePath)


#add more from
#http://www.algorytm.org/przetwarzanie-obrazow/filtrowanie-obrazow.html

#image parameters


def getImageParameters(filename, save_source=True, defaultImagePath= "./public/python/images/"):
    """Reading image from file and save parameters in the json file

    Keyword argument:
    filename -- relative path to image

    Return:
    image as Numpy Array
    """

    #removeFiles()
    np_image = bs.readImage(filename)
    if save_source:
        extension = ".png"
        bs.saveImage(np_image, defaultImagePath + '0' + extension)

    if save_source:
        extension = ".png"
        bs.saveImage(np_image, defaultImagePath + '0' + extension)

    parameters = {}
    parameters['Type'] = bs.getImageColorType(np_image)
    parameters['x_res'] = int(np_image.shape[0])
    parameters['y_res'] = int(np_image.shape[1])

    tmp_minmax = bs.getMinMaxPix(np_image) 
    parameters = dict(list(parameters.items()) + list(tmp_minmax.items()))
    tmp_stat = bs.getStatisticImageParameters(np_image)
    parameters = dict(list(parameters.items()) + list(tmp_stat.items()))

    if bs.isColorImage(np_image):
        parameters['R_histogram'] = bs.getImageHistogram(np_image[:,:,0]).tolist()
        parameters['G_histogram'] = bs.getImageHistogram(np_image[:,:,1]).tolist()
        parameters['B_histogram'] = bs.getImageHistogram(np_image[:,:,2]).tolist()
    else:
        parameters['histogram'] = bs.getImageHistogram(np_image).tolist()

    #error catching
    #print(json.dumps(parameters, default=convert))

    #ZMIANA W ŚCIEŻCE
    with open('public/python/images/data_color.json', 'w+') as fp:
        json.dump(parameters, fp, default=bs.convert)

def toGrayscale(filename, gray="human"):
    """Convert image to  grayscale 

    Keyword argument:
    filename --  filename or path to image
    gray:
        human - convert to "human" grayscale (0,215*R+0.7151*G+0.0721*B)
        machine - convert to "machine" grayscale (R+G+B)/3
    Return:
        
    Important:
    !!! This operation reducing Array dimension from 3 to 2 !!!
    """
    getImageParameters(filename)
    np_image = bs.readImage(filename)
    #isRGB?
    if bs.isColorImage(np_image):
        if gray == "human":
            np_final = bs.getHumanGrayscale(np_image)

        elif gray == "machine":
            np_final = bs.getMachineGrayscale(np_image)
        else: #default -- human
            np_final = bs.getHumanGrayscale(np_image)
    else:
        np_final = np_image

    bs.saveImage(np_final, defaultImagePath+"1.png")
    return 1

def getBinaryzedImage(filename, threshold, number_of_inters=1): #tested
    """Return binaryzed image based on threshold given by user

    Keyword argument:
    filename --  filename or path to image
    thershold -- integer value in range (0,255)
    number_of_inters -- number of inter images which should be generated by this operation
    Return:
        1 if operation was succesful
        0 if threshold is out of range
    """

    getImageParameters(filename)
    if threshold > 255 or threshold < 0:
        return 0
    start_image_number = 0
    np_image = bs.readImage(filename)
    np_image_2D, isConverted = bs.ensureGrayscale(np_image, info = True)
    if isConverted:
        bs.saveImage(np_image_2D, defaultImagePath+"1.png")
        start_image_number += 1
    np_final = bn.thresholdBinaryzation(np_image_2D, threshold)

   #generate name
    bs.generateInterImages(np_image_2D, np_final, number_of_inters, start_image_number)

    return 1

def getOtsuBinaryzedImage(filename, number_of_inters=1):#tested, read about images where is zeros in histogram
    """Return binaryzed image, by threshold which is generated by OTSU method.
    Algorithm calculated Otsu using maximalization between class variance.

    Keyword argument:
    filename --  filename or path to image
    number_of_inters -- number of inter images which should be generated by this operation
    Return:
        1 if operation was succesful.
    """

    getImageParameters(filename)
    np_image = bs.readImage(filename)
    np_image_2D, isConverted = bs.ensureGrayscale(np_image, info = True)
    start_image_number = 0
    if isConverted:
        bs.saveImage(np_image_2D, defaultImagePath+"1.png")
        start_image_number += 1
    #print(filters.threshold_otsu(np_image_2D))
    np_final = bn.otsuBinaryzation(np_image_2D)

    #generate names
    bs.generateInterImages(np_image_2D, np_final, number_of_inters, start_image_number)
    return 1

def getDilate(filename, struct_elem='rect', size=3,  number_of_inters=3): #tested
    """Execute dilate morphological operation on image
    Function binaryzed image by OTSU method, so pass RGB or grayscale images possible.

    Keyword argument:
    filename --  filename or path to image
    struct_elem:
        cross - cross structural element
        rect - rectangle structural element
        circ -- cricle structural element(maybe implemente)
    size-- size of struct element, should be 2N+1
    number_of_inters -- number of inter images which should be generated by this operation
    Return:
        1 if operation was succesful.
    """

    getImageParameters(filename)
    np_image = bs.readImage(filename)
    np_image_2D, isConverted = bs.ensureGrayscale(np_image, info = True)
    start_image_number = 0
    if isConverted:
        bs.saveImage(np_image_2D, defaultImagePath+"1.png")
        start_image_number += 1
    np_image_bin = bn.otsuBinaryzation(np_image_2D)
    bs.saveImage(np_image_bin, defaultImagePath+"1.png")
    start_image_number += 1
    np_final = bn.dilate(np_image_bin, struct_elem, size)

    #generate names
    bs.generateInterImages(np_image_bin, np_final, number_of_inters, start_image_number)
    return 1

def getErode(filename, struct_elem='cross', size=3, number_of_inters=3): #tested
    """Execute erode morphological operation on image
    Function binaryzed image by OTSU method, so pass RGB or grayscale images possible.

    Keyword argument:
    filename --  filename or path to image
    struct_elem:
        cross - cross structural element
        rect - rectangle structural element
        circ -- cricle structural element(maybe implemente)
    size: size of struct element, should be 2N+1
    number_of_inters -- number of inter images which should be generated by this operation
    Return:
        1 if operation was succesful.
    """
    
    start_image_number = 0
    np_image = bs.readImage(filename)
    np_image_2D, isConverted = bs.ensureGrayscale(np_image, info = True)
    if isConverted:
        bs.saveImage(np_image_2D, defaultImagePath+"1.png")
        start_image_number += 1
    np_image_bin = bn.otsuBinaryzation(np_image_2D)
    bs.saveImage(np_image_bin, defaultImagePath+"2.png")
    start_image_number += 1
    np_final = bn.erode(np_image_bin, struct_elem, size)

    #generate names
    bs.generateInterImages(np_image_bin, np_final, number_of_inters, defaultImagePath+"1.png", start_image_number)

def getOpenly(filename, struct_elem='rect', size=3, number_of_inters=4): #Tested
    """Execute openly(erode and dilate on the same image) morphological operation on image
    Function binaryzed image by OTSU method, so pass RGB or grayscale images possible.

    Keyword argument:
    filename --  filename or path to image
    struct_elem:
        cross - cross structural element
        rect - rectangle structural element
        circ -- cricle structural element(maybe implemente)
    size: size of struct element, should be 2N+1
    number_of_inters -- number of inter images which should be generated by this operation
    Return:
        1 if operation was succesful.
    """

    start_image_number = 0
    np_image = bs.readImage(filename)
    np_image_2D, isConverted = bs.ensureGrayscale(np_image, info = True)
    if isConverted:
        bs.saveImage(np_image_2D, defaultImagePath+"1.png")
        start_image_number += 1
    np_image_bin = bn.otsuBinaryzation(np_image_2D)
    bs.saveImage(np_image_bin, defaultImagePath+"2.png")
    start_image_number += 1

    np_image_er = bn.erode(np_image_bin, struct_elem, size)
    bs.generateInterImages(np_image_bin, np_image_er, int(number_of_inters/2), int(start_image_number/2))

    start_image_number += int(number_of_inters/2)
    rest = number_of_inters % 2
    number_of_inters = int(number_of_inters/2) + rest

    np_final = bn.dilate(np_image_er, struct_elem, size)

    #filenames
    bs.generateInterImages(np_image_er, np_final, number_of_inters, start_image_number)

def getClosely(filename, struct_elem='rect', size=3, number_of_inters=4): #Tested
    """Execute openly(erode and dilate on the same image) morphological operation on image
    Function binaryzed image by OTSU method, so pass RGB or grayscale images possible.

    Keyword argument:
    filename --  filename or path to image
    struct_elem:
        cross - cross structural element
        rect - rectangle structural element
        circ -- cricle structural element(maybe implemente)
    size: size of struct element, should be 2N+1
    number_of_inters -- number of inter images which should be generated by this operation
    Return:
        1 if operation was succesful.
    """

    np_image = bs.readImage(filename)
    np_image_2D, isConverted = bs.ensureGrayscale(np_image, info = True)
    start_image_number = 0
    if isConverted:
        bs.saveImage(np_image_2D, defaultImagePath+"1.png")
        start_image_number += 1
    np_image_bin = bn.otsuBinaryzation(np_image_2D)
    bs.saveImage(np_image_bin, defaultImagePath+"2.png")
    start_image_number += 1
    
    np_image_dil = bn.dilate(np_image_bin, struct_elem, size)
    bs.generateInterImages(np_image_bin, np_image_dil, int(number_of_inters/2))

    rest = number_of_inters % 2
    number_of_inters = int(number_of_inters/2) + rest
    np_final = bn.erode(np_image_dil, struct_elem, size)

    #filenames
    bs.generateInterImages(np_image_dil, np_final, number_of_inters, start_image_number)

def filteringImage(filename, np_mask_pom, number_of_inters=1): #Tested
    """
    Processing filtering with given kernel.
    If RGB image is passed, then each channel will be filter separately.

    Keyword argument:
    filename --  filename or path to image
    np_mask -- mask matrix as numpy array
    number_of_inters -- number of inter images which should be generated by this operation
    Return:
        1 if operation was succesful.
    """
    
    getImageParameters(filename)
    if np_mask_pom == 'LP1':
        np_mask = np_LP1;
    elif np_mask_pom == 'LP2':
        np_mask = np_LP2;
    elif np_mask_pom == 'LP3':
        np_mask = np_LP3;
    elif np_mask_pom == 'LP4':
        np_mask = np_LP4;
    elif np_mask_pom == 'HP1':
        np_mask = np_HP1;
    elif np_mask_pom == 'HP2':
        np_mask = np_HP2;
    elif np_mask_pom == 'HP3':
        np_mask = np_HP3;
    else: np_mask = np_HP4;
        
    np_image = bs.readImage(filename)   

    if bs.isColorImage(np_image):
        np_final = np.zeros(np_image.shape, dtype = np.uint8)
        np_final[:,:,0] = fil.matrixFilter(np_image[:,:,0], np_mask) 
        np_final[:,:,1] = fil.matrixFilter(np_image[:,:,1], np_mask)
        np_final[:,:,2] = fil.matrixFilter(np_image[:,:,2], np_mask)
    else:
        np_final = fil.matrixFilter(np_image, np_mask)

    #filenames
    bs.generateInterImages(np_image, np_final, number_of_inters)

def medianFiltergingImage(filename, struct_elem='rect', size=3, number_of_inters=1):#Tested
    """
    Processing median filtering with specified shape and size on image given by filename
    If RGB image is passed, then each channel will be filter separately.

    Keyword argument:
    filename --  filename or path to image
    struct_elem:
        cross -- cross structural element
        rect -- rectangle structural element
        circ -- cricle structural element(maybe will be implemented)
    size: size of struct element, should be 2N+1
    number_of_inters -- number of inter images which should be generated by this operation
    Return:
        1 if operation was succesful. 
    """

    getImageParameters(filename)
    np_image = bs.readImage(filename)  
    np_image = ns.saltPepperNoising(np_image)
    bs.saveImage(np_image, filename)

    if bs.isColorImage(np_image):
        np_final = np.zeros(np_image.shape, dtype = np.uint8)
        np_final[:,:,0] = fil.medianFilter(np_image[:,:,0], struct_elem, size) 
        np_final[:,:,1] = fil.medianFilter(np_image[:,:,1], struct_elem, size)
        np_final[:,:,2] = fil.medianFilter(np_image[:,:,2], struct_elem, size)
    else:
        np_final = fil.medianFilter(np_image, struct_elem, size)

    #filenames
    bs.generateInterImages(np_image, np_final, number_of_inters)

def gammaCorrection(filename, gamma, number_of_inters=1): #Tested
    """
    Processing gamma correction with specified gamma correction atribute
    If RGB image is passed, then each channel will be correct separately.
    Keyword argument:
    filename --  filename or path to image
    gamma - gamma correction parameter
    number_of_inters -- number of inter images which should be generated by this operation
    Return:
        1 if operation was succesful. 
    """
    
    getImageParameters(filename)
    np_image = bs.readImage(filename)  

    if bs.isColorImage(np_image):
        np_final = np.zeros(np_image.shape, dtype = np.uint8)
        np_final[:,:,0] = fil.gammaCorrection(np_image[:,:,0], gamma) 
        np_final[:,:,1] = fil.gammaCorrection(np_image[:,:,1], gamma)
        np_final[:,:,2] = fil.gammaCorrection(np_image[:,:,2], gamma)
    else:
        np_final = fil.gammaCorrection(np_image, gamma)

    #filenames
    bs.generateInterImages(np_image, np_final, number_of_inters)

def addGaussianNoise(filename, std_dev=0.05, mean=0, number_of_inters=1): #Grayscale looks good, test color
    """Adding gaussian noise with to given image

    Keyword argument:
    filename --  filename or path to image
    std_dev -- standard deviation parameter
    mean -- mean parameter (default = 0) 
    number_of_inters -- number of inter images which should be generated by this operation
    Return:
        1 if operation was succesful. 
    """

    getImageParameters(filename)
    np_image = bs.readImage(filename)    
    np_image_3D = bs.ensure3D(np_image)
    np_final = ns.gaussianNoise(np_image_3D, std_dev, mean)

    #filenames
    bs.generateInterImages(np_image, np_final, number_of_inters)
    
def addSaltPepperNoise(filename, propability = 0.05, saltPepperRatio = 0.5, number_of_inters=1): #Tested
    
    """Adding salt pepper nois to given image
    Keyword argument:
    filename --  filename or path to image
    propability -- how much of image should be noising. Propability that single pixel become salt/pepper noise.
    (default = 0.5) 
    saltPepperRatio -- specified salt to pepper ratio (default 0.5):
        1.0 -- only salt
        0.5 -- equal propability of salt and pepper
        0.0 -- only pepper
    Return:
        1 if operation was succesful. 
    """

    getImageParameters(filename)
    np_image = bs.readImage(filename)  
    np_final =  ns.saltPepperNoising(np_image, propability, saltPepperRatio)

    #filenames
    bs.generateInterImages(np_image, np_final, number_of_inters)
    
def removeFiles():   
    folder = "./public/python/images"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return 1

#getImageParameters("bin2.jpg")
getImageParameters("bin2.jpg")
getClosely("bin2.jpg")
#getImageParameters("cameraman.png",save_source=False)

if __name__ == "__main__":
    print("main")


    functions = {
        'getImageParameters': getImageParameters,
        'toGrayscale': toGrayscale,
        'getBinaryzedImage': getBinaryzedImage,
        'getOtsuBinaryzedImage': getOtsuBinaryzedImage,
        'getDilate': getDilate,
        'getErode': getErode,
        'getOpenly': getOpenly,
        'getClosely': getClosely,
        'filteringImage': filteringImage,
        'medianFiltergingImage': medianFiltergingImage,
        'gammaCorrection': gammaCorrection,
        'addGaussianNoise': addGaussianNoise,
        'addSaltPepperNoise': addSaltPepperNoise,
        'removeFiles': removeFiles
    }

    option = sys.argv[1]
    parameters = sys.argv[2:]

    for i in range(len(parameters)):
        try:
            parameters[i] = int(parameters[i])
        except ValueError:
            try:
                parameters[i] = float(parameters[i])
            except ValueError:
                continue
    if len(sys.argv) >=1:
        functions[option](*parameters)