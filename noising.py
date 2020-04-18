import random
import numpy as np
import basic_operations as bs


def saltPepperNoising(np_image, propability = 0.05, saltPepperRatio = 0.5):
    """Adding salt pepper nois to given image
    Keyword argument:
    np_image -- image to apply noising
    propability -- how much of image should be noising. Propability that single pixel become salt/pepper noise.
    (default = 0.5) 
    saltPepperRatio -- specified salt to pepper ratio (default 0.5):
        1.0 -- only salt
        0.5 -- equal propability of salt and pepper
        0.0 -- only pepper
    Return:
        Image noised with specified values. Dimension the same as given.
    """
    if len(np_image.shape) == 3:
            salt = (255,255,255)
            pepper = (0,0,0)
    else:
        salt = 255
        pepper = 0
    
    #calculate piksels to noise
    total_piksels = propability * np_image.shape[0] * np_image.shape[1]
    salt_piksels = int(total_piksels * saltPepperRatio)
    pepper_piksels = int(total_piksels - salt_piksels)

    xySalt = [(random.randrange(0, np_image.shape[0]), random.randrange(0, np_image.shape[1])) for i in range(salt_piksels)]
    xyPepper = [(random.randrange(0, np_image.shape[0]), random.randrange(0, np_image.shape[1])) for i in range(pepper_piksels)]

    #sprawdzanie czy się nie powtarzają??
    #sprawdzanie czy nie jest kwadratowy, wtedy oszczednosc w iteracjach, oraz czy nie jest wyzerowane

    for x,y in xySalt:
        np_image[x,y] = salt

    for x,y in xyPepper:
        np_image[x,y] = pepper

    return np_image

def gaussianNoise(np_image_3D, std_dev=0.1, mean=0):
    """Adding gaussian noise with to given image
    Keyword argument:
    np_image_3D -- image with 3 dimensions to apply noising
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

data = bs.readImage("Lena.png", verbose=True)
#data2 = bs.readImage("bin1.png", verbose=False)
data = bs.getHumanGrayscale(data)
#data_gray = bs.grayTo3D(data_gray)
data = bs.grayTo3D(data)
data_noise = gaussianNoise(data)
#data_noise2 = saltPepperNoising(data2, propability=0.4)
bs.saveImage(data_noise, "Bin.png", verbose=True)
#bs.saveImage(data_noise2, "Bin.png", verbose=True)
