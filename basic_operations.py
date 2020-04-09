#To activate env in cmd type in workspace directory:
#ImageApp\Scripts\activate
import time
import numpy as np 
from PIL import Image

start = time.time()
#odczyt obrazu:
image = Image.open("obraz.bmp")
#podstawowe informacje o obrazie
print(image.format)
print(image.size)
print(image.mode)

#przejście do NumPy array
data = np.array(image, dtype=np.uint8)
data.setflags(write=1)
print(type(data))
print(data.shape)

to_mono_vector = [0.2125 , 0.7154 , 0.0721 ]  #human grayscale, for machine (R+G+B)/
grayscale = np.zeros((data.shape[0],data.shape[1]), dtype=np.uint8)
grayscale = np.around(to_mono_vector[0]*data[:,:,0] + to_mono_vector[1]*data[:,:,1] + to_mono_vector[2]*data[:,:,2])
grayscale_machine = np.around(np.mean(data, axis=2))


grayscale = grayscale.astype(np.uint8)
grayscale_machine = grayscale_machine.astype(np.uint8)
#pętla przez wszystkie piksele
for xy in np.ndindex(data.shape[:2]):
    print(str(grayscale[xy])+", " + str(grayscale_machine[xy]))
    time.sleep(0.3)
    

#powrót do Pillow image
image2 = Image.fromarray(grayscale, mode ='L')
image3 = Image.fromarray(grayscale_machine, mode='L')
#image2 = Image.fromarray(datas)
image2.show(title="human")
image3.show(title="machine")
print("image2 parameters:")
print(image2.format)
print(image2.size)
print(image2.mode)
image2.save("grayscale.bmp")
end = time.time()
print(end - start)