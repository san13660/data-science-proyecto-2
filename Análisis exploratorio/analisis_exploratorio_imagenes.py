from PIL import Image
import os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from os import walk

### Se hace una lectura de las imagenes
heightArray = []
widthArray = []
cantidadPixeles = []
pathImagenPequena = ''
pathImagenGrande = ''
for (dirpath, dirnames, filenames) in walk('boneage-training-dataset'):
    for filename in filenames:
        ### Se extrae las dimensiones de las imagenes
        with Image.open('boneage-training-dataset/' + filename) as image: 
            width, height = image.size
            heightArray.append(height)
            widthArray.append(width)
            cantidadPixeles.append(height * width)
            if max(cantidadPixeles) == height * width:
                pathImagenGrande = 'boneage-training-dataset/' + filename
            if min(cantidadPixeles) == height * width:
                pathImagenPequena = 'boneage-training-dataset/' + filename    


### Se revisa la distribucion que hay en las alturas de las imagenes
plt.hist(heightArray, bins=10, color='blue')
plt.show()

### Se revisa la distribucion que hay en los anchos de las imagenes
plt.hist(widthArray, bins=10, color='orange')
plt.show()

### Se revisa la distribucion que hay en la cantidad de pixeles en las imagenes
plt.hist(cantidadPixeles, bins=10, color='red')
plt.show()

### Se hace una revision de cuales son las dimension mas pequeñas y mas grandes
print("La altura mas grande es: ", str(max(heightArray)))
print("La altura mas pequeña es: ", str(min(heightArray)))
print("La anchura mas grande es: ", str(max(widthArray)))
print("La anchura mas pequeña es: ", str(min(widthArray)))
print("La cantidad de pixeles mas grande en un imagen es: ", str(max(cantidadPixeles)))
print("La cantidad de pixeles mas pequeña en un imagen es: ", str(min(cantidadPixeles)))


### Se hace una comparacion de la imagen mas pequeña y la mas grande para darnos una idea
### de que tanto afectaria redimensionar el tamaño de la imagen
img = mpimg.imread(pathImagenPequena)
imgplot = plt.imshow(img)
plt.show()

img = mpimg.imread(pathImagenGrande)
imgplot = plt.imshow(img)
plt.show()
