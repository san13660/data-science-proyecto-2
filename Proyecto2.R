#Universidad del Valle de Guatemala
#Data Science
#Proyecto 1 - Limpieza Datos
#Integrantes 
#Maria Fernanada Estrada 14198
#Christopher Sandoval 13660
#Rodrigo Samayoa 17332
#Ana Lucia Villela
#David Soto 17551

### Librerias a utilizar en R para la limpieza de datos
library(dplyr)
library(tools)

### Leer el CSV con todos los departamentos, separado por comas, cambiando los campos vacios y espacios en blanco por NA
registros <- read.csv("boneage-training-dataset.csv", stringsAsFactors = FALSE, na.strings=c("", " ","NA"), sep = ",")

### Se hace una vista para ver que los datos fueron cargados correctamente
View(registros)

### Se revisa que la columna Male contenga solo valores True y False
registrosMaleIncorrectos <- subset(registros, male!='False' && male!='True')

### Revisamos si existe un registro con algun contenido diferente al esperado
View(registrosMaleIncorrectos)

### Se revisa que la columna BoneAge contenga edades positivas
registrosBoneageIncorrectos <- subset(registros, boneage<0)

### Revisamos si existe un registro con algun contenido diferente al esperado
View(registrosBoneageIncorrectos)

### Se revisa que el contenido de Bonage sea un valor numerico en cada registro
View(which(!grepl('^[0-9]', registros$boneage)))

