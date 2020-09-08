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

### Se revisa que el contenido de ID sea un valor numerico en cada registro
View(which(!grepl('^[0-9]', registros$id)))

### Sustituir False/True por 0/1
registros$male <- gsub("False", "0", registros$male)
registros$male <- gsub("True", "1", registros$male)

### Se revisa la cantidad de archivos de imagenes en el DataSet que son 12611
length(list.files("boneage-training-dataset/boneage-training-dataset"))

### Se revisa que la cantidad de registros haga match con los archivos de imagenes que son 12611
nrow(registros)

### Se revisa el resumen de las variables del DataFrame para revisar que la limpieza fuera correcta
summary(registros)

### Se hace una vista para ver que los datos fueron limpiados correctamente
View(registros)