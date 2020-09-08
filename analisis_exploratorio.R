# Universidad del Valle de Guatemala
# Data Science 1 - Seccion 10
# Analisis exploratorio - Proyecto 2
# Christopher Sandoval 13660
# Fernanda Estrada 14198
# Rodrigo Samayoa 17332
# David Soto 17551
# Ana Villela 18903
# 08/09/2020



# Importar librerias
library(ggplot2)

# Especifico para trabajar en Kaggle
list.files(path = "../input/rsna-bone-age")

# Leer csv (este path es especifico para trabajar en Kaggle)
data_training <- read.csv("../input/rsna-bone-age/boneage-training-dataset.csv", stringsAsFactors = TRUE)


# ******************** ANALISIS EXPLORATORIO ********************

# Resumen de los datos (se incluye la tabla de frecuencias de la variable male)
summary(data_training)

# Histograma de la variable boneage
ggplot(data_training, aes(x = boneage)) +
  geom_histogram(aes(y = stat(count)), bins = 12, color="black", fill="grey") +
  scale_y_continuous()

# Diagrama de caja y bigotes de la variable boneage
ggplot(data_training, aes(x = boneage)) +
  geom_boxplot()

# Cambiar los datos de True y False a 1 y 0, respectivamente
data_training$male <- as.numeric(data_training$male) - 1

# Grafica de barras de la variable male
barplot(table(data_training$male))

# Tabla de proporciones de la variable male
tabla <- table(data_training$male)
prop.table(tabla)
