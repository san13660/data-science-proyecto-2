# Universidad del Valle de Guatemala
# Data Science 1 - Proyecto 2
# David Soto
# Christopher Sandoval
# Fernanda Estrada
# Rodrigo Samayoa
# Ana Villela
# Noviembre 2020

# Aplicacion del modelo MXNet al set de datos de edad osea



# Librerias y paquetes
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")
install.packages("imager")
library(EBImage)
library(pbapply)
library(caret)
library(mxnet)
library(imager)

# Cargar set de training y test
data_training <- read.csv("../input/rsna-bone-age/boneage-training-dataset.csv", stringsAsFactors = TRUE)
data_test <- read.csv("../input/rsna-bone-age/boneage-test-dataset.csv", stringsAsFactors = TRUE)

# Resumen datos
summary(data_training)


# ----- PREPROCESAMIENTO DE LAS IMAGENES -----

width <- 28
height <- 28

extract_feature <- function(dir_path = "../input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset", width, height, add_label = TRUE) {
  img_size <- width*height
  images_names <- list.files(dir_path)
  print(paste("Start processing", length(images_names), "images"))
  # Escala de grises
  feature_list <- pblapply(images_names, function(imgname) {
    img <- readImage(file.path(dir_path, imgname))
    # Cambiar tamano
    img_resized <- resize(img, w = width, h = height)
    # Escala de grises
    grayimg <- channel(img_resized, "gray")
    # Matriz
    img_matrix <- grayimg@.Data
    # Convertir a vector
    img_vector <- as.vector(t(img_matrix))
    return(img_vector)
  })
  # Juntar la lista de vectores
  feature_matrix <- do.call(rbind, feature_list)
  feature_matrix <- as.data.frame(feature_matrix)
  # Nombres
  names(feature_matrix) <- paste0("pixel", c(1:img_size))
  if (add_label) {
    # Labels
    feature_matrix <- cbind(label = label, feature_matrix)
  }
  return(feature_matrix)
}


# ----- TRAINING DEL MODELO -----

# Modelo MXNET
mx_data <- mx.symbol.Variable('data')
# 1era capa con 7x7 kernel y 30 filtros
conv_1 <- mx.symbol.Convolution(data = mx_data, kernel = c(7, 7), num_filter = 30)
tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")
pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2,2 ))
# 2da capa con 7x7 kernel y 60 filtros
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(7,7), num_filter = 60)
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 <- mx.symbol.Pooling(data = tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 1era capa totalmente conectada
flat <- mx.symbol.Flatten(data = pool_2)
fcl_1 <- mx.symbol.FullyConnected(data = flat, num_hidden = 500)
tanh_3 <- mx.symbol.Activation(data = fcl_1, act_type = "tanh")
# 2da capa totalmente conectada
fcl_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 2)
# Modelo generado
NN_model <- mx.symbol.SoftmaxOutput(data = fcl_2)

# Entrenando el modelo
model <- mx.model.FeedForward.create(NN_model, X = data_training, y = data_training,
                                     ctx = mx.cpu(),
                                     num.round = 30,
                                     array.batch.size = 100,
                                     learning.rate = 0.05,
                                     momentum = 0.9,
                                     wd = 0.00001,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))


# ----- PREDICCION -----

predict_probs <- predict(model, data_test)
predicted_labels <- max.col(t(predict_probs)) - 1
table(test_data[, 1], predicted_labels)
sum(diag(table(test_data[, 1], predicted_labels)))/2500

saveRDS(model, "model.rds")