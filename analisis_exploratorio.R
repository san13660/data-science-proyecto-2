library(ggplot2)

list.files(path = "../input/rsna-bone-age")

data_training <- read.csv("../input/rsna-bone-age/boneage-training-dataset.csv", stringsAsFactors = TRUE)

summary(data_training)

ggplot(data_training, aes(x = boneage)) +
  geom_histogram(aes(y = stat(count)), bins = 12, color="black", fill="grey") +
  scale_y_continuous()

ggplot(data_training, aes(x = boneage)) +
  geom_boxplot()

data_training$male <- as.numeric(data_training$male) - 1

barplot(table(data_training$male))

tabla <- table(data_training$male)
prop.table(tabla)