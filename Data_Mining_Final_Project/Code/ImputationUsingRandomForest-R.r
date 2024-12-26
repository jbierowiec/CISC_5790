setwd("/Users/janbierowiec/Desktop/Eddy")
library(missForest)
data <- read.csv("census-income.data.csv", check.name = FALSE)
data[data == " ?"] <- NA
summary(data)
data.imp <- missForest(data)
data.imp$OOBerror
summary(data.imp)
data.imp$ximp
nrow(data.imp$ximp)
write.csv(data.imp$ximp, file="census-income.data_WithImputation_UsingRandomForest.csv", row.names=FALSE)