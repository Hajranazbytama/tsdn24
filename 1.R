data <- read.csv(file = "C:/Users/acer/Documents/Kuliah/Lomba/2024/TSDN/Praktek/dataset.csv")
View(data)

data <- data[, -c(1, 2, 5)]
View(data)

sum(is.na(data))

data <- na.omit(data)
sum(is.na(data))

View(data)

str(data)

data$copd <- as.factor(data$copd)

str(data)

library(caret)
set.seed(2024)
control <- rfeControl(functions = treebagFuncs, 
                      method = "LGOCV", 
                      number = 20)
pilih_var <- rfe(data[, -14], 
                 data[, 14], 
                 sizes = c(1:20), 
                 rfeControl = control)

pilih_var

var_pilihan <- predictors(pilih_var)
var_pilihan

View(data)


