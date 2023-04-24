data <- read.csv('processed_data.csv')
boxplot(data$lstn_Jun)
boxplot(data$tomato_production.tonnes.acre.)
min(data$tomato_farmland_area.acres.)

pdata <- read.csv('processed_data_without_outliers.csv')
barplot(table(pdata$CA_county_ansi))
summary(pdata$tomato_production.tonnes.acre.)

library(rpart)
library(dplyr)
library(caTools)
library(party)
train_index <- sample(nrow(data), 0.7 * nrow(data))
train <- data[train_index, ]
test <- data[-train_index, ]

temp <- data[data$CA_county_ansi == 47, ]
barplot(temp$tomato_production.tonnes.acre., names.arg = 2011:2021)
