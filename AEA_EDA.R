library(tidyverse)
#install.packages('tidymodels')
#install.packages('tidyverse')
library(tidymodels)
#install.packages('DataExplorer')
#install.packages("poissonreg")
library(poissonreg)
# install.packages("glmnet")
library(glmnet)
library(patchwork)
# install.packages("rpart")
# install.packages('ranger')
library(ranger)
#install.packages('stacks')
library(stacks)
library(vroom)
library(parsnip)
# install.packages('dbarts')
# library(dbarts)



data_train <- vroom("train.csv") # grab training data

factor_cols <- c(1:10)

data_train <- data_train %>%
  mutate(across(factor_cols, as.factor)) # convert factor variables

data_train


##### EDA #####
library(ggplot2)

boxplot(data_train$ROLE_CODE ~ data_train$ACTION,
        col='steelblue',
        main='action by role code',
        xlab='Action',
        ylab='ROLE_CODE')

boxplot(data_train$ROLE_TITLE ~ data_train$ACTION,
        col='steelblue',
        main='action by role title',
        xlab='Action',
        ylab='ROLE_TITLE')




