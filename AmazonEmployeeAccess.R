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
install.packages('embed')
library(embed)

## 112 Cols

data_train <- vroom("train.csv") # grab training data

factor_cols <- c(1:10)

data_train <- data_train %>%
  mutate(across(factor_cols, as.factor)) # convert factor variables

data_train


###############
##### EDA #####
###############

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


#######################
##### Recipe/Bake #####
#######################

rFormula <- ACTION ~ .

my_recipe <- recipe(rFormula, data = data_train) %>% # set model formula and dataset
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(c(RESOURCE, MGR_ID, ROLE_ROLLUP_1, ROLE_ROLLUP_2,
               ROLE_DEPTNAME, ROLE_TITLE, ROLE_FAMILY_DESC,
               ROLE_FAMILY, ROLE_CODE), threshold = .01) %>% # get hours
  step_dummy(all_nominal_predictors()) # get dummy variables

prepped_recipe <- prep(my_recipe) # preprocessing new data
baked_data1 <- bake(prepped_recipe, new_data = data_train)

ncol(baked_data1)



