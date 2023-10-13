#install.packages('tidyverse')
library(tidyverse)
#install.packages('tidymodels')
library(tidymodels)
#install.packages('DataExplorer')
#install.packages("poissonreg")
# library(poissonreg)
#install.packages("glmnet")
library(glmnet)
#library(patchwork)
# install.packages("rpart")
#install.packages('ranger')
library(ranger)
#install.packages('stacks')
library(stacks)
#install.packages('vroom')
library(vroom)
#install.packages('parsnip')
library(parsnip)
# install.packages('dbarts')
# library(dbarts)
#install.packages('embed')
library(embed)

# rm(list=ls()) use to erase environment

## 112 Cols

data_train <- vroom("./data/train.csv") %>%
  mutate(ACTION=factor(ACTION))# grab training data


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
  step_other(all_nominal_predictors(), threshold = .01) %>% # get hours
  step_dummy(all_nominal_predictors()) # get dummy variables

prepped_recipe <- prep(my_recipe) # preprocessing new data
baked_data1 <- bake(prepped_recipe, new_data = data_train)

# ncol(baked_data1)


##################################################
##### Logistic Regression: Bin Cross Entropy #####
##################################################

log_reg <- logistic_reg() %>% #Type of model
  set_engine("glm")

amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(log_reg) %>%
  fit(data = data_train) # Fit the workflow

data_test <- vroom("./data/test.csv") # grab testing data

amazon_predictions <- predict(amazon_workflow,
                         new_data=data_test,
                         type="prob") %>% # "class" or "prob"
  mutate(Id = data_test$id) %>%
  mutate(ACTION = ifelse(.pred_1 > .95, 1, 0)) %>%
  select(-.pred_0, -.pred_1)

vroom_write(amazon_predictions, "amazon_pred_logreg.csv", delim = ",")
save(file = 'amazon_wf.RData', list = c('amazon_workflow'))
load('amazon_wf.RData')


################################################
##### Logistic Regression: target encoding #####
################################################

rFormula <- ACTION ~ .

my_recipe <- recipe(rFormula, data = data_train) %>% # set model formula and dataset
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))# get hours
  

prepped_recipe <- prep(my_recipe) # preprocessing new data
baked_data1 <- bake(prepped_recipe, new_data = data_train)

log_reg <- logistic_reg(mixture = tune(), penalty = tune()) %>% #Type of model
  set_engine("glmnet")

pretune_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(log_reg)

# Grid for CV
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 tuning possibilities

# Split data for CV
folds <- vfold_cv(data_train, v = 10, repeats = 1)

# Run CV
CV_results <- pretune_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best('roc_auc')

final_wf <- pretune_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = data_train)

data_test <- vroom("./data/test.csv") # grab testing data

amazon_predictions <- predict(final_wf,
                              new_data=data_test,
                              type="prob") %>% # "class" or "prob"
  mutate(Id = data_test$id) %>%
  mutate(ACTION = ifelse(.pred_1 > .95, 1, 0)) %>%
  select(-.pred_0, -.pred_1)

vroom_write(amazon_predictions, "amazon_logreg_target.csv", delim = ",")
save(file = 'amazon_penalized_wf.RData', list = c('final_wf'))
load('amazon_penalized_wf.RData')



