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

# library(ggplot2)
# 
# boxplot(data_train$ROLE_CODE ~ data_train$ACTION,
#         col='steelblue',
#         main='action by role code',
#         xlab='Action',
#         ylab='ROLE_CODE')
# 
# boxplot(data_train$ROLE_TITLE ~ data_train$ACTION,
#         col='steelblue',
#         main='action by role title',
#         xlab='Action',
#         ylab='ROLE_TITLE')


#######################
##### Recipe/Bake #####
#######################

rFormula <- ACTION ~ .

# my_recipe <- recipe(rFormula, data = data_train) %>% # set model formula and dataset
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   step_other(all_nominal_predictors(), threshold = .01) %>% # get hours
#   step_dummy(all_nominal_predictors()) # get dummy variables
# 
# prepped_recipe <- prep(my_recipe) # preprocessing new data
# baked_data1 <- bake(prepped_recipe, new_data = data_train)

### For target encoding/Random Forests: ###

my_recipe <- recipe(rFormula, data = data_train) %>% # set model formula and dataset
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))# get hours

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



########################################
##### Classification Random Forest #####
########################################

class_rf_mod <- rand_forest(mtry = tune(), 
                            min_n = tune(),
                            trees = 800) %>% #Type of model
  set_engine('ranger') %>%
  set_mode('classification')

pretune_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(class_rf_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(mtry(range = c(1,ncol(data_train)-1)),
                            min_n(),
                            levels = 3) ## L^2 total tuning possibilities

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
  #mutate(ACTION = ifelse(.pred_1 > .95, 1, 0)) %>%
  mutate(ACTION = .pred_1) %>%
  select(-.pred_0, -.pred_1)

vroom_write(amazon_predictions, "amazon_pred_rf.csv", delim = ",")
save(file = 'amazon_penalized_wf.RData', list = c('final_wf'))
load('amazon_penalized_wf.RData')



################################
##### Naive Bayes Approach #####
################################

install.packages('discrim')
library(discrim)
install.packages('naivebayes')
library(naivebayes)

# nb model
nb_mod <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
                        set_mode('classification') %>%
                        set_engine('naivebayes')

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_mod)

tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5) ## L^2 total tuning possibilities

# Split data for CV
folds <- vfold_cv(data_train, v = 10, repeats = 1)

# Run CV
CV_results <- nb_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best('roc_auc')

final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = data_train)

data_test <- vroom("./data/test.csv") # grab testing data

amazon_predictions <- predict(final_wf,
                              new_data=data_test,
                              type="prob") %>% # "class" or "prob"
  mutate(Id = data_test$id) %>%
  #mutate(ACTION = ifelse(.pred_1 > .95, 1, 0)) %>%
  mutate(ACTION = .pred_1) %>%
  select(-.pred_0, -.pred_1)

vroom_write(amazon_predictions, "amazon_pred_nb.csv", delim = ",")
save(file = 'amazon_penalized_wf.RData', list = c('final_wf'))
load('amazon_penalized_wf.RData')



###############################
##### K-Nearest Neighbors #####
###############################

install.packages('kknn')
library(kknn)

my_recipe <- recipe(rFormula, data = data_train) %>% # set model formula and dataset
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% # get hours
  step_normalize()

prepped_recipe <- prep(my_recipe) # preprocessing new data
baked_data1 <- bake(prepped_recipe, new_data = data_train)

## knn model
knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

## Fit or Tune MOdel
tuning_grid <- grid_regular(neighbors(),
                            levels = 5) ## L^2 total tuning possibilities

# Split data for CV
folds <- vfold_cv(data_train, v = 10, repeats = 1)

# Run CV
CV_results <- knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best('roc_auc')

final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = data_train)

data_test <- vroom("./data/test.csv") # grab testing data

amazon_predictions <- predict(final_wf,
                              new_data=data_test,
                              type="prob") %>% # "class" or "prob"
  mutate(Id = data_test$id) %>%
  #mutate(ACTION = ifelse(.pred_1 > .95, 1, 0)) %>%
  mutate(ACTION = .pred_1) %>%
  select(-.pred_0, -.pred_1)

vroom_write(amazon_predictions, "amazon_pred_knn.csv", delim = ",")


####################################################
##### Naive Bayes Principle Comp Dim Reduction #####
####################################################

install.packages('discrim')
library(discrim)
install.packages('naivebayes')
library(naivebayes)

my_recipe <- recipe(rFormula, data = data_train) %>% # set model formula and dataset
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))%>% # get hours
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = 0.9) # Threshold between 0 and 1

prepped_recipe <- prep(my_recipe) # preprocessing new data
baked_data1 <- bake(prepped_recipe, new_data = data_train)

# nb model
nb_mod <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode('classification') %>%
  set_engine('naivebayes')

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_mod)

tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5) ## L^2 total tuning possibilities

# Split data for CV
folds <- vfold_cv(data_train, v = 10, repeats = 1)

# Run CV
CV_results <- nb_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best('roc_auc')

final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = data_train)

data_test <- vroom("./data/test.csv") # grab testing data

amazon_predictions <- predict(final_wf,
                              new_data=data_test,
                              type="prob") %>% # "class" or "prob"
  mutate(Id = data_test$id) %>%
  #mutate(ACTION = ifelse(.pred_1 > .95, 1, 0)) %>%
  mutate(ACTION = .pred_1) %>%
  select(-.pred_0, -.pred_1)

vroom_write(amazon_predictions, "./data/amazon_nb_dim_red.csv", delim = ",")
save(file = 'amazon_penalized_wf.RData', list = c('final_wf'))
load('amazon_penalized_wf.RData')


##################################
##### KNN Comp Dim Reduction #####
##################################

install.packages('kknn')
library(kknn)

my_recipe <- recipe(rFormula, data = data_train) %>% # set model formula and dataset
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% # get hours
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = 0.9) # Threshold between 0 and 1

prepped_recipe <- prep(my_recipe) # preprocessing new data
baked_data1 <- bake(prepped_recipe, new_data = data_train)

## knn model
knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

## Fit or Tune Model
tuning_grid <- grid_regular(neighbors(),
                            levels = 5) ## L^2 total tuning possibilities

# Split data for CV
folds <- vfold_cv(data_train, v = 10, repeats = 1)

# Run CV
CV_results <- knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best('roc_auc')

final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = data_train)

data_test <- vroom("./data/test.csv") # grab testing data

amazon_predictions <- predict(final_wf,
                              new_data=data_test,
                              type="prob") %>% # "class" or "prob"
  mutate(Id = data_test$id) %>%
  #mutate(ACTION = ifelse(.pred_1 > .95, 1, 0)) %>%
  mutate(ACTION = .pred_1) %>%
  select(-.pred_0, -.pred_1)

vroom_write(amazon_predictions, "./data/amazon_knn_dim_red.csv", delim = ",")







