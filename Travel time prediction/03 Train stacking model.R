
# Load relevant libraries
library(data.table)
library(dplyr)
library(ggplot2)
library(h2o)
library(bit64)
library(tidyr)

# Set working directory
setwd("E:/MSc thesis")

# Import the data
train_vars <- c("TRIP_ID", "DURATION", "START_CELL", "END_CELL", "MINUTE", "HOUR", "WDAY", "WEEK", "DISTANCE")
train <- as_data_frame(fread("./Processed data/train_lm.csv", header = TRUE, sep = ",", select = train_vars, colClasses(TRIP_ID = "bit64")))

# Make sure HOUR, WDAY and WEEK are factors
train$MINUTE <- as.factor(train$MINUTE)
train$HOUR <- as.factor(train$HOUR)
train$WDAY <- factor(train$WDAY, levels = 0:6, labels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
train$WEEK <- as.factor(train$WEEK)

# Transform the start cell and destination cell to a factor
train$START_CELL <- factor(train$START_CELL, levels = 1:7500)
train$END_CELL <- factor(train$END_CELL, levels = 1:7500)

# RMSLE
RMSLE <- function(preds, obs) {
  return(sqrt(mean((log(preds + 1) - log(obs + 1))^2)))
}

# Take natural logarithm of duration
train$LOG_DURATION = log(train$DURATION)

# Initialize h2o server
localH2O <- h2o.init(nthreads = 6)

# Define variables for the models to use
independent <- c("START_CELL", "END_CELL", "MINUTE", "HOUR", "WDAY", "WEEK")
dependent <- c("LOG_DURATION")
lambda_seq <- 10^(seq(from = -6, to = 1, by = 1))
n_trees <- 500

# Create the folds
set.seed(30061992)
n_folds <- 5
n_train <- dim(train)[1]
folds <- sample(1:n_folds, replace = TRUE, size = n_train)

# Create the file for stacking training data
empty_data <- data.frame(fold = numeric(0), GBM_pred = numeric(0), RF_pred = numeric(0), OBS_DURATION = numeric(0))
#write.table(empty_data, "./Travel time prediction/Models/stacking_training_data.csv", row.names = FALSE)

# Loop through the folds to build the dataset for the ridge regression
for(fold in seq_len(n_folds)) {
  print(paste("Processing fold", fold))
  # Get the data rows that are in the current fold
  rows_holdout <- (fold == folds)
  
  # Load the corresponding train and holdout set in H2O
  trainH2O <- as.h2o(conn = localH2O, 
                     object = train[-(rows_holdout), ] %>% 
                       select(-TRIP_ID, -DISTANCE), 
                     destination_frame = "train.hex")
  
  holdoutH2O <- as.h2o(conn = localH2O,
                       object = train[rows_holdout, ] %>%
                         select(-TRIP_ID, -DISTANCE),
                       destination_frame = "holdout.hex")
  
  # Fit the GBM and RF with the optimal parameters
  gbm_model <- h2o.gbm(x = independent,
                      y = dependent,
                      training_frame = trainH2O,
                      ntrees = 250,
                      max_depth = 6,
                      distribution = "gaussian",
                      learn_rate = 0.1)

  rf_model <- h2o.randomForest(x = independent,
                              y = dependent,
                              training_frame = trainH2O,
                              ntrees = 160,
                              max_depth = 13,
                              mtries = 5,
                              min_rows = 14)
  
  # Now build the predictions of these models
  preds_RF <- h2o.predict(object = rf_model, newdata = holdoutH2O)
  preds_GBM <- h2o.predict(object = gbm_model, newdata = holdoutH2O)
  
  # Store these into a dataframe
  stacking_data <- data.frame(fold = fold,
                              GBM_pred = as.data.frame(preds_GBM),
                              RF_pred = as.data.frame(preds_RF),
                              OBS_DURATION = train[rows_holdout, ]$LOG_DURATION)
  
  # Write the concatenated data to the disk
  write.table(x = stacking_data,
              file = "./Travel time prediction/Models/stacking_training_data.csv",
              append = TRUE,
              row.names = FALSE,
              col.names = FALSE)
  
}

# Train full models
trainH2O <- as.h2o(conn = localH2O, 
                   object = train %>% 
                     select(-TRIP_ID, -DISTANCE, -DURATION), 
                   destination_frame = "train.hex")

# Fit the GBM and RF with the optimal parameters
gbm_model <- h2o.gbm(x = independent,
                     y = dependent,
                     model_id = "gbm_val",
                     training_frame = "train.hex",
                     #validation_frame = "holdout.hex",
                     ntrees = 250,
                     max_depth = 6,
                     distribution = "gaussian",
                     learn_rate = 0.1)

rf_model <- h2o.randomForest(x = independent,
                             y = dependent,
                             training_frame = "train.hex",
                             #validation_frame = "holdout.hex",
                             ntrees = 160,
                             max_depth = 13,
                             mtries = 5,
                             min_rows = 14)

h2o.saveModel(object = gbm_model, path = "./Travel time prediction/Models/FINAL_GBM_150909", force = TRUE)
h2o.saveModel(object = rf_model, path = "./Travel time prediction/Models/FINAL_RF_150909", force = TRUE)

# Now we are going to use this data to train the Ridge regression.

# First, we load the complete data back into R
stack_data <- as_data_frame(fread("./Travel time prediction/Models/stacking_training_data.csv", header = TRUE))

# Use glmnet to estimate the model
library(glmnet)
lambda_grid <- 10^seq(from = 2, to = -4, length = 125)
stack_model_glmnet <- cv.glmnet(x = stack_data %>% select(GBM_pred, RF_pred) %>% as.matrix(),
                                y = stack_data %>% select(OBS_DURATION) %>% as.matrix(),
                                alpha = 0,
                                standardize = TRUE,
                                lambda = lambda_grid,
                                nfolds = 5,
                                intercept = FALSE,
                                upper.limits = 1,
                                lower.limits = 0)

# Plot this in ggplot
plot_df <- with(stack_model_glmnet, data.frame(log.lambda  = log(lambda), cvm, cvup, cvlo))

pdf("./Visualizations/stacked_cv.pdf", width = 6, height = 4)
plot_df %>%
  ggplot(aes(x = log.lambda, y = sqrt(cvm))) +
  geom_line() +
  geom_point() + 
  #geom_errorbar(aes(ymax = sqrt(cvup), ymin = sqrt(cvlo))) +
  geom_vline(xintercept = log(stack_model_glmnet$lambda.min), linetype = 2, ) +
  geom_vline(xintercept = log(stack_model_glmnet$lambda.1se), linetype = 2, color = "blue") +
  labs(x = expression(log(lambda)), y = "RMSLE") +
  scale_x_continuous(breaks = seq(from = -12, to = 6, by = 1), limits = c(-5, 5)) + 
  scale_y_continuous(breaks = seq(from = -0.2, to = 1, by = 0.05), limits = c(NA, 1))
dev.off()

# Now estimate the full model
lambda.1se <- stack_model_glmnet$lambda.1se
# 0.0009284145
stack_model_glmnet_full <- glmnet(x = stack_data %>% select(GBM_pred, RF_pred) %>% as.matrix(),
                                  y = stack_data %>% select(OBS_DURATION) %>% as.matrix(),
                                  alpha = 0,
                                  standardize = TRUE,
                                  intercept = FALSE,
                                  upper.limits = 1,
                                  lower.limits = 0,
                                  lambda = lambda_grid)

save(stack_model_glmnet_full, file = "./Travel time prediction/Models/stacked_ridge_constrained.RData")





