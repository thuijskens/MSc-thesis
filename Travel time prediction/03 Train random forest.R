# 03 Train random forest.R

# This script trains a random forest regressor for 
#
#   T = T_0 + T_a,
#
# where T_0 is the already traversed time of a truncated trip and T_a is the additional travel time,
# that is predicted by the random forest regressor. The explanatory variables used are
#   1. Hour of the day
#   2. Day of the week
#   3. Week of the year
#   4. Start cell
#   5. End cell
#   6. Start*end cell (interaction effect)

# Load relevant libraries
library(data.table)
library(dplyr)
library(ggplot2)
library(h2o)
library(bit64)

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

# Define variables for random forest to use
independent <- c("START_CELL", "END_CELL", "MINUTE", "HOUR", "WDAY", "WEEK")
dependent <- c("LOG_DURATION")

##############################################
# Grid search
##############################################

# Store the data as a h2o instance
trainH2O <- as.h2o(localH2O, 
                   train %>% 
                     select(-TRIP_ID, -DISTANCE), 
                   key = "train.hex")

RF_mod <- h2o.randomForest(x = independent, 
                           y = dependent, 
                           ntree = c(100, 250, 500),
                           mtries = c(1,2,3,4,5,6),
                           depth = c(5, 10, 20),
                           holdout.fraction = 0.1,
                           verbose = TRUE
                           seed = 12345,
                           data = trainH2O, 
                           classification = FALSE,
                           type = "Bigdata")

# Save the model to the disk
h2o.saveModel(object = RF_mod, dir = "E:/MSc thesis/Travel time prediction/Models", name = "RandomForest", force = TRUE)

##############################################
# Randomized grid search 
##############################################

# Fit random forest on the data (randomized grid search)
n <- dim(train)[1]
n_models <- 50
set.seed(7654321)
rows_holdout <- sample(1:n, size = floor(n/4), replace = FALSE)

trainH2O <- as.h2o(conn = localH2O, 
                   object = train[-rows_holdout, ] %>% 
                     select(-TRIP_ID, -DISTANCE), 
                   destination_frame = "train.hex")

holdoutH2O <- as.h2o(conn = localH2O,
                     object = train[rows_holdout, ] %>%
                       select(-TRIP_ID, -DISTANCE),
                     destination_frame = "holdout.hex")

fit_ntree <- 50
for(i in 23:50) {
  rand_depth <- sample(5:30, 1)
  rand_mtries <- sample(1:length(independent), 1)
  rand_nmin <- sample(1:20, 1)
  model_name <- paste0("RandomForest_", i,
                       "_ntree", fit_ntree,
                       "_depth", rand_depth,
                       "_mtries", rand_mtries,
                       "_nmin", rand_nmin)
  
  model <- h2o.randomForest(x = independent,
                            y = dependent,
                            training_frame = trainH2O,
                            model_id = model_name,
                            #validation_frame = holdoutH2O,
                            ntrees = fit_ntree,
                            max_depth = rand_depth,
                            mtries = rand_mtries,
                            min_rows = rand_nmin,
                            verbose = TRUE,
                            classification = FALSE,
                            type = "BigData")
  
  # Predict the validation set
  mod_preds <- h2o.predict(object = model, newdata = holdoutH2O)
  errors <- RMSLE(preds = exp(as.data.frame(mod_preds)), obs = train[rows_holdout, ]$DURATION)
  error_df <- data.frame(model = model_name, error = errors)
  
  # Write results to file
  if(i == 1) {
    write.table(error_df, "./Travel time prediction/Models/RF_CV_results.csv", row.names = FALSE, append = FALSE)
  } else {
    write.table(error_df, "./Travel time prediction/Models/RF_CV_results.csv", row.names = FALSE, col.names = FALSE, append = TRUE)
  }
  
  print(paste("Processed model", model_name))
}


# Load the file and get the model with the minimum RMSLE
cv_results <- read.table("./Travel time prediction/Models/RF_CV_results.csv", header = TRUE)
min_model <- cv_results[which.min(cv_results$error), ]

# Just get the params by hand
depth <- 13
mtries <- 5
nmin <- 14
n_max <- 50

# Fit the full model
early_stopping_RF <- h2o.randomForest(x = independent,
                                      y = dependent,
                                      training_frame = trainH2O,
                                      validation_frame = holdoutH2O,
                                      score_each_iteration = TRUE,
                                      model_id = "Earlystopping_RF",
                                      ntrees = 1000,
                                      max_depth = depth,
                                      mtries = mtries,
                                      min_rows = nmin,
                                      seed = 1234)

h2o.scoreHistory(early_stopping_RF) %>%
  ggplot(aes(x = number_of_trees)) +
  geom_line(aes(y = sqrt(training_MSE),color = "Training set")) +
  geom_line(aes(y = sqrt(validation_MSE), color = "Holdout set")) +  
  scale_x_continuous(breaks = seq(from = 0, to = 600, by = 50)) + 
  scale_y_continuous(breaks = seq(from = 0.25, to = 0.6, by = 0.05)) +
  labs(x = "Number of trees", y = "RMSLE") +
  theme(legend.title = element_blank())


write.csv(early_stopping_RF@model$scoring_history, "./Travel time prediction/Models/early_stopping_rf.csv", row.names = FALSE)
write.csv(early_stopping_RF@model$variable_importances, "./Travel time prediction/Models/early_stopping_rf_varimp.csv", row.names = FALSE)













##############################################
# Randomized grid search (old)
##############################################

# Fit random forest on the data (randomized grid search)
n <- dim(train)[1]
n_models <- 50
set.seed(7654321)
rows_holdout <- sample(1:n, size = floor(n/4), replace = FALSE)

trainH2O <- as.h2o(conn = localH2O, 
                   object = train[-rows_holdout, ] %>% 
                     select(-TRIP_ID, -DISTANCE), 
                   destination_frame = "train.hex")

holdoutH2O <- as.h2o(conn = localH2O,
                     object = train[rows_holdout, ] %>%
                       select(-TRIP_ID, -DISTANCE),
                     destination_frame = "holdout.hex")

models <- c()

fit_ntree <- 50

for(i in seq_len(n_models)) {
  rand_depth <- sample(5:30, 1)
  rand_mtries <- sample(1:length(independent), 1)
  rand_nmin <- sample(1:20, 1)
  model_name <- paste0("RandomForest_", i,
                       "_ntree", fit_ntree,
                       "_depth", rand_depth,
                       "_mtries", rand_mtries,
                       "_nmin", rand_nmin)
  
  model <- h2o.randomForest(x = independent,
                            y = dependent,
                            training_frame = trainH2O,
                            model_id = model_name,
                            ntrees = fit_ntree,
                            max_depth = rand_depth,
                            mtries = rand_mtries,
                            min_rows = rand_nmin,
                            verbose = TRUE,
                            classification = FALSE,
                            type = "BigData")
  
  models <- c(models, model)
  print(paste("Processed model", model_name))
}

errors <- numeric(n_models)
for(i in 1:8) {
  mod_preds <- h2o.predict(object = models[[i]], newdata = holdoutH2O)
  errors[i] <- RMSLE(preds = exp(as.data.frame(mod_preds)), obs = train[rows_holdout, ]$DURATION)
}

# Write the errors to a table 
errors_df <- data.frame(model_name = sapply(models, function(x) x@model_id),
                        errors = errors)
write.csv(errors_df, "./Travel time prediction/Models/random_forest_cv_results.csv", row.names = FALSE)




# Now, we fit the random forest with one tree at a time
model <- models[[which.min(errors)]]
params <- model@model$params
n_max <- 800

early_stopping_RF <- h2o.randomForest(x = independent,
                          y = dependent,
                          training_frame = trainH2O,
                          validation_frame = holdoutH2O,
                          model_id = "Early stopping RF",
                          ntrees = n_max,
                          max_depth = params$max_depth,
                          mtries = params$mtries,
                          min_rows = params$min_rows,
                          verbose = TRUE,
                          classification = FALSE,
                          type = "BigData",
                          score_each_iteration = TRUE,
                          seed = 1234)

write.csv(early_stopping_RF@model$scoring_history, "./Travel time prediction/Models/early_stopping_rf.csv", row.names = FALSE)
write.csv(early_stopping_RF@model$variable_importances, "./Travel time prediction/Models/early_stopping_rf_varimp.csv", row.names = FALSE)






# Get the best model and the best parameters
model <- models[[which.min(errors)]]
params <- model@model$params

trainH2O <- as.h2o(localH2O, 
                   train %>% 
                     select(-TRIP_ID, -DISTANCE), 
                   key = "train.hex")

# Build the final model with the optimal parameters
RF_mod <- h2o.randomForest(x = independent,
                           y = dependent,
                           key = "FinalModel",
                           ntree = params$ntree,
                           depth = params$depth,
                           mtries = params$mtries,
                           classification = FALSE,
                           verbose = TRUE,
                           data = trainH2O,
                           type = "BigData",
                           score_each_iteration = TRUE)

# Save the model
h2o.saveModel(object = RF_mod, dir = "E:/MSc thesis/Travel time prediction/Models", name = "Final_RandomForest", force = TRUE)

# In-sample error
RF_preds <- h2o.predict(object = RF_mod, newdata = trainH2O)
mse <- RMSLE(exp(as.data.frame(RF_preds)), train$DURATION)
print(mse)