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
localH2O <- h2o.init(nthreads = -1)

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

GBM_mod <- h2o.gbm(x = independent,
                   y = dependent,
                   distribution = "gaussian",
                   data = trainH2O,
                   n.trees = c(100, 250, 500),
                   interaction.depth = c(5, 10, 20),
                   shrinkage = c(1.0, 0.1, 0.01),
                   verbose = TRUE,
                   seed = 123456,
                   holdout.fraction = 0.25)

# Save the model to the disk
h2o.saveModel(object = GBM_mod, dir = "E:/MSc thesis/Travel time prediction/Models", name = "GBM", force = TRUE)

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
                   destination_frame = "train_holdout.hex")

holdoutH2O <- as.h2o(conn = localH2O,
                     object = train[rows_holdout, ] %>%
                     select(-TRIP_ID, -DISTANCE),
                     destination_frame = "holdout.hex")

models <- c()

for(i in seq_len(n_models)) {
  rand_ntree <- sample(1:200, 1)
  rand_depth <- sample(5:25, 1)
  rand_shrinkage <- sample(10^-(1:5), 1)
  model_name <- paste0("GBM_", i,
                       "_ntree", rand_ntree,
                       "_depth", rand_depth,
                       "_shrinkage", rand_shrinkage)
  
  print(paste("Training model", model_name))
  model <- h2o.gbm(x = independent,
                   y = dependent,
                   key = model_name,
                   distribution = "gaussian",
                   data = trainH2O,
                   n.trees = rand_ntree,
                   interaction.depth = rand_depth,
                   shrinkage = rand_shrinkage)
  
  models <- c(models, model)
}

errors <- numeric(n_models)
for(i in seq_len(n_models)) {
  mod_preds <- h2o.predict(object = models[[i]], newdata = holdoutH2O)
  errors[i] <- RMSLE(preds = exp(as.data.frame(mod_preds)), obs = train[rows_holdout, ]$DURATION)
}

# Write the errors to a table 
errors_df <- data.frame(model_name = sapply(models, function(x) x@key),
                        errors = errors)
write.csv(errors_df, "./Travel time prediction/Models/gbm_cv_results.csv", row.names = FALSE)


# Do the early stopping training
model <- models[[which.min(errors)]]
params <- model@model$params
n_max <- 1000

GBM_early_stopping <- h2o.gbm(x = independent,
                              y = dependent,
                              training_frame = "train.hex",
                              validation_frame = "holdout.hex",
                              ntrees = n_max,
                              max_depth = 6,
                              distribution = "gaussian",
                              learn_rate = 0.1,
                              seed = 1992,
                              score_each_iteration = TRUE)

# Get the metrics on the validation set
pdf("./Visualizations/gbm_early_stopping.pdf", width = 8, height= 4)
GBM_early_stopping@model$scoring_history  %>%
  ggplot(aes(x = number_of_trees)) +
  geom_line(aes(y = sqrt(training_MSE),color = "Training set")) +
  geom_line(aes(y = sqrt(validation_MSE), color = "Holdout set")) +  
  scale_x_continuous(breaks = seq(from = 0, to = 600, by = 50)) + 
  scale_y_continuous(breaks = seq(from = 0.25, to = 0.6, by = 0.05)) +
  labs(x = "Number of trees", y = "RMSLE") +
  theme(legend.title = element_blank())
dev.off()

write.csv(GBM_early_stopping@model$scoring_history, "./Travel time prediction/Models/early_stopping_gbm.csv", row.names = FALSE)
write.csv(GBM_early_stopping@model$variable_importances, "./Travel time prediction/Models/early_stopping_gbm_varimp.csv", row.names = FALSE)

h2o.saveModel(object = GBM_early_stopping, dir = "E:/", name = "EarlyStoppingGBM", force = TRUE)

# Get the best model and the best parameters
model <- models[[which.min(errors)]]
params <- model@model$params

trainH2O <- as.h2o(localH2O, 
                   train %>% 
                     select(-TRIP_ID, -DISTANCE), 
                   key = "train.hex")

# Build the final model with the optimal parameters
GBM_mod <- h2o.gbm(x = independent,
                    y = dependent,
                    distribution = "gaussian",
                    data = trainH2O,
                    n.trees = params$n.trees
                    interaction.depth = params$interaction.depth,
                    shrinkage = params$shrinkage)

# Save the model
h2o.saveModel(object = GBM_mod, dir = "E:/MSc thesis/Travel time prediction/Models", name = "Final_GBM", force = TRUE)





####
library(xtable)
library(stringr)
results <- read.csv("./Travel time prediction/Models/gbm_cv_results.csv", header = TRUE)
ntree_pos <- 4
depth_pos <- 6
shrinkage_pos <- 8

params_string <- sapply(str_split(results$model_name, "_"), function(x) data.frame(depth = as.integer(x[depth_pos]), 
                                                                                   shrinkage = as.numeric(x[shrinkage_pos])), simplify = FALSE)
params_df <- cbind(do.call(rbind, params_string), errors = results$errors)
print(xtable(params_df, digits = 5), include.rownames=FALSE)

results <- read.table("./Travel time prediction/Models/RF_CV_results.csv", header = TRUE, stringsAsFactors = FALSE)
splitted_names <- str_split(results$model, "_")

nmin <- sapply(splitted_names, function(x) x[6])
nmin_vals <- sapply(str_replace(nmin, "nmin", ""), as.integer)

depth <- sapply(splitted_names, function(x) x[4])
depth_vals <- sapply(str_replace(depth, "depth", ""), as.integer)

mtries <- sapply(splitted_names, function(x) x[5])
mtries_vals <- sapply(str_replace(mtries, "mtries", ""), as.integer)

params_df <- data.frame(nmin = nmin_vals,
                        depth = depth_vals,
                        mtries = mtries_vals,
                        RMSLE = results$error)
print(xtable(params_df, digits = 4), include.rownames = FALSE)


# Variable importance plot
library(ggplot2)
library(gridExtra)
varimp_gbm <- read.csv("./Travel time prediction/Models/early_stopping_gbm_varimp.csv", header = TRUE)
varimp_gbm$variable <- factor(varimp_gbm$variable, 
                              levels = c("END_CELL", "START_CELL", "HOUR", "WEEK", "WDAY", "MINUTE"),
                              labels = c("Destination cell", "Start cell", "Minute of hour", "Week of year", "Hour of day", "Day of week"))

varimp_plot_gbm <- varimp_gbm %>%
  ggplot(aes(x = reorder(variable, -percentage), y = percentage)) +
  geom_bar(stat = "identity", colour = "black", fill = "white") +
  scale_y_continuous(breaks = seq(from = 0, to = 0.55, by = 0.05), limits = c(NA, 0.55)) + 
  labs(x = "Variable", y = "Percentage of total importance") 

varimp_rf <- read.csv("./Travel time prediction/Models/early_stopping_rf_varimp.csv", header = TRUE)
varimp_rf$variable <- factor(varimp_rf$variable, 
                             levels = c("END_CELL", "START_CELL", "HOUR", "WEEK", "WDAY", "MINUTE"),
                             labels = c("Destination cell", "Start cell", "Minute of hour", "Week of year", "Hour of day", "Day of week"))

varimp_plot_rf <- varimp_rf %>%
  ggplot(aes(x = reorder(variable, -percentage), y = percentage)) +
  geom_bar(stat = "identity", colour = "black", fill = "white") +
  scale_y_continuous(breaks = seq(from = 0, to = 0.55, by = 0.05), limits = c(NA, 0.55)) + 
  labs(x = "Variable", y = "Percentage of total importance") 

pdf("./Visualizations/variable_importance.pdf", height = 6, width = 14)
grid.arrange(varimp_plot_rf, varimp_plot_gbm, ncol = 2)
dev.off()

##################################################
# OLD
##################################################

# Get model names
model_names <- data.frame(model_name = character(n_models),
                          errors = errors,
                          stringsAsFactors = FALSE)
for(i in seq_len(n_models)) {
  params <- models[[i]]@model$params
  
  model_names[i, "model_name"] = paste0("GBM_", i, 
                                        "_ntree_", params$n.trees,
                                        "_depth_", params$interaction.depth,
                                        "_shrinkage_", params$shrinkage)
  
}

# Do the early stopping training
early_stopping_error <- numeric(n_max)

for(i in seq(from = 112, to = n_max)) {
  # Train the model
  set.seed(1234)
  gbm_mod <- h2o.gbm(x = independent,
                     y = dependent,
                     distribution = "gaussian",
                     data = trainH2O,
                     ntrees = i,
                     interaction.depth = params$interaction.depth,
                     shrinkage = params$shrinkage)
  
  # Predict the holdout
  holdout_pred <- h2o.predict(object = gbm_mod, newdata = holdoutH2O)
  
  # Get error
  early_stopping_error[i] <- RMSLE(preds = exp(as.data.frame(holdout_pred)), obs = train[rows_holdout, ]$DURATION)
}

write.csv(early_stopping_error, "./Travel time prediction/Models/gbm_early_stopping_errors_R.csv", row.names = FALSE)


# In-sample error
GBM_preds <- h2o.predict(object = GBM_mod, newdata = trainH2O)
mse <- RMSLE(exp(as.data.frame(GBM_preds)), train$DURATION)
print(mse)