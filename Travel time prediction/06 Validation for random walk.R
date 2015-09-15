# 06 Validation for random walk.R

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
library(glmnet)

# Set working directory
setwd("E:/MSc thesis")

# Import the data
val <- as_data_frame(fread("./Processed data/val_lm.csv", header = TRUE, colClasses = c(TRIP_ID = "bit64")))

# Make sure HOUR, WDAY and WEEK are factors
val$HOUR <- as.factor(val$HOUR)
val$WDAY <- factor(val$WDAY, levels = 0:6, labels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
val$WEEK <- as.factor(val$WEEK)

# Transform the start cell and destination cell to a factor
val$START_CELL <- factor(val$START_CELL, levels = 1:7500)
val$TRUNC_CELL <- factor(val$TRUNC_CELL, levels = 1:7500)
val$END_CELL <- factor(val$END_CELL, levels = 1:7500)

# Store the number of unique TRIP_ID's (#trips) in the validation set for sanity check
unique_ids <- length(unique(val$TRIP_ID))

# Initialize h2o server
localH2O <- h2o.init(nthreads = 6)

# Store the data as a h2o instance
trainH2O <- as.h2o(localH2O, 
                   train %>% 
                     select(-TRIP_ID, -DISTANCE), 
                   key = "train.hex")

# Load the randomForest model
GBM_mod <- h2o.loadModel(path = "./Travel time prediction/Models/FINAL_GBM_150909/gbm_val", conn = localH2O)
RF_mod <- h2o.loadModel("./Travel time prediction/Models/FINAL_RF_150909/DRF_model_R_1441878627938_8", conn = localH2O)

# RMSLE
RMSLE <- function(preds, obs) {
  return(sqrt(mean((log(preds + 1) - log(obs + 1))^2)))
}

# Load the validation files
val_posteriors_rw <- as_data_frame(fread("./Processed data/val_posteriors_random_walk.csv", header = TRUE, sep = ",", colClasses = c(TRIP_ID = "bit64")))
val_posteriors_2nd_rw <- as_data_frame(fread("./Processed data/val_posteriors_2nd_order_random_walk.csv", header = TRUE, sep = ",", colClasses = c(TRIP_ID = "bit64")))

# Sanity check
if(length(unique(val_posteriors_rw$TRIP_ID)) != unique_ids) {
  stop("Number of trip_id's do not match for first-order model")
} else if(length(unique(val_posteriors_2nd_rw$TRIP_ID)) != unique_ids) {
  stop("Number of trip_id's do not match for second-order model")
} else {
  print("Passed sanity check")
}

# Make sure the destination is a factor variable
val_posteriors_rw$DEST_CELL <- factor(val_posteriors_rw$DEST_CELL, levels = 1:7500)
val_posteriors_2nd_rw$DEST_CELL <- factor(val_posteriors_2nd_rw$DEST_CELL, levels = 1:7500)

# Prepare the validation dataset for easy prediction. Note:
# - We want to predict the additional time, hence we set the start cell equal to the trunc_cell field
print("Joining posterior distributions on validation set")
val_pred_rw <- val %>% 
  left_join(val_posteriors_rw %>%
              select(TRIP_ID, DEST_CELL, PROB), 
            by = c("TRIP_ID" = "TRIP_ID")) %>%
  select(TRIP_ID, START_CELL = TRUNC_CELL, END_CELL = DEST_CELL, PROB, HOUR, WDAY, WEEK, TRUNC_DURATION, DURATION)

val_pred_2nd_rw <- val %>% 
  left_join(val_posteriors_2nd_rw %>%
              select(TRIP_ID, DEST_CELL, PROB), 
            by = c("TRIP_ID" = "TRIP_ID")) %>%
  select(TRIP_ID, START_CELL = TRUNC_CELL, END_CELL = DEST_CELL, PROB, HOUR, WDAY, WEEK, TRUNC_DURATION, DURATION)

# Sanity check
if(length(unique(val_pred_rw$TRIP_ID)) != unique_ids | dim(val_pred_rw)[1] != dim(val_posteriors_rw)[1]) {
  stop("Join failed for first-order model")
} else if(length(unique(val_pred_2nd_rw$TRIP_ID)) != unique_ids | dim(val_pred_2nd_rw)[1] != dim(val_posteriors_2nd_rw)[1]) {
  stop("Join failed for second-order model")
} else {
  print("Passed sanity check")
}

# Store this dataset in H2O
print("Loading data in H2O")
valH2O_rw <- as.h2o(localH2O, val_pred_rw, destination_frame = "val_pred_rw.hex")
valH2O_2nd_rw <- as.h2o(localH2O, val_pred_2nd_rw, destination_frame = "val_pred_2nd_rw.hex")

# Predict the validation set with the random forest
print("Predicting validation set")
RF_preds_rw <- h2o.predict(object = RF_mod, newdata = valH2O_rw)
RF_preds_2nd_rw <- h2o.predict(object = RF_mod, newdata = valH2O_2nd_rw)

# Predict the validation set with the GBM
print("Predicting validation set")
GBM_preds_rw <- h2o.predict(object = GBM_mod, newdata = valH2O_rw)
GBM_preds_2nd_rw <- h2o.predict(object = GBM_mod, newdata = valH2O_2nd_rw)

# Predict the validation set with the stacked Ridge regression
load("./Travel time prediction/Models/stacked_ridge_constrained.RData")
lambda.1se <- stack_model_glmnet$lambda.1se

# Compute the predictions from the ridge regression
ridge_preds_rw <- predict(object = stack_model_glmnet_full, 
                       newx = data.frame(GBM_pred = as.data.frame(GBM_preds_rw),
                                         RF_pred = as.data.frame(RF_preds_rw)) %>% as.matrix(),
                       s = lambda.1se)

ridge_preds_rw2 <- predict(object = stack_model_glmnet_full, 
                           newx = data.frame(GBM_pred = as.data.frame(GBM_preds_2nd_rw),
                                             RF_pred = as.data.frame(RF_preds_2nd_rw)) %>% as.matrix(),
                           s = lambda.1se)

# Obtain the total travel time predictions for the validation set
val_pred_rw <- val_pred_rw %>%
  cbind(data.frame(predict = as.data.frame(RF_preds_rw), predict.1 = as.data.frame(GBM_preds_rw), ridge_preds = ridge_preds_rw)) %>%
  group_by(TRIP_ID) %>%
  summarise(OBS_DURATION = mean(DURATION), 
            TRUNC_DURATION = mean(TRUNC_DURATION), 
            PREDICT_DURATION_RF = exp(weighted.mean(predict, PROB)),
            PREDICT_DURATION_GBM = exp(weighted.mean(predict.1, PROB)),
            PREDICT_DURATION_RIDGE = exp(weighted.mean(X1, PROB))) %>%
  mutate(TOTAL_DURATION_RF = TRUNC_DURATION + PREDICT_DURATION_RF,
         TOTAL_DURATION_GBM = TRUNC_DURATION + PREDICT_DURATION_GBM,
         TOTAL_DURATION_RIDGE = TRUNC_DURATION + PREDICT_DURATION_RIDGE)

val_pred_2nd_rw <- val_pred_2nd_rw %>%
  cbind(data.frame(as.data.frame(RF_preds_2nd_rw), as.data.frame(GBM_preds_2nd_rw), ridge_preds = ridge_preds_rw2)) %>%
  group_by(TRIP_ID) %>%
  summarise(OBS_DURATION = mean(DURATION), 
            TRUNC_DURATION = mean(TRUNC_DURATION), 
            PREDICT_DURATION_RF = exp(weighted.mean(predict, PROB)),
            PREDICT_DURATION_GBM = exp(weighted.mean(predict.1, PROB)),
            PREDICT_DURATION_RIDGE = exp(weighted.mean(X1, PROB))) %>%
  mutate(TOTAL_DURATION_RF = TRUNC_DURATION + PREDICT_DURATION_RF,
         TOTAL_DURATION_GBM = TRUNC_DURATION + PREDICT_DURATION_GBM,
         TOTAL_DURATION_RIDGE = TRUNC_DURATION + PREDICT_DURATION_RIDGE)


if(dim(val_pred_rw)[1] != dim(val)[1]) {
  stop("Prediction set size does not match validation set size for first-order model")
} else if(dim(val_pred_2nd_rw)[1] != dim(val)[1]) {
  stop("Prediction set size does not match validation set size for second-order model")
} else {
  print("Passed sanity check, writing to file")
}

write.table(x = val_pred_rw, file = "./Processed data/val_predictions_rw.csv", sep = ",", row.names = FALSE)
write.table(x = val_pred_2nd_rw, file = "./Processed data/val_predictions_2nd_rw.csv", sep = ",", row.names = FALSE)

# Compute RMSLE
RMSLE_rw <- c(RMSLE(val_pred_rw$TOTAL_DURATION_RF, val_pred_rw$OBS_DURATION),
              RMSLE(val_pred_rw$TOTAL_DURATION_GBM, val_pred_rw$OBS_DURATION),
              RMSLE(val_pred_rw$TOTAL_DURATION_RIDGE, val_pred_rw$OBS_DURATION))

RMSLE_rw2 <- c(RMSLE(val_pred_2nd_rw$TOTAL_DURATION_RF, val_pred_2nd_rw$OBS_DURATION),
               RMSLE(val_pred_2nd_rw$TOTAL_DURATION_GBM, val_pred_2nd_rw$OBS_DURATION),
               RMSLE(val_pred_2nd_rw$TOTAL_DURATION_RIDGE, val_pred_2nd_rw$OBS_DURATION))

rmsle_df <- as.data.frame(rbind(RMSLE_rw, RMSLE_rw2))
row.names(rmsle_df) <- c("First-order random walk", "Second-order random walk")
colnames(rmsle_df) <- c("RF", "GBM", "Ridge")
write.table(rmsle_df, file = "./Processed data/val_rmsle_rw.csv", row.names = TRUE, sep = ",")

library(xtable)
print(xtable(rmsle_df, digits = c(1, 4, 4, 4)), include.rownames = TRUE)
