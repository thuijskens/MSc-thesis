# 03 Random forest for travel time.R

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
library(xtable)

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

# Load the randomForest model
GBM_mod <- h2o.loadModel(path = "./Travel time prediction/Models/FINAL_GBM_150909/gbm_val", conn = localH2O)
RF_mod <- h2o.loadModel("./Travel time prediction/Models/FINAL_RF_150909/DRF_model_R_1441878627938_8", conn = localH2O)

# RMSLE
RMSLE <- function(preds, obs) {
  return(sqrt(mean((log(preds + 1) - log(obs + 1))^2)))
}

# Define the cut-off value for the posterior distribution
epsilon <- 0.0001

# Build the sequence of alpha's
alpha_values <- seq(from = 0.0, to = 1.0, by = 0.2)
rmsle_df <- data.frame(alpha = numeric(0), beta = numeric(0), rmsle_rf = numeric(0), rmsle_gbm = numeric(0))
if(!file.exists("./Travel time prediction/Models/efficient_driving_rmsle.csv")) {
  write.table(rmsle_df, "./Travel time prediction/Models/efficient_driving_rmsle.csv", col.names = TRUE, row.names = FALSE)
}

for(alpha in alpha_values) {
  # Build the sequence of corresponding beta's
  beta_values <- seq(from = 0.0, to = 1 - alpha, by = 0.2)
  for(beta in beta_values) {
    # Construct the filepath and load the corresponding validation file
    filepath <- sprintf("./Processed data/Cross-validation efficient driving likelihood/alpha = %.1f and beta = %.1f/val_posteriors.csv", alpha, beta)

    print(paste("Loading file", filepath))
    val_posteriors <- as_data_frame(fread(filepath, header = TRUE, sep = ",", colClasses = c(TRIP_ID = "bit64")))
    
    # Filter out the candidate positions with a very low probability
    val_posteriors <- val_posteriors %>% filter(PROB >= epsilon)
    
    # Sanity check
    if(length(unique(val_posteriors$TRIP_ID)) != unique_ids) {
      print(paste("Error: number of trip_id's do not match for alpha =", alpha, "and beta =", beta))
      write.table(cbind(alpha, beta, 0, 0), "./Travel time prediction/Models/efficient_driving_rmsle.csv", col.names = FALSE, row.names = FALSE, append = TRUE)
      next
    } else {
      print("Passed sanity check")
    }
    
    # Make sure the destination is a factor variable
    val_posteriors$DEST_CELL <- factor(val_posteriors$DEST_CELL, levels = 1:7500)
    
    # Prepare the validation dataset for easy prediction. Note:
    # - We want to predict the additional time, hence we set the start cell equal to the trunc_cell field
    print("Joining posterior distributions on validation set")
    val_pred <- val %>% 
      left_join(val_posteriors %>%
                  select(TRIP_ID, DEST_CELL, PROB), 
                by = c("TRIP_ID" = "TRIP_ID")) %>%
      select(TRIP_ID, START_CELL = TRUNC_CELL, END_CELL = DEST_CELL, PROB, HOUR, WDAY, WEEK, TRUNC_DURATION, DURATION) %>%
      filter(!is.na(END_CELL))
    
    # Sanity check
    if(length(unique(val_pred$TRIP_ID)) != unique_ids | dim(val_pred)[1] != dim(val_posteriors)[1]) {
      print(paste("Join failed for alpha =", alpha, "and beta =", beta))
      write.table(cbind(alpha, beta, 0, 0), "./Travel time prediction/Models/efficient_driving_rmsle.csv", col.names = FALSE, row.names = FALSE, append = TRUE)
      next
    } else {
      print("Passed sanity check")
    }
    
    # Store this dataset in H2O
    print("Loading data in H2O")
    valH2O <- as.h2o(conn = localH2O, object = val_pred)
    
    # Predict the validation set with the random forest
    print("Predicting validation set")
    RF_preds <- h2o.predict(object = RF_mod, newdata = valH2O)
    GBM_preds <- h2o.predict(object = GBM_mod, newdata = valH2O)
    
    # Obtain the total travel time predictions for the validation set
    val_pred <- val_pred %>%
      cbind(data.frame(RF_pred = as.data.frame(RF_preds), GBM_pred = as.data.frame(GBM_preds))) %>%
      group_by(TRIP_ID) %>%
      summarise(OBS_DURATION = mean(DURATION), 
                TRUNC_DURATION = mean(TRUNC_DURATION), 
                PREDICT_DURATION_RF = exp(weighted.mean(predict, PROB)),
                PREDICT_DURATION_GBM = exp(weighted.mean(predict.1, PROB))) %>%
      mutate(TOTAL_DURATION_RF = TRUNC_DURATION + PREDICT_DURATION_RF,
             TOTAL_DURATION_GBM = TRUNC_DURATION + PREDICT_DURATION_GBM)
    
    if(dim(val_pred)[1] != dim(val)[1]) {
      print(paste("Prediction set size does not match validation set size"))
      write.table(cbind(alpha, beta, 0, 0), "./Travel time prediction/Models/efficient_driving_rmsle.csv", col.names = FALSE, row.names = FALSE, append = TRUE)
      next
    } else {
      print("Passed sanity check, writing to file")
    }
    
    write.table(x = val_pred, 
                file = sprintf("./Processed data/Cross-validation efficient driving likelihood/alpha = %.1f and beta = %.1f/val_predictions.csv", alpha, beta), 
                sep = ",", row.names = FALSE)
    
    # Compute RMSLE
    
    write.table(data.frame(alpha, beta, RMSLE_rf = RMSLE(preds = val_pred$TOTAL_DURATION_RF, obs = val_pred$OBS_DURATION), RMSLE_gbm = RMSLE(preds = val_pred$TOTAL_DURATION_GBM, obs = val_pred$OBS_DURATION)),
                "./Travel time prediction/Models/efficient_driving_rmsle.csv", col.names = FALSE, row.names = FALSE, append = TRUE)
    
    print(paste("Processed predictions for alpha =", alpha, "and beta =", beta))
  }
}

# Load the ridge regression 
load("./Travel time prediction/Models/stacked_ridge.RData")
lambda.1se <- 0.0009284145

load("./Travel time prediction/Models/stacked_ridge_constrained.RData")
lambda.1se <- stack_model_glmnet$lambda.1se

# Iterate through the alpha and beta values again
for(alpha in alpha_values) {
  beta_values <- seq(from = 0.0, to = 1 - alpha, by = 0.2)
  for(beta in beta_values) {
    # Load in the predictions
    filename <- sprintf("./Processed data/Cross-validation efficient driving likelihood/alpha = %.1f and beta = %.1f/val_predictions.csv", alpha, beta)
    if(file.exists(filename)) {
      val_preds <- as_data_frame(fread(filename, header = TRUE))
      
      # Compute the predictions from the ridge regression
      ridge_preds <- predict(object = stack_model_glmnet_full, 
                             newx = val_preds %>%
                               mutate(GBM_pred = log(PREDICT_DURATION_GBM), RF_pred = log(PREDICT_DURATION_RF)) %>%
                               select(GBM_pred, RF_pred) %>%
                               as.matrix(), 
                             s = lambda.1se)
    
      # Add these predictions to the data and save
      val_preds <- val_preds %>% 
        mutate(PREDICT_DURATION_RIDGE = exp(ridge_preds),
               TOTAL_DURATION_RIDGE = TRUNC_DURATION + PREDICT_DURATION_RIDGE)
      
      write.csv(val_preds, filename, row.names = FALSE)
    }
    print(paste("Processed alpha =", alpha, "and beta =", beta))
  }
}

# Compute the RMSLE
rmsle_df <- data.frame(alpha = numeric(0), beta = numeric(0), rmsle_ridge = numeric(0))
for(alpha in alpha_values) {
  beta_values <- seq(from = 0.0, to = 1 - alpha, by = 0.2)
  for(beta in beta_values) {
    filename <- sprintf("./Processed data/Cross-validation efficient driving likelihood/alpha = %.1f and beta = %.1f/val_predictions.csv", alpha, beta)
    if(file.exists(filename)) {
      val_preds <- as_data_frame(fread(filename, header = TRUE))
      RMSLE_ridge <- RMSLE(preds = val_preds$TOTAL_DURATION_RIDGE, obs = val_preds$OBS_DURATION)
      
      # save RMSLE
      rmsle_df <- rmsle_df %>% rbind(c(alpha, beta, RMSLE_ridge))
    } else {
      rmsle_df <- rmsle_df %>% rbind(c(alpha, beta, 0))
    }
  }
}

colnames(rmsle_df) <- c("alpha", "beta", "rmsle_ridge")
# Load the RMSLE file
rmsle_file <- read.csv("./Travel time prediction/Models/efficient_driving_rmsle.csv", header = TRUE)

# join the ridge RMSLE df on it and save
rmsle_final <- cbind(rmsle_file, rmsle_ridge_constrained = rmsle_df$rmsle_ridge)

write.csv(rmsle_final, "./Travel time prediction/Models/efficient_driving_rmsle.csv",row.names = FALSE)
print(xtable(rmsle_final, digits = c(1, 1, 1, 4, 4, 4)), include.rownames = FALSE)