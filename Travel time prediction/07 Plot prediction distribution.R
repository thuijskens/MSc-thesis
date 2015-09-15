# Load relevant libraries
library(data.table)
library(dplyr)
library(ggplot2)
library(h2o)
library(bit64)
library(glmnet)
library(xtable)
library(gridExtra)

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

# Start H2O
localH2O <- h2o.init(nthreads = 6)

# Load the models in H2O
rf_model <- h2o.loadModel("./Travel time prediction/Models/FINAL_RF_150909/DRF_model_R_1441878627938_8", conn = localH2O)

# Load appropriate posterior distribution
alpha <- 0.8
beta <- 0.2
filepath <- sprintf("./Processed data/Cross-validation efficient driving likelihood/alpha = %.1f and beta = %.1f/val_posteriors.csv", alpha, beta)

val_posterior <- as_data_frame(fread(filepath)) %>% filter(TRIP_ID == "1372937349620000138") # all predictions for one observation
val_posterior$DEST_CELL <- factor(val_posterior$DEST_CELL, levels = 1:7500)

# Get validation data for the first trip
destination <- val_posterior$DEST_CELL
prob <- val_posterior$PROB
val_posterior <- as_data_frame(data.frame(destination, prob, val %>% filter(TRIP_ID == "1372937349620000138"))) %>%
  select(-(TRIP_ID:START_CELL), END_CELL = destination, PROB = prob, START_CELL = TRUNC_CELL, -END_CELL, -TRUNC_DISTANCE)

val_post_H2O <- as.h2o(val_posterior)

# Predict the duration for each cell
val_preds <- val_posterior$TRUNC_DURATION + exp(as.data.frame(predict(rf_model, newdata = val_post_H2O)))

# Add the predictions to the original data
val_posterior <- cbind(val_posterior, val_preds) %>% as_data_frame

# Compute the density
pred_density <- with(val_posterior, density(x = predict, weights = ifelse(PROB > 0.0000000001, PROB, 0)))

# Actual prediction
prediction <- val_posterior %>%
  summarise(weighted.mean(x = predict, w = PROB))

pdf("./Visualizations/prediction_density.pdf", width = 6, height = 4)
data.frame(time = pred_density$x / 60, prob = pred_density$y) %>%
  ggplot(aes(time, prob)) +
  geom_line() +
  geom_vline(aes(xintercept = 1380 / 60), color = "red", linetype = 2) +
  geom_vline(aes(xintercept = 1361.314 / 60), color = "blue", linetype = 2) +
  scale_y_continuous(breaks = seq(from = 0, to = 0.02, by = 0.0025)) +
  scale_x_continuous(breaks = seq(from = 0, to = 2000/60, by = 1)) +
  labs(x = "Travel time (min)", y = "Density", title = "Posterior distribution for trip 1372937349620000138") +
  theme(legend.title = element_blank())
dev.off()


#########################################################
# Plot of RMSLE vs trip length
#########################################################

alpha <- 0.8
beta <- 0.2
filepath <- sprintf("./Processed data/Cross-validation efficient driving likelihood/alpha = %.1f and beta = %.1f/val_predictions.csv", alpha, beta)

# RMSLE
RMSLE <- function(preds, obs) {
  return(sqrt(mean((log(preds + 1) - log(obs + 1))^2)))
}

val_rmsle <- as_data_frame(read.csv(filepath, header = TRUE))

rmsle_val <- val_rmsle %>%
  mutate(TRIP_LENGTH = floor(OBS_DURATION/60)) %>%
  group_by(TRIP_LENGTH = cut(TRIP_LENGTH, breaks = seq(from = 0, to = 90, by = 5))) %>%
  summarise(RMSLE_RF = RMSLE(TOTAL_DURATION_RF, OBS_DURATION),
            RMSLE_GBM = RMSLE(TOTAL_DURATION_GBM, OBS_DURATION),
            RMSLE_RIDGE = RMSLE(TOTAL_DURATION_RIDGE, OBS_DURATION),
            nobs = n())

rmsle_plot <- rmsle_val %>%
  filter(!is.na(TRIP_LENGTH)) %>%
  ggplot(aes(x = TRIP_LENGTH)) +
  geom_point(aes(y = RMSLE_RF, colour = "RF"), size = 2.5) +
  geom_point(aes(y = RMSLE_GBM, colour = "GBM"), size = 2.5) +
  #geom_point(aes(y = RMSLE_RIDGE, colour = "Stacked Ridge")) +
  scale_y_continuous(limits = c(0, 1.5), breaks = seq(from = 0, to = 1.25, by = 0.25)) +
  labs(x = "Trip duration (min)", y = "RMSLE", colour = "Model") +
  theme(legend.position="top")

n_plot <- rmsle_val %>%
  filter(!is.na(TRIP_LENGTH)) %>%
  ggplot(aes(x = TRIP_LENGTH)) + 
  geom_bar(aes(y = nobs), stat = "identity", fill = "white", color = "black")  + 
  scale_y_continuous(breaks = seq(from = 0, to = 800, by = 100)) +
  labs(x = "Trip duration (min)", y = "Number of observations")

pdf("./Visualizations/rmsle_validation.pdf", width = 10, height = 8)
grid.arrange(rmsle_plot, n_plot, nrow = 2)
dev.off()

#########################################################
# Plot of top 5 destination missclassification rate
#########################################################
alpha <- 0.8
beta <- 0.2
filepath <- sprintf("./Processed data/Cross-validation efficient driving likelihood/alpha = %.1f and beta = %.1f/val_posteriors.csv", alpha, beta)

post_val <- as_data_frame(fread(filepath, header = TRUE, colClasses = c(TRIP_ID = "bit64")))

top5_agg <- post_val %>%
  group_by(TRIP_ID) %>%
  filter(row_number(-PROB) <= 75) %>%
  left_join(val %>% 
              select(TRIP_ID, END_CELL, OBS_DURATION = DURATION), 
            by = c("TRIP_ID" = "TRIP_ID")) %>%
  summarise(CORRECT_CLASS = ifelse(sum(DEST_CELL == END_CELL) > 0, 1, 0),
            TRIP_LENGTH = floor(mean(OBS_DURATION)/60)) 

misclass_plot <- top5_agg %>%
  group_by(TRIP_LENGTH = cut(TRIP_LENGTH, breaks = seq(from = 0, to = 90, by = 5))) %>%
  summarise(MISCLASS = 1 - mean(CORRECT_CLASS)) %>%
  filter(!is.na(TRIP_LENGTH)) %>%
  ggplot(aes(x = TRIP_LENGTH)) +
  geom_bar(aes(y = MISCLASS), stat = "identity", fill = "white", color = "black")  + 
  scale_y_continuous(breaks = seq(from = 0, to = 1, by = 0.2)) +
  labs(x = "Trip duration (min)", y = "Wrongly predicted destination (%)")

pdf("./Visualizations/misclassification_plot.pdf", width = 10, height = 4)
print(misclass_plot)
dev.off()

pdf("./Visualizations/total_rmsle_plot.pdf", width = 10, height = 10)
grid.arrange(rmsle_plot, n_plot, misclass_plot, nrow = 3)
dev.off()

# Can make the same plot for random walk
val_posteriors_rw <- as_data_frame(fread("./Processed data/val_posteriors_random_walk.csv", header = TRUE, sep = ",", colClasses = c(TRIP_ID = "bit64")))
val_posteriors_2nd_rw <- as_data_frame(fread("./Processed data/val_posteriors_2nd_order_random_walk.csv", header = TRUE, sep = ",", colClasses = c(TRIP_ID = "bit64")))

top5_agg_rw <- val_posteriors_2nd_rw %>%
  group_by(TRIP_ID) %>%
  filter(row_number(-PROB) <= 75) %>%
  left_join(val %>% 
              select(TRIP_ID, END_CELL, OBS_DURATION = DURATION), 
            by = c("TRIP_ID" = "TRIP_ID")) %>%
  summarise(CORRECT_CLASS = ifelse(sum(DEST_CELL == END_CELL) > 0, 1, 0),
            TRIP_LENGTH = floor(mean(OBS_DURATION)/60)) 

misclass_plot_rw <- top5_agg_rw %>%
  group_by(TRIP_LENGTH = cut(TRIP_LENGTH, breaks = seq(from = 0, to = 90, by = 5))) %>%
  summarise(MISCLASS = 1 - mean(CORRECT_CLASS)) %>%
  filter(!is.na(TRIP_LENGTH)) %>%
  ggplot(aes(x = TRIP_LENGTH)) +
  geom_bar(aes(y = MISCLASS), stat = "identity", fill = "white", color = "black")  + 
  scale_y_continuous(breaks = seq(from = 0, to = 1, by = 0.1)) +
  labs(x = "Trip duration (min)", y = "Misclassification rate")

