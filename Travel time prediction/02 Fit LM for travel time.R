# 03 Fit LM for travel time.R

# This script fits a linear model where the response is the travel time of a
# taxi trip. It uses the output of the script 01c Format training data for LM.py
# as input for the model.

# Load relevant libraries
library(data.table)
library(dplyr)
library(ggplot2)

# Set working directory
setwd("E:/MSc thesis")

# Import the data
train_vars = c("START_POINT_LON", "START_POINT_LAT", "END_POINT_LON", "END_POINT_LAT", "START_CELL", "END_CELL", "HOUR", "WDAY", "WEEK", "DISTANCE", "DURATION")
train <- as_data_frame(fread("./Processed data/train_lm.csv", header = TRUE, sep = ",", select = train_vars))
val <- as_data_frame(fread("./Processed data/val_lm.csv", header = TRUE, colClasses = c(TRIP_ID = "character")))

# Make sure HOUR, WDAY and WEEK are factors
train$HOUR <- as.factor(train$HOUR)
train$WDAY <- factor(train$WDAY, levels = 0:6, labels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
train$WEEK <- as.factor(train$WEEK)

val$HOUR <- as.factor(val$HOUR)
val$WDAY <- factor(val$WDAY, levels = 0:6, labels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
val$WEEK <- as.factor(val$WEEK)

# Transform the start cell and destination cell to a factor
train$START_CELL <- factor(train$START_CELL, levels = 1:7500)
train$END_CELL <- factor(train$END_CELL, levels = 1:7500)

val$START_CELL <- factor(val$START_CELL, levels = 1:7500)
val$TRUNC_CELL <- factor(val$TRUNC_CELL, levels = 1:7500)
val$END_CELL <- factor(val$END_CELL, levels = 1:7500)

# Transform the response so that its distribution is more alike a normal distribution
# train$DURATION <- log(train$DURATION)

# Fit an OLS model
travel_lm <- lm(formula = log(DURATION) ~ START_CELL + END_CELL + START_CELL*END_CELL + HOUR + WDAY + WEEK,
                data = train)

# Summary of the model
summary(travel_lm)

# As expected, all the explanatory variables are highly significant

# Residual analysis
par(mfrow = c(2,2)) 
plot(travel_lm)

# Load the posterior distributions of the partial trips
post_val <- as_data_frame(fread("./Processed data/val_posteriors.csv", header = TRUE, sep = ",", nrows = 10))

# Get the coefficients of the model
travel_lm_coef <- coef(travel_lm)

# Predict the travel time for each partial trip
for(trip in 1:nrows(val)) {
  # Get the necessary predictors from the validation set
  trip_expl <- val %>% 
    filter(TRIP_ID == val[trip]$TRIP_ID) %>%
    select(-DURATION)
  
  # Get the destination posterior for the trip
  trip_dest <- post_val %>%
    filter(TRIP_ID = post_val[trip]$TRIP_ID)
  
  # Build the model matrix for the computation of the predictions
  trip_expl_mat <- model.matrix(formula(travel_lm), cbind(trip_expl, trip_dest[c("PROB")])
  
  prediction <- coef %*% trip_expl_mat 
  
}
