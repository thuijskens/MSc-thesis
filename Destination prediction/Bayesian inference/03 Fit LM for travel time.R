# 03 Fit LM for travel time.R

# This script fits a linear model where the response is the travel time of a
# taxi trip. It uses the output of the script 01c Format training data for LM.py
# as input for the model.#

# Load relevant libraries
library(data.table)
library(dplyr)
library(ggplot2)

# Set working directory
setwd("E:/MSc thesis")

# Import the data
train <- as_data_frame(fread("./Processed data/train_lm.csv", header = TRUE, sep = ","))
val <- as_data_frame(fread("./Processed data/val_lm.csv", header = TRUE, sep = ",", nrows = 10))

# Make sure HOUR, WDAY and WEEK are factors
train$HOUR <- as.factor(train$HOUR)
train$WDAY <- as.factor(train$WDAY)
train$WEEK <- as.factor(train$WEEK)

# Transform the response so that its distribution is more alike a normal distribution
train$DURATION <- log(train$DURATION)

# Fit an OLS model
travel_lm <- lm(formula = DURATION ~ HOUR + WDAY + WEEK + DISTANCE,
                data = train)

# Summary of the model
summary(travel_lm)

# As expected, all the explanatory variables are highly significant

# Residual analysis
par(mfrow = c(2,2)) 
plot(travel_lm)

# Load the posterior distributions of the partial trips
post_val <- as_data_frame(fread("./Processed data/val_posteriors.csv", header = TRUE, sep = ",", nrows = 10))

# Get the coefficients of the miodel
travel_lm_coef <- coef(travel_lm)

# Predict the travel time for each partial trip
for(trip in 1:nrows(val)) {
  trip_expl <- val %>% filter(TRIP_ID == val[trip]$TRIP_ID)
}
