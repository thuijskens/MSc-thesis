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

# Set working directory
setwd("E:/MSc thesis")

# Import the data
train_vars = c("TRIP_ID", "DURATION", "START_CELL", "END_CELL", "HOUR", "WDAY", "WEEK", "DISTANCE")
train <- as_data_frame(fread("./Processed data/train_lm.csv", header = TRUE, sep = ",", select = train_vars, nrows = 1000, colClasses = c(TRIP_ID = "character")))
val <- as_data_frame(fread("./Processed data/val_lm.csv", header = TRUE, colClasses = c(TRIP_ID = "character")))
val_posteriors <- as_data_frame(fread("./Processed data/val_posteriors.csv", header = TRUE, sep = ",", nrows = 10, colClasses = c(TRIP_ID = "character")))

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

val_posteriors$DEST_CELL <- factor(val_posteriors$DEST_CELL, levels = 1:7500)

# Take natural logarithm of duration
train$DURATION = log(train$DURATION)

# Prepare the validation dataset for easy prediction. Note:
# - We want to predict the additional time, hence we set the start cell equal to the trunc_cell field
val <- val %>% 
  left_join(val_posteriors %>%
              select(TRIP_ID, DEST_CELL, PROB), 
            by = c("TRIP_ID" = "TRIP_ID")) %>%
  select(TRIP_ID, START_CELL = TRUNC_CELL, END_CELL = DEST_CELL, PROB, HOUR, WDAY, WEEK, TRUNC_DURATION = log(TRUNC_DURATION))

# Initialize h2o server
localH2O <- h2o.init(nthreads = 6)

# Store the data as a h2o instance
trainH2O <- as.h2o(localH2O, 
                   train %>% 
                     select(-TRIP_ID, -DISTANCE), 
                   key = "train")
valH2O <- as.h2o(localH2O, val, key = "val")

# Fit a random forest on the data
independent <- c("START_CELL", "END_CELL", "HOUR", "WDAY", "WEEK")
dependent <- c("DURATION")
RF_mod <- h2o.randomForest(x = independent, 
                           y = dependent, 
                           data = trainH2O, 
                           classification = FALSE,
                           type = "Bigdata")

# Predict the validation set with the random forest
RF_preds <- h2o.predict(object = RF_mod, newdata = valH2O)

# Obtain the total travel time predictions for the validation set
val <- val %>%
  cbind(as.data.frame(RF_preds)) %>%
  group_by(TRIP_ID, START_CELL, HOUR, WDAY, WEEK, exp(TRUNC_DURATION)) %>%
  summarise(PREDICT_DURATION = exp(weighted.mean(predict, PROB)) %>%
  mutate(TOTAL_DURATION = exp(TRUNC_DURATION) + exp(PREDICT_DURATION))
  
