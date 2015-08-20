# 01 Predictor analysis for linear model.R

# This scripts creates a number of plots that show the relationships between
# the variables in the training set and the response for the linear model (travel time)

# Load relevant libraries
library(data.table)
library(dplyr)
library(ggplot2)

# Set working directory
setwd("E:/MSc thesis")

# Import the data
train <- as_data_frame(fread("./Processed data/train_lm.csv", header = TRUE, sep = ","))

# Make sure HOUR, WDAY and WEEK are factors
train$HOUR <- as.factor(train$HOUR)
train$WDAY <- as.factor(train$WDAY) # 0 = Monday
train$WEEK <- as.factor(train$WEEK)

# Travel time over hours, days and week
train %>%
  group_by(HOUR) %>%
  summarise(MEAN_DURATION = mean(DURATION)) %>%
  ggplot(aes(x = HOUR, y = MEAN_DURATION)) +
  geom_bar(stat = "identity", fill = "white", colour = "black")
# Travel times during the day are shorter on average than travel times in the late evening or early morning

train %>%
  group_by(WDAY) %>%
  summarise(MEAN_DURATION = mean(DURATION)) %>%
  ggplot(aes(x = WDAY, y = MEAN_DURATION)) +
  geom_bar(stat = "identity", fill = "white", colour = "black")
# Travel times in the weekend are shorter on average than travel times during the week

train %>%
  group_by(WEEK) %>%
  summarise(MEAN_DURATION = mean(DURATION)) %>%
  ggplot(aes(x = as.numeric(WEEK), y = MEAN_DURATION)) +
  geom_line()
# There is a big drop in mean travel time in the summer weeks (max 1min)

# Scatter plot of distances versus travel time
train %>%
  filter(DURATION <= 500*15) %>%
  ggplot(aes(x = log(DISTANCE), y = DURATION)) +
  geom_point()

# Histogram of distance
train %>%
  filter(DURATION <= 500*15) %>%
  ggplot(aes(x = sqrt(DISTANCE))) +
  geom_histogram(aes(y = ..count../sum(..count..)), fill = "white", colour = "black")
# Square root transformation stabilizes the histogram of the distance




