# 02 Fit mixed effects model.R

# This scripts estimates a mixed effects model for
#
# T = T_0 + T_a,
#
# where T_0 is the already traversed time of a truncated trip and T_a is the additional travel time,
# that is to be estimated by the mixed effects model
#
# log(T_a) = alpha + beta_{s_i} + gamma_{d_i} + delta{s_i, d_i} + bX, 
#
# where s_i is the starting cell of trip i, d_i is the destination cell of trip i, delta is the interaction
# effect of these two, alpha is a global intercept and where bX are remaining fixed effects (hour, day, week, etc.)

# Load relevant libraries
library(data.table)
library(dplyr)
library(ggplot2)
library(lme4)

# Set working directory
setwd("E:/MSc thesis")

# Import the data
train_vars = c("TRIP_ID", "START_POINT_LON", "START_POINT_LAT", "END_POINT_LON", "END_POINT_LAT", "START_CELL", "END_CELL", "HOUR", "WDAY", "WEEK", "DISTANCE", "DURATION")
train <- as_data_frame(fread("./Processed data/train_lm.csv", header = TRUE, sep = ",", select = train_vars, colClasses = c(TRIP_ID = "character")))
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

# Calculate average speed
train <- train %>% 
  mutate(AVG_SPEED = DISTANCE / (DURATION/3600))

val <- val %>%
  mutate(TRUNC_AVG_SPEED = TRUNC_DISTANCE / TRUNC_DURATION)

#############
# Mixed effects model analysis
#############

# First, we estimate a complete random effects model (no fixed effects)
full_re <- lmer(formula = log(DURATION) ~ (1 | HOUR) + (1 | WDAY) + (1 | WEEK) + (1 | START_CELL) + (1 | END_CELL),
                data = train,
                REML = TRUE)

# Look at the summary of the model
summary(full_re)

# Then, we look at a reduced mixed effects model
travel_me <- lmer(formula = log(DURATION) ~ 1 + HOUR + WDAY + WEEK + (1 | START_CELL) + (1 | END_CELL),
                  data = train,
                  REML = TRUE)

summary(travel_me)

# Finally, consider the mixed effects model, but nest end cell withing the starting cell
interaction_me <- lmer(formula = log(DURATION) ~ (1 | START_CELL/END_CELL),
                       data = train,
                       REML = TRUE)

# Look at some diagnostic plots
plot(travel_me) # residuals vs fitted
qqnorm(resid(travel_me, type = "pearson"))
hist(resid(travel_me, type= "pearson"))
qqnorm(ranef(travel_me, drop = TRUE))

# There might be an interaction effect between the starting cell and the end cell. Plot to check
train %>%
  ggplot(aes(x = START_CELL, y = END_CELL)) +
  geom_point()

# Estimate a mixed effects model with interaction term
travel_me_inter <- lmer(formula = log(DURATION) ~ 1 + HOUR + WDAY + WEEK + (1 | START_CELL) + (1 | END_CELL) + (1 | START_CELL:END_CELL),
                  data = train,
                  REML = TRUE)

summary(travel_me_inter)