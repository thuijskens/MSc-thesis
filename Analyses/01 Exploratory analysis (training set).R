# Preliminary analysis plots.R
# 
# This script produces plots that are used for the explanatory analysis 
# chapter of the thesis.

# Load relevant libraries
library(data.table)
library(dplyr)
library(rjson)
library(ggplot2)
library(scales)
library(gridExtra)

# Set the working directory
setwd("E:/MSc thesis")

# Load training data and meta data
train <- as_data_frame(fread("./Processed data/train_lm.csv", header = TRUE, sep = ",", drop = c("TRIP_ID")))
meta <- as_data_frame(fread('./Raw data/metadata_taxistands.csv', header = TRUE))
colnames(meta) <- c("ID", "Stand", "Latitude", "Longitude")

# Plot the distribution of the travel time in the training set
pdf("./Visualizations/duration_hist.pdf", width = 8, height = 4)
train %>%
  filter(DURATION <= 1.5 * 3600) %>%
  ggplot(aes(x = DURATION / 60)) +
  geom_histogram(aes(y = ..count../sum(..count..)), fill = "white", colour = "black", binwidth = 75 / 60) + 
  scale_x_discrete(breaks = seq(from = 0, to = 100, by = 10)) +
  labs(x = "Total travel time (m)", y = "Probability", title = "Histogram of the total travel time")
dev.off()

# How is the distribution of the travel time across CALL_TYPE?
train %>% 
  group_by(CALL_TYPE) %>%
  summarise(count = n()) %>%
  ggplot(aes(x = CALL_TYPE, y = count)) +
  geom_bar(stat = "identity", fill = "white", colour = "black")

train %>% 
  group_by(CALL_TYPE) %>%
  summarise(mt = mean(DURATION)) %>%
  ggplot(aes(x = CALL_TYPE, y = mt)) +
  geom_bar(stat = "identity", fill = "white", colour = "black")


# Most of the taxis are called from a stand. For the taxis called from a stand, how well is the ORIGIN_STAND
# column filled?
train %>%
  filter(CALL_TYPE == "B") %>%
  summarise(n_total = n(),
            n_missing = sum(is.na(ORIGIN_STAND)))

# 9144 out of 664138 entries are missing. These trips will be treated as if they were called from the street in the modelling pipeline
train <- train %>%
  mutate(CALL_TYPE_NEW = ifelse(CALL_TYPE == "B" & is.na(ORIGIN_STAND), "C", CALL_TYPE))

# Summarize the distributions in a table
train %>%
  group_by(CALL_TYPE_NEW) %>%
  summarise(mean_time = mean(DURATION),
            sd_time = sd(DURATION),
            count = n())
# Get total statistics
train %>%
  summarise(mean_time = mean(DURATION),
            sd_time = sd(DURATION),
            count = n())

# How is the distribution of called taxis across stands?
stand_time_mean <- train %>%
  filter(CALL_TYPE_NEW == "B") %>%
  left_join(meta, by = c("ORIGIN_STAND" = "ID")) %>%
  group_by(Stand) %>%
  summarise(count = mean(DURATION)) %>%
  filter(!is.na(Stand)) %>%
  ggplot(aes(x = reorder(Stand, count), y = count)) +
  geom_point() +
  coord_flip() + 
  #labs(x = "Stand", y = "Number of taxi trips")
  labs(x = "Stand", y = "Mean travel time", title = "Mean travel time per stand")


stand_time_count <- train %>%
  filter(CALL_TYPE_NEW == "B") %>%
  left_join(meta, by = c("ORIGIN_STAND" = "ID")) %>%
  group_by(Stand) %>%
  summarise(count = n()) %>%
  filter(!is.na(Stand)) %>%
  ggplot(aes(x = reorder(Stand, count), y = count)) +
  geom_point() +
  coord_flip() + 
  labs(x = "Stand", y = "Number of taxi trips", title = "Number of trips per stand")

pdf('./Visualizations/stand_time_distribution.pdf', height = 6, width = 12)
grid.arrange(stand_time_mean, stand_time_count, ncol = 2)
dev.off()

pdf('./Visualizations/stand_time_distribution.pdf')
stand_time_mean
dev.off()

pdf('./Visualizations/stand_count_distribution.pdf')
stand_time_count
dev.off()

# For the taxis that are ordered by the central, how well is the ORIGIN_CALL filled?
train %>%
  filter(CALL_TYPE == "A") %>%
  summarise(sum(is.na(ORIGIN_CALL) | is.null(ORIGIN_CALL)))


# There is a pattern visible here. A weekly pattern is visible as well as a small seasonal pattern
# There are less taxi trips in the summer --> less busy on the roads?
train_agg_hour <- train %>%
  group_by(hour = HOUR, minute = MINUTE - (MINUTE %% 10)) %>%
  summarise(count = n(), mean_time = mean(DURATION)) %>%
  mutate(hm = as.POSIXct(sprintf("1992-06-30 %s:%s:00", hour, minute))) %>%
  select(hm, count, mean_time)

p_hour_count <- train_agg_hour %>%
  ggplot(aes(x = hm, y = count)) +
  geom_line() +
  geom_smooth() +
  scale_x_datetime(breaks = date_breaks("2 hours"), labels = date_format("%H:%M")) +
  labs(x = "Time (h)", y = "Number of taxi trips per hour")

p_hour_duration <- train_agg_hour %>%
  ggplot(aes(x = hm, y = mean_time/60)) +
  geom_line() +
  #geom_smooth() +
  scale_y_continuous(breaks = seq(from = 9, to = 15, by = 1.0)) +
  scale_x_datetime(breaks = date_breaks("2 hours"), labels = date_format("%H:%M")) +
  labs(x = "Time (h)", y = "Mean travel time (m)")

# What is the distribution of travel times over a week, and over a day?
p_wday_duration <- train %>%
  group_by(weekday = WDAY) %>%
  summarise(count = n(), mean_time = mean(DURATION)) %>%
  select(weekday, count, mean_time) %>%
  ggplot(aes(x = weekday, y = mean_time)) +
  geom_bar(stat = "identity", colour = "black", fill = "white")

train_agg_week %>%
  ggplot(aes(x = weekday, y = mean_time)) +
  geom_line()

pdf('./Visualizations/trips_per_hour.pdf', width = 8, height = 4)
print(p_hour_duration)
dev.off()
# Travel times are longer in morning and afternoon rush hour
