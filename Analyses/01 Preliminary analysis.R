# Preliminary analysis of the Porto taxi data set
# This script preprocesses the original data by
# 1. Making sure all the columns have the correct types
# 2. Transforming the UNIX timestamp to a GMT timestamp
# 3. Adding the travel time for each trip
# 4. Removing the entries for which MISSING_DATA = TRUE
# 5. Removing the entries for which the number of pairs of GPS coordinates is smaller than 1

# Furthermore, the script outputs a number of visualizations of the dataset.

# Set the working directory
setwd("E:/MSc thesis")

# Load relevant libraries
library(data.table)
library(dplyr)
library(rjson)
library(ggplot2)
library(scales)

# Load auxilary script for preprocessing utility functions
source('./R scripts/preprocessing.r')

# Load the training data and the metadata
train <- as_data_frame(fread('./Raw data/train.csv', header = TRUE, sep = ","))
test <- as_data_frame(fread('./Raw data/test.csv', header = TRUE, sep = ","))
holidays <- as_data_frame(fread('./Raw data/holidays.csv', header = TRUE, sep = ","))
meta <- as_data_frame(fread('./Raw data/metadata_taxistands.csv', header = TRUE))
colnames(meta) <- c("ID", "Stand", "Latitude", "Longitude")

# Preprocess the data
train <- processData(train)
test <- processData(test)

################################################################
#
#       Preprocessing
#
################################################################

# How many trips are there with zero points, or just one point?
train %>%
  filter(NRPOINTS <= 5) %>%
  group_by(NRPOINTS) %>%
  summarise(count = n())

# There are 5901 trips with no GPS coordinates. These trips should be removed, since there is no way
# we can infer the GPS coordinates and the travel time accurately

# The trips with only one pair of GPS coordinates also needs further consideration

# Filter out the entries with no GPS coordinates
train <- train %>%
  filter(NRPOINTS >= 2) 

# Save the data
if(!file.exists("./Processed data/train_processed.csv")) {
  write.csv(train, "./Processed data/train_processed.csv", row.names = FALSE)
}

if(!file.exists("./Processed data/test_processed.csv")) {
  write.csv(test, "./Processed data/test_processed.csv", row.names = FALSE)
}

################################################################
#
#       Exploratory data analysis
#
################################################################

train <- train %>%
  rowwise() %>%
  mutate(TRAVERSED_DISTANCE = distanceTraversed(POLYLINE))

train %>%
  ggplot(aes(x = TRAVERSED_DISTANCE*1000)) +
  geom_histogram(aes(y = ..count../sum(..count..)), fill = "white", colour = "black", binwidth = 50) +
  scale_x_continuous(limits = c(NA, 5000))

train %>% 
  group_by(bins = cut(TRAVERSED_DISTANCE, breaks = seq(from = 0, to = 5000, by = 50), 1000000)) %>%
  summarise(count = n())


# What is the distribution of the length of the trips?
pdf("duration_hist.pdf")
train %>%
  select(NRPOINTS) %>%
  filter(NRPOINTS <= 500) %>%
  ggplot(aes(x = NRPOINTS)) +
  geom_histogram(aes(y = ..count../sum(..count..)), fill = "white", colour = "black", binwidth = 5)
dev.off()

train %>%
  filter(DURATION_RE <= 10000) %>%
  ggplot(aes(x = DURATION_RE)) +
  geom_histogram(aes(y = ..count../sum(..count..)), fill = "white", colour = "black", binwidth = 120)

train %>% filter(NRPOINTS > 500) %>% summarise(n())
# 2060 trips that have more than 500 points

# How is the distribution across CALL_TYPE?
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

# 11302 out of 817878 entries are missing. How is the distribution of called taxis across stands?
pdf('./Visualizations/stand_time_distribution.pdf')
train %>%
  filter(CALL_TYPE == "B") %>%
  left_join(meta, by = c("ORIGIN_STAND" = "ID")) %>%
  group_by(Stand) %>%
  #summarise(count = n()) %>%
  summarise(count = mean(DURATION)) %>%
  filter(!is.na(Stand)) %>%
  ggplot(aes(x = reorder(Stand, count), y = count)) +
  geom_point() +
  coord_flip() + 
  #labs(x = "Stand", y = "Number of taxi trips")
  labs(x = "Stand", y = "Mean travel time")
dev.off()

# For the taxis that are ordered by the central, how well is the ORIGIN_CALL filled?
train %>%
  filter(CALL_TYPE == "A") %>%
  summarise(sum(is.na(ORIGIN_CALL) | is.null(ORIGIN_CALL)))

# The ORIGIN_CALL column is dense. Does it actually contain useful information? 
pdf('origin_call_example.pdf')
train %>%
  filter(CALL_TYPE == "A" & ORIGIN_CALL == 31508) %>%
  plotTrip()
# Definetely holds information. These trips are very similar
dev.off()

# How is the distribution across DAY_TYPE?
train %>% 
  group_by(DAY_TYPE) %>%
  summarise(count = n()) %>%
  ggplot(aes(x = DAY_TYPE, y = count)) +
  geom_bar(stat = "identity", fill = "white", colour = "black")
# There are no entries for which DAY_TYPE != "A", so for now this column is useless.

# How is the distribution of the number of trips and total travel time over the year?
train_agg_day <- train %>%
  group_by(year = year(TIMESTAMP), month = month(TIMESTAMP), day = mday(TIMESTAMP)) %>%
  summarise(count = n(), mean_time = mean(DURATION)) %>%
  mutate(date = as.Date(sprintf("%s-%s-%s", year, month, day))) %>%
  select(date, count, mean_time)

p_day_count <- train_agg_day %>%
  ggplot(aes(x = date, y = count)) +
  geom_line() +
  scale_y_continuous(breaks = seq(from = 2000, to = 8000, by = 1000)) +
  scale_x_date(breaks = date_breaks("months"), labels = date_format("%b %Y")) +
  labs(x = "Date", y = "Number of taxi trips per day")

p_day_duration <- train_agg_day %>%
  ggplot(aes(x = date, y = mean_time/60)) +
  geom_line() +
  scale_y_continuous(breaks = seq(from = 10, to = 14, by = 0.5)) +
  scale_x_date(breaks = date_breaks("months"), labels = date_format("%b %Y")) +
  labs(x = "Date", y = "Mean travel time per day (m)")

pdf('trips_per_day.pdf', width = 16, height = 12)
grid.arrange(p_day_count, p_day_duration)
dev.off()

# There is a pattern visible here. A weekly pattern is visible as well as a small seasonal pattern
# There are less taxi trips in the summer --> less busy on the roads?
train_agg_hour <- train %>%
  group_by(hour = chron::hours(TIMESTAMP), minute = chron::minutes(TIMESTAMP) - (chron::minutes(TIMESTAMP) %% 10)) %>%
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
  geom_smooth() +
  scale_y_continuous(breaks = seq(from = 9, to = 15, by = 1.0)) +
  scale_x_datetime(breaks = date_breaks("2 hours"), labels = date_format("%H:%M")) +
  labs(x = "Time (h)", y = "Mean travel time (m)")

pdf('trips_per_hour.pdf')
grid.arrange(p_hour_count, p_hour_duration)
dev.off()

# Travel times are longer in morning and afternoon rush hour

# Interpreting the travel times of a given taxi driver as a time series, is there any autocorrelation between the times?
taxiID <- sample(unique(train$TAXI_ID), size = 1)

train_single_taxi <- train %>%
  filter(TAXI_ID == taxiID) %>%
  arrange(TIMESTAMP) %>%
  select(DURATION)

pdf('Trips_taxi.pdf', width = 10)
train_single_taxi %>%
  ggplot(aes(x = 1:nrow(train_single_taxi), y = DURATION)) +
  geom_line() +
  labs(x = "Trip number", y = "Duration", title = paste("Trips of taxi", taxiID))
dev.off()

p_acf <- acf(train_single_taxi)
p_pacf <- pacf(train_single_taxi)


ntrips <- nrow(train_single_taxi)
lag_df <- data_frame(lag1 = train_single_taxi[1:(ntrips - 1),]$DURATION, current = train_single_taxi[2:ntrips,]$DURATION) 

lag_df %>%
  ggplot(aes(x = lag1, y = current)) +
  geom_point()

# What is the distribution of travel times over a week, and over a day?
train_agg_week <- train %>%
  group_by(weekday = wday(TIMESTAMP)) %>%
  summarise(count = n(), mean_time = mean(DURATION)) %>%
  select(weekday, count, mean_time)

train_agg_week %>%
  ggplot(aes(x = weekday, y = count)) +
  geom_line()

train_agg_week %>%
  ggplot(aes(x = weekday, y = mean_time)) +
  geom_line()

################################################################
#
#       Split data on CALL_TYPE
#
################################################################

train_call_A <- train %>%
  filter(CALL_TYPE == "A")

train_call_B <- train %>%
  filter(CALL_TYPE == "B")

grid.arrange(
train_call_A %>% 
  filter(NRPOINTS <= 500) %>%
  ggplot(aes(x = DURATION)) +
  geom_histogram(aes(y = ..count../sum(..count..)), fill = "white", colour = "black", binwidth = 15),

train_call_B %>% 
  filter(NRPOINTS <= 500) %>%
  ggplot(aes(x = DURATION)) +
  geom_histogram(aes(y = ..count../sum(..count..)), fill = "white", colour = "black", binwidth = 15),

train %>% 
  filter(NRPOINTS <= 500) %>%
  ggplot(aes(x = DURATION)) +
  geom_histogram(aes(y = ..count../sum(..count..)), fill = "white", colour = "black", binwidth = 15),

nrow = 3)

