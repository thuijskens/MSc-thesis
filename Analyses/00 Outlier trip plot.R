# Load relevant libraries
library(data.table)
library(dplyr)
library(rjson)
library(ggplot2)
library(scales)
library(ggmap)
library(rjson)

# Set the working directory
setwd("E:/MSc thesis")

# Define some necessary utility functions
get_positions <- function(trip) {
  # positions is an utility function that outputs a dataframe with the longitude and latitude coordinates of a trip
  # Arguments:
  # -- trip: A list of latitude and longitude coordinates (POLYLINE)
  # Returns:
  # -- A data.frame with the latitude and longitude coordinates
  
  out <- as.data.frame(do.call(rbind, fromJSON(trip)))
  if (ncol(out) == 2) { # Cath the error in the case trip = "[[]]"
    colnames(out) <- c("Longitude", "Latitude")
  }
  
  return(out)
}

# Define global variables
bounding_box <- c(left = -8.67, bottom = 41.135, right = -8.58, top = 41.16)

# Load the raw training data
train_raw <- as_data_frame(fread("./Raw data/train.csv", header = TRUE, sep = ","))

# Get a trip were the GPS coordinates show outliers and store the data in a data frame
trip_df <- train_raw %>%
  filter(TRIP_ID == "1372636951620000320") %>%
  select(POLYLINE) %>%
  as.character() %>%
  get_positions()

trip_df <- cbind(trip_df, nr = 1:dim(trip_df)[1])

# Plot the trip
map <- get_openstreetmap(bbox = bounding_box, 
               scale = 13030,
               filename = "./Visualizations/porto_osm_map")

pdf("./Visualizations/outlier_trip.pdf", width = 8, height = 4)
map %>%
  ggmap() +
  geom_path(data = trip_df,
            aes(x = Longitude, y = Latitude, colour = nr), 
            lineend = "round", 
            linejoin = "round",
            size = 1.25) +
  scale_fill_brewer(palette = "RdBu") + 
  labs(x = "Longitude", y = "Latitude", title = "Path of taxi trip with ID 1372636951620000320") +
  theme(legend.position = "none")
dev.off()

# Now plot all of the trips for a trip that is called from a taxi central
trips_df  <- train_raw %>%
  filter(ORIGIN_CALL == 31508)

coords_df <- data_frame(tripid = numeric(),
                        Longitude = numeric(),
                        Latitude = numeric())

for(i in 1:nrow(trips_df)) {
  coords <- get_positions(trips_df[i, ]$POLYLINE)
  coords_df <- rbind(coords_df, data_frame(tripid = i,
                                           Longitude = coords$Longitude,
                                           Latitude = coords$Latitude))
}

# Define global variables
bounding_box_city <- c(left = -8.68, bottom = 41.13, right = -8.55, top = 41.19)
map <- get_openstreetmap(bbox = bounding_box_city, 
                         scale = 26055,
                         filename = "./Visualizations/porto_osm_map_caller")

pdf("./Visualizations/caller_trips.pdf", width = 8, height = 6)
map %>%
  ggmap() +
  geom_path(data = coords_df,
            aes(x = Longitude, y = Latitude, colour = as.factor(tripid)), 
            lineend = "round",
            size = 1.25) +
  labs(x = "Longitude", y = "Latitude", title = "Path of taxi trips with caller ID 31508") +
  theme(legend.position = "none")
dev.off()