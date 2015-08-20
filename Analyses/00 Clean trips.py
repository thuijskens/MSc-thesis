""" 00 Clean trips.py
    This scripts loads the file train_processed.csv and cleans each trip by 
    adjusting 'strange' GPS coordinates. The coordinates are not removed 
    because we compute trip length by counting the number of recorded coordinates
    
    Outlier detection:
      1.  If distance between start and end point is less than 100 meters, 
          remove the trip.
      2.  If the speed between two consecutive pairs of coordinates lies above
          a given threshold, impute the coordinates by the mean of the previous
          and the next pair of coordinates.
"""

from __future__ import division

import numpy as np
import pandas as pd
import datetime
import json
import sys


def haversine(p1, p2):
  """ haversine
      Calculate the great circle distance between two points 
      on the earth (specified in decimal degrees)
      
      Arguments:
      ----------
      p1: 2d numpy array
        Two-dimensional array containing longitude and latitude coordinates of points.
      p2: 2d numpy array
        Two-dimensional array containing longitude and latitude coordinates of points.
        
      Returns:
      --------
      dist: 1d numpy array 
        One-dimensional array with the distances between p1 and p2.
  """
  # convert decimal degrees to radians 
  if p1.ndim == 1:
    p1 = p1.reshape(-1,2)
  if p2.ndim == 1:
    p2 = p2.reshape(-1,2)
    
  lon1, lat1, lon2, lat2 = map(np.radians, [p1[:, 0], p1[:, 1], p2[:, 0], p2[:, 1]])
  
  # haversine formula 
  dlon = lon2 - lon1 
  dlat = lat2 - lat1 
  a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
  c = 2 * np.arcsin(np.sqrt(a)) 
  r = 6371 
  return c * r

def remove_outliers(trip, threshold_speed = 120):
  """ remove_outliers
      Removes points from a trip that can be considered outliers. A point is 
      considered an outlier if the speed difference between a point and the 
      first previous, non-outlier point is bigger than threshold_speed km/h
      
      Arguments:
      ----------
      trip: 2-dimensional numpy array
        Array that contains the longitude and latitude coordinates of the trip.
      threshold_speed: scalar
        Speed threshold for outlier detection.
        
      Returns:
      --------
      new_trip: 2-dimensional numpy array
        Array with the cleaned coordinates of the trip.
  """
  c = threshold_speed / 3600 * 15 # km per 15s threshold

  # Initialize the list with the cleaned coordinates
  new_trip = [trip[0]]

  for point in trip[1:]:
    # Compare the distance between the current point and the first previous
    # non-outlier point
    if haversine(point, new_trip[-1]) < c:
      new_trip.append(point)
      
  return np.array(new_trip)
    
def is_outlier(trip, threshold_speed = 120):
  """ is_outlier
      Checks if a trip contains a point that can be considered as an outlier.
  """
  # Define a conversion for the threshold comparison
  c = threshold_speed / 3600 * 15 # km per 15s threshold

  if type(trip) != np.ndarray:
    try:
      trip = np.array(trip)
    except:
      print "Fuck"
      return True
    
  # Compute distances
  distances = haversine(trip[:-1], trip[1:])
  
  if np.any(distances >= c):
    return True
  else:
    return False
    
def is_trip_in_grid(trip, lon_vals, lat_vals):
  """ is_trip_in_grid
      Checks if the all the coordinates in the trip fall within the grid 
      defined by lon and lat
  """
  if len(trip) is not 0:
    in_lon_grid = np.all(trip[:, 0] >= lon_vals[0]) and np.all(trip[:, 0] <= lon_vals[1])
    in_lat_grid = np.all(trip[:, 1] >= lat_vals[0]) and np.all(trip[:, 1] <= lat_vals[1])
  else: 
    # This filters out the empty trips
    return False
  
  return in_lon_grid and in_lat_grid
    
    
if __name__ == "__main__":
  print "- Starting cleaning script"
  
  # Get the location of the data file
  args = sys.argv[1:]
  if args[0] is None:
    filepath = "E:/MSc thesis/Raw data/train.csv"
  else:
    filepath = args[0]
  
  if args[1] is None:
    filepath_clean = "E:/MSc thesis/Processed data/train_cleaned.csv"
  else:
    filepath_clean = args[1]
  
  # Use pandas read_csv function to read the data in chunks
  data_chunks = pd.read_csv(filepath_or_buffer = filepath,
                            sep = ",",
                            chunksize = 10000, 
                            converters = {'POLYLINE': lambda x: json.loads(x)})

  # From visual inspection of porto map. We are only focusing on the city centre
  # 15-07-14: We continue with this bounding box as of now
  lon_vals = (-8.73, -8.5)
  lat_vals = (41.1, 41.25)   

  # Define cutoff thresholds for duration
  threshold_min = 3 # A trip needs to last 30 seconds
  threshold_max = 1.5 * 3600 # A trip should be finished in 1.5 hours
  
  # The max threshold is based on the following. The distance between the lower
  # -left corner and the top-right corner of the bounding box is approx. 25.5 km
  # Given an average speed of 20 kmh (cycling speed), you need 76.5 minutes to
  # cross the bounding box.
                   
  # Define a lambda function that can be passed to pd.map
  is_in_grid = lambda x: is_trip_in_grid(np.array(x), lon_vals, lat_vals)
  
  for idx, chunk in enumerate(data_chunks):
    # Remove the points that have MISSING_DATA = TRUE
    missing = (chunk["MISSING_DATA"] == True)
    # Check if each trip is contained in the grid representation
    in_grid = chunk["POLYLINE"].map(is_in_grid)   
    # Remove the trips that have outlier points
    outliers = chunk["POLYLINE"].map(is_outlier)
    # Remove trips that last for less than 30 seconds
    too_short = chunk["POLYLINE"].map(len) <= threshold_min
    # Remove trips that last for longer than 1.5 hours
    too_long = chunk["POLYLINE"].map(len) >= threshold_max
    
    # Concatenate these if-statements together
    remove = missing | outliers | ~ in_grid | too_short | too_long
    
    # Remove these rows from the dataframe
    chunk = chunk[-remove].reset_index(drop = True)
    
    """
    # Clean the trips by removing outlier points
    chunk["POLYLINE"] = chunk["POLYLINE"].map(lambda x: remove_outliers(np.array(x)))
    """
    
    # Store start and end points of trips for later use
    chunk["START_POINT"] = chunk["POLYLINE"].apply(lambda x: x[0])
    chunk["END_POINT"] = chunk["POLYLINE"].apply(lambda x: x[-1])
    
    # Store UNIX timestamp as POSIXct
    chunk["TIMESTAMP"] = chunk["TIMESTAMP"].apply(datetime.datetime.utcfromtimestamp)
    
    # Get hour of the day and day of the week from timestamp
    chunk["HOUR"] = chunk["TIMESTAMP"].dt.hour
    chunk["WDAY"] = chunk["TIMESTAMP"].dt.weekday
    # IMPORTANT:
    # Python: Monday = 0
    # R: Sunday = 0
    
    # Calculate length of trip
    chunk["DURATION"] = chunk["POLYLINE"].apply(lambda x: (len(x) - 1) * 15)
    
    # Back transform POLYLINE
    chunk["POLYLINE"] = chunk["POLYLINE"].apply(json.dumps)
    
    # Append chunk to file
    if idx == 0:
      chunk.to_csv(filepath_clean, header = True, index = False)
    else:
      chunk.to_csv(filepath_clean, mode = "a", header = False, index = False)
      
    print "Processed chunk %d of 171" % idx
