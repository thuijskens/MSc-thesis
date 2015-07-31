# -*- coding: utf-8 -*-

"""
01c Format training data for LM.py

This script formats the training data so that it can be used for easy import in
R. Currently, the linear model includes the following explanatory variables:
    - Hour of the day
    - Day of the week
    - Week of the year
    - Haversine distance between starting and end point of the trip
    - Starting point
    - Destination
"""

from __future__ import division

import numpy as np
import pandas as pd

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
  # Convert decimal degrees to radians 
  lon1, lat1, lon2, lat2 = map(np.radians, [p1[0], p1[1], p2[0], p2[1]])
  
  # Haversine formula 
  dlon = lon2 - lon1 
  dlat = lat2 - lat1 
  a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
  c = 2 * np.arcsin(np.sqrt(a)) 
  r = 6371 
  
  return c * r
  
if __name__ == "__main__":
  print "- 01d Format validation data for LM.py"
  
  # Define the filepaths
  filepath = "E:/MSc thesis/Processed data/train_binarized_trips_validation.csv"
  filepath_processed = "E:/MSc thesis/Processed data/val_lm.csv"
  chunk_size = 1000
  
  # Define variables to export
  expl_vars = ["START_POINT_LON", "START_POINT_LAT", "HOUR", "WDAY", "WEEK", "DURATION"]
  
  # Read in the file
  data_chunks = pd.read_csv(filepath_or_buffer = filepath,
                            sep = ",",
                            chunksize = chunk_size,
                            usecols = ["TIMESTAMP", "START_POINT", "HOUR", "WDAY", "DURATION"],
                            converters = {"START_POINT": lambda x: eval(x)})
                                          
  # Iterate through the chunks
  for idx, chunk in enumerate(data_chunks): 
    # Convert string timestamp to datetime
    chunk["TIMESTAMP"] = pd.to_datetime(chunk["TIMESTAMP"])
    
    # Compute start points
    chunk["START_POINT_LON"] = chunk["START_POINT"].map(lambda x: x[0])
    chunk["START_POINT_LAT"] = chunk["START_POINT"].map(lambda x: x[1])

    # Compute week of year
    chunk["WEEK"] = chunk["TIMESTAMP"].dt.week
    
    # Save the chunk to a file
    if idx == 0:
      chunk.to_csv(filepath_processed, header = True, index = False, columns = expl_vars)
    else:
      chunk.to_csv(filepath_processed, mode = "a", header = False, index = False, columns = expl_vars)
                                          
    print "-- Processed chunk %d of 11" % idx                      

    