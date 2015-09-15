# -*- coding: utf-8 -*-
"""
02 Output example trip for grid.py
This script formats the necessary data for an example of the discretization
for a trip
"""

# Import relevant libraries
from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
  print "- 02 Output example trip for grid.py"
  
  # Define filepaths and grid variables
  filepath = "E:/MSc thesis/Processed data/train_binarized_trips_train.csv"
  filepath_processed_grid = "E:/MSc thesis/Processed data/discretized_trip_example_grid.csv"
  filepath_processed_gps = "E:/MSc thesis/Processed data/discretized_trip_example_gps.csv"
  N, M = (100, 75)
  
  # Define the boundaries of the grid
  lon_vals = (-8.73, -8.5)
  lat_vals = (41.1, 41.25)   
  
  lon_step = (lon_vals[1] - lon_vals[0]) / (N - 1)
  lat_step = (lat_vals[1] - lat_vals[0]) / (M - 1)
  # We do N - 1 and M - 1 for compatibility with np.linspace.
  # Definitely check this out!

  # Define function that transforms cells to ids
  id_to_nr = lambda (i, j): N * j + i # (i + 1) 
  
  # Use pandas read_csv function to read the data in chunks
  data_chunks = pd.read_csv(filepath_or_buffer = filepath,
                            sep = ",",
                            nrows = 10,
                            usecols = ["GRID_POLYLINE", "POLYLINE"],
                            converters = {'GRID_POLYLINE': lambda x: eval(x),
                                          "POLYLINE": lambda x: json.loads(x)})
# Select an example trip
  trip = data_chunks.loc[3]
  
  # Create dictionary to hold the grid
  grid = {}
  for i in xrange(N):
    for j in xrange(M):
      grid[(i,j)] = 0
  
  # Loop through GRID_POLYLINE
  for cell in trip.GRID_POLYLINE:
    grid[cell] = 1
  
  # Create a new DataFrame that stores the cells and GPS points of this trip
  trip_data = pd.DataFrame({"CELL": grid.keys(),
                            "IN_TRIP": grid.values()})
                            
  # Compute longitude and latitude values of the cell                        
  trip_data["LON_MIN"] = trip_data["CELL"].map(lambda x: lon_vals[0] + x[0] * lon_step)
  trip_data["LAT_MIN"] = trip_data["CELL"].map(lambda x: lat_vals[0] + x[1] * lat_step)

  trip_data["LON_MAX"] = trip_data["LON_MIN"] + lon_step
  trip_data["LAT_MAX"] = trip_data["LAT_MIN"] + lat_step                         
  
  # Get the ID of each cell
  trip_data["CELL_ID"] = trip_data.CELL.map(lambda x: id_to_nr(x))
  
  # Save the data
  trip_data.to_csv(filepath_processed_grid, columns = ["CELL_ID", "LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX", "IN_TRIP"], index = False, header = True) 

  trip_data_gps = pd.DataFrame({"LON": [coord[0] for coord in trip.POLYLINE],
                                "LAT": [coord[1] for coord in trip.POLYLINE]})
                                
  trip_data_gps.to_csv(filepath_processed_gps, index = False, header = True)
                            
  