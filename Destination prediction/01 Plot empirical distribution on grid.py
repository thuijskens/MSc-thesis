# -*- coding: utf-8 -*-
"""
01 Plot empirical distribution on grid
This script computes the heatmap for the map discretization and formats the
data for easy import in R.
"""

# Import relevant libraries
from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
  print "- 01 Plot empirical distribution on grid"
  
  # Define filepaths and grid variables
  filepath = "E:/MSc thesis/Processed data/train_binarized_trips_train.csv"
  filepath_processed = "E:/MSc thesis/Processed data/grid_distribution.csv"
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
  
  # Use a dictionary to represent the grid
  grid = {}
  for i in xrange(N):
    for j in xrange(M):
      grid[(i,j)] = 0
  
  # Use pandas read_csv function to read the data in chunks
  data_chunks = pd.read_csv(filepath_or_buffer = filepath,
                            sep = ",",
                            chunksize = 10000,
                            usecols = ["GRID_POLYLINE"],
                            converters = {'GRID_POLYLINE': lambda x: eval(x)})
                            
  for idx, chunk in enumerate(data_chunks):
    for trip_id, trip in chunk.iterrows():
      # Ugly: allocate grid cells to histogram
      for cell in trip.GRID_POLYLINE:
        grid[cell] += 1
      
    print "-- Processed chunk %d" % idx
  
  # Store the data from the grid into a Pandas DataFrame
  grid_data = pd.DataFrame({"CELL": grid.keys(),
                            "COUNT": grid.values()})
  
  # Add the GPS coordinates of the cell boundaries                          
  grid_data["LON_MIN"] = grid_data["CELL"].map(lambda x: lon_vals[0] + x[0] * lon_step)
  grid_data["LAT_MIN"] = grid_data["CELL"].map(lambda x: lat_vals[0] + x[1] * lat_step)

  grid_data["LON_MAX"] = grid_data["LON_MIN"] + lon_step
  grid_data["LAT_MAX"] = grid_data["LAT_MIN"] + lat_step
  
  # Get the ID of each cell
  grid_data["CELL_ID"] = grid_data.CELL.map(lambda x: id_to_nr(x))
  
  # Save the data
  grid_data.to_csv(filepath_processed, columns = ["CELL_ID", "COUNT", "LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"], index = False, header = True)
  
  """ 
    This part plots the heatmap in python
  """
  
  # Transform the data to a matrix
  hist = np.zeros((N, M))
  
  for cell, prob in grid.iteritems():
    hist[cell] = prob
  
  hist = hist[:, ::-1].T

  plt.imshow(np.log(hist), extent = [lon_vals[0], lon_vals[1], lat_vals[0], lat_vals[1]])
  plt.xlabel("Longitude")
  plt.ylabel("Latitude")
  plt.savefig("E:/MSc thesis/Visualizations/grid_heatmap.pdf")

  

