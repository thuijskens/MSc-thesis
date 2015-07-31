""" 04 Random walk.py
    Constructs a lattice based on the historical trips in the dataset. A random
    walk can then be simulated by using the counts as weights on the edges.
"""
from __future__ import division
from lattice import *
from DestinationGrid import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
  # Location of the data
  filepath = "E:/MSc thesis/Processed data/train_binarized_trips_train.csv"
  filepath_val = "E:/MSc thesis/Processed data/train_binarized_trips_validation.csv"
  
  # From visual inspection of porto map. We are only focusing on the city centres
  lon_vals = (-8.73, -8.5)
  lat_vals = (41.1, 41.25)   
  
  # Number of points used for the discretization
  N = 100
  M = 75
  
  # Number of simulations of the random walk
  S = 1
  
  # Construct the lattice
  lon_bins, lon_step = np.linspace(lon_vals[0], lon_vals[1], N, retstep = True)
  lat_bins, lat_step = np.linspace(lat_vals[0], lat_vals[1], M, retstep = True)
  
  ltc = Lattice(leftCorner = (lon_vals[0], lat_vals[0]), 
                rightCorner = (lon_vals[1], lat_vals[1]), nrCells = (N, M))
  
  # Read in the data (only 100 rows to test)
  trips = pd.read_csv(filepath_or_buffer = filepath,
                      sep = ",",
                      nrows = 100000,
                      usecols = ["POLYLINE", "GRID_POLYLINE"],
                      converters = {"POLYLINE": lambda x: json.loads(x),
                                    "GRID_POLYLINE": lambda x: eval(x)})

  """ Build the lattice and compute the weights for each edge
  """
  # Loop through every trip
  rows = trips.iterrows()
  
  for row in rows:
    # Acces the trip as a list
    trip = row[1][1]
    
    # Initialize lists that keep track of which cells and links are counted
    vertices = []
    edges = []
    
    for cell, next_cell in zip(trip[:-1], trip[1:]):
      # Increase the counter for the current cell
      if cell not in vertices:
        ltc.getCell(cell).timesReached += 1
        vertices.append(cell)
      
      # Increase the counter for the link
      if (cell, next_cell) not in edges and cell != next_cell:
        ltc.increaseWeight(cell, next_cell, directed_lattice = True)
        edges.append((cell, next_cell))
   
    
  """ Remark: Since it is possible that we observe a sequence like
        ((1,1), (2,2), (1,1), (1,2))
      we get that (1,1).timesReached is unequal to the sum of the outgoing
      weights from (1,1).
  """
 
  """ Random walk approach on weighted graph
  """
  # Now we have the weights for the edges, we can simulate a random walk on the graph
  
  # Load in the validation data
  validation = pd.read_csv(filepath_or_buffer = filepath_val,
                           sep = ",",
                           nrows = 10,
                           converters = {"POLYLINE": lambda x: json.loads(x),
                                         "START_POINT": lambda x: eval(x),
                                         "GRID_POLYLINE": lambda x: eval(x),
                                         "TRUNC_GRID_POLYLINE": lambda x: eval(x)})
  # Initialize partial trip                                      
  partial_trip = validation.loc[0]
  start_simulation = partial_trip.GRID_POLYLINE[-1]  
  
  # Initialize list of simulated destinations
  destinations = []
  
  for simulation in xrange(S):
    # Initialize the walker
    wlk = Walker(lattice = ltc, start = start_simulation, dest_threshold = 1)
    
    # Simulate a random walk and record the destination
    destinations.append(wlk.simulateWalker())

  # Use pandas facilities
  table = pd.DataFrame({"DEST": pd.Series(destinations)})
  table["COUNT"] = 1
  
  prob_table = table.groupby(["DEST"], as_index = False).aggregate({"COUNT": "sum"})
  prob_table.COUNT = prob_table.COUNT / np.sum(prob_table.COUNT)
  
  # Store the probabilities in a DestinationGrid object
  posterior = DestinationGrid(N, M)
  
  for idx, row in prob_table.iterrows():
    posterior.setProb(row.DEST, row.COUNT)
    
    
  # Plot the new distribution
  plt.subplot(1,2,1)
  plt.imshow(np.log(posterior.as_array() + trip_to_array(partial_trip.GRID_POLYLINE, N, M)), interpolation = "nearest")
  plt.title("Complete trip superimposed")
  
  plt.subplot(1,2,2)
  plt.imshow(np.log(posterior.as_array() + trip_to_array(partial_trip.TRUNC_GRID_POLYLINE, N, M)), interpolation = "nearest")
  plt.title("Partial trip superimposed")

  
  
  