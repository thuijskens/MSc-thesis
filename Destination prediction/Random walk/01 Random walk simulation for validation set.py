""" 01 Random walk simulation for validation set.py
    Constructs a lattice based on the historical trips in the dataset. S random
    walks are then simulated for every partial trip in the validation data set, 
    where S is pre-specified.
    
    The destination of each random walk is recorded, and the distribution of
    simulated destinations is used as input for the travel time prediction
    model.
"""

from __future__ import division
from lattice import *
from DestinationGrid import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
 
def coords_from_cell(cell, lon = [-8.73, -8.50], lat = [41.10, 41.25], N = 100, M = 75):
  """ coords_from_cell
      Calculates the longitude and latitude coordinates of the middle of a 
      given cell.
      
      Arguments:
      ----------
      cell: tuple
        Tuple of length two containing the cell.
      
      Returns:
      --------
      coords: list
        list of length two containing the longitude and latitude coordinates
        of the middle of the cell.
  """
  lon_step = (lon[1] - lon[0]) / N 
  lat_step = (lat[1] - lat[0]) / M
  
  middle_lon = lon[0] + cell[0] * lon_step + lon_step / 2
  middle_lat = lat[0] + cell[1] * lat_step + lat_step / 2
  
  return [middle_lon, middle_lat]
  
if __name__ == "__main__":
  # Location of the data
  filepath = "E:/MSc thesis/Processed data/train_binarized_trips_train.csv"
  filepath_val = "E:/MSc thesis/Processed data/train_binarized_trips_validation.csv"
  filepath_processed = "E:/MSc thesis/Processed data/val_posteriors_random_walk.csv"
  
  # From visual inspection of porto map. We are only focusing on the city centres
  lon_vals = (-8.73, -8.5)
  lat_vals = (41.1, 41.25)   
  
  # Number of points used for the discretization
  N = 100
  M = 75
  
  # Number of simulations of the random walk
  S = 1000
  
  # Define function that transforms cells to ids
  id_to_nr = lambda (i, j): N * j + i # (i + 1) 
  
  # Define distance function
  dist = lambda x1, x2: np.sqrt(np.power(x1[0] - x2[0], 2) + np.power(x1[1] - x2[1], 2))
  
  # Construct the lattice
  lon_bins, lon_step = np.linspace(lon_vals[0], lon_vals[1], N, retstep = True)
  lat_bins, lat_step = np.linspace(lat_vals[0], lat_vals[1], M, retstep = True)
  
  ltc = Lattice(leftCorner = (lon_vals[0], lat_vals[0]), 
                rightCorner = (lon_vals[1], lat_vals[1]), nrCells = (N, M))
  
  # Read in the data (only 100 rows to test)
  train = pd.read_csv(filepath_or_buffer = filepath,
                      sep = ",",
                      chunksize = 100,
                      usecols = ["GRID_POLYLINE"],
                      converters = {"GRID_POLYLINE": lambda x: eval(x)})

  """ Build the lattice and compute the weights for each edge
  """

  print "-- Computing transitional probabilities on the road graph"
  # Loop through every trip  
  for i, chunk in enumerate(train):
    for idx, trip in chunk.iterrows():    
      # Initialize lists that keep track of which cells and links are counted
      vertices = []
      edges = []
      
      for cell, next_cell in zip(trip.GRID_POLYLINE[:-1], trip.GRID_POLYLINE[1:]):
        # Increase the counter for the current cell
        if cell not in vertices:
          ltc.getCell(cell).timesReached += 1
          vertices.append(cell)
        
        # Increase the counter for the link
        if (cell, next_cell) not in edges and cell != next_cell:
          ltc.increaseWeight(cell, next_cell, directed_lattice = True)
          edges.append((cell, next_cell))
    print "--- Processed %d rows" % ((i + 1)*100000)
     
    
  """ Remark: Since it is possible that we observe a sequence like
        ((1,1), (2,2), (1,1), (1,2))
      we get that (1,1).timesReached is unequal to the sum of the outgoing
      weights from (1,1).
  """
 
  """ Random walk approach on weighted graph
  """
  # Now we have the weights for the edges, we can simulate a random walk on the graph
  
  # Load in the validation data
  print "-- Loading validation data"
  validation = pd.read_csv(filepath_or_buffer = filepath_val,
                           sep = ",",
                           converters = {"TRIP_ID": lambda x: str(x),
                                         "START_POINT": lambda x: eval(x),
                                         "TRUNC_GRID_POLYLINE": lambda x: eval(x)})
  
  print "-- Starting simulations"
  # Loop through every trip to make a prediction
  for idx, partial_trip in validation.iterrows():
    start_simulation = partial_trip.TRUNC_GRID_POLYLINE[-1]  
    
    # Initialize list of simulated destinations
    destinations = []
    
    for simulation in xrange(S):
      # Initialize the walker
      wlk = Walker(lattice = ltc, start = start_simulation, dest_threshold = 3)
      
      # Simulate a random walk and record the destination
      destinations.append(wlk.simulateWalker())
  
    # Store distances to destinations
    dests = [id_to_nr(dest) for dest in destinations]
    #dists = [haversine(coords_from_cell(dest), partial_trip.START_POINT) for dest in destinations]
  
    # Use pandas to compute the frequency distribution over the destinations                       
    table = pd.DataFrame({"DEST_CELL": dests})
    table["PROB"] = 1
    
    prob_table = table.groupby(["DEST_CELL"], as_index = False).aggregate({"PROB": "sum"})
    prob_table.PROB = prob_table.PROB / np.sum(prob_table.PROB)
    
    # Add trip ID and distance as columns
    prob_table["TRIP_ID"] = partial_trip.TRIP_ID
    #prob_table["DISTANCE"] = prob_table.apply()
    
    if idx == 0:
      prob_table.to_csv(filepath_processed, header = True, index = False)
    else:
      prob_table.to_csv(filepath_processed, mode = "a", header = False, index = False)
    
    print "--- Processed trip %d of %d" % (idx + 1, len(validation))
  
  print "- Finished random walk simulation script"
    

  