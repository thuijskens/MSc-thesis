# -*- coding: utf-8 -*-
""" 02 Second-order random walk for validation set.py
    Constructs a lattice based on the historical trips in the dataset. In this
    lattice the nodes represent the edges from the city graph and the edges 
    represent movements between these edges.
    
    S second-order random walks are then simulated for every partial trip in the validation data set, 
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
import json


if __name__ == "__main__":
  # Location of the data
  filepath = "E:/MSc thesis/Processed data/train_binarized_trips_train.csv"
  filepath_val = "E:/MSc thesis/Processed data/train_binarized_trips_validation.csv"
  filepath_processed = "E:/MSc thesis/Processed data/val_posteriors_2nd_order_random_walk.csv"
  filepath_plot = "E:/MSc thesis/Processed data/plot_posteriors_2nd_order_random_walk.csv"
  filepath_plot_trip = "E:/MSc thesis/Processed data/plot_trip_2nd_order_random_walk.csv"
  chunk_size = 10000
  
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
  
  ltc = SecondOrderLattice(leftCorner = (lon_vals[0], lat_vals[0]),
                           rightCorner = (lon_vals[1], lat_vals[1]), nrCells = (N, M))
  
  # Read in the data (only 100 rows to test)
  train = pd.read_csv(filepath_or_buffer = filepath,
                      sep = ",",
                      chunksize = chunk_size,
                      usecols = ["GRID_POLYLINE"],
                      converters = {"GRID_POLYLINE": lambda x: eval(x)})
                      
  """ Build the lattice and compute the weights for each edge
  """

  print "-- Computing transitional probabilities on the road graph"
  # Loop through every trip  
  for i, chunk in enumerate(train):
    for idx, trip in chunk.iterrows():    
      # Transform the list of cells into a list of edges, where moves to the 
      # same cell (Ex: (0,0) --> (0,0)) are removed.
      trip_edges = [(cell, next_cell) for cell, next_cell in zip(trip.GRID_POLYLINE[:-1], trip.GRID_POLYLINE[1:]) if cell != next_cell]
      
      # Initialize lists that keep track of which edges and moves are counted  
      # we do not want to count the same edge twice
      edges = []
      moves = []
           
      for edge, next_edge in zip(trip_edges[:-1], trip_edges[1:]):
        # Increase the counter for the current cell
        if edge not in edges:
          ltc.getCell(edge).timesReached += 1
          edges.append(edge)
        
        # Increase the counter for the link
        # Thomas @ 23-8: edge != next_edge can be removed now?
        if (edge, next_edge) not in moves and edge != next_edge:
          ltc.increaseWeight(edge, next_edge, directed_lattice = True)
          moves.append((edge, next_edge))
    print "--- Processed %d rows" % ((i + 1) * chunk_size)
    
  # Load in the validation data
  print "-- Loading validation data"
  validation = pd.read_csv(filepath_or_buffer = filepath_val,
                           sep = ",",
                           converters = {"POLYLINE": lambda x: json.loads(x),
                                         "TRUNC_POLYLINE": lambda x: json.loads(x),
                                         "START_POINT": lambda x: eval(x),
                                         "GRID_POLYLINE": lambda x: eval(x),
                                         "TRUNC_GRID_POLYLINE": lambda x: eval(x)})
  
  print "-- Starting simulations"
  np.random.seed(seed = 654321)
  # Loop through every trip to make a prediction
  for idx, partial_trip in validation.iterrows():
    # Get unique moves, so that we dont start from a (cell, cell) edge
    trip_moves = [(cell, next_cell) for cell, next_cell in zip(partial_trip.TRUNC_GRID_POLYLINE[:-1], partial_trip.TRUNC_GRID_POLYLINE[1:]) if cell != next_cell]
    if not trip_moves:
      # In this case, the partial trip consists of only moves within the same cell.
      # To start the random walk, we need to move from one cell to the current cell.
      # Therefore we randomly sample a previous cell prev_cell, and start the
      # random walk from the edge (prev_cell, current_cell)
      possible_moves = [-1, 0, 1]
      move = [0, 0]
      
      while move[0] == 0 and move[1] == 0:
        move = np.random.choice(a = possible_moves, size = 2)
      
      current_cell = partial_trip.TRUNC_GRID_POLYLINE[-1]
      prev_cell = (current_cell[0] + move[0], current_cell[1] + move[1])
      
      start_edge = (prev_cell, current_cell)
    else:
      start_edge = trip_moves[-1]
      
    # Initialize list of simulated destinations
    destinations = []
    
    for simulation in xrange(S):
      # Initialize the walker
      wlk = Walker(lattice = ltc, start = start_edge, alpha = 0.005, dest_threshold = 1)
      
      # Simulate a random walk and record the destination
      move_to_dest = wlk.simulateWalker()
      destinations.append(move_to_dest[1])
  
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
  
  print "- Finished second-order random walk simulation script"