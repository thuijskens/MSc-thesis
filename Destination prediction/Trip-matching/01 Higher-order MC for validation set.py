# -*- coding: utf-8 -*-
""" 01 Higher-orer MC for validation set.py
This script assumes a higher order Markov chain model for a given taxi trip.
Given a partial trip, we loop through the complete training set and find all 
trajectories that that contain the sequence of last k transitions of the 
partial trip.

A quick destination density estimate can then be obtained by using the
empirical distribution of the destinations of the matched trips.

Alternatively, we can employ a random walk model where the transitional 
probabilities are only estimated from the matched trips.
"""

from __future__ import division
from DestinationGrid import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def match(segment, trip, k = 3):
  """ match
      Returns true if the segmenth of length k can be found in trip.
      
      Arguments:
      ----------
      segment: list of tuples
        List that contains the grid cells of the segment.
      trip: list of tuples
        List that contains the grid cells of the trip.
      k: integer
        Length of segment that has to be matched.
        
      Returns:
      --------
      match: Boolean
        True if segment is found in trip, False otherwise.
  """
  for idx in range(k, len(trip) + 1):
    partial = trip[(idx - k):idx]
    if partial == segment:
      return True
    
  return False
  
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
  filepath_processed = "E:/MSc thesis/Processed data/val_posteriors_trip_matching.csv"
  
  # From visual inspection of porto map. We are only focusing on the city centre
  lon_vals = (-8.73, -8.5)
  lat_vals = (41.1, 41.25)   
  
  # Number of points used for the discretization
  N, M = (100, 75)
  
  # Markov chain order
  k = 3
  
  # Define function that transforms cells to ids
  id_to_nr = lambda (i, j): N * j + i # (i + 1) 
  
  # Define distance function
  dist = lambda x1, x2: np.sqrt(np.power(x1[0] - x2[0], 2) + np.power(x1[1] - x2[1], 2))
  
  # Import the training data in chunks
  train = pd.read_csv(filepath_or_buffer = filepath,
                      sep = ",",
                      #nrows = 1000,
                      chunksize = 10000,
                      usecols = ["GRID_POLYLINE"],
                      converters = {"GRID_POLYLINE": lambda x: eval(x)})
  
  # Import the validation data
  validation = pd.read_csv(filepath_or_buffer = filepath_val,
                           sep = ",",
                           usecols = ["TRIP_ID", "START_POINT", "GRID_POLYLINE", "TRUNC_GRID_POLYLINE"],
                           converters = {"START_POINT": lambda x: eval(x),
                                         "GRID_POLYLINE": lambda x: eval(x),
                                         "TRUNC_GRID_POLYLINE": lambda x: eval(x)})
                                         
  # Loop through every trip to make a prediction
  for nr, partial_trip in validation.iterrows():                                                                
    # Select the last segment of the partial trip
    end = partial_trip.TRUNC_GRID_POLYLINE[-k:]
    
    # Create a DestinationGrid object
    posterior = DestinationGrid(N, M)
  
    # Match this segment with trips in the training set
    for i, chunk in enumerate(train):
      # Only look at trips that start from the same position
      trips = chunk[chunk.START_CELL == partial_trip.GRID_POLYLINE[0]]
      
      for idx, row in trips.iterrows():
        if match(end, row.GRID_POLYLINE, k):
          destination = row.GRID_POLYLINE[-1]
          posterior._table[destination] += 1
          
    posterior.normalizeProbs()   
    
    # Export the probabilities
    dests = [id_to_nr(dest) for dest in posterior]
    dists = [haversine(coords_from_cell(dest), partial_trip.START_POINT) for dest in posterior]
    probs = posterior._table.values()
    
    df_probs = pd.DataFrame({"TRIP_ID": partial_trip.TRIP_ID,
                             "DEST_CELL": dests,
                             "PROB": probs,
                             "DISTANCE": dists})
                             
    # Filter out the cells that never are a destination
    df_probs = df_probs[df_probs.PROB != 0.0]
    
    if idx == 0:
      df_probs.to_csv(filepath_processed, header = True, index = False)
    else:
      df_probs.to_csv(filepath_processed, mode = "a", header = False, index = False)
    
    print "--- Processed trip %d of %d" % (idx + 1, len(validation))