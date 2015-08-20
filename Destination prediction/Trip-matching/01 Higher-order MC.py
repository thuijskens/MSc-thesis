# -*- coding: utf-8 -*-
""" 01 Higher-orer MC.py
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

def match(segment, trip, k):
  for idx in range(k, len(trip) + 1):
    partial = trip[(idx - k):idx]
    if partial == segment:
      return True
    
  return False
  
if __name__ == "__main__":
  # Location of the data
  filepath = "E:/MSc thesis/Processed data/train_binarized_trips_train.csv"
  filepath_val = "E:/MSc thesis/Processed data/train_binarized_trips_validation.csv"
  
  # From visual inspection of porto map. We are only focusing on the city centre
  lon_vals = (-8.73, -8.5)
  lat_vals = (41.1, 41.25)   
  
  # Number of points used for the discretization
  N = 100
  M = 75
  
  # Markov chain order
  k = 3
  
  # Import the training data in chunks
  trips = pd.read_csv(filepath_or_buffer = filepath,
                      sep = ",",
                      nrows = 1000,
                      #chunksize = 10000,
                      usecols = ["POLYLINE", "GRID_POLYLINE"],
                      converters = {"POLYLINE": lambda x: json.loads(x),
                                    "GRID_POLYLINE": lambda x: eval(x)})
  
  # Import the validation data
  test = pd.read_csv(filepath_or_buffer = filepath_val,
                     sep = ",",
                     nrows = 10,
                     converters = {#"TIMESTAMP" : lambda x: datetime.datetime.fromtimestamp(x),
                                   "POLYLINE": lambda x: json.loads(x),
                                   "START_POINT": lambda x: eval(x),
                                   "GRID_POLYLINE": lambda x: eval(x),
                                   "TRUNC_POLYLINE": lambda x: eval(x),
                                   "TRUNC_GRID_POLYLINE": lambda x: eval(x)})
                                                                      
  # Select a partial trip                              
  partial_trip = test.loc[0]
  
  trips["DEST_CELL"] = trips.GRID_POLYLINE.map(lambda x: x[-1])
  
  # Select the last segment of the partial trip
  end = partial_trip.TRUNC_GRID_POLYLINE[-k:]
  
  # Create a dummy flag to indicate if a trip matches or not
  trips["MATCH"] = False
  
  # Match this segment with trips in the training set
  for idx, row in trips.iterrows():
    if match(end, row.GRID_POLYLINE, k):
      trips.set_value(idx, 'MATCH', True)
    
  # We are only really interested in the matched trips now
  trips = trips[trips.MATCH]
  
  # Compute the destination distribution
  trips_agg = trips.groupby(["DEST_CELL"], as_index = False).aggregate({"MATCH": "sum"})
  trips_agg.MATCH = trips_agg.MATCH / np.sum(trips_agg.MATCH)
  
  # Store it in a DestinationGrid object
  grid = DestinationGrid(N, M)
  grid.setProbs(trips_agg.DEST_CELL.values, trips_agg.MATCH.values)
  
  # Plot the new distribution
  plt.subplot(1,2,1)
  plt.imshow(np.log(grid.as_array() + trip_to_array(partial_trip.GRID_POLYLINE, N, M)), interpolation = "nearest")
  plt.title("Complete trip superimposed")
  
  plt.subplot(1,2,2)
  plt.imshow(np.log(grid.as_array() + trip_to_array(partial_trip.TRUNC_GRID_POLYLINE, N, M)), interpolation = "nearest")
  plt.title("Partial trip superimposed")