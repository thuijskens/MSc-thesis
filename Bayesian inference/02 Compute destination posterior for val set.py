# -*- coding: utf-8 -*-
"""
02 Compute destination posterior for val set.py

This script computes posterior distribution of the destination for a partial
trajectory, based on Bayes' formula

P[D | X^P] \propto P[X^P] P[D],

where P[D] represents a prior distribution based on the origin (call, stand, 
street) of the trip. X^P = (X^P_0, ..., X^P_N_i) is the partial trajectory, 
and we can write

P[X^P | D] = \prod_{j = 2}^N_i  P[X^P_j | D, X_{1:(j-1)}]

with

X^P_j | D, X_{1:(j-1)} = 1_{if ||X_j - D|| < ||X_k - D|| for all k < j},

so that X^P_j | D, X_{1:(j-1)} ~ Ber(p). [Efficient driving likelihood]
"""

from __future__ import division

import json
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

class DestinationGrid(object):         
  def __init__(self, N, M, init_prob = 0):   
    self.N = N
    self.M = M     
    
    # Initialize the table by filling it with zeros
    self._table = {}
    
    for i in xrange(N):
      for j in xrange(M):
        self._table[(i, j)] = init_prob
      
  def __iter__(self):
    return iter(self._table.keys())     
    
  def as_array(self):
    # Initialize array
    arr = np.zeros((self.N, self.M))
    
    # Fill array with probabilities
    for cell, prob in self._table.items():
      arr[cell] = prob
    
    # Flip matrix for easy plotting
    return arr[:, ::-1].T
  
  def getProb(self, cell):
    return self._table[cell]
    
  def importProbs(self, filepath, standID = None, callID = None):
    # Read in probabilities from filepath
    data = pd.read_csv(filepath_or_buffer = filepath, 
                       sep = ",",
                       converters = {"DESTINATION" : lambda x: eval(x)})   
    
    if standID is None and callID is None:
      # Hailed from the streets
      mask = (data.ORIGIN_STREET == True)
    elif standID is None and callID is not None:
      # Ordered by phone
      mask = (data.ORIGIN_CALL == callID)
    elif callID is None and standID is not None:
      # Ordered from a stand
      mask = (data.ORIGIN_STAND == standID)
      
    # Only look at the required probabilities
    data = data[mask]
        
    # Reset current probabilities
    for key in self._table.keys():
      self._table[key] = 0
    
    # Pass the cells and probabilities as numpy arrays to setProbs 
    self.setProbs(data.DESTINATION.values, data.PERCENT.values)
  
  def melt(self, other, lmbd = None):
    if self.N != other.N or self.M != other.M:
      raise ValueError("Different grids must be of the same size.")
    if lmbd is not None and (lmbd < 0 or lmbd > 1):
      raise ValueError("Lambda should be between 0 and 1, got %s" % lmbd)
      
    result = DestinationGrid(self.N, self.M)    
    
    if lmbd is None:
      # Use shrinkage estimator
      lmbd = (1 - sum([target ** 2 for target in other._table.values()])) / ((self.N * self.M  - 1) * sum([(target - orig) ** 2 for (orig, target) in zip(self._table.values(), other._table.values())]))
     
    # Format probabilities
    new_probs = [(1 - lmbd) * orig + lmbd * target for (orig, target) in zip(self._table.values(), other._table.values())]
    result.setProbs(self._table.keys(), new_probs)

    return result
    
  def normalizeProbs(self):
    # Compute the normalizing constant
    probSum = np.sum([prob for prob in self._table.values()])
    
    # Normalize the probabilities
    for cell, prob in self._table.items():
      self.setProb(cell, prob / probSum)
    
  def setProb(self, cell, probability):
    self._table[cell] = probability
    
  def setProbs(self, cells, probability):
    for idx in xrange(len(cells)):
      self._table[cells[idx]] = probability[idx]
     
     
def trip_to_array(trip, N = 100, M = 75):
  """ trip_to_array
  Converts a trip to a two-dimensional binary numpy array
  
  Arguments:
  ----------
  trip: list of tuples
  
  Returns:
  --------
    out: numpy array
  """
  
  # Initialize the array
  grid_array = np.zeros((N, M))
  
  # Go through the list linearly and set each bit on when necessary
  for x, y in trip:
    grid_array[x, y] = 1
  
  # Flip the matrix so that the array reflects the pattern in the same orientation
  # as the map    
  return grid_array[:, ::-1].T
   
   
if __name__ == "__main__":
  print "- 02 Compute destination posterior.py"
  
  # Define the filepaths
  filepath_val = "E:/MSc thesis/Processed data/train_binarized_trips_validation.csv"
  filepath_probs = "E:/MSc thesis/Processed data/prob_tables.csv"
  filepath_target = "E:/MSc thesis/Processed data/prob_tables_shrinkage.csv"
  filepath_processed = "E:/MSc thesis/Processed data/val_posteriors.csv"
  
  # Grid parameters
  N, M = (100, 75)
  
  # Probability parameter for efficient driving likelihood
  p = 0.6
  
  # Define distance function
  dist = lambda x1, x2: np.sqrt(np.power(x1[0] - x2[0], 2) + np.power(x1[1] - x2[1], 2))
    
  # Import the data in chunks with pandas read_csv
  validation = pd.read_csv(filepath_or_buffer = filepath_val,
                           sep = ",",
                            #nrows = 10,
                            #chunksize = chunk_size,
                           converters = {#"TIMESTAMP" : lambda x: datetime.datetime.fromtimestamp(x),
                                         "POLYLINE": lambda x: json.loads(x),
                                         "START_POINT": lambda x: eval(x),
                                         "GRID_POLYLINE": lambda x: eval(x),
                                         "TRUNC_POLYLINE": lambda x: eval(x),
                                         "TRUNC_GRID_POLYLINE": lambda x: eval(x)})
  
  # Load the probabilities for the target distribution
  target_prob = pd.read_csv(filepath_or_buffer = filepath_target,
                            sep = ",",
                            converters = {"START_CELL": lambda x: eval(x),
                                          "DESTINATION": lambda x: eval(x)})
                                          
  # Process each partial trip one at a time                                        
  for idx, partial_trip in validation.iterrows():
    # Initialize the prior distribution for the partial trip
    posterior = DestinationGrid(N, M)
    prior = DestinationGrid(N, M)  
    
    if partial_trip.ORIGIN_CALL >= 0:
      prior.importProbs(filepath_probs, callID = partial_trip.ORIGIN_CALL)
    elif partial_trip.ORIGIN_STAND >= 0:
      prior.importProbs(filepath_probs, standID = partial_trip.ORIGIN_STAND)
    else:
      prior.importProbs(filepath_probs)
    
    # Shrinkage estimator (same as categorical likelihood and dirichlet prior?)
    shrinkage = DestinationGrid(N, M, 1 / (N*M))
    
    # We set up the target distribution
    target = DestinationGrid(N, M)
                                            
    # Get the probabilities for the current starting cell                                        
    target_data = target_prob[target_prob.START_CELL == partial_trip.TRUNC_GRID_POLYLINE[0]]
      
    # Update the target
    target.setProbs(target_data.DESTINATION.values, target_data.PERCENT.values)
    
    # Melt this target with a uniform prior to get a "open-world" prior
    target = target.melt(shrinkage)
    
    # Finally melt the target with the data-based prior
    prior = prior.melt(target)
    
    # Update the posterior probability for every destination cell
    for destination in posterior:
      # Compute the probability P[X^p | D]
      probability = 1 # Probability of partial trip
      dist_to_dest = dist(destination, partial_trip.TRUNC_GRID_POLYLINE[0]) # Distance of start cell to destination
      closest = dist_to_dest # Closest distance to the destination
      
      for cell in partial_trip.TRUNC_GRID_POLYLINE[1:]:
        dist_to_dest = dist(destination, cell)
        if dist_to_dest < closest:
          closest = dist_to_dest
          probability *= p
        else:
          probability *= 1 - p
        
      # Compute the posterior probability
      posterior.setProb(destination, probability * prior.getProb(destination))
      
    # Normalize the posterior distribution
    posterior.normalizeProbs()
    
    # Export the probabilities
    df_probs = pd.DataFrame(columns = ["TRIP_ID", "DEST_CELL", "PROB", "DISTANCE"])
    for dest, prob in posterior._table.iteritems():
      dst = haversine(coords_from_cell(dest), partial_trip.START_POINT)
      new_line = pd.DataFrame({"TRIP_ID": partial_trip.TRIP_ID,
                               "DEST_CELL": str([dest[0], dest[1]]),
                               "PROB": prob,
                               "DISTANCE": dst},
                               index = np.arange(1))
      df_probs = df_probs.append(new_line, ignore_index = True)

    if idx == 0:
      df_probs.to_csv(filepath_processed, header = True, index = False)
    else:
      df_probs.to_csv(filepath_processed, mode = "a", header = False, index = False)
    
    print "-- Processed trip %d" % idx
    
  

                                   
                                   
                                   
                                   
                                   
                                   
                                   
                                   
                                   