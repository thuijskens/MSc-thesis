# -*- coding: utf-8 -*-
"""
02 Compute destination posterior.py

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
  
  # Grid parameters
  #N, M = (100, 75)
  N, M = (150, 100)  
  
  # Probability parameter for efficient driving likelihood
  p = 0.6
  
  # Import the data in chunks with pandas read_csv
  test = pd.read_csv(filepath_or_buffer = filepath_val,
                     sep = ",",
                     nrows = 10,
                     #chunksize = chunk_size,
                     converters = {#"TIMESTAMP" : lambda x: datetime.datetime.fromtimestamp(x),
                                   "POLYLINE": lambda x: json.loads(x),
                                   "GRID_POLYLINE": lambda x: eval(x)})
  
  # Begin by just looking at one trip
  partial_trip = test.iloc[1, :]
  
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
  target_data = pd.read_csv(filepath_or_buffer = filepath_target,
                            sep = ",",
                            converters = {"START_CELL": lambda x: eval(x),
                                          "DESTINATION": lambda x: eval(x)})
  # Get the probabilities for the current starting cell                                        
  target_data = target_data[target_data.START_CELL == partial_trip.GRID_POLYLINE[0]]
    
  # Update the target
  target.setProbs(target_data.DESTINATION.values, target_data.PERCENT.values)
  
  # Melt this target with a uniform prior to get a "open-world" prior
  target = target.melt(shrinkage)
  
  # Finally melt the target with the data-based prior
#  prior = prior.melt(target)
  prior = prior.melt(shrinkage)
  
  # Define distance function
  dist = lambda x1, x2: np.sqrt(np.power(x1[0] - x2[0], 2) + np.power(x1[1] - x2[1], 2))
  
  # Define a time decay function
  alpha = 0.0001
  decay = lambda n: 1/(1 + alpha) ** n
  
  #middle of trip
  mid = len(partial_trip.GRID_POLYLINE) // 2
  
  # Update the posterior probability for every destination cell
  for destination in posterior:
    # Compute the probability P[X^p | D]
    probability = 1 # Probability of partial trip
    dist_to_dest = dist(destination, partial_trip.GRID_POLYLINE[0]) # Distance of start cell to destination
    closest = dist_to_dest # Closest distance to the destination
    
    for cell in partial_trip.GRID_POLYLINE[1:]:
      dist_to_dest = dist(destination, cell)
      if dist_to_dest < closest:
        closest = dist_to_dest
        probability *= p
      else:
        probability *= 1 - p
    
    # Decay for far away trips (distance between end point of a trip and the
    # current destination is used to judge "far away")
    #probability *= decay(dist_to_dest)
      
    # Compute the posterior probability
    posterior.setProb(destination, probability * prior.getProb(destination))
    
  # Normalize the posterior distribution
  posterior.normalizeProbs()
  
  # Plot the new distribution
  
  # Trip superimposed on posterior
  plt.subplot(1,2,1)
  plt.imshow(np.log(posterior.as_array() + trip_to_array(partial_trip.GRID_POLYLINE, N, M)), interpolation = "nearest")
  
  plt.subplot(1,2,2)
  plt.imshow(np.log(prior.as_array()), interpolation = "nearest")
  
  # Log-scale
  plt.subplot(1,3,1)
  plt.imshow(np.log(posterior.as_array()), interpolation = "nearest")
  
  plt.subplot(1,3,2)
  plt.imshow(np.log(trip_to_array(partial_trip.GRID_POLYLINE)), interpolation = "nearest")
  
  plt.subplot(1,3,3)
  plt.imshow(np.log(prior.as_array()), interpolation = "nearest")
  
  # Normal scale
  plt.subplot(1,3,1)
  plt.imshow(posterior.as_array(), interpolation = "nearest")
  
  plt.subplot(1,3,2)
  plt.imshow(trip_to_array(partial_trip.GRID_POLYLINE, N, M), interpolation = "nearest")
  
  plt.subplot(1,3,3)
  plt.imshow(prior.as_array(), interpolation = "nearest")
  
  
  

                                   
                                   
                                   
                                   
                                   
                                   
                                   
                                   
                                   