# -*- coding: utf-8 -*-
""" destinationGrid.py
    This script contains code for the DestinationGrid class used to store the
    posterior distributions over the destinations.
"""
from __future__ import division

import numpy as np
import pandas as pd

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
   