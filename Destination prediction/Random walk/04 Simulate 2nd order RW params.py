# -*- coding: utf-8 -*-
"""
04 Simulate RW params.py
"""

from __future__ import division
from lattice import *
from DestinationGrid import *
from random import choice
from scipy.stats import entropy

import numpy as np
import pandas as pd
from numpy import cumsum
from numpy.random import rand

def weightedChoice(weights, objects):
    """Return a random item from objects, with the weighting defined by weights 
    (which must sum to 1)."""
    cs = cumsum(weights) #An array of the weights, cumulatively summed.
    idx = sum(cs < rand()) #Find the index of the first weight over a random value.
    return objects[idx]

if __name__ == "__main__":
  filepath = "E:/MSc thesis/Processed data/train_binarized_trips_train.csv"
  filepath_len_distr = "E:/Msc thesis/Processed data/length_distribution.csv"
  
  len_distr = pd.read_csv(filepath_or_buffer = filepath_len_distr)
  
  # Define the k and lambda-grid to search over
  k_grid = range(1, 11)
  gamma_grid = [0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
  
  # Need to get the transitional probabilities from the original data
  
  """
  # Port code from 01 Random walk.py
  """
  
  # From visual inspection of porto map. We are only focusing on the city centres
  lon_vals = (-8.73, -8.5)
  lat_vals = (41.1, 41.25)   
  
  # Number of points used for the discretization
  N = 100
  M = 75
  
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
                      chunksize = 100000,
                      usecols = ["GRID_POLYLINE"],
                      converters = {"GRID_POLYLINE": lambda x: eval(x)})

  """ Build the lattice and compute the weights for each edge
  Alse keep track of all the possible starting cells
  """

  print "-- Computing transitional probabilities on the road graph"
  starting_cells = {}
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
      
      if trip.GRID_POLYLINE[0] not in starting_cells:
        starting_cells[trip.GRID_POLYLINE[0]] = 1
      else:
        starting_cells[trip.GRID_POLYLINE[0]] += 1
        
           
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
    print "--- Processed %d rows" % ((i + 1) * 100000)
    
     
  sum_weights = sum(starting_cells.values())
  for cell in starting_cells.keys():
    starting_cells[cell] = starting_cells[cell] / sum_weights
    
  """
  Now we have the transitional probabilities, so we can sample a random walk on
  the city grid. We will sample S random walks from a random starting point in 
  the city, and record the final length of the trip.
  
  This is done for each possible configuration of k and gamma. The KL-divergence
  between the simulated and empirical distribution is calculated and stored
  """
  
  # Define number of simulations
  S = 10000
  
  # Get the empirical distribution of the trip length
  q = {}
  for idx, row in len_distr.iterrows():
    q[row.LENGTH] = row.COUNT
  
  # Placeholder variable for KL-divergence
  KL_divergence = {}

  # Set the seed
  np.random.seed(987654321)
  
  for k in k_grid:
    for gamma in gamma_grid:
      # Placeholder variable for the simulated trip lengths
      sim_lengths = {}
      for sim in range(S):
        # Sample a random starting point
        current_cell = weightedChoice(weights = starting_cells.values(), objects = starting_cells.keys())
        
        # Get unique moves, so that we dont start from a (cell, cell) edge        
        possible_moves = [-1, 0, 1]
        move = [0, 0]
        
        while (move[0] == 0 and move[1] == 0) or (prev_cell[0] < 0 or prev_cell[1] < 0 or prev_cell[0] > 99 or prev_cell[1] > 74): 
          move = np.random.choice(a = possible_moves, size = 2)
          prev_cell = (current_cell[0] + move[0], current_cell[1] + move[1])

        start_edge = (prev_cell, current_cell)
        
        # Simulate a random walk
        rw = Walker(lattice = ltc, start = start_edge, dest_threshold = k, alpha = gamma)
        rw.simulateWalker()        
        
        # Get the length
        length_rw = len(rw.path) - 1
        
        if length_rw not in sim_lengths:
          sim_lengths[length_rw] = 1
        else:
          sim_lengths[length_rw] += 1
          
      print "Processed simulations for k = %d and gamma = %.3f" % (k, gamma)
      # compute the KL-divergence with the original distribution
      
      KL_divergence[(k, gamma)] = compute_KL_divergence(sim_lengths, q)
      
      print "KL-divergence for k = %d and gamma = %.3f is %.5f" % (k, gamma, KL_divergence[(k, gamma)])
        
  k_vals = []
  gamma_vals = []
  KL_div = []
  
  for key, val in KL_divergence.iteritems():
    k_vals.append(key[0])
    gamma_vals.append(key[1])
    KL_div.append(val)
    
  df_KL_divergence = pd.DataFrame({"k": k_vals,
                                   "gamma": gamma_vals,
                                   "KL_divergence": KL_div})
  df_KL_divergence.to_csv("E:/MSc thesis/Destination prediction/Random walk/second_order_params.csv", header = True, index = False)
  
  min_params = df_KL_divergence.loc[df_KL_divergence["KL_divergence"].idxmin()]
      
def compute_KL_divergence(p, q, shrinkage = 0.001):
  """
  Input:
    p: dict
    q: dict
  """
  p_max = max(p.keys())
  q_max = max(q.keys())
  
  range_max = max(p_max, q_max)
  
  #prob_array = np.zeros((range_max, 2)) + 0.000000001
  prob_array = np.zeros((range_max, 2))  
  
  for l in range(range_max):
    if l in p:
      prob_array[l, 0] = p[l]
    if l in q:
      prob_array[l, 1] = q[l]
  
  # Use a bayesian estimate
  # source: http://mathoverflow.net/questions/72668/how-to-compute-kl-divergence-when-pmf-contains-0s
  nr_atoms = np.sum(prob_array[:, 0] == 0)
  total_nr = np.sum(prob_array[:, 0])
  
  for l in range(range_max):
    prob_array[l, 0] = (prob_array[l, 0] + 1)/(total_nr + nr_atoms)
    
  prob_array[:, 1] = prob_array[:, 1] / np.sum(prob_array[:, 1])
  
  return entropy(pk = prob_array[:, 1], qk = prob_array[:, 0])
  


