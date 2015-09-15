# -*- coding: utf-8 -*-
""" lattice.py
    Implements class types for Lattice and Node objects.
"""
from __future__ import division

import numpy as np

class Cell(object):
  def __init__(self, key, cellAsEdge = False, cellWeight = 0):
    self.id = key
    self.neighbours = {}
    self.timesReached = 0
    
    if cellAsEdge:
      self.addNeighbour(self)
    
    if not isinstance(self.id, tuple):
      raise ValueError("key has to be a tuple, got %s" % self.id)
    
  def addNeighbour(self, neighbour, weight = 0):
    self.neighbours[neighbour] = weight
  
  def getNeighbours(self, returnWeights = False):
    if returnWeights:
      return self.neighbours
    else:
      return self.neighbours.keys()
  
  def getWeight(self, neighbour):
    if neighbour in self.neighbours:
      return self.neighbours[neighbour]
    else:
      return None
      
  def removeNeighbour(self, neighbour):
    del self.neighbours[neighbour]
      
  def setWeight(self, neighbour, weight):
    if neighbour in self.neighbours:
      self.neighbours[neighbour] = weight
    else:
      return None
  
  def getTransProbs(self):
    normConstant = sum([ weight for weight in self.neighbours.values() ])
      
    transProbs = {}
    for neighbour, weight in self.neighbours.iteritems():
      if normConstant == 0:
        transProbs[neighbour] = 0
      else:
        transProbs[neighbour] = weight / normConstant
    
    return transProbs
    
class Lattice(object):
  def __init__(self, leftCorner, rightCorner, nrCells):
    self._cells_x = nrCells[0]
    self._cells_y = nrCells[1]
    self._min_x = leftCorner[0]
    self._min_y = leftCorner[1]
    self._max_x = rightCorner[0]
    self._max_y = rightCorner[1]
    
    # Create the lattice
    self._lattice = {}
    
    # First, we add only the cells in the lattice
    self.addCells( (i,j) for i in range(self._cells_x) for j in range(self._cells_y) )
    
    # Then we add the edges in the lattice    
    for i in xrange(self._cells_x):
      for j in xrange(self._cells_y):
        if i - 1 > 0:
          self.addEdge((i, j), (i - 1, j))
          if j - 1 > 0:
            self.addEdge((i, j), (i - 1, j - 1))
          if j + 1 < self._cells_y:
            self.addEdge((i, j), (i - 1, j + 1))
            
        if i + 1 < self._cells_x:
          self.addEdge((i, j), (i + 1, j))
          if j - 1 > 0:
            self.addEdge((i, j), (i + 1, j - 1))
          if j + 1 < self._cells_y:
            self.addEdge((i, j), (i + 1, j + 1))
            
        if j - 1 > 0:
          self.addEdge((i, j), (i, j - 1))
        
        if j + 1 < self._cells_y:
          self.addEdge((i, j), (i, j + 1))
          
  def __iter__(self):
    return iter(self._lattice.values())  
    
  def addCell(self, key):
    self._lattice[key] = Cell(key)
    
  def addCells(self, keys):
    for key in keys:
      self.addCell(key)
  
  def addEdge(self, u, v, weight = 0, directed_lattice = False):
    if u not in self._lattice:
      self.addCell(u)
    if v not in self._lattice:
      self.addCell(v)
    
    self._lattice[u].addNeighbour(self._lattice[v], weight)
    
    if not directed_lattice:
      self._lattice[v].addNeighbour(self._lattice[u], weight)
  
  def addEdges(self, edges):
    """
    edges must be given as a 3-tuple (u, v, w)
    """
    for edge in edges:
      self.addEdge(edge[0], edge[1], edge[2])
  
  def getCell(self, cell):
    if cell in self._lattice:
      return self._lattice[cell]
    else:
      return None
  
  def getNeighbours(self, cell, returnWeights = False):
    if cell in self._lattice:
      neighbours = self._lattice[cell].getNeighbours(returnWeights)
      
      if returnWeights:
        out = {}
        
        for neighbour, weight in neighbours.iteritems():
          out[neighbour.id] = weight
          
        return out
      else:
        return [neighbour.id for neighbour in neighbours]
        
    else:
      return None
           
  def increaseWeight(self, u, v, by = 1, directed_lattice = False):
    self.setWeight(u, v, self._lattice[u].getWeight(self._lattice[v]) + by, directed_lattice)
      
  def removeEdge(self, u, v, directed_lattice = False):
    if u not in self._lattice:
      raise ValueError("Cell %s is not in the lattice" % u)
    if v not in self._lattice:
      raise ValueError("Cell %s is not in the lattice" % v)
    
    self._lattice[u].removeNeighbour(v)
    
    if not directed_lattice:
      self._lattice[v].removeNeighbour(u)
  
  def setWeight(self, u, v, weight, directed_lattice = False):
    if u not in self._lattice:
      raise ValueError("Cell %s is not in the lattice" % u)
    if v not in self._lattice:
      raise ValueError("Cell %s is not in the lattice" % v)
    
    self._lattice[u].setWeight(self._lattice[v], weight)
    
    if not directed_lattice:
      self._lattice[v].setWeight(self._lattice[u], weight)

class SecondOrderLattice(Lattice):
  """ SecondOrderLattice class
      This class inherits from Lattice and is used to model the second-order 
      random walk.
      
      Instead of regular cells, the nodes in this lattice represent the edges 
      from the city graph. Instead of the cell number (i, j), we now encode 
      keys according to
        (old_cell_id, new_cell_id)
      where the cell_id is computed from (i, j) --> j*N + i. This way the node
      actually represents a movement. 
  """
    
  def __init__(self, leftCorner, rightCorner, nrCells):
    self._cells_x = nrCells[0]
    self._cells_y = nrCells[1]
    self._min_x = leftCorner[0]
    self._min_y = leftCorner[1]
    self._max_x = rightCorner[0]
    self._max_y = rightCorner[1]
    
    # Create the lattice
    self._lattice = {}
    
    # Now we need to add each edge as a node in the lattice.
    # First get a list of all possible cells in the city grid
    grid_cells = [(i, j) for i in xrange(self._cells_x) for j in xrange(self._cells_y)]
    
    # Then we fill the lattice with all possible moves as keys
    for grid_cell in grid_cells:
      moves = self.get_moves(grid_cell)
      for move in moves:
        self._lattice[move] = Cell(key = move)
        
    # Now we need to add the possible moves between edges as neighbours 
    for edge in self._lattice.keys():
      new_cell = edge[1] # Get the end cell of a move
      # Compute all possible new moves from that end cell
      new_moves = self.get_moves(new_cell)
      for new_move in new_moves:
        self.addEdge(edge, new_move, directed_lattice = True)
        
  def get_moves(self, cell): 
    i, j = cell
    moves = []
    
    if i - 1 >= 0:
      moves.append(((i, j), (i - 1, j)))
      if j - 1 >= 0:
        moves.append(((i, j), (i - 1, j - 1)))
      if j + 1 < self._cells_y:
        moves.append(((i, j), (i - 1, j + 1)))
        
    if i + 1 < self._cells_x:
      moves.append(((i, j), (i + 1, j)))
      if j - 1 >= 0:
        moves.append(((i, j), (i + 1, j - 1)))
      if j + 1 < self._cells_y:
        moves.append(((i, j), (i + 1, j + 1)))

    if j - 1 >= 0:
      moves.append(((i, j), (i, j - 1)))
    
    if j + 1 < self._cells_y:
      moves.append(((i, j), (i, j + 1)))

    return moves
    
    
    
class Walker(object):
  def __init__(self, lattice, start, alpha = 0.01, dest_threshold = 3, rng_seed = None):
    self.position = lattice.getCell(start)
    self.path = [self.position.id]
    self.iter = 0
    self.decay = lambda t: 1 / (1 + alpha) ** t
    
    self._dest_threshold = dest_threshold
    self._lattice = lattice
    self._same_cell_count = 0
    self._rng = np.random.RandomState(rng_seed)
    
    if self.position not in self._lattice:
      raise ValueError("Starting cell %s is not a cell in the given lattice" % start)
    
  def simulateStep(self):
    # Increate the time
    self.iter += 1
    
    # Get the neighbouring cells and the transition probabilities
    probs = self.position.getTransProbs()
      
    # To force the random walk to stay in a cell (the destination) over time,
    # we use a decay function for the transition probabilities to the 
    # neighbouring cells
    scale = self.decay(self.iter)
    probSum = 0
    for cell in probs.keys():
      if cell is not self.position:
        probs[cell] *= scale
        probSum += probs[cell]

    # Add the probability that the walker does not move
    probs[self.position] = 1 - scale
    
    """ Thomas @ 29-07: This should do the same as the above line
    # Make sure the probabilities sum up to 1
    """
    probs[self.position] = 1 - probSum

    
    # Sample one of the possibilities
    newCell = self._rng.choice(a = probs.keys(), p = probs.values())
    
    # Check if the new cell is the same as the old cell
    if self.position == newCell:
      self._same_cell_count += 1
    else:
      self._same_cell_count = 0
    
    # Set the new cell as the current position and add it to the walker's path
    self.position = newCell
    self.path.append(newCell.id)
    
    return self.position
  
  def simulateWalker(self, T = None):
    if T is not None:
      for t in xrange(T):
        self.simulateStep()
    
    else:
      # If the walker stays in the same cell longer than the threshold, then the
      # walker has reached its destination
      while self._same_cell_count <= self._dest_threshold:
        self.simulateStep()
    
    return self.path[-1]
  