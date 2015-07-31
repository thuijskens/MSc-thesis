# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class Grid:
  """ Grid
      Placeholder class that holds the grid representation of a trip
  """
  def __init__(self, grid, lon_grid, lon_step, lat_grid, lat_step):
    self.grid = grid
    self.lon_grid = lon_grid
    self.lon_step = lon_step
    self.lat_grid = lat_grid
    self.lat_step = lat_step
    
  def __str__(self):
    return str(self.grid)
    
  def grid_to_array(self):
    N = len(self.grid)
    grid_array = np.zeros((N, N))
    
    for cell in self.grid:
      grid_array[(cell[0] - 1, cell[1] - 1)] += 1
      
    return grid_array
    
  def PlotGrid(self, ax, plot_grid_lines = False):
    """ PlotGrid
        Plots the grid representation of the gray, where is cell is colored 
        gray if the trip passed through that cell.
        
        Arguments:
        fig: matplotlib figure
        plot_grid_lines: boolean
        
        Returns:
        fig: matplotlib figure
        ax: matplotlib axes
    """    
    
    # If there is no figure given to the function
    #if fig is None or ax is None:
    #  fig, ax = plt.subplots()
    
    #For each cell in the grid representation, we plot a shaded grey box
    for cell in self.grid:
      # Compute the coordinates of the cell
      lon_min = self.lon_grid[0] + cell[0] * self.lon_step
      lon_max = lon_min + self.lon_step
      
      lat_min = self.lat_grid[0] + cell[1] * self.lat_step
      lat_max = lat_min + self.lat_step
      
      # Plot the shaded cell
      ax.fill([lon_min, lon_max, lon_max, lon_min], [lat_min, lat_min, lat_max, lat_max], 'k', alpha = 0.05)
      
    if plot_grid_lines:
      # Hack so we get the same x and y axis ranges
      x_range = ax.get_xlim()
      y_range = ax.get_ylim()
      
      # We plot the grid lines as well
      ax.set_yticks(self.lat_grid, minor = False)
      ax.set_xticks(self.lon_grid, minor = False)
      
      # Show the gridlines
      ax.yaxis.grid(True)
      ax.xaxis.grid(True)
      
      # Reset the ranges for the x and y axis
      ax.set_xlim(x_range)
      ax.set_ylim(y_range)
      
    return ax