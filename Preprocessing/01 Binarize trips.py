""" 01 binarize trips.py
    This scripts loads the file train_processed.csv and binarizes each trip by
    defining a grid over the city, and listing the grid cells the trip passes
"""

# Import relevant libraries
from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys


def find_direct_path(start, end):
  """ find_direct_path
      Connects the start and end cell by Bresenham's line algorithm and outputs a list
      of the grid cells the line crosses.
      
      Arguments:
      ----------
      start: tuple
        tuple with the coordinates of the start cell.
      end: tuple
        tuple with the coordinates of the terminal cell.
        
      Returns:
      --------
      path: list
        list of tuples containing the direct path between start and end
      
  """ 
  x1, y1 = start
  x2, y2 = end
  
  dy = y2 - y1
  dx = x2 - x1
  
  # Check if the connecting line is steep
  steep = abs(dy) > abs(dx)
  if steep:
    # If so, rotate the line
    x1, y1 = y1, x1
    x2, y2 = y2, x2

  # Swap the start and end points if necessary
  swapped = False
  if x1 > x2:
    x1, x2 = x2, x1
    y1, y2 = y2, y1
    swapped = True
    
  # Recalculate the differentials
  dy = y2 - y1
  dx = x2 - x1
  
  # Calculate the error
  error = int(dx / 2.0)
  
  # Are we moving left or right?
  ystep = 1 if y1 < y2 else -1
  
  # Iterate over the possible points between start and end
  y = y1
  path = []
  for x in range(x1, x2 + 1):
    # Bresenham's algorithm
    point = (y, x) if steep else (x, y)
    path.append(point)
    error -= abs(dy)
    
    if error < 0:
      y += ystep
      error += dx
  
  # Reverse the list if the start and end points were swapped
  if swapped:
    path.reverse()
  
  return path
  

def trip_to_grid(coords, lon_bins, lon_step, lat_bins, lat_step):
  """ trip_to_grid
      Takes the GPS coordinates of a trip and outputs a list with the grid 
      cells the trip passes through.
      
      Arguments:
      ----------
      coords: 2-dimensional numpy array
      lon_bins: 1-dimensional numpy array
      lon_step: scalar
      lat_bins: 1-dimensional numpy array
      lat_step: scaler
      
      Returns:
      --------
      grid: Grid-object
  """
  # Initialize list with grid coordinates of the trip
  grid_trip = []

  for coord in coords:
    # Determine the cell for each pair of coordinates
    cell_lon = np.max(np.where(coord[0] > lon_bins))
    cell_lat = np.max(np.where(coord[1] > lat_bins))
    new_cell = (cell_lon, cell_lat)
     
    # Add the cell coordinates to the list if it is empty
    if len(grid_trip) == 0:
      grid_trip.append(new_cell)
      continue
    
    """ Uncomment this if we only used the 'passed here'-indicator
    # If the new cell is the same as the old cell, do not add it
    if new_cell == grid_trip[-1]:
      continue
    """
    
    # Otherwise, add the cell to the list and interpolate the cells in 
    # between, if necessary. 
    if abs(new_cell[0] - grid_trip[-1][0]) >= 2 or abs(new_cell[1] - grid_trip[-1][1]) >= 2:
      # Compute the direct path between the gap start and end points
      gap_path = find_direct_path(start = grid_trip[-1], end = new_cell)
      
      # Add the path to the coordinate list
      grid_trip.extend(gap_path[1:])
    else:
      grid_trip.append(new_cell)
      
  return Grid(grid_trip, lon_bins, lon_step, lat_bins, lat_step)

def plot_grid_representation(coords, grid, plot_grid_lines = False):
  """ plot_grid_representation
      Plots the coordinates of the trip as a line plot, with the grid representation
      overlayed on the trip coordinates.
      
      Arguments:
      ----------
      coords: list
        List of GPS coordinates of the trip.
      grid: Grid-object
        Grid-object containing the grid representation of coords
      plot_grid_lines: boolean
        Boolean that indicates if the complete grid should be plotted as well.
        
      Returns:
      --------
      grid_plot: plot
        Plot of the trip with grid representation overlay.
  """  
  # Get the handle of the figure
  fig, ax = plt.subplots()
  
  # Plot the GPS coordinates of the trip
  ax.plot(coords[:,0], coords[:,1], color = "black")
  ax.scatter(coords[:,0], coords[:,1], s = 10, color = "black")

  # Plot the grid representation
  ax = grid.PlotGrid(ax, plot_grid_lines)
  
  return fig, ax
  
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
    
  def grid_to_string(self):
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

if __name__ == "__main__":
  print "- Starting binarizing script"
  
  # Get the location of the data file
  filepath = "E:/MSc thesis/Processed data/train_cleaned.csv"
  filepath_clean = "E:/MSc thesis/Processed data/train_binarized_trips.csv"
  
  # Define the bins for the grid
  # Note: 1 degree of longitude/latitude is approximately 111.38 km
  N = 100
  M = 75
  
  # From visual inspection of porto map. We are only focusing on the city centre
  # 15-07-14: We continue with this bounding box as of now
  lon_vals = [-8.73, -8.5]
  lat_vals = [41.1, 41.25]   
  
  lon_bins, lon_step = np.linspace(lon_vals[0], lon_vals[1], N, retstep = True)
  lat_bins, lat_step = np.linspace(lat_vals[0], lat_vals[1], M, retstep = True)
     
  # Use pandas read_csv function to read the data in chunks
  data_chunks = pd.read_csv(filepath_or_buffer = filepath,
                            sep = ",",
                            chunksize = 10000,
                            converters = {'POLYLINE': lambda x: json.loads(x)})
                            
  # Define a function that transforms the POLYLINE data                  
  to_grid = lambda polyline: trip_to_grid(np.array(polyline), lon_bins, lon_step, lat_bins, lat_step)
                            
  for idx, chunk in enumerate(data_chunks):
    # Compute the grid representations of these trips
    chunk["GRID_POLYLINE"] = chunk["POLYLINE"].map(to_grid) 

    # Transform into a string
    chunk["GRID_POLYLINE"] = chunk["GRID_POLYLINE"].map(lambda x: x.grid_to_string())
    
    # Append data to a csv file
    if idx == 0:
      chunk.to_csv(filepath_clean, header = True)
    else:
      chunk.to_csv(filepath_clean, mode = "a", header = False)
      
    print "Processed chunk %d of 140" % idx
