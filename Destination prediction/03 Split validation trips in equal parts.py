# -*- coding: utf-8 -*-
"""03 Split validation trips in equal parts.py
This script takes the trips from the validation set, and splits the complete
trajectories in a 25-50-75% split. This way, the performance of the algorithms
can be compared to see if the predictions get more accurate as we know more
about the trips
"""

import numpy as np
import pandas as pd

if __name__ == "__main__":
  print "- 03 Split validation trips in equal parts.py"
  
  # Define the filepaths
  filepath_val = "E:/MSc thesis/Processed data/train_binarized_trips_validation.csv"
  filepath_processed = "E:/MSc thesis/Processed data/validation_25_50_75_split.csv"
  
  # Import the data in chunks with pandas read_csv
  validation = pd.read_csv(filepath_or_buffer = filepath_val,
                           sep = ",",
                           converters = {"GRID_POLYLINE": lambda x: eval(x)})
  
  truncate = lambda x, y: x[:y]
      
  # Truncate the paths (UGLY WAY)
  validation["GRID_POLYLINE_25"] = None
  validation["GRID_POLYLINE_50"] = None  
  validation["GRID_POLYLINE_75"] = None   
          
  # Now we split the trips for every row
  for row, trip in validation.iterrows():
    length = len(trip.GRID_POLYLINE)
    splits = [length / 4, length / 2, 3 * length / 4] # use python integer division
    print(splits)
    
    validation = validation.set_value(row, "GRID_POLYLINE_25", truncate(validation.loc[row, "GRID_POLYLINE"], splits[0]))
    validation = validation.set_value(row, "GRID_POLYLINE_50", truncate(validation.loc[row, "GRID_POLYLINE"], splits[1]))
    validation = validation.set_value(row, "GRID_POLYLINE_75", truncate(validation.loc[row, "GRID_POLYLINE"], splits[2]))
  
  validation.to_csv(filepath_processed, header = True, index = False)
