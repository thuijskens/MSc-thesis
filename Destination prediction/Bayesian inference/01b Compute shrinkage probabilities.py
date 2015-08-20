# -*- coding: utf-8 -*-
"""
01b Compute shrinkage probabilities.py

This script computes the empirical probability distributions of

  P[D | {S_i}_{i = 1,...,Q} ],

where D is the destination cell of a trip, and the {S_i} are the starting 
cells of the trips in the training set

"""

from __future__ import division

import numpy as np
import pandas as pd

if __name__ == "__main__":
  print "- 01b Compute shrinkage probabilities.py"
  
  # Define the filepaths
  filepath = "E:/MSc thesis/Processed data/train_binarized_trips_train.csv"
  filepath_processed = "E:/MSc thesis/Processed data/prob_tables_shrinkage.csv"
  chunk_size = 1000
  
  # Import the data in chunks with pandas read_csv
  """
  data_chunks = pd.read_csv(filepath_or_buffer = filepath,
                            sep = ",",
                            chunksize = chunk_size,
                            usecols = ["GRID_POLYLINE", "DESTINATION"],
                            converters = {"DESTINATION": lambda x: eval(x),
                                          "GRID_POLYLINE": lambda x: eval(x)[0]})
  """
  data_chunks = pd.read_csv(filepath_or_buffer = filepath,
                            sep = ",",
                            chunksize = chunk_size,
                            usecols = ["GRID_POLYLINE"],
                            converters = {"GRID_POLYLINE": lambda x: eval(x)})
                          
  for idx, chunk in enumerate(data_chunks):   
    # Get destination and start cells
    chunk["DESTINATION"] = chunk.GRID_POLYLINE.map(lambda x: x[-1])
    chunk["START_CELL"] = chunk.GRID_POLYLINE.map(lambda x: x[0])  
    
    # Dummy so that we can count the rows   
    chunk["COUNT"] = 1
    
    # We need to compute the occurences of each destination cell 
    prob_table = chunk.groupby(["START_CELL", "DESTINATION"], as_index = False).sum()

    # Save the chunk to a file
    if idx == 0:
      prob_table.to_csv(filepath_processed, header = True, index = False)
    else:
      prob_table.to_csv(filepath_processed, mode = "a", header = False, index = False)
      
    print "-- Processed chunk %d of 1392" % idx
    
  # Since we loaded the original data in chunks, we now have to load the 
  # complete data set and aggregate again so that we have unique groups
  
  print "- Aggregating tables"
  table = pd.read_csv(filepath_or_buffer = filepath_processed, sep = ",")
  table = table.groupby(["START_CELL", "DESTINATION"], as_index = False).sum() 
    
  # Compute the total counts for each group, and join them on the original table
  table_groupsums = table.groupby(["START_CELL"], as_index = False).aggregate({"COUNT": "sum"})
  table = pd.merge(table, table_groupsums, on = ["START_CELL"], how = "left")
  table = table.rename(columns = {"COUNT_x" : "COUNT", "COUNT_y" : "TOTAL"})
  
  # Compute the group percentages
  table["PERCENT"] = table["COUNT"] / table["TOTAL"] 
  
  # Sanity check: do group percentages count up to one
  check = table.groupby(["START_CELL"]).agg({"PERCENT" : "sum"}).sort(ascending = False)
  check = (np.sum(check.PERCENT) == check.PERCENT.count())  
  
  if check:
    print "-- Passed sanity check after aggregating."
  else:
    raise Exception("-- Sanity check failed. Probabilities do not sum up to one.")
  
  # Overwrite the old data
  table.to_csv(filepath_processed, header = True, index = False)
  
  print "- Finished computing shrinkage probability tables"
    
    
  
  
                            