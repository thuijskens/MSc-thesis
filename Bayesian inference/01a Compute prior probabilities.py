# -*- coding: utf-8 -*-
"""
01 Compute prior probabilities.py

This script computes the empirical probability distributions of

  P[D | X, C], 
  P[D | X, S],
  P[D | X, N],
  
where D is the destination of a trip, X is the full set of complete taxi trips
and S is a stand ID, C is a caller ID and N a binary variable indicating if the
taxi was called from the streets.

"""

from __future__ import division

import numpy as np
import pandas as pd

if __name__ == "__main__":
  print "- 01 Compute prior probabilities.py"
  
  # Define the filepaths
  filepath = "E:/MSc thesis/Processed data/train_binarized_trips_train.csv"
  filepath_processed = "E:/MSc thesis/Processed data/prob_tables.csv"
  chunk_size = 1000
  
  # Import the data in chunks with pandas read_csv
  data_chunks = pd.read_csv(filepath_or_buffer = filepath,
                            sep = ",",
                            #nrows = 100,
                            chunksize = chunk_size,
                            usecols = ["ORIGIN_CALL", "ORIGIN_STAND", "DESTINATION"],
                            converters = {"DESTINATION": lambda x: eval(x)})
                            
  for idx, chunk in enumerate(data_chunks):   
    # Add an extra variable to indicate if a cab was hailed from the street
    chunk["ORIGIN_STREET"] = pd.isnull(chunk["ORIGIN_CALL"]) & pd.isnull(chunk["ORIGIN_STAND"])
    
    # Change NaN values for ORIGIN_CALL and ORIGIN_STREET to make sure they do not
    # disappear when grouping
    chunk["ORIGIN_CALL"] = chunk["ORIGIN_CALL"].fillna(-1)
    chunk["ORIGIN_STAND"] = chunk["ORIGIN_STAND"].fillna(-1)

    # Dummy so that we can count the rows   
    chunk["COUNT"] = 1
    
    # We need to compute the occurences of each destination cell 
    prob_table = chunk.groupby(["ORIGIN_CALL", "ORIGIN_STAND", "ORIGIN_STREET", "DESTINATION"], as_index = False).sum()

    # Save the chunk to a file
    if idx == 0:
      prob_table.to_csv(filepath_processed, header = True, index = False)
    else:
      prob_table.to_csv(filepath_processed, mode = "a", header = False, index = False)
      
    print "-- Processed chunk %d of 1404" % idx
    
    
  # Since we loaded the original data in chunks, we now have to load the 
  # complete data set and aggregate again so that we have unique groups
  
  print "- Aggregating tables"
  table = pd.read_csv(filepath_or_buffer = filepath_processed, sep = ",")
  table = table.groupby(["ORIGIN_CALL", "ORIGIN_STAND", "ORIGIN_STREET", "DESTINATION"], as_index = False).sum() 
    
  # Compute the total counts for each group, and join them on the original table
  table_groupsums = table.groupby(["ORIGIN_CALL", "ORIGIN_STAND", "ORIGIN_STREET"], as_index = False).aggregate({"COUNT": "sum"})
  table = pd.merge(table, table_groupsums, on = ["ORIGIN_CALL", "ORIGIN_STAND", "ORIGIN_STREET"], how = "left")
  table = table.rename(columns = {"COUNT_x" : "COUNT", "COUNT_y" : "TOTAL"})
  
  # Compute the group percentages
  table["PERCENT"] = table["COUNT"] / table["TOTAL"] 
  
  # Sanity check: do group percentages count up to one
  check = table.groupby(["ORIGIN_CALL", "ORIGIN_STAND", "ORIGIN_STREET"]).agg({"PERCENT" : "sum"}).sort(ascending = False)
  check = (np.sum(check.PERCENT) == check.PERCENT.count())  
  
  if check:
    print "-- Passed sanity check after aggregating."
  else:
    raise Exception("-- Sanity check failed. Probabilities do not sum up to one.")
  
  # Overwrite the old data
  table.to_csv(filepath_processed, header = True, index = False)
  
  print "- Finished computing probability tables"
    
    
  
  
                            