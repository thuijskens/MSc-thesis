# -*- coding: utf-8 -*-

"""
00 Split data in train-val.py

This script splits the training data for the Bayesian approach in a training
and a validation set. A list of datetimes (cutoff points) is required as input, 
and the script truncates each trip that is active during a cutoff point and
allocates it to the validation set. Inactive trips are allocated to the training set.
"""

from __future__ import division

import json
import numpy as np
import pandas as pd

if __name__ == "__main__":
  print "- 00 Split data in train-val.py"
  
  # Define the filepaths
  filepath = "E:/MSc thesis/Processed data/train_binarized_trips.csv"
  filepath_processed = "E:/MSc thesis/Processed data/"
  chunk_size = 10000  

  # Generate a list of cutoff dates
  seconds_per_day = 24 * 60 * 60
  day_gap = 5
  
  start_unix = 1372633253 # Minimum datetime (2013-07-01 00:00:53) in seconds (unix)
  end_unix = 1404169154 # Maximum datetime (2014-06-30 23:59:14) in seconds (unix)
  
  dates = np.arange(start_unix, end_unix, day_gap * seconds_per_day)
  cutoff_dates = np.array([], dtype = 'datetime64[s]')
  
  # Set the seed for reproducibility
  np.random.seed(30061992)
  
  for begin, end in zip(dates[:-1], dates[1:]):
    cutoff_date = np.random.randint(low = begin, high = end)
    cutoff_dates = np.append(cutoff_dates, np.datetime64(cutoff_date, 's'))
  
  # Import the data in chunks with pandas read_csv
  data_chunks = pd.read_csv(filepath_or_buffer = filepath,
                            sep = ",",
                            #nrows = 2000,
                            chunksize = chunk_size,
                            converters = {#"TIMESTAMP" : lambda x: datetime.datetime.fromtimestamp(x),
                                          "POLYLINE": lambda x: json.loads(x),
                                          "GRID_POLYLINE": lambda x: eval(x)})
  
  # Utilities for processing
  savedTest = False
  truncate = lambda x, y: x[:y]
                                           
  for idx, chunk in enumerate(data_chunks):    
    # Convert timestamp to datetime series
    chunk.TIMESTAMP = pd.to_datetime(chunk.TIMESTAMP)
    
    # Store the destination and starting cells
    chunk["DESTINATION"] = chunk.GRID_POLYLINE.map(lambda x: x[-1])
    chunk["START_CELL"] = chunk.GRID_POLYLINE.map(lambda x: x[0])
    
    # Loop through the cutoff dates. For every cutoff date:
    # 1. The active trips are removed from the chunk (training) set
    # 2. These trips are truncated and saved into the test set.
    # 3. Iteratively, the size of chunk is reduced until we have passed
    #    through all the cutoff dates.
    # 4. This final remained forms the training set.
    
    for cutoff_date in cutoff_dates:
      # Allocate the inactive trips to the training set. Add 30 seconds for 
      # the boundary cases (if time difference between cutoff_date and 
      # TIMESTAMP is less than 30 seconds: not enough data to truncate 
      active = ((chunk.TIMESTAMP + pd.to_timedelta(30, unit = 's')) <= cutoff_date) & ((chunk.TIMESTAMP + pd.to_timedelta(chunk.DURATION, unit = 's')) >= cutoff_date)
      
      # For the active trips, the trip is truncated at the cutoff time
      if np.sum(active) > 0:
        validation = chunk[active].reset_index(drop = True)

        # Compute elapsed time in seconds
        elapsed = np.abs((cutoff_date.astype(np.int64) - (validation.TIMESTAMP.astype(np.int64) // 10 ** 9))) # astype(np.int64) returns unix in nanoseconds!
        
        # Get the (integer) cutoff point from the elapsed time. (15 seconds between each measurement)
        validation["CUTOFF"] = np.floor(elapsed/15).astype(int) 
        
        # Truncate the paths (UGLY WAY)
        validation["TRUNC_POLYLINE"] = None
        validation["TRUNC_GRID_POLYLINE"] = None
        
        for row in xrange(len(validation)):
          validation = validation.set_value(row, "TRUNC_POLYLINE", truncate(validation.loc[row, "POLYLINE"], validation.loc[row, "CUTOFF"]))
          validation = validation.set_value(row, "TRUNC_GRID_POLYLINE", truncate(validation.loc[row, "GRID_POLYLINE"], validation.loc[row, "CUTOFF"]))
          
        """ 15-07-13: This does not seem to work for some reason, gives a broadcasting error. I think having list as the element type in the column
            messes something up.
            
        # Truncate the paths   
        validation["TRUNC_POLYLINE"] = validation[["POLYLINE", "CUTOFF"]].apply(lambda x: x["POLYLINE"][:x["CUTOFF"]], axis = 1)
        validation["GRID_POLYLINE"] = validation[["GRID_POLYLINE", "CUTOFF"]].apply(lambda x: x["GRID_POLYLINE"][:x["CUTOFF"]], axis = 1)
        
        """
        
        # Add some extra variables
        validation["TRUNC_END_CELL"] = validation["TRUNC_GRID_POLYLINE"].map(lambda x: x[-1])
        validation["TRUNC_POINT"] = validation["TRUNC_POLYLINE"].map(lambda x: x[-1])
        validation["TRUNC_DURATION"] = validation["TRUNC_GRID_POLYLINE"].map(lambda x: 15*(len(x) - 1))
        
        # Save validation set
        if not savedTest:
          validation.to_csv(filepath_processed + "train_binarized_trips_validation.csv", header = True, index = False)
          savedTest = True
        else:
          validation.to_csv(filepath_processed + "train_binarized_trips_validation.csv", mode = 'a', header = False, index = False)
      
      # Update the chunk by removing the active trips
      chunk = chunk[~active].reset_index(drop = True)
      
    # The remaining trips from the training set. 
    if idx == 0:
      chunk.to_csv(filepath_processed + "train_binarized_trips_train.csv", header = True, index = False)
    else:
      chunk.to_csv(filepath_processed + "train_binarized_trips_train.csv", mode = 'a', header = False, index = False)

    print "-- Processed chunk %d of 140" % idx 

  # Sanity check
  tr = pd.read_csv(filepath_processed + "train_binarized_trips_train.csv", sep = ",", chunksize = 1000)      
  te = pd.read_csv(filepath_processed + "train_binarized_trips_validation.csv", sep = ",")   
  tot = pd.read_csv(filepath, sep = ",", chunksize = 1000)
  
  count = 0
  for ch in tr:
    count += ch.TRIP_ID.count()
  
  tot_count = 0 
  for ch in tot:
    tot_count += ch.TRIP_ID.count()
    
  count += te.TRIP_ID.count()
  check = (count == tot_count)

  if check:
    print "- Passed sanity check."
  else:
    raise Exception("Number of training trips and validation trips do not sum up to original number of trips")          





