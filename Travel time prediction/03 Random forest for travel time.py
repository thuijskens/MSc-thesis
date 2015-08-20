# -*- coding: utf-8 -*-
"""
03 Random forest for travel time.py

This script estimates a random forest for the natural logarithm of the travel
time based on the features in the training set:
  - Hour of the day
  - Day of the week
  - Week of the year
  - Starting cell
  - End cell
  - ...?
  
The scripts also outputs predictions for the validation set. Since the input 
here is a distribution over destinations, a prediction is made for each 
destination and the total travel time prediction is weighter by the probabilities
for each destination.
"""

# Import relevant libraries
from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble.forest import RandomForestRegressor

if __name__ == "__main__":
  print "03 - Random forest for travel time.py"
  
  # Define the filepaths
  filepath = "E:/MSc thesis/Processed data/train_binarized_trips_train.csv"
  filepath_val = "E:/MSc thesis/Processed data/train_binarized_trips_validation.csv"
  
  print "-- Fitting random forest on training data"
  
  # Read in the training data
  train = pd.read_csv(filepath_or_buffer = filepath,
                       sep = ",",
                       usecols = ["START_CELL", "END_CELL", "HOUR", "WDAY", "WEEK", "DURATION"],
                       nrows = 100,
                       converters = {"START_CELL": lambda x: eval(x),
                                     "END_CELL": lambda x: eval(x)})
  
  # Take natural logarithm of travel time
  train.DURATION = np.log(train.DURATION)
                       