
import numpy as np
import pandas as pd
import json
import sys

from __future__ import division

def haversine(p1, p2):
  """ haversine
      Calculate the great circle distance between two points 
      on the earth (specified in decimal degrees)
      
      Arguments:
      ----------
      p1: 2d numpy array
        Two-dimensional array containing longitude and latitude coordinates of points.
      p2: 2d numpy array
        Two-dimensional array containing longitude and latitude coordinates of points.
        
      Returns:
      --------
      dist: 1d numpy array 
        One-dimensional array with the distances between p1 and p2.
  """
  # convert decimal degrees to radians 
  if p1.ndim == 1:
    p1 = p1.reshape(-1,2)
  if p2.ndim == 1:
    p2 = p2.reshape(-1,2)
    
  lon1, lat1, lon2, lat2 = map(np.radians, [p1[:, 0], p1[:, 1], p2[:, 0], p2[:, 1]])
  
  # haversine formula 
  dlon = lon2 - lon1 
  dlat = lat2 - lat1 
  a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
  c = 2 * np.arcsin(np.sqrt(a)) 
  r = 6371 
  return c * r
  
def is_outlier(trip, threshold_speed = 160):
  # Define a conversion for the threshold comparison
  c = threshold_speed / 3600 * 15 # km per 15s threshold

  if type(is_outlier) != np.ndarray:
    try:
      trip = np.array(trip)
    except:
      print "Fuck"
      return True
    
  # Compute distances
  distances = haversine(trip[:-1], trip[1:])
  
  if np.any(distances >= c):
    return True
  else:
    return False
    
if __name__ == "__main__":
  print "Starting cleaning script"
  
  # Get the location of the data file
  args = sys.argv[1:]
  if args[0] is None:
    filepath = "E:/MSc thesis/Processed data/train_processed.csv"
  else:
    filepath = args[0] 
  
  # Use pandas read_csv function to read the data in chunks
  data_chunks = pd.read_csv(filepath_or_buffer = filepath,
                            sep = ",",
                            chunksize = 1000, 
                            converters = {'POLYLINE': lambda x: json.loads(x)})
  # Declare outlier counter                         
  outliers = 0

  for idx, chunk in enumerate(data_chunks):
    # Compute outliers
    chunk["IS_OUTLIER"] = chunk["POLYLINE"].map(is_outlier)
    
    # Add to previous count
    outliers += np.sum(chunk["IS_OUTLIER"])
    
    if outliers > 0:
      break
    
    print "Processed chunk %d of 1675" % idx
  
  print "Number of outliers is %d" % outliers#
  
  # 197702 trips for threshold = 140
  # 174905 trips for threshold = 150
  # 145437 trips for threshold = 160
  # 38211 for threshold = 200
    