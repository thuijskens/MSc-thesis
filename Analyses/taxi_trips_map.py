import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Store working directory
wd = "C:/Users/Thomas/Documents/MSc in Applied Statistics/MSc thesis/Data/"

lat_mid = 41.1496100
lon_mid = -8.6109900

# Load the data in chunks
data_chunks = pd.read_csv("E:/MSc thesis/Raw data/train.csv",
                          sep = ",",
                          chunksize = 1000,
                          iterator = True,
                          usecols = ['ORIGIN_STAND', 'POLYLINE'],
                           converters={'POLYLINE': lambda x: json.loads(x)})
                 
# Define the number of bins to look at
nrbins = 2000

# Define the stands to look at
stands = range(64)
stands.remove(0)

# Initialize the histogram
hist = []

for stand in stands:
  hist.append(np.zeros((nrbins,nrbins)))


for data in df:
  for idx, stand in enumerate(stands):
    # Only take the rows that start from the appropriate stand
    data_stand = data[data.ORIGIN_STAND == stand]
    
    if not data_stand.empty:
      # Get just the longitude and latitude coordinates for each trip
      latlong = np.array([ coord for coords in data_stand['POLYLINE'] for coord in coords if len(coords) > 0])
    
      if latlong.size is not 0:
        # Compute the histogram with the longitude and latitude data as a source
        hist_new, _, _  = np.histogram2d(x = latlong[:,1], y = latlong[:,0], bins = nrbins, 
                                         range = [[lat_mid - 0.1, lat_mid + 0.1], [lon_mid - 0.1, lon_mid + 0.1]])
                                         
        # Add the new counts to the previous counts
        hist[idx] = hist[idx] + hist_new
    

for idx, stand in enumerate(stands):
  # We consider the counts on a logarithmic scale
  img = np.log(hist[idx][::-1,:] + 1)
  
  # Plot the counts
  plt.figure(figsize = (50, 50))
  ax = plt.subplot(1,1,1)
  plt.imshow(img)
  plt.axis('off')
        
  plt.savefig('trips_density_stand_' + str(stand) + '.png')
  plt.close()
  
# Save data for later use
np.savez("hist_plots_data", *hist)

for idx, stand in enumerate(stands):
  np.save("hist_data_stand" + str(idx), hist[idx])
