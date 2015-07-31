import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lat_mid = 41.1496100
lon_mid = -8.6109900
width = 0.2

# Load the data in chunks
data_chunks = pd.read_csv("E:/MSc thesis/Raw data/train.csv",
                          sep = ",",
                          chunksize = 10000,
                          iterator = True,
                          usecols = ['ORIGIN_STAND', 'POLYLINE'],
                          converters={'POLYLINE': lambda x: json.loads(x)})
                 
# Define the number of bins to look at
nrbins = 1250

# Initialize the histogram
hist = np.zeros((nrbins, nrbins))

for idx, chunk in enumerate(data_chunks):
  # Get just the longitude and latitude coordinates for each trip
  latlon = np.array([ (lat, lon) for trip in chunk.POLYLINE for lon, lat in trip if len(trip) > 0])

  # Compute the histogram with the longitude and latitude data as a source
  hist += np.histogram2d(*latlon.T, bins = nrbins, 
                         range = [[lat_mid - width, lat_mid + width], 
                                  [lon_mid - width, lon_mid + width]])[0]
                                  
  print "Processed chunk %d of 171" % idx
                                       
# We consider the counts on a logarithmic scale
img = np.log(hist[::-1,:] + 1)

# Plot the counts
plt.figure(figsize = (50, 50))
ax = plt.subplot(1,1,1)
plt.imshow(img)
plt.axis('off')
      
plt.savefig('gps_heatmap.png')
plt.close()

# Save data for later use
np.save("hist_plots_data", hist)

