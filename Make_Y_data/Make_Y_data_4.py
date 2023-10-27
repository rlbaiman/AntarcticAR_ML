# Import necessary libraries
import numpy as np
from numpy import savetxt
import xarray as xr
import pandas as pd
import os


y_data = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/Y_labels_AR_PRECSN_T/*')


## only label landfalling AR
y_data['Y'] = xr.where(y_data.Y.isin([ 1100, 1101, 1110, 1111]),1, 0) # extreme temp


# split continent into longitude slices
slice_start = np.array([0, 36, 72, 108, 144, -180 ,-144, -108, -72,-36])

slice_end = slice_start + 36
list_results = []
for s in range(len(slice_start)):
    list_results.append(y_data.sel(lon = slice(slice_start[s], slice_end[s])).Y.sum(dim = ('lat_index','lon')))
ar_data = np.vstack(list_results)



Y_data = np.zeros((np.shape(ar_data)[1]), dtype = int)
for t in range(np.shape(ar_data)[1]):
    if np.max(ar_data[:,t]) !=0:
        Y_data[t] = (np.argmax(ar_data[:,t])+1)
        
pd_Y_data = pd.DataFrame(Y_data)

pd_Y_data.to_csv('/rc_scratch/reba1583/Y_data/AR_lonslice.csv', index = False)
