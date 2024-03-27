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
slice_start = np.array([0, 36.14, 72.28, 108.42, 144.56, -180 ,-143.86, -107.72, -71.58,-35.44])

slice_end = slice_start + 36.14
list_results = [np.zeros(len(y_data.time))]
for s in range(len(slice_start)):
    list_results.append(y_data.sel(lon = slice(slice_start[s], slice_end[s])).Y.sum(dim = ('lat_index','lon')).values)
ar_data = np.vstack(list_results)


# make 0,1 data where ARs occur
ar_data[ar_data!=0] = 1
ar_data[0, np.max(ar_data, axis = 0)==0] = 1

        
pd_Y_data = pd.DataFrame(ar_data)

pd_Y_data.to_csv('/scratch/alpine/reba1583/AR_lonslice_multiple.csv', index = False)
