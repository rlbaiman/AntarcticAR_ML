# Import necessary libraries
import numpy as np
import xarray as xr
import pandas as pd
import os


y_data = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/Y/*')


## only label landfalling AR
y_data['Y'] = xr.where(y_data.Y.isin([1100, 1101, 1110, 1111]),1, 0) # extreme temp


y_data.to_netcdf('/rc_scratch/reba1583/Y_data/Y_AR_only.nc')

