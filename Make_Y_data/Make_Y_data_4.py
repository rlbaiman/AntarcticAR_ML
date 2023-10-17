# Import necessary libraries
import numpy as np
import xarray as xr
import pandas as pd
import os


y_data = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/Y/*')


## only label landfalling AR
y_data['labels_2d'] = xr.where(y_data.labels_2d.isin([1100, 1101, 1110, 1111]),1, 0) # extreme temp


## get even number of timesteps with/without ARs
AR_index = np.squeeze(np.where(y_data.labels_2d.max(dim = ('lat','lon')).load()==1))
noAR_index = np.setdiff1d(np.arange(len(y_data.time)), AR_index)
np.random.shuffle(noAR_index)
noAR_index = noAR_index[0:len(AR_index)]
select = np.sort(np.concatenate((noAR_index, AR_index)))
y_data = y_data.isel(time = select)


# split into training, validating, and testing
index = np.arange(len(y_data.time))
split1, split2 = int(.7*len(index)), int(.85*len(index))

np.random.shuffle(index)
index_train, index_validate, index_test = index[:split1], index[split1:split2], index[split2:]
index_train.sort()
index_validate.sort()
index_test.sort()

ds_train = y_data.isel(time = index_train)
ds_test = y_data.isel(time = index_test)
ds_validate = y_data.isel(time = index_validate)

ds_test.to_netcdf('/rc_scratch/reba1583/Y_data/test.nc')
ds_train.to_netcdf('/rc_scratch/reba1583/Y_data/train.nc')
ds_validate.to_netcdf('/rc_scratch/reba1583/Y_data/validate.nc')