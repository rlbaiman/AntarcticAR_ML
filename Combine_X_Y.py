# Import necessary libraries
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os

# X data
fp = '/rc_scratch/reba1583/variable_yr_files_4/'

IWV = xr.open_mfdataset(fp+'IWV').IWV.transpose('time', 'lon','lat_index').values
EFLUX = xr.open_mfdataset(fp+'EFLUX').EFLUX.transpose('time', 'lon','lat_index').values
LWTNET = xr.open_mfdataset(fp+'LWTNET').LWTNET.transpose('time', 'lon','lat_index').values
SF = xr.open_mfdataset(fp+'sf').sf.transpose('time', 'lon','lat_index').values
SLP = xr.open_mfdataset(fp+'SLP').SLP.transpose('time', 'lon','lat_index').values
T = xr.open_mfdataset(fp+'T').T.transpose('time', 'lon','lat_index').values
U = xr.open_mfdataset(fp+'U').U.transpose('time', 'lon','lat_index').values
V = xr.open_mfdataset(fp+'V').V.transpose('time', 'lon','lat_index').values

data = np.stack([ IWV, EFLUX, LWTNET,
                 SF, SLP, T, U, V], axis = 3)

del V 
del U
del T
del IWV
del EFLUX
del SF
del SLP
del LWTNET

# Y data
Y = xr.open_mfdataset('/rc_scratch/reba1583/Y_data/Y_AR_only.nc').Y.transpose('time','lon','lat_index').values

# times for final xarray
variable_times = pd.to_datetime(np.array(xr.open_mfdataset(fp+'U').time))

var_data = dict(
    features = ([ 'time', 'lon', 'lat','n_channel' ], data),
    labels_2d = (['time', 'lon', 'lat'], Y)
)

coords = dict(
    time = (['time'], variable_times), 
    n_channel = (['n_channel'], np.array(['IWV', 'EFLUX', 'LWTNET', 'SF', 'SLP', 'T', 'U', 'V'])),
      
)

ds = xr.Dataset(
    data_vars = var_data, 
    coords = coords
)

ds = ds.fillna(0)
ds.to_netcdf('/rc_scratch/reba1583/full_data.nc')

## get even number of timesteps with/without ARs
AR_index = np.squeeze(np.where(ds.labels_2d.max(dim = ('lat','lon')).load()==1))
noAR_index = np.setdiff1d(np.arange(len(ds.time)), AR_index)
np.random.shuffle(noAR_index)
noAR_index = noAR_index[0:len(AR_index)]
select = np.sort(np.concatenate((noAR_index, AR_index)))
data = ds.isel(time = select)

# split into training, validating, and testing
index = np.arange(len(data.time))
split1, split2 = int(.7*len(index)), int(.85*len(index))

np.random.shuffle(index)
index_train, index_validate, index_test = index[:split1], index[split1:split2], index[split2:]
index_train.sort()
index_validate.sort()
index_test.sort()

ds_train = data.isel(time = index_train)
ds_test = data.isel(time = index_test)
ds_validate = data.isel(time = index_validate)

ds_train.to_netcdf('/rc_scratch/reba1583/train.nc')
ds_test.to_netcdf('/rc_scratch/reba1583/test.nc')
ds_validate.to_netcdf('/rc_scratch/reba1583/validate.nc')

