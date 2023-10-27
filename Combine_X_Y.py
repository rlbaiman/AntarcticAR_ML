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
Y = pd.read_csv('/rc_scratch/reba1583/Y_data/AR_lonslice.csv', index_col = False)
Y = np.array(Y['0'])


# times for final xarray
variable_times = pd.to_datetime(np.array(xr.open_mfdataset(fp+'U').time))

var_data = dict(
    features = ([ 'time', 'lon', 'lat','n_channel' ], data),
    labels_1d = (['time'], Y)
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



## only include the number of no AR examples that is equivalent in size to the most common label
## This leaves us with a total dataset of 11,553 timesteps and 8,087 training timesteps
num_noAR = np.max((len(Y[Y==1]), len(Y[Y==2]), len(Y[Y==3]), len(Y[Y==4]), len(Y[Y==5]), len(Y[Y==6]), len(Y[Y==7]) , len(Y[Y==8]), len(Y[Y==9]), len(Y[Y==10]))) 

AR_index = np.argwhere(Y!=0).squeeze()
noAR_index = np.argwhere(Y == 0).squeeze()
np.random.shuffle(noAR_index)
noAR_index = noAR_index[0:num_noAR]
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

ds_train.to_netcdf('/rc_scratch/reba1583/CNN_data_limitNoAR/train.nc')
ds_test.to_netcdf('/rc_scratch/reba1583/CNN_data_limitNoAR/test.nc')
ds_validate.to_netcdf('/rc_scratch/reba1583/CNN_data_limitNoAR/validate.nc')








# ## to get all ARs instead of landfalling ARs
# ds = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/Combined_Training_data/full_data.nc')
# y = xr.open_mfdataset('/rc_scratch/reba1583/Y_data/Y_fullAR.nc')

# new_y = xr.DataArray(np.transpose(y.Y.values, (0,2,1)),
#                      coords = {'time': y.time.values, 'lon':ds.lon.values, 'lat':ds.lat.values},
#                      dims = ['time', 'lon', 'lat'])
# ds['labels_2d'] = new_y
# ds = ds.load()
# ### 

