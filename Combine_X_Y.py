# Import necessary libraries
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os


# X data
fp = '/rc_scratch/reba1583/variable_yr_files4/'

IWV = xr.open_mfdataset(fp+'IWV').IWV.transpose('time', 'lon','lat_index').values
EFLUX = xr.open_mfdataset(fp+'EFLUX').EFLUX.transpose('time', 'lon','lat_index').values
LWTNET = xr.open_mfdataset(fp+'LWTNET').LWTNET.transpose('time', 'lon','lat_index').values
SF = xr.open_mfdataset(fp+'sf').sf.transpose('time', 'lon','lat_index').values
SLP = xr.open_mfdataset(fp+'SLP').SLP.transpose('time', 'lon','lat_index').values
U800 = xr.open_mfdataset(fp+'U800').U.transpose('time', 'lon','lat_index').values
V800 = xr.open_mfdataset(fp+'V800').V.transpose('time', 'lon','lat_index').values
U950 = xr.open_mfdataset(fp+'U950').U.transpose('time', 'lon','lat_index').values
V950 = xr.open_mfdataset(fp+'V950').V.transpose('time', 'lon','lat_index').values
AODANA = xr.open_mfdataset(fp+'AODANA').AODANA.transpose('time', 'lon','lat_index').values

data = np.stack([ IWV, EFLUX, LWTNET,
                 SF, SLP, U800, V800, U950, V950, AODANA], axis = 3)

del V800 
del U800
del V950 
del U950
del IWV
del EFLUX
del SF
del SLP
del LWTNET
del AODANA

# Y data
Y = pd.read_csv('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/AR_binary_6hrly.csv', index_col = False)
Y = np.array(Y).T


# times for final xarray
variable_times = pd.to_datetime(np.array(xr.open_mfdataset(fp+'U800').time))

var_data = dict(
    features = ([ 'time', 'lon', 'lat','n_channel' ], data),
    labels_1d = (['time','categories'], Y)
)

coords = dict(
    time = (['time'], variable_times), 
    n_channel = (['n_channel'], np.array(['IWV', 'EFLUX', 'LWTNET', 'SF', 'SLP', 'U800', 'V800', 'U950', 'V950', 'AODANA'])),
      
)

ds = xr.Dataset(
    data_vars = var_data, 
    coords = coords
)

ds = ds.fillna(0)

del data
del var_data


fp_out = '/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/Combined_Data_CNN/'
ds.to_netcdf(fp_out+'full_X_and_Y.nc')

# split into training, validating, and testing
index = np.arange(len(ds.time))
split1, split2 = int(.7*len(index)), int(.85*len(index))

np.random.shuffle(index)
index_train, index_validate, index_test = index[:split1], index[split1:split2], index[split2:]
index_train.sort()
index_validate.sort()
index_test.sort()

ds_train = ds.isel(time = index_train)
ds_test = ds.isel(time = index_test)
ds_validate = ds.isel(time = index_validate)

# ds_train.to_netcdf(fp_out+'train.nc')
ds_test.to_netcdf(fp_out+'test.nc')
ds_validate.to_netcdf(fp_out+'validate.nc')

del ds_test
del ds_validate
print('trimming training data')

# trim the training dataset to have equal timesteps for each label
label_data = np.array(ds_train.labels_1d)
num_limit = np.sum(label_data, axis = 0).min()

true_index = np.argwhere(label_data[:,5] == 1).squeeze()# get index for each category
select = np.random.choice(true_index, num_limit) # choose num_limit indeces for each category

selection = np.empty((0), int)
for i in range(np.shape(label_data)[1]):
    true_index = np.argwhere(label_data[:,i] == 1).squeeze() # get index for each category
    
    already_selected = np.intersect1d(true_index, selection)
    available_to_select = np.setdiff1d(true_index, selection)
    select = np.random.choice(available_to_select, np.max((0, num_limit - len(already_selected))), replace = False) # choose num_limit indeces for each category

    selection = np.append(selection, select)
    
ds_train_trim = ds_train.isel(time = np.sort(np.unique(selection)))
ds_train_trim.to_netcdf(fp_out+'train_trim.nc')