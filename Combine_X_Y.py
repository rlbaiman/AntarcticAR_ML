# Import necessary libraries
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os

import sys
sys.path.insert(1, '/projects/reba1583/Research3/AntarcticAR_ML/')
from functions import flatten_X_variable 
from define_variables import get_variables,get_variable_names,get_variable_lats


# X data
fp = '/rc_scratch/reba1583/variable_yr_files4/'
variables = get_variables()
variable_names = get_variable_names()
variable_lats = get_variable_lats()


#for each timestep, get the data flattned by variable, lon, lat
time_data = xr.open_mfdataset(fp+variable_names[0])[variables[0]].time.values
variable_lat_lengths = [len(variable_lats[i]) for i in range(len(variable_lats))]
data = np.empty((len(time_data), sum(variable_lat_lengths)*144))
data[:,:] = np.nan
for t in range(len(time_data)):
    storage = []
    for v in range(len(variable_names)):
        tempdata = xr.open_mfdataset(fp+variable_names[v])[variables[v]].isel(time = t)
        tempdata = flatten_X_variable(tempdata)
        storage.append(tempdata)
    data[t] = [item for row in storage for item in row]


# Y data
Y = pd.read_csv('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/AR_binary_daily.csv', index_col = False)
Y = np.array(Y).T


var_data = dict(
    features = ([ 'time', 'data' ], data),
    labels_1d = (['time','categories'], Y)
)

coords = dict(
    time = (['time'], time_data), 
      
)

ds = xr.Dataset(
    data_vars = var_data, 
    coords = coords
)

ds = ds.fillna(0)

del data
del var_data


fp_out = '/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/daily_data_XGboost/'
ds.to_netcdf(fp_out+'full_data.nc')


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


ds_train.to_netcdf(fp_out+'train_full.nc')
ds_test.to_netcdf(fp_out+'test_full.nc')
ds_validate.to_netcdf(fp_out+'validate_full.nc')

del ds_test
del ds_validate




# print('trimming training data')

# # trim the training dataset to have equal timesteps for each label
# label_data = np.array(ds_train.labels_1d)
# num_limit = np.sum(label_data, axis = 0).min()

# # select about an even number of AR days from each sector
# selection = np.empty((0), int)
# for i in range(np.shape(label_data)[1]):
#     true_index = np.argwhere(label_data[:,i] == 1).squeeze() # get index for each category
#     already_selected = np.intersect1d(true_index, selection)
#     available_to_select = np.setdiff1d(true_index, selection)
#     select = np.random.choice(available_to_select, np.max((0, num_limit - len(already_selected))), replace = False) # choose num_limit indeces for each category

#     selection = np.append(selection, select)
# #select some timesteps without ARs
# num_limit_noAR = np.sum(label_data[np.sort(np.unique(selection))], axis = 0).max()*2
# select = np.random.choice(np.argwhere(np.sum(label_data,axis =1)==0).squeeze(), num_limit_noAR)

# selection = np.append(selection, select)
    
    
# ds_train_trim = ds_train.isel(time = np.sort(np.unique(selection)))
# ds_train_trim.to_netcdf(fp_out+'train_trim.nc')