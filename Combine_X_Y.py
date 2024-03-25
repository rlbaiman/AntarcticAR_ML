# Import necessary libraries
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os


# X data
fp = '/rc_scratch/reba1583/variable_yr_files4/'
variables = [
    'H','H','H',
    
    'U',
    
    'V',
    
    'SLP','SLP','SLP',
    
    'EFLUX','EFLUX','EFLUX',
    
    'LWTNET','LWTNET','LWTNET',
    
    'sf', 'sf',
    
    'IWV','IWV','IWV',
    
#     'AODANA', 'AODANA', 'AODANA'  
]
variable_names = [
    'H500_lead0', 'H500_lead1', 'H500_lead2',
    
    'U800_lead0',
    
    'V800_lead0',
    
    'SLP_lead0', 'SLP_lead1', 'SLP_lead2',
    
    'EFLUX_lead0', 'EFLUX_lead1', 'EFLUX_lead2',
    
    'LWTNET_lead3', 'LWTNET_lead4', 'LWTNET_lead5',
    
    'sf_lead0','sf_lead4',
    
    'IWV_lead0', 'IWV_lead1', 'IWV_lead2',

#     'AODANA_lead0', 'AODANA_lead1', 'AODANA_lead2'
]

data_list = []
for v in range(len(variable_names)):
    data = xr.open_mfdataset(fp+variable_names[v])[variables[v]].transpose('time', 'lon','lat_index').values
    data_list.append(data)
data = np.stack([data_list], axis = 3).squeeze()
data = np.rollaxis(data, 0, start=4)# switch coordinates to order time, lon, lat, n_channel 



# Y data
Y = pd.read_csv('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/AR_binary_daily.csv', index_col = False)
Y = np.array(Y).T


# times for final xarray
variable_times = pd.to_datetime(np.array(xr.open_mfdataset(fp+'U800_lead0').time))

var_data = dict(
    features = ([ 'time', 'lon', 'lat','n_channel' ], data),
    labels_1d = (['time','categories'], Y)
)

coords = dict(
    time = (['time'], variable_times), 
    n_channel = (['n_channel'], np.array(variable_names)),
      
)

ds = xr.Dataset(
    data_vars = var_data, 
    coords = coords
)

ds = ds.fillna(0)

del data
del var_data


fp_out = '/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/Combined_Daily_Data_CNN/'
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

# ds_train.to_netcdf(fp_out+'train.nc')
ds_test.to_netcdf(fp_out+'test.nc')
ds_validate.to_netcdf(fp_out+'validate.nc')

del ds_test
del ds_validate
print('trimming training data')

# trim the training dataset to have equal timesteps for each label
label_data = np.array(ds_train.labels_1d)
num_limit = np.sum(label_data, axis = 0).min()

# select about an even number of AR days from each sector
selection = np.empty((0), int)
for i in range(np.shape(label_data)[1]):
    true_index = np.argwhere(label_data[:,i] == 1).squeeze() # get index for each category
    already_selected = np.intersect1d(true_index, selection)
    available_to_select = np.setdiff1d(true_index, selection)
    select = np.random.choice(available_to_select, np.max((0, num_limit - len(already_selected))), replace = False) # choose num_limit indeces for each category

    selection = np.append(selection, select)
#select some timesteps without ARs
num_limit_noAR = np.sum(label_data[np.sort(np.unique(selection))], axis = 0).max()
select = np.random.choice(np.argwhere(np.sum(label_data,axis =1)==0).squeeze(), num_limit_noAR)

selection = np.append(selection, select)
    
    
ds_train_trim = ds_train.isel(time = np.sort(np.unique(selection)))
ds_train_trim.to_netcdf(fp_out+'train_trim.nc')