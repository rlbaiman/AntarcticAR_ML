# Import necessary libraries
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

variable_index = int(sys.argv[1])


# Functions

def make_IWV_climo_stats(data_input):
    """
    input data and create a symmetric distribution based on the right hand side of the distribution. 
    Use this for IWV before calculating the std and mean so that the std is not skewed by the limit
    of the distribution at 0. 
    """
    mean = data_input.mean(dim = 'time')
    data1 = data_input.where(data_input>= mean)
    data2 = mean - np.abs(data1 - mean)
    data2['time'] = data2.time +pd.Timedelta('1H')
    
    data_updated = xr.concat((data1, data2), dim = 'time')
    
    mean_out = data_updated.groupby("time.month").mean('time')
    std_out = data_updated.groupby("time.month").std('time')
    return(mean_out, std_out)


fp_out_3 = '/rc_scratch/reba1583/variable_yr_files3/'


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


variable_name = variable_names[variable_index]    
if os.path.exists('/rc_scratch/reba1583/variable_yr_files4/'+variable_name):
    print(variable_name+' already processed')

else:
    print('creating '+variable_name)  

    data = xr.open_mfdataset(fp_out_3+variable_name+'*', chunks = 'auto').load()
    if 'lev' in data.dims:
        data = data.squeeze()
        data = data.drop('lev')

    if variable_name =='IWV':
        # base standard deviation off of right half of IWV distribution
        climo_mean, climo_std = make_IWV_climo_stats(data) 
    else:
        climo_mean = data.groupby("time.month").mean('time',skipna=True).load()
        climo_std = data.groupby("time.month").std('time').load()

    stand_anomalies = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        data.groupby("time.month"),
        climo_mean,
        climo_std,
    )
    stand_anomalies = stand_anomalies.drop('month')

     
    #add uniform lat_index
    lat_index = np.arange(0,90)
    stand_anomalies = stand_anomalies.assign_coords(lat_index=("lat", lat_index))
    stand_anomalies = stand_anomalies.swap_dims({'lat':'lat_index'})

    stand_anomalies = stand_anomalies.astype('float32') # lower precision to save memory
    stand_anomalies.to_netcdf('/rc_scratch/reba1583/variable_yr_files4/'+variable_name)
