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
    data1 = data.where(data>= mean)
    data2 = mean - np.abs(data1 - mean)
    data2['time'] = data2.time +pd.Timedelta('1H')
    
    data_updated = xr.concat((data1, data2), dim = 'time')
    
    mean_out = data_updated.groupby("month").mean('time')
    std_out = data_updated.groupby("month").std('time')
    return(mean_out, std_out)


fp_out_3 = '/rc_scratch/reba1583/variable_yr_files_3/'
variables = [
    'U',
    'V',
    'T',
    'SLP',
    'EFLUX',
    'LWTNET',
    'sf',
    'IWV',
]
variable = variables[variable_index]


data = xr.open_mfdataset(fp_out_3+variable+'*').load()


#add uniform lat_index
lat_index = np.arange(0,32)
data = data.assign_coords(lat_index=("lat", lat_index))
data = data.swap_dims({'lat':'lat_index'})
    
if variable == 'LWTNET': # LWTNET is binary so it does not need to be normalized
    data.to_netcdf('/rc_scratch/reba1583/variable_yr_files_4/'+variable)
else:
    if variable =='IWV':
        # base standard deviation off of right half of IWV distribution
        climo_mean, climo_std = make_IWV_climo_stats(data) 
    else:
        climo_mean = data.groupby("time.month").mean('time')
        climo_std = data.groupby("time.month").std('time')

    stand_anomalies = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        data.groupby("time.month"),
        climo_mean,
        climo_std,
    )

    stand_anomalies.to_netcdf('/rc_scratch/reba1583/variable_yr_files_4/'+variable)
