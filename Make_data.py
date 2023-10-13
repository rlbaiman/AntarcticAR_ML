# Import necessary libraries
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

year = 1980+int(sys.argv[1])


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

def make_LWTNET_binary(lwtnet_data):
    lwtnet_data = xr.where(lwtnet_data<220,2,0)
    return(lwtnet_data)


def resample(file_name, index): 
        data =  xr.open_mfdataset(fp_out_2+file_name)
        if variables[i] == 'LWTNET':
            lwtnet_data = xr.where(lwtnet_data<220,3,0)
        data = data.resample(time = '6H').mean()
        if variable_leadtimes[index] !=0:
            data = data.shift(time = int(hours/6), fill_value = np.nan)
        data = data.interp(lon = np.linspace(-180,180,256), lat = (variable_lats[index][0], variable_lats[index][1], 32))
        data.to_netcdf(fp_out_3+file_name+'.nc')



fp = '/pl/active/ATOC_SynopticMet/data/ar_data/Research2/3hrly_merra2_hemisphere/'
fp_out_1 = '/rc_scratch/reba1583/variable_yr_files/'
fp_out_2 = '/rc_scratch/reba1583/variable_yr_files_2/'
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
variable_files = [
    fp+str(year)+'*',
    fp+str(year)+'*',
    fp+str(year)+'*',
    fp+str(year)+'*',
    fp+'EFLUX/EFLUX_'+str(year)+'*',
    fp+'LWTNET/LWTNET_'+str(year)+'*',
    fp+'200streamfunc/sf_'+str(year)+'*',
    fp+'IWV/'+str(year)+'*',
    
]

variable_levels = [
    '950',
    '950',
    '950',
    None,
    None,
    None,
    None, 
    None,
]

variable_lats = [
    [-75,-45],
    [-75,-45],
    [-50,-80],
    [-75,-45],
    [-50,-20],
    [-30,0],
    [-90,0],
    [-70,-40]
]

# time before that you would like to select (in hours)
variable_leadtimes = [
    0,0,0,0,0,96,48,0
]




for i in range(len(variables)):
    # select the variable
    command = 'cdo -select,name='+variables[i]+' '+variable_files[i]+' '+fp_out_1+variables[i]+'_'+str(year)
    os.system(command)
    
    # select the level if necessary
    if variable_levels[i] is None:
        command_2 = 'cp '+fp_out_1+variables[i]+'_'+str(year)+' '+fp_out_2+variables[i]+'_'+str(year)
        os.system(command_2)

    else:
        command_3 = 'cdo -sellevel,'+variable_levels[i]+' '+fp_out_1+variables[i]+'_'+str(year)+' '+fp_out_2+variables[i]+'_'+str(year)
        os.system(command_3)

    print(variables[i]+'_'+str(year)+' in variable_yr_files_2')
    


    resample(variables[i]+'_'+str(year), i)
    print(variables[i]+'_'+str(year)+' in variable_yr_files_3')
  

    os.system('rm '+fp_out_1+variables[i]+'_'+str(year))
    os.system('rm '+fp_out_2+variables[i]+'_'+str(year))
        
        
#     #Standardized anomalies
#     data = xr.open_mfdataset(fp_out_3+variables[i]+'*').load()
# #add this in
# lat_index = np.arange(0,32)
# test = test.assign_coords(lat_index=("lat", lat_index))
# test = test.swap_dims({'lat':'lat_index'})
    
#     if variables[i] == 'LWTNET':
#         data.to_netcdf('/rc_scratch/reba1583/variable_yr_files_4/'+variables[i])
#     else:
#         if variables[i] =='IWV':
#             climo_mean, climo_std = make_IWV_climo_stats(data)
#         else:
#             climo_mean = data.groupby("time.month").mean('time')
#             climo_std = data.groupby("time.month").std('time')

#         stand_anomalies = xr.apply_ufunc(
#             lambda x, m, s: (x - m) / s,
#             data.groupby("time.month"),
#             climo_mean,
#             climo_std,
#         )

#         stand_anomalies 
#         stand_anomalies.to_netcdf('/rc_scratch/reba1583/variable_yr_files_4/'+variables[i])
