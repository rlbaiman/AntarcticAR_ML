# Import necessary libraries
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

year = 1980+int(sys.argv[1])


def make_LWTNET_binary(lwtnet_data):
    """
    LWTNET is labeled 0 (no convection) or 2(strong tropical convection).
    We use 2 instead of 1 so it is more comparable to normalized anomaly data
    """
    lwtnet_data = xr.where(lwtnet_data<220,2,0)
    return(lwtnet_data)

def resample(file_name, index): 
    """
    Function to open the variable file at a single level and resample according training needs.
    This includes smoothing in time, lat, and lon. Also includes making LWTNET binary and 
    shifting variables to selected variable_leadtimes
    """
    data =  xr.open_mfdataset(fp_out_2+file_name)
    
    if variables[i] == 'sf':
        data = data.reindex(lat=list(reversed(data.lat)))
        
    data = data.resample(time = '6H').mean()
    
    if variable_leadtimes[index] !=0:
        data = data.shift(time = int(variable_leadtimes[index]/6), fill_value = np.nan)
    
    data = data.interp(lon = np.linspace(-180,180,256), lat = np.linspace(variable_lats[index][0], variable_lats[index][1], 32))
            
    if variables[i] == 'LWTNET':
        data = xr.where(data<220,3,0)
 
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
    [-80,-50],
    [-75,-45],
    [-50,-20],
    [-30,0],
    [-90,0],
    [-70,-40]
]

# time before that you would like to select (in hours)
# note: positive values means we are looking at _hours before AR landfall
variable_leadtimes = [
    0,0,0,0,0,96,48,0
]




for i in range(len(variables)):
    
    if False == os.path.exists('/rc_scratch/reba1583/variable_yr_files_3/'+variables[i]+'_'+str(year)+'.nc'):

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
