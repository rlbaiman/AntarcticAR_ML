# Import necessary libraries
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os


year = 1980+int(sys.argv[1])

fp = '/pl/active/ATOC_SynopticMet/data/ar_data/Research2/3hrly_merra2_hemisphere/'
fp_out_1 = '/rc_scratch/reba1583/variable_yr_files/'
fp_out_2 = '/rc_scratch/reba1583/variable_yr_files_2/'
fp_out_3 = '/rc_scratch/reba1583/variable_yr_files_3/'

variable_files = [
    fp+str(year)+'*',
    fp+str(year)+'*',
    fp+str(year)+'*',
    fp+str(year)+'*',
    fp+str(year)+'*',
    fp+'EFLUX/EFLUX_'+str(year)+'*',
    fp+'LWTNET/LWTNET_'+str(year)+'*',
    fp+'200streamfunc/sf_'+str(year)+'*',
    
]
variables = [
    'U',
    'V',
    'T',
    'SLP',
    'H',
    'EFLUX',
    'LWTNET',
    'sf',
    
]
variable_levels = [
    '950',
    '950',
    '950',
    None,
    '500',
    None,
    None,
    None, 
]

def resample(file_name): 
        data =  xr.open_mfdataset(fp_out_2+file_name)
        period = 1460
        if year in [1980,1984,1988,1992,1996,2000,2004,2008,2012,2016,2020]:
            period = 1464
        data = data.interp(lon = np.arange(-180,181,1), lat = np.arange(-90,1,1), time = pd.date_range(str(year)+"-01-01T00:00:00.000000000", periods= period, freq="6H"))
        data.to_netcdf(fp_out_3+file_name+'.nc')

for i in range(len(variables)):

    command = 'cdo -select,name='+variables[i]+' '+variable_files[i]+' '+fp_out_1+variables[i]+'_'+str(year)
    os.system(command)

    if variable_levels[i] is None:
        command_2 = 'cp '+fp_out_1+variables[i]+'_'+str(year)+' '+fp_out_2+variables[i]+'_'+str(year)
        os.system(command_2)

    else:
        command_3 = 'cdo -sellevel,'+variable_levels[i]+' '+fp_out_1+variables[i]+'_'+str(year)+' '+fp_out_2+variables[i]+'_'+str(year)
        os.system(command_3)

    print(variables[i]+'_'+str(year)+' in variable_yr_files_2')
    
    resample(variables[i]+'_'+str(year))
    print(variables[i]+'_'+str(year)+' in variable_yr_files_3')
  

    os.system('rm '+fp_out_1+variables[i]+'_'+str(year))
    os.system('rm '+fp_out_2+variables[i]+'_'+str(year))
        