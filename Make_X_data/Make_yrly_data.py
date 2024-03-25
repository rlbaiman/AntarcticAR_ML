# Import necessary libraries
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import xesmf as xe
import glob

year = 1980+int(sys.argv[1])

def resample(file_name, directory, index): 
    """
    Function to open the variable file at a single level and resample according training needs.
    This includes smoothing in time, lat, and lon. Also includes making LWTNET binary and 
    shifting variables to selected variable_leadtimes
    """
    
    if directory == 1:
        year_data = xr.open_mfdataset(glob.glob(fp_out_1+variables[index]+'*'+str(year)+'*')[0])

    elif directory ==2:
        year_data = xr.open_mfdataset(glob.glob(fp_out_2+variables[index]+'*'+str(year)+'*')[0])
    
    if variables[i] == 'sf':
        year_data = year_data.reindex(lat=list(reversed(year_data.lat)))
    
    if variable_leadtimes[index] !=0:
        shift_delta = int(variable_leadtimes[index]/3)
        year_data = year_data.shift(time = shift_delta, fill_value = np.nan) 
        if variables[i] == 'sf': #for sf variable replace the first days with day number 1 of each year so the average over time doesn't turn to Nan
            year_data = year_data.load()
            # replace the first day(s) with day 1 values so you don't have NANs
            year_data.sf[0:shift_delta] = np.array([year_data.sf[shift_delta].values]*shift_delta)
        
    
    data = year_data.resample(time = '24H').mean()
    
    if variables[i] == 'sf': #take 4-day mean leading up to each day
        roller = data.rolling(time=4, center = 0, min_periods = 1)
        data = roller.mean()
    
    # regrid spatial
    ds_out = xr.Dataset({"lat":(["lat"], np.linspace(variable_lats[index][0], variable_lats[index][1], 90), {"units": "degrees_north"}), "lon":(["lon"], np.arange(-180, 180, 2.5), {"units": "degrees_east"})})
    regridder = xe.Regridder(data, ds_out, "bilinear", periodic = True)
    ds_out = regridder(data, keep_attrs=True)

    ds_out.to_netcdf(fp_out_3+file_name+'.nc')



fp = '/pl/active/ATOC_SynopticMet/data/ar_data/Research2/3hrly_merra2_hemisphere/'
fp_out_1 = '/rc_scratch/reba1583/variable_yr_files1/'
fp_out_2 = '/rc_scratch/reba1583/variable_yr_files2/'
fp_out_3 = '/rc_scratch/reba1583/variable_yr_files3/'

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
    'LWTNET_lead3', 'LWTNET_lead4', 'LWTNET_lead5', #change these to 4, 6, 8
    'sf_lead0','sf_lead4',
    'IWV_lead0', 'IWV_lead1', 'IWV_lead2',
#     'AODANA_lead0', 'AODANA_lead1', 'AODANA_lead2'
]
variable_files = [
    fp+str(year)+'*',fp+str(year)+'*',fp+str(year)+'*',    
    fp+str(year)+'*',    
    fp+str(year)+'*',    
    fp+str(year)+'*', fp+str(year)+'*', fp+str(year)+'*',    
    fp+'EFLUX/EFLUX_'+str(year)+'*', fp+'EFLUX/EFLUX_'+str(year)+'*', fp+'EFLUX/EFLUX_'+str(year)+'*',    
    fp+'LWTNET/LWTNET_'+str(year)+'*', fp+'LWTNET/LWTNET_'+str(year)+'*', fp+'LWTNET/LWTNET_'+str(year)+'*',        
    fp+'200streamfunc/sf_'+str(year)+'*', '200streamfunc/sf_'+str(year)+'*',    
    fp+'IWV/'+str(year)+'*', fp+'IWV/'+str(year)+'*', fp+'IWV/'+str(year)+'*',    
#     fp+'aerosol/'+str(year)+'*', fp+'aerosol/'+str(year)+'*', fp+'aerosol/'+str(year)+'*',   
]
variable_levels = [
    '500', '500', '500',    
    '800',    
    '800',    
    None,  None,  None,   
    None, None, None,     
    None,None, None,    
    None, None,    
    None, None, None,     
#     None, None, None, 
]
variable_lats = [
    [-75,-40], [-75,-40], [-75,-40],    
    [-75,-40],    
    [-75,-40],    
    [-75,-40], [-75,-40],[-75,-40],    
    [-75,-20], [-75,-20], [-75,-20],    
    [-20,0], [-20,0], [-20,0],     
    [-90,0], [-90,0],    
    [-75,-40], [-75,-40], [-75,-40],
#     [-75,-40], [-75,-40], [-75,-40],
]
# time before that you would like to select (in hours)
# note: positive values means we are looking at _hours before AR landfall
variable_leadtimes = [
    0, 24,48,
    0,
    0,
    0, 24,48,
    0, 24,48,
    96, 144, 192,
    0,96,
    0,24,48,
#     0, 24,48,
]


for i in range(len(variables)):

    variable = variables[i]
    file_name = variable_names[i]+'_'+str(year)
    
    if os.path.exists(fp_out_3+file_name+'.nc'): #skip if already processed 
        print(file_name+' already processed')
    
    else:
        print('creating '+file_name)  

        if len(glob.glob(fp_out_1+variables[i]+'*'+str(year)+'*'))==0: #if that variable hasn't already been loaded for that year
            # select the variable
            command = 'cdo -select,name='+variable+' '+variable_files[i]+' '+fp_out_1+file_name
            os.system(command)

        # select the level if necessary  
        if variable_levels[i] is None:
            resample(file_name, directory =1 ,index = i)

        else:
            command_3 = 'cdo -sellevel,'+variable_levels[i]+' '+glob.glob(fp_out_1+variables[i]+'*'+str(year)+'*')[0]+' '+fp_out_2+file_name
            os.system(command_3)
            resample(file_name, directory = 2, index = i)
    
        os.system('rm '+fp_out_2+file_name)
        

        if ('2' in variable_names[i][-1]) | ('5' in variable_names[i][-1]): #if it is the last file of that year and that variable
            os.system('rm '+fp_out_1+file_name)
        