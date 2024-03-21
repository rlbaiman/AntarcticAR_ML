# Import necessary libraries
import numpy as np
import xarray as xr
import pandas as pd
import sys


year=1980+int(sys.argv[1])
variables = ['T2M', 'PRECSN']

for v in range(len(variables)):
    # climatologicaly percentiles (by month)
    percentiles = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/ML_testing/PRECSN_T2M_percentiles/'+variables[v]+'.nc').load()

    percentiles = percentiles[variables[v]]
    
    #hourly variable data
    timesteps = xr.open_mfdataset('/pl/active/icesheetsclimate/MERRA2/'+variables[v]+'_hourly_'+str(year)+'*').sel(lat = slice(-90,-60))
    timesteps = timesteps[variables[v]]
    
    #here select quantile 2, which is 98th percentile. This can be adjusted
    timesteps = timesteps*(timesteps >= percentiles.isel(quantile = 2).sel(month = (timesteps.time.dt.month)))
    
    #label anything about 98th percentile 1, everything else zero
    timesteps = xr.where(timesteps>0,1,0)
    #resample to 6hr max to match X data
    new_xarray = timesteps.resample(time = '6H').max().load()
    
    #only select timesteps where there are locations with values above percentile
    new_xarray = new_xarray.sel(time = new_xarray.where(new_xarray>0, drop = True).time)
    new_xarray = new_xarray.drop('quantile')

    # coarsen lat lon to 1x1 grid to match x data
    new_xarray = new_xarray.interp(lon = np.linspace(-180,180,256), lat = np.linspace(-90,-60,32))
    
    # Make lat index dimension that will be uniform for all data
    lat_index = np.arange(0,32)
    new_xarray = new_xarray.assign_coords(lat_index=("lat", lat_index))
    new_xarray = new_xarray.swap_dims({'lat':'lat_index'})
    
    if v == 0:  
        new_xarray = xr.where(new_xarray>0,1,0)
    elif v ==1:
        new_xarray = xr.where(new_xarray>0,10,0)
    
    new_xarray.to_netcdf('/rc_scratch/reba1583/Y_variables_98th/'+variables[v]+str(year))
    

    
    