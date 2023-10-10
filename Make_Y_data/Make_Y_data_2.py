# Import necessary libraries
import numpy as np
import xarray as xr
import pandas as pd
import sys


year=1980+int(sys.argv[1])

#mask for the ice shelves and ice sheet
basins = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/AIS_basins_Zwally_MERRA2grid.nc').sel(lat = slice(-90,-60)).load()
basins = basins.Zwallybasins > 0

variables = ['T2M', 'PRECSN']

for v in range(len(variables)):
    # climatologicaly percentiles (by month)
    percentiles = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/ML_testing/PRECSN_T2M_percentiles/'+variables[v]+'.nc').load()

    percentiles = percentiles[variables[v]]*basins
    
    #hourly variable data
    timesteps = xr.open_mfdataset('/pl/active/icesheetsclimate/MERRA2/'+variables[v]+'_hourly_'+str(year)+'*').sel(lat = slice(-90,-60))
    timesteps = timesteps[variables[v]]*basins
    
    #here select qyantile 2, which is 98th percentile. This can be adjusted
    timesteps = timesteps*(timesteps>=percentiles.isel(quantile = 2).sel(month = (timesteps.time.dt.month)))
    
    #label anything about 98th percentile 1, everything else zero
    timesteps = xr.where(timesteps>0,1,0)
    #resample to 6hr max to match X data
    timesteps = timesteps.resample(time = '6H').max().load()
    
    #only select timesteps where there are locations with values above percentile
    timesteps = timesteps.sel(time = timesteps.where(timesteps>0, drop = True).time)
    
    

    # get the full hemisphere lat lon 
    latlon_data = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research2/3hrly_merra2_hemisphere/19800101*')
    lats = latlon_data.lat
    lons = latlon_data.lon
    
    # Make array of zeros to fill in hemisphere
    pad_data = xr.DataArray(np.zeros((len(timesteps.time),len(lats.sel(lat = slice(-59.9,1))), len(lons))),
                       dims = ['time','lat','lon'],
                       coords = dict(time = timesteps.time,
                                    lat = lats.sel(lat = slice(-59.9,1)),
                                    lon = lons))
    # Make a new array of padding plus timesteps
    new_xarray = xr.concat([timesteps, pad_data], dim = 'lat').drop('quantile')
    
    # coarsen lat lon to 1x1 grid to match x data
    new_xarray = new_xarray.interp(lon = np.arange(-180,181,1), lat = np.arange(-90,1,1))
    
    if v == 0:  
        new_xarray = xr.where(new_xarray>0,1,0)
    elif v ==1:
        new_xarray = xr.where(new_xarray>0,10,0)
    
    new_xarray.to_netcdf('/rc_scratch/reba1583/Y_variables_98th/'+variables[v]+str(year))
    

    
    