# Import necessary libraries
import numpy as np
import xarray as xr
import pandas as pd
import sys


year=1980+int(sys.argv[1])

#AR mask
ar = xr.open_mfdataset('/pl/active/icesheetsclimate/ARTMIP/Wille_AR_catalogues/MERRA2.ar_tag.Wille_v2.3_vIVT.3hourly.'+str(year)+'*').sel(lat = slice(-90,-60))
ar = ar.resample(time="6H").max()
ar = ar.interp(lon = np.linspace(-180,180,256), lat = np.linspace(-90,-60,32))
ar = xr.where(ar.ar_binary_tag>0,100,0)
ar = ar.sel(time = ar.where(ar>0, drop = True).time)
ar = ar.rename('Y')

# 98th percentile T2M
t = xr.open_mfdataset('/rc_scratch/reba1583/Y_variables_98th/T2M'+str(year)+'*')
t = t.T2M.rename('Y')

# 98th percentile PRECSN
p = xr.open_mfdataset('/rc_scratch/reba1583/Y_variables_98th/PRECSN'+str(year)+'*')
p = p.PRECSN.rename('Y')

year_times = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research2/3hrly_merra2_hemisphere/'+str(year)+'*')
year_times = year_times.time.resample(time="6H").min()

# basins
basins = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/AIS_basins_Zwally_MERRA2grid.nc').sel(lat = slice(-90,-60)).load()
basins = basins.interp(lon = np.linspace(-180,180,256), lat = np.linspace(-90,-60,32))
basins = basins.Zwallybasins > 0
basins = xr.where(basins,1000,0)

Y_dataset = xr.DataArray(np.zeros((len(year_times),len(t.lat_index), len(t.lon))),
                   dims = ['time','lat_index','lon'],
                   coords = dict(time = year_times,
                                lat = t.lat_index,
                                lon = t.lon),
                     name = 'Y')

for time in range(len(Y_dataset.time)):

    Y_dataset[time,:,:] = basins.values

    if Y_dataset.time[time].isin(t.time):
        Y_dataset[time,:,:] = Y_dataset[time,:,:] + t.sel(time = Y_dataset.time[time]).values
    if Y_dataset.time[time].isin(p.time):
        Y_dataset[time,:,:] = Y_dataset[time,:,:] + p.sel(time = Y_dataset.time[time]).values
    if Y_dataset.time[time].isin(ar.time):
        Y_dataset[time,:,:] = Y_dataset[time,:,:] + ar.sel(time = Y_dataset.time[time]).values


Y_dataset.to_netcdf('/rc_scratch/reba1583/yrly_Y_files/'+str(year))

print(year)