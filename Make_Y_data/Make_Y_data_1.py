# Import necessary libraries
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os


i = int(sys.argv[1])


variables = [
    'PRECSN',
    'T2M',    
]
variable_files = [
            '/pl/active/icesheetsclimate/MERRA2/PRECSN_hourly_*',
            '/pl/active/icesheetsclimate/MERRA2/T2M_hourly_*'
        ]
   
for n in range(len(variables)):
    data = xr.open_mfdataset(variable_files[n]).sel(time = slice('1980-01-01','2019-12-31'), lat = slice(-90,-60))
    data = data.where(data.time.dt.month.isin([i]), drop=True)
    data = data.chunk({'time':len(data.time),'lon':2, 'lat':2})  
    data = data.assign_coords({'month':('month',[i])})
    monthly_quantiles = data.quantile([.90, .95, .98, .99], dim = 'time')
    monthly_quantiles.to_netcdf('/rc_scratch/reba1583/monthly_quantiles/'+variables[n]+'_month'+str(i))
    del data
    del monthly_quantiles






    
    
    
    
    
    
    
    
    
    
    
    
    
# os.system('TMPDIR=/rc_scratch/reba1583')
# # split longitudes into 30 lon chunks for memory issues
# splits_start = np.arange(-180,180,30)
# splits_end = np.arange(-150.01,180.01,30)
    
    
    
    
    
# for m in range(1,13):
#     for n in range(len(variables)):
#         variable_files = [
#             '/pl/active/icesheetsclimate/MERRA2/PRECSN_hourly_*',
#             '/pl/active/icesheetsclimate/MERRA2/T2M_hourly_*'
#         ]

#         variable = xr.open_mfdataset(variable_files[n]).sel(time = slice('1980-01-01','2019-12-31'))
#         variable = variable.sel(lat = slice(-90,0))
#         variable = variable.where(variable.time.dt.month.isin([m]), drop=True)
#         variable = variable.assign_coords({'month':('month',[m])})
#         variable = variable.interp(lon = np.linspace(-180,180,256), lat = np.linspace(-90,0,64))
#         variable = variable.sel(lon = slice(splits_start[i], splits_end[i]))
#         variable = variable.resample(time = '6H').max()
# #         variable.to_netcdf('/rc_scratch/reba1583/splits_climo/splits_data/'+variables[n]+'_'+str(i)+'_month'+str(m))


#         basins = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/AIS_basins_Zwally_MERRA2grid.nc').sel(lat = slice(-90,0))
#         basins = basins.interp(lon = np.linspace(-180,180,256), lat = np.linspace(-90,0,64))
#         basins = basins.sel(lon = slice(splits_start[i], splits_end[i]))
#         basins = basins.Zwallybasins > 0


#         climo_mean = variable.groupby('time.month').mean('time')
#         climo_mean =  climo_mean.where(basins.values)
#         climo_mean.to_netcdf('/rc_scratch/reba1583/splits_climo/'+variables[n]+'_'+str(i)+'_month'+str(m))
#         del climo_mean

#         climo_std = variable.groupby('time.month').std('time')
#         climo_std =  climo_std.where(basins.values)
#         climo_std.to_netcdf('/rc_scratch/reba1583/splits_std/'+variables[n]+'_'+str(i)+'_month'+str(m))
#         del climo_std

# #         quantiles = variable.quantile([90, 95, 98, 99], dim = 'time')
# #         quantiles =  quantiles.where(basins.values)
# #         quantiles.to_netcdf('/rc_scratch/reba1583/splits_quantile/'+variables[n]+'_'+str(i))
# #         del quantiles 

#         monthly_quantiles = variable.groupby('time.month').quantile([90, 95, 98, 99], dim = 'time')
#         monthly_quantiles = monthly_quantiles.where(basins.values).load()
#         monthly_quantiles.to_netcdf('/rc_scratch/reba1583/splits_monthly_quantile/'+variables[n]+'_'+str(i)+'_month'+str(m))
#         del monthly_quantiles 
        
#         del variable