# Import necessary libraries
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# load in data to put together

fp = '/rc_scratch/reba1583/variable_yr_files_4/'

H = xr.open_mfdataset(fp+'H').H.values
IWV = xr.open_mfdataset(fp+'IWV').IWV.values
EFLUX = xr.open_mfdataset(fp+'EFLUX').EFLUX.values
LWTNET = xr.open_mfdataset(fp+'LWTNET').LWTNET.values
SF = xr.open_mfdataset(fp+'SF').sf.values
SLP = xr.open_mfdataset(fp+'SLP').SLP.values
T = xr.open_mfdataset(fp+'T').T.values
U = xr.open_mfdataset(fp+'U').U.values
V = xr.open_mfdataset(fp+'V').V.values

data = np.stack([H, IWV, EFLUX, LWTNET,
                 SF, SLP, T, U, V])

del H
del V 
del U
del T
del IWV
del EFLUX
del SF
del SLP
del LWTNET


#category 2d label data: spatial AR mask
ar_catalog = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/ML_testing/3yr_norm_anom/ar_ds.nc')
variable_times = ar_catalog.time.values
ar_catalog = ar_catalog.ar_binary_tag.values


# category 1d label data: is there a landfalling AR or no
centers = pd.read_csv('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/ML_testing/ML_test_artimes_centers.csv')
centers = centers[pd.to_datetime(centers.AR_time)<pd.to_datetime('2003-01-01')]

ar_times = pd.to_datetime(np.array(centers.AR_time))

times_resampled = []
for i in range(len(ar_times)):
    if ar_times[i].hour in [0,6,12,18]:
        times_resampled.append(ar_times[i])
    else:
        times_resampled.append(ar_times[i] - pd.Timedelta(hours = 3))

ar_times_6hrly = pd.to_datetime(np.unique(times_resampled))
    
times = pd.DataFrame({'time':pd.to_datetime(np.array(variable_times))}).set_index('time')

times['AR_flag'] = np.zeros(len(times))
times.AR_flag[times.index.isin(ar_times_6hrly)] = 1

ar_flag = np.array(times.AR_flag)




var_data = dict(
    features = (['n_channel', 'time', 'lat', 'lon' ], data),
    label_1d_AR = (['time'], ar_flag),
    label_2d_AR = (['time', 'lat', 'lon'], ar_catalog)
)

coords = dict(
    n_channel = (['n_channel'], np.array(['H', 'IWV', 'EFLUX', 'LWTNET', 'SF', 'SLP', 'T', 'U', 'V'])),
    time = (['time'], pd.to_datetime(np.array(variable_times))),
    
)

ds = xr.Dataset(
    data_vars = var_data, 
    coords = coords
)

ds.to_netcdf('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/ML_testing/3yr_norm_anom/data')