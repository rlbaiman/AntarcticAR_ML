import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import sys
import xgboost as xgb
from xgboost import XGBClassifier
import shap

region_id = 0

ds_train = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/Combined_Daily_Data_CNN/train_full.nc')
ds_val = xr.open_mfdataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/Combined_Daily_Data_CNN/validate_full.nc')

#select a reasonable amount of times with and without ARs in region (ocean and land)
all_y = np.array(ds_train.labels_1d)
regional_ar_index = np.argwhere(np.max(all_y[:,[region_id,region_id+10]],1)==1)[:,0]
No_regional_ar_index = (np.argwhere(np.max(all_y[:,[region_id,region_id+10]],1)!=1)[:,0])
np.random.shuffle(No_regional_ar_index)
No_regional_ar_index  = np.sort(No_regional_ar_index[0:2*len(regional_ar_index)]) #here I choose twice as many no AR landfalls as landfall
ds_train_region = ds_train.isel(time = np.sort(np.append(regional_ar_index, No_regional_ar_index)))

X_train = ds_train_region.stack(data=['n_channel','lon', 'lat']).features #make x data 1d
Y_train = np.max(ds_train_region.labels_1d.values[:,[region_id,region_id+10]],1)

X_val = ds_val.stack(data=['n_channel','lon', 'lat']).features
Y_val = np.max(ds_val.labels_1d.values[:,[region_id,region_id+10]],1)

data_test = xgb.DMatrix(X_train,Y_train)


m = XGBClassifier(
    learning_rate = .1, 
    n_estimators=500,
    max_depth = 10, #max depth of trees
    min_child_weight = 1, #smaller to allow small leves
    gamma = 0,
    subsample = .8,
    colsample_bytree = 0.8,
    objective = 'binary:logistic',
    scale_pos_weight=1,
    seed=27)



m.fit(X_train, Y_train)

Y_pred = m.predict(X_val)

results = pd.DataFrame({'time':np.array(X_val.time), 'Y_Val':Y_val, 'Y_pred':Y_pred}).set_index('time')


new_folder = '/rc_scratch/reba1583/XGB_test2_region0/'
os.system('mkdir '+new_folder)

results.to_csv(new_folder+'results.csv')


shap_values = shap.TreeExplainer(m).shap_values(X_val)

parameters= []
list_channels = np.array(X_train.n_channel)
for i in range(len(list_channels)):
    parameters.append(list_channels[i]+'_'+str(i))

shap = pd.DataFrame(shap_values, columns = parameters)

variable_names = np.array(ds_train.n_channel)
data = []
for i in range(len(variable_names)):
    variable_data = np.reshape(np.array(shap[shap.columns[pd.Series(shap.columns).str.startswith(variable_names[i])]]),(len(shap),144,90))
    data.append(variable_data)

var_data = dict(
    shap_values = ([ 'n_channel', 'time', 'lon', 'lat'], data)
)

coords = dict(
    time = (['time'], np.array(X_val.time)), 
    n_channel = (['n_channel'], np.array(variable_names)),
      
)
ds_shap = xr.Dataset(
    data_vars = var_data, 
    coords = coords
)

ds_shap.to_netcdf(new_folder+'shap.nc')


m.save_model(new_folder+'model.json')
