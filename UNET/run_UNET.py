import xarray as xr 
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
from keras_unet_collection import models
import pandas as pd
import os


batch_size = 32
epoch_num = 10

# Landfalling ARs only
ds_train_xr = xr.open_dataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/Combined_Training_data/train.nc')
ds_val_xr = xr.open_dataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/Combined_Training_data/validate.nc')

# # All ARs
# ds_train_xr = xr.open_dataset('/rc_scratch/reba1583/data_fullAR/train.nc')
# ds_val_xr = xr.open_dataset('/rc_scratch/reba1583/data_fullAR/validate.nc')


#make datasets 
ds_train = tf.data.Dataset.from_tensor_slices((ds_train_xr.features.values,ds_train_xr.labels_2d.values))
ds_val = tf.data.Dataset.from_tensor_slices((ds_val_xr.features.values,ds_val_xr.labels_2d.values))

# #shuffle only the training
ds_train = ds_train.shuffle(ds_train.cardinality().numpy())

#batch both 
ds_train = ds_train.batch(batch_size)
ds_val = ds_val.batch(batch_size)

for batch in ds_train:
    break 
    
model = models.unet_2d([256, 32, 8],[2,4],1,stack_num_down=1,stack_num_up=1,output_activation='Sigmoid',weights=None)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

history = model.fit(ds_train,validation_data=ds_val,epochs=epoch_num)

y_preds = model.predict(ds_val)

import copy 
for i,(_,y) in enumerate(ds_val):
    if i == 0:
        y_v = copy.deepcopy(y)
    else:
        y_v = np.concatenate([y_v,y])

#ravel both
y_val = y_v.reshape((np.shape(np.squeeze(y_v))[0], np.shape(np.squeeze(y_v))[1]*np.shape(np.squeeze(y_v))[2]))
y_pred = np.squeeze(y_preds).reshape((np.shape(np.squeeze(y_preds))[0], np.shape(np.squeeze(y_preds))[1]*np.shape(np.squeeze(y_preds))[2]))

y_val_ravel = y_val.ravel()
y_pred_ravel = y_pred.ravel()


# new_folder = '/rc_scratch/reba1583/unet_testing_results/fullAR_epochs'+str(epoch_num)+'_bsize'+str(batch_size)
new_folder = '/rc_scratch/reba1583/unet_testing_results/epochs'+str(epoch_num)+'_bsize'+str(batch_size)

os.system('mkdir '+new_folder)
np.savetxt(new_folder+'/val.csv', y_val, delimiter=",")
np.savetxt(new_folder+'/pred.csv', y_pred, delimiter=",")



## Make evaluation data ###
#probability threholds 
thresh = np.arange(0.05,1.05,0.05)
#statsitcs we need for performance diagram 
tp = tf.keras.metrics.TruePositives(thresholds=thresh.tolist())
fp = tf.keras.metrics.FalsePositives(thresholds=thresh.tolist())
fn = tf.keras.metrics.FalseNegatives(thresholds=thresh.tolist())

#get performance diagram line by getting tp,fp and fn 
tp.reset_state()
fp.reset_state()
fn.reset_state()

tps = tp(y_val_ravel,y_pred_ravel)
fps = fp(y_val_ravel,y_pred_ravel)
fns = fn(y_val_ravel,y_pred_ravel)

#calc x,y of performance diagram 
pods = tps/(tps + fns)
srs = tps/(tps + fps)
csis = tps/(tps + fns + fps)

#save evaluation data
df = pd.DataFrame(data = {'pods':pods.numpy(),
                          'srs':srs.numpy(),
                          'csis': csis.numpy()},
                 index = np.arange(0.05,1.05,0.05))
df.to_csv(new_folder+'/evaluation.csv', index = 0)

