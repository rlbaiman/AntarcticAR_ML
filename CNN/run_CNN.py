import xarray as xr 
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
import pandas as pd
import os

import sys
sys.path.insert(1, '/projects/reba1583/Research3/WAF_ML_Tutorial_Part2/scripts/')
from gewitter_functions import get_contingency_table,make_performance_diagram_axis,get_acc,get_pod,get_sr,csi_from_sr_and_pod
from gewitter_functions import get_acc

## Inputs
batch_size = 32
epoch_num = 50


ds_train = xr.open_dataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/Combined_Daily_Data_CNN/train_full.nc')
ds_val = xr.open_dataset('/pl/active/ATOC_SynopticMet/data/ar_data/Research3/Data/Combined_Daily_Data_CNN/validate_full.nc')

train_random_shuffle = np.arange(len(ds_train.features))
np.random.shuffle(train_random_shuffle )

X_train = ds_train.features.values[train_random_shuffle]
Y_train = ds_train.labels_1d.values[train_random_shuffle] 
time_train = ds_train.time[train_random_shuffle]

val_random_shuffle = np.arange(len(ds_val.features))
np.random.shuffle(val_random_shuffle)
X_val = ds_val.features.values[val_random_shuffle]
Y_val = ds_val.labels_1d.values[val_random_shuffle]
time_val = ds_val.time[val_random_shuffle]

train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
val_data = tf.data.Dataset.from_tensor_slices((X_val, Y_val))

#batch both 
train_data = train_data.batch(batch_size)
val_data = val_data.batch(batch_size)

for batch in train_data:
    break 
for batch in val_data:
    break 


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(train_data.element_spec[0].shape[1:]), activation ='relu'),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(train_data.element_spec[0].shape[1:]), activation ='relu'),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=(train_data.element_spec[0].shape[1:]), activation ='relu'),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Flatten(),
    

    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(20,activation='sigmoid'),
])

model.summary()
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = .01,patience=5,  verbose = 1, )
model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2,from_logits=False),
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), metrics=['accuracy'])
history = model.fit(train_data,validation_data=val_data,epochs=epoch_num, callbacks = [callback])
history_pd = pd.DataFrame(history.history)
y_preds_distribution = model.predict(val_data)


results_pd = pd.DataFrame(y_preds_distribution)
results_pd = results_pd.set_index(np.array(time_val))

val_pd = pd.DataFrame(Y_val)
val_pd = val_pd.set_index(np.array(time_val))


new_folder = '/rc_scratch/reba1583/CNN_daily_test3_fulldata'
os.system('mkdir '+new_folder)
results_pd.to_csv(new_folder+'/results.csv')
val_pd.to_csv(new_folder+'/val.csv')
model.save(new_folder+'/model.keras')
history_pd.to_csv(new_folder+'/history.csv',index= False)