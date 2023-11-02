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
batch_size = 64
epoch_num = 10


ds_train = xr.open_dataset('/rc_scratch/reba1583/CNN_data_limitNoAR/train.nc')
ds_val = xr.open_dataset('/rc_scratch/reba1583/CNN_data_limitNoAR/validate.nc')

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

train_data = tf.data.Dataset.from_tensor_slices((X_train, utils.to_categorical(Y_train)))
val_data = tf.data.Dataset.from_tensor_slices((X_val, utils.to_categorical(Y_val)))

#batch both 
train_data = train_data.batch(batch_size)
val_data = val_data.batch(batch_size)

for batch in train_data:
    break 
for batch in val_data:
    break 


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=(train_data.element_spec[0].shape[1:])),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # Add max pooling layer

    tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),

    tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),

    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # Add max pooling layer

    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),

    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # Add max pooling layer

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(0.3),  # Add dropout layer
    tf.keras.layers.Dense(64, activation='relu'),  # Add dense layer
    tf.keras.layers.Dropout(0.3),  # Add dropout layer
    tf.keras.layers.Dense(32, activation='relu'),  # Add dense layer
    tf.keras.layers.Dropout(0.3),  # Add dropout layer
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dropout(0.3),  # Add dropout layer
    #output layer 
    tf.keras.layers.Dense(11,activation='softmax'),
])

model.summary()


model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3), metrics=['accuracy'])
# try optimizer = 'adam'
history = model.fit(train_data,validation_data=val_data,epochs=epoch_num)
history_pd = pd.DataFrame(history.history)
y_preds_distribution = model.predict(val_data)


results_pd = pd.DataFrame(y_preds_distribution)
results_pd.insert(0,'predY_3',np.zeros((len(results_pd))))
results_pd.insert(0,'predY_2',np.zeros((len(results_pd))))
results_pd.insert(0,'predY_1',np.zeros((len(results_pd))))
results_pd.insert(0,'trueY',Y_val)
results_pd.insert(0,'timestep',np.array(time_val))


for n in range(len(results_pd)):
    results_pd.at[n , 'predY_1'] = int(np.argwhere(np.array(results_pd.iloc[n, 5:16]) == np.sort(results_pd.iloc[n, 5:16])[-1]))
    results_pd.at[n , 'predY_2'] = int(np.argwhere(np.array(results_pd.iloc[n, 5:16]) == np.sort(results_pd.iloc[n, 5:16])[-2]))
    results_pd.at[n , 'predY_3'] = int(np.argwhere(np.array(results_pd.iloc[n, 5:16]) == np.sort(results_pd.iloc[n, 5:16])[-3]))

new_folder = '/rc_scratch/reba1583/CNN_test2'
os.system('mkdir '+new_folder)
results_pd.to_csv(new_folder+'/results.csv')
model.save(new_folder+'/model.keras')
history_pd.to_csv(new_folder+'/history.csv',index= False)