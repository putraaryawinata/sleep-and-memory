#!/usr/bin/python
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
import tensorflow as tf

from utils.utils import ReadDataset, PartialNumpyArray, ObjectManip
from utils.model_generative import GenerativeModel

# Data Preprocessing
## Load Dataset
data = ReadDataset("dataset_psqi_memory.csv", dir="./dataset")
x, y = data()
# print(f"Input data shape: {x.shape}")
# print(f"Output data shape: {y.shape}")

x_arr, y_arr = PartialNumpyArray(x), PartialNumpyArray(y)
x, y = x_arr(row=169), y_arr(row=169)
print(f"Input data shape: {x.shape}")
print(f"Output data shape: {y.shape}")

## Split Dataset (train, val)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Build Generative Model
latent_dim = x.shape[1] // 2
shape_in = x.shape[1:]
shape_out = y.shape[1:]
early_stop = callbacks.EarlyStopping(monitor='loss', patience=100)
def R_squared(y, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1.0, tf.divide(residual, total))
    return r2

generative_model = GenerativeModel(latent_dim=latent_dim, shape_in=shape_in, shape_out=shape_out)
## Compile Model
generative_model.compile(optimizer="adam", loss="mse", metrics=[R_squared])
## Train Model
history = generative_model.fit(x, y, epochs=10000, validation_data=(x_test, y_test), callbacks=[early_stop])

ObjectManip("generative_model.pickle", obj=history).save_obj()
generative_model.save("generative_model.keras")