#!/usr/bin/python
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf

from utils.utils import ReadDataset, PartialNumpyArray, ObjectManip, Normalize
from utils.model_generative import GenerativeModel
from utils.tf_utils import R2

# Data Preprocessing
## Load Dataset
data = ReadDataset("dataset_psqi_memory.csv", dir="./dataset")
x, y = data()
x, y = x[:169], y[:169]
# x, y = Normalize(x)(axis=0), Normalize(y)(axis=0)
# print(f"Input data shape: {x.shape}")
# print(f"Output data shape: {y.shape}")

## Split Dataset (train, val)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Build Generative Model
latent_dim = x.shape[1] * 2
shape_in = x.shape[1:]
shape_out = y.shape[1:]
## Set Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
# best_ckpt = callbacks.ModelCheckpoint("saved_model/generative_model", monitor='val_loss', save_best_only=True)
callbacks = [early_stop,]# best_ckpt]

generative_model = GenerativeModel(latent_dim=latent_dim, shape_in=shape_in, shape_out=shape_out)
## Compile Model
generative_model.compile(optimizer="adam", loss="mse", metrics=[R2])
## Train Model
history = generative_model.fit(x, y, epochs=10000, validation_data=(x_test, y_test), callbacks=callbacks)

ObjectManip("generative_model.pickle", obj=history).save_obj()
generative_model.save("generative_model.keras")
print("keras model success")

