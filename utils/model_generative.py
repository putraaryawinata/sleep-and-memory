#!/usr/bin/python
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

class GenerativeModel(Model):
    def __init__(self, latent_dim, shape_in, shape_out):
        super(GenerativeModel, self).__init__()
        self.latent_dim = latent_dim
        self.shape_in = shape_in
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim * 2, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(latent_dim * 2, activation='relu'),
            layers.Dense(tf.math.reduce_prod(shape_out), activation='relu'),
            layers.Reshape(shape_out)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded