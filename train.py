#!/usr/bin/python
import numpy as np
import os

from utils.utils import ReadDataset, PartialNumpyArray

# Data Preprocessing
data = ReadDataset("dataset_psqi_memory.csv", dir="./dataset")
x, y = data()
print(f"Input data shape: {x.shape}")
print(f"Output data shape: {y.shape}")

x_arr, y_arr = PartialNumpyArray(x), PartialNumpyArray(y)
x, y = x_arr(row=169), y_arr(row=169)
print(f"Input data shape: {x.shape}")
print(f"Output data shape: {y.shape}")


