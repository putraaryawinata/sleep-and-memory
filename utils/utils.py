#!/usr/bin/python
from typing import Any
import pandas as pd
import numpy as np
import os
import pickle

class ReadDataset:
    def __init__(self, data_file, dir="./dataset"):
        self.data = pd.read_csv(os.path.join(dir, data_file))
    
    def __call__(self, split_in_out=True, num_split=7):
        if split_in_out:
            return self.data.iloc[:, :num_split].values, self.data.iloc[:, num_split:].values
        return self.data.values

class PartialNumpyArray:
    def __init__(self, np_array):
        self.np_array = np_array
    
    def __call__(self, row=0, col=0):
        if row:
            return self.np_array[:row]
        if col:
            return self.np_array[:, :col]
            
class ObjectManip:
    def __init__(self, name, obj=None):
        self.obj = obj
        self.name = name
    
    def save_obj(self):
        try:
            with open(self.name, 'wb') as f:
                pickle.dump(self.obj, f)
            print(f"{self.name} is saved successfully!")
        except:
            raise Exception(f"Error occurs while saving {self.name}")
    
    def load_obj(self):
        try:
            with open(self.name, 'rb') as f:
                self.obj = pickle.load(f)
            print(f"{self.name} is loaded successfully!")
            return self.obj
        except:
            raise Exception(f"Error occurs while loading {self.name}")
        
class Normalize:
    def __init__(self, arr):
        self.arr = arr
    
    def __call__(self, axis=0):
        assert axis == 0 or 1, "Only 0 and 1 accepted for the axis"
        if axis == 1:
            return (self.arr - np.min(self.arr)) / (np.max(self.arr)-np.min(self.arr))
        else:
            x = np.zeros(self.arr.shape)
            for i in range(self.arr.shape[1]):
                x[:,i] = (self.arr[:,i] - np.min(self.arr[:,i])) / (np.max(self.arr[:,i])-np.min(self.arr[:,i]))
            return x