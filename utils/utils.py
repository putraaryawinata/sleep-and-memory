#!/usr/bin/python
import pandas as pd
import os

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
            
