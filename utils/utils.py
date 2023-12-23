#!/usr/bin/python
import pandas as pd
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