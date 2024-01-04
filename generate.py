import tensorflow as tf
from tensorflow.keras.models import load_model

from utils.tf_utils import R2

from utils.utils import ReadDataset, Normalize
# Data Preprocessing
## Load Dataset
data = ReadDataset("dataset_psqi_memory.csv", dir="./dataset")
x, y = data()
x, y = Normalize(x)(axis=0), Normalize(y)(axis=0)
x, y = x[169:], y[169:]

model = load_model("generative_model.keras", compile=False)
model.compile(loss="mse", metrics=[R2])

print(model.predict(x)[40:,-1])
print(y[40:,-1])
