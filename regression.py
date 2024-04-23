import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import pandas as pd

dataset = pd.read_csv("dataset/psqi_memory_update.csv")
X = dataset.iloc[:, :7].values
y_scene = dataset.iloc[:, 7].values
y_read = dataset.iloc[:, 8].values
y_digit = dataset.iloc[:, 9].values

X_ols = sm.add_constant(X)
regressor_OLS = sm.OLS(endog = y_read, exog = X).fit()
# print(regressor_OLS.summary())
print(dataset.keys())