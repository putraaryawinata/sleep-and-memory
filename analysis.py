#!/usr/bin/python
# STATISTICAL ANALYSIS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from statsmodels.multivariate.manova import MANOVA
# from statsmodels.stats.outliers_influence import variance_inflation_factor 

dataset = pd.read_csv("dataset/psqi_memory_update.csv")
predictors = list(dataset.keys())[:-3]
X = dataset[predictors]

# # MANOVA
# # Reference: 
# MANOVA_analysis = MANOVA(dataset[:, -3:], dataset[:, :-3])
# print(MANOVA_analysis.mv_test())

# # Correlation Analysis
# # Reference: https://link.springer.com/article/10.1007/s11325-020-02150-w#MOESM1
# corr_matrix = X.corr()
# print(corr_matrix)
# plt.subplots(figsize=(25,20))
# sn.heatmap(corr_matrix, linewidths=.5, annot=True,)
# plt.savefig("corr_matrix.png")

# # Variance Inflation Analysis
# # Reference: https://tesble.com/10.1080/03610918.2017.1371750, https://cir.nii.ac.jp/crid/1130000795085690880,
# # https://iopscience.iop.org/article/10.1088/1742-6596/949/1/012009/meta
# vif_data = pd.DataFrame()
# vif_data["feature"] = X.columns
# vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
# # vif_data["Tolerance"] = 1 / vif_data["VIF"]
# print(vif_data) # all feature less than 10
# vif_data.to_csv("vif_data.csv")

# PCA
sc = StandardScaler()
X_std = sc.fit_transform(X)
pca = PCA(n_components=None)
X_pca = pca.fit_transform(X_std)

var_exp = pca.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)

print(f"PCA cum_sum: {cum_var_exp}") # Exceed the 90% of the cum_sum at index 5 (n_components=5)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_std)
sc_inv = StandardScaler().fit(np.ones(X_pca.shape))
X_pca = sc_inv.inverse_transform(X_pca)
# print(X_pca)
np.save("dataset/psqi_pca.npy", X_pca)