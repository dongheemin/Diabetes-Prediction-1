import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# PD Setting
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Data Information
# print(datas.head(5))
# print('data shape: ', datas.shape)
# print('-----------[info]-----------')
# print(datas.info())
#
# # Make MAP
# # datas.hist(figsize=(15,15))
# # sns.clustermap(data=datas.corr(), annot = True, cmap = 'RdYlBu_r', vmin = -1, vmax = 1)
# # sns.heatmap(data=datas.corr(), annot=True, fmt='.2f', linewidths=.5, cmap="Blues")

# datas.info(verbose=True)
# print(datas.describe().T)
#
# print(datas.isnull().sum())
# color_wheel = {1: "#0392cf",
#                2: "#7bc043"}
# colors = datas["DE1_dg"].map(lambda x: color_wheel.get(x + 1))
# print(datas.DE1_dg.value_counts())
# datas.DE1_dg.value_counts().plot(kind="bar")
#
# plt.show()
