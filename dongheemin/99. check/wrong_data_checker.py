import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# PD Setting
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

real_datas = pd.read_csv('./210312_real_data(7-layer, norm).csv')
datas = pd.read_csv('./210312_wrong_data(7-layer, norm).csv')

# Data Information
# print(datas.head(5))
# print('data shape: ', datas.shape)
# print('-----------[info]-----------')
# print(datas.info())

# Make MAP
# datas.hist(figsize=(15,15))
# sns.clustermap(data=datas.corr(), annot = True, cmap = 'RdYlBu_r', vmin = -1, vmax = 1)
# sns.heatmap(data=datas.corr(), annot=True, fmt='.2f', linewidths=.5, cmap="Blues")

# print(datas.describe().T)
# print(datas.isnull().sum())
# color_wheel = {1: "#0392cf",
#                2: "#7bc043"}
# colors = datas["DE1_DG_PRED"].map(lambda x: color_wheel.get(x + 1))

# print(datas.DE1_DG_PRED.value_counts())
# print(real_datas.DE1_DG_PRED.value_counts())

sns.pairplot(real_datas, hue='DE1_DG_PRED')
sns.pairplot(datas, hue='DE1_DG_PRED')
plt.show()
