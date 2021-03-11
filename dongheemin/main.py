import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 최대 줄 수 설정
pd.set_option('display.max_rows', 500)
# 최대 열 수 설정
pd.set_option('display.max_columns', 500)
# 표시할 가로의 길이
pd.set_option('display.width', 1000)



datas = pd.read_csv('./diabetes2.csv')

# Data Information
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

#Training

np.random.seed(20210302)
dataset = datas.to_numpy()

train, test = train_test_split(dataset, test_size=0.2)
train, val = train_test_split(train, test_size=0.4)

train_X = train[:,0:11]
train_Y = train[:,11]
test_X = test[:,0:11]
test_Y = test[:,11]
val_X = val[:,0:11]
val_Y = val[:,11]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_dim=11),
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model2 = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units=30,
                              activation='tanh',
                              input_shape=train_X.shape),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
model2.summary()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model2.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

history = model.fit(train_X,train_Y, epochs=800, batch_size=100, verbose=0, validation_data=(val_X,val_Y))
history2 = model.fit(train_X,train_Y, epochs=800, batch_size=100, verbose=0, validation_data=(val_X,val_Y))

scores = model.evaluate(test_X,test_Y)
scores2 = model2.evaluate(test_X, test_Y)

print("\nANN %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("\nRNN %s: %.2f%%" % (model2.metrics_names[1], scores2[1]*100))

fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2, sharey=False, figsize=(10, 5))
fig1, (ax0_1, ax1_1) = plt.subplots(nrows=1,ncols=2, sharey=False, figsize=(10, 5))


ax0.plot(history.history['accuracy'])
ax0.set(title='model accuracy', xlabel='epoch', ylabel='accuracy')

# 모델의 오차를 그립니다.
ax1.plot(history.history['loss'])
ax1.set(title='model loss', xlabel='epoch', ylabel='loss')

ax0_1.plot(history2.history['accuracy'])
ax0_1.set(title='model accuracy', xlabel='epoch', ylabel='accuracy')

# 모델의 오차를 그립니다.
ax1_1.plot(history2.history['loss'])
ax1_1.set(title='model loss', xlabel='epoch', ylabel='loss')

plt.show()
#                       ht      wt   wc    bmi bo1_1 bo2_1 bd1_11 bd2 bs1_1 bs2_1 bs3_1
patient_1 = np.array([[156  , 68.1, 81.2, 27.98,    2,    4,    0,  0,    0,    0,    0]]) # dg = 1 (코호트)
patient_2 = np.array([[157.1, 64.2, 86.8, 26.01,    1,    1, 	0,	0,	  3,	0,    0]]) # dg = 1 (국민건강)
patient_3 = np.array([[177.9, 74.7, 80.4, 23.60,	1,	  1,	2, 16,    2,   16,    3]]) # dg = 0 (국민건강)

print(model.predict_classes(patient_1)*100)
print(model.predict_classes(patient_2)*100)
print(model.predict_classes(patient_3)*100)

print(model2.predict_classes(patient_1)*100)
print(model2.predict_classes(patient_2)*100)
print(model2.predict_classes(patient_3)*100)

predict_class_1 = model.predict_classes(test_X)
predict_class_2 = model2.predict_classes(test_X)

unique, counts = np.unique((test_Y==predict_class_1.T), return_counts=True)
unique_1, counts_1 = np.unique((test_Y==predict_class_2.T), return_counts=True)
ounique, ocounts = np.unique((test_Y), return_counts=True)

print(dict(zip(ounique, ocounts)))
print(dict(zip(unique, counts)))
print(dict(zip(unique_1, counts_1)))

# print(datas.head(5))
# print('data shape: ', datas.shape)
# print('-----------[info]-----------')
# print(datas.info())
#
# # datas.hist(figsize=(15,15))
# # sns.clustermap(data=datas.corr(), annot = True, cmap = 'RdYlBu_r', vmin = -1, vmax = 1)
# #sns.heatmap(data=datas.corr(), annot=True, fmt='.2f', linewidths=.5, cmap="Blues")

