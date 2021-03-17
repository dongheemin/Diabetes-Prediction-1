import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Normalization Function
def z_score_normalize(lst):
    normalized = []
    for value in lst:
        normalized_num = (value - np.mean(lst)) / np.std(lst)
        normalized.append(normalized_num)
    return np.array(normalized)

np.random.seed(20210302)

# Data Load
datas = pd.read_csv('../0. dataset/diabetes2.csv')
dataset = datas.to_numpy()

# Data Preprocessing
rnn_dataset = np.empty((0,12,1), dtype=np.float32)

for i in dataset:
    lsts = np.empty((0,1), np.float)

    for j in range(0,12):
        lst = np.array([i[j]], dtype=np.float32)
        lsts = np.append(lsts, [lst], axis=0)
    rnn_dataset = np.append(rnn_dataset, [lsts], axis=0)
print(rnn_dataset.shape)

# Data Split
train, test = train_test_split(rnn_dataset, test_size=0.3)

train_X = np.array(train[:,0:11], dtype=np.float32)
train_Y = np.array(train[:,11:12], dtype=np.float32)
test_X = np.array(test[:,0:11], dtype=np.float32)
test_Y = np.array(test[:,11:12], dtype=np.float32)

# Model Learning & Testing
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=100, return_sequences=True, input_shape=[11, 1]),
    tf.keras.layers.LSTM(units=30),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
history = model.fit(train_X,train_Y, epochs=800, batch_size=100, verbose=0, validation_split=0.2)
scores = model.evaluate(test_X,test_Y)
print("\nRNN %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Model Test
#                       ht      wt   wc    bmi bo1_1 bo2_1 bd1_11 bd2 bs1_1 bs2_1 bs3_1
patient_1 = np.array([[[156]  , [68.1], [81.2], [27.98],    [2],    [4],    [0],  [0],   [0],    [0],    [0]]]) # dg = 1 (코호트)
patient_2 = np.array([[[157.1], [64.2], [86.8], [26.01],    [1],    [1], 	[0],  [0],	 [3],	 [0],    [0]]]) # dg = 1 (국민건강)
patient_3 = np.array([[[177.9], [74.7], [80.4], [23.60],	[1],	[1],	[2], [16],   [2],   [16],    [3]]]) # dg = 0 (국민건강)

print(model.predict(patient_1)*100)
print(model.predict(patient_2)*100)
print(model.predict(patient_3)*100)
#
predict_class_1 = model.predict_classes(test_X)

unique, counts = np.unique((test_Y==predict_class_1.T), return_counts=True)
ounique, ocounts = np.unique((test_Y), return_counts=True)

print(dict(zip(ounique, ocounts)))
print(dict(zip(unique, counts)))

# Show Figure
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'val', 'loss', 'val_loss'], loc='upper left')

plt.show()
