import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def z_score_normalize(lst):
    normalized = []
    for value in lst:
        normalized_num = (value - np.mean(lst)) / np.std(lst)
        normalized.append(normalized_num)
    return np.array(normalized)

np.random.seed(20210302)

#Load Data
datas = pd.read_csv('../0. dataset/diabetes2.csv')
dataset = datas.to_numpy()

# DataSet Split
train, test = train_test_split(dataset,test_size=0.3)

# Normalization
train_X = z_score_normalize(train[:, 0:11])
train_Y = train[:, 11]
test_X = z_score_normalize(test[:, 0:11])
test_Y = test[:, 11]

#Model Making
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_dim=11),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.GaussianDropout(0.25),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.GaussianDropout(0.25),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.GaussianDropout(0.25),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

#Model Training with Validation
history = model.fit(train_X,train_Y, epochs=10, batch_size=1000, verbose=0, validation_split=0.2)

#Model Test
scores = model.evaluate(test_X,test_Y)
print("\nANN %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Make PLT
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'val', 'loss', 'val_loss'], loc='upper left')
plt.show()

#Custom Data Testing
#                       ht      wt   wc    bmi bo1_1 bo2_1 bd1_11 bd2 bs1_1 bs2_1 bs3_1
patient_1 = np.array([[156  , 68.1, 81.2, 27.98,    2,    4,    0,  0,    0,    0,    0]]) # dg = 1 (코호트)
patient_2 = np.array([[157.1, 64.2, 86.8, 26.01,    1,    1, 	0,	0,	  3,	0,    0]]) # dg = 1 (국민건강)
patient_3 = np.array([[177.9, 74.7, 80.4, 23.60,	1,	  1,	2, 16,    2,   16,    3]]) # dg = 0 (국민건강)

print(model.predict_classes(patient_1)*100)
print(model.predict_classes(patient_2)*100)
print(model.predict_classes(patient_3)*100)

#Wrong Data Check
predict_class_1 = model.predict_classes(test_X)

unique, counts = np.unique((test_Y==predict_class_1.T), return_counts=True)
ounique, ocounts = np.unique((test_Y), return_counts=True)

print(dict(zip(ounique, ocounts)))
print(dict(zip(unique, counts)))

# for i in range(0, len(test_Y)):
#     str_1 = "";
#     if(test_Y[i] != predict_class_1[i]):
#         for j in test_X[i]:
#             str_1 = str_1 + ", " + str(j)
#
#         print("input : ",str_1,"output : ",predict_class_1[i][0],", but real is",test_Y[i])