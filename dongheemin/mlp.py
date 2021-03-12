import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Load Data
datas = pd.read_csv('./diabetes2.csv')

#Training

np.random.seed(20210302)
dataset = datas.to_numpy()

# Normalization


# DataSet Split
train, test = train_test_split(dataset, test_size=0.2)
train, val = train_test_split(train, test_size=0.4)

train_X = train[:, 0:11]
train_Y = train[:, 11]
test_X = test[:, 0:11]
test_Y = test[:, 11]
val_X = val[:, 0:11]
val_Y = val[:, 11]

#Model Making
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_dim=11),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

#Model Training with Validation
history = model.fit(train_X,train_Y, epochs=800, batch_size=100, verbose=0, validation_data=(val_X,val_Y))

#Model Test
scores = model.evaluate(test_X,test_Y)
print("\nANN %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Make Picture
fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2, sharey=False, figsize=(10, 5))

ax0.plot(history.history['accuracy'])
ax0.set(title='model accuracy', xlabel='epoch', ylabel='accuracy')
ax1.plot(history.history['loss'])
ax1.set(title='model loss', xlabel='epoch', ylabel='loss')

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

for i in range(0, len(test_Y)):
    str_1 = "";
    if(test_Y[i] != predict_class_1[i]):
        for j in test_X[i]:
            str_1 = str_1 + ", " + str(j)

        print("input : ",str_1,"output : ",predict_class_1[i][0],", but real is",test_Y[i])

