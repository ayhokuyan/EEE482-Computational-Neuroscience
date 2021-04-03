import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy

DATA_DIR = '3D_Data_for_CNN/'
features = np.load(os.path.join(DATA_DIR,'3d_features.npy'), allow_pickle=True)
labels = np.load(os.path.join(DATA_DIR, '3d_labels.npy'), allow_pickle=True)
print(features.shape)
print(labels.shape)

LABELS = ['scissors', 'face', 'cat', 'scrambledpix', 'bottle', 'chair', 'shoe', 'house']
LABELS_INT = [0,1,2,3,4,5,6,7]
label_dict = dict()
rev_label_dict = dict()
i=0
for label in LABELS:
    label_dict[label] = i
    rev_label_dict[i] = label
    i+=1
print(label_dict)
print(rev_label_dict)
int_labels = []
for i in range(labels.shape[0]):
    int_labels.append(label_dict[labels[i]])
print(labels[0:20])
print(int_labels[0:20])
int_labels = np.asarray(int_labels)
print(int_labels.shape)

print(features.shape)
print(int_labels.shape)
features = np.transpose(features, (0,2,3,1))

X_train, X_test, y_train, y_test = train_test_split(features, int_labels, test_size=0.1, random_state=24)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

model1 = Sequential()
model1.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu',kernel_initializer='he_uniform', input_shape=(64,64,40)))
model1.add(layers.BatchNormalization())
model1.add(layers.MaxPooling2D((2,2)))
model1.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform'))
model1.add(layers.BatchNormalization())
model1.add(layers.MaxPooling2D((2,2)))
model1.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform'))
model1.add(layers.BatchNormalization())
model1.add(layers.MaxPooling2D((2,2)))
model1.add(layers.Flatten())
model1.add(layers.Dense(258, activation='relu', kernel_initializer='he_uniform'))
model1.add(layers.Dense(64, activation='relu', kernel_initializer='he_uniform'))
model1.add(layers.Dense(len(LABELS), activation='softmax'))
model1.summary()

model1.compile(loss=SparseCategoricalCrossentropy(), optimizer=SGD(learning_rate=0.00075), metrics=['accuracy'])
history = model1.fit(X_train, y_train, batch_size=64, epochs=95, validation_split=0.1113)

print(features.shape)
plt.imshow(features[0,:,:,20], cmap='gray')
plt.show()
plt.imshow(features[0,32,:,:], cmap='gray')
plt.show()
plt.imshow(features[0,:,32,:], cmap='gray')
plt.show()
