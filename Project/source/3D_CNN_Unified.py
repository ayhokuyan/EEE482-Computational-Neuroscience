import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, optimizers, losses, models
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

DATA_DIR = 'data_final'

def accuracy(y_pred, y_real):
    num_true = np.sum(y_pred == y_real)
    return num_true / y_real.shape[0]

le = LabelEncoder()
LABELS = ['scissors', 'face', 'cat', 'scrambledpix', 'bottle', 'chair', 'shoe', 'house']
NUM_LABELS = len(LABELS)
le.fit(LABELS)

SUBJ_DIR = os.path.join(DATA_DIR, 'subj_uni_mixed')

X_train = np.load(os.path.join(SUBJ_DIR, 'train_features.npy'), allow_pickle=True)
X_train = np.transpose(X_train, (0, 2, 3, 4, 1))
y_train = np.load(os.path.join(SUBJ_DIR, 'train_labels.npy'), allow_pickle=True)
y_train = le.transform(y_train)

print('Train data is loaded: %s %s' % (X_train.shape, y_train.shape))
# del model

batch_size = 1
no_epochs = 40
learning_rate = 0.0005
best_val_acc = 0

model = models.Sequential()
model.add(
    layers.Conv3D(16, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', kernel_initializer='he_uniform',
                  input_shape=(40, 64, 64, 9)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
model.add(
    layers.Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', kernel_initializer='he_uniform'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
model.add(
    layers.Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', kernel_initializer='he_uniform'))
model.add(layers.SpatialDropout3D(0.2))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu', kernel_initializer='he_uniform'))
model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(layers.Dense(NUM_LABELS, activation='softmax'))

model.summary()

model.compile(loss=losses.sparse_categorical_crossentropy,
              optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.4, beta_2=0.8), metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, epochs=no_epochs, validation_split=0.1)

X_test = np.load(os.path.join(SUBJ_DIR, 'test_features.npy'), allow_pickle=True)
X_test = np.transpose(X_test, (0, 2, 3, 4, 1))
y_test = np.load(os.path.join(SUBJ_DIR, 'test_labels.npy'), allow_pickle=True)
y_test = le.transform(y_test)

print('Test data is loaded: %s %s' % (X_test.shape, y_test.shape))

pred = model.predict(X_test)
print(accuracy(np.argmax(pred, axis=1), y_test))

plt.imshow(X_train[0, :, :, 20, 0])
plt.show()
plt.imshow(X_train[0, :, 20, :, 0])
plt.show()
plt.imshow(X_train[0, 20, :, :, 0])
plt.show()

plt.imshow(X_test[0, :, :, 20, 0])
plt.show()
plt.imshow(X_test[0, :, 20, :, 0])
plt.show()
plt.imshow(X_test[0, 20, :, :, 0])
plt.show()
