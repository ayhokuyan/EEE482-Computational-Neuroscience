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

# PCA training and validation
best_models = []
for i in range(1, 7):
    SUBJ_DIR = os.path.join(DATA_DIR, 'subj' + str(i))
    LABELS_DIR = os.path.join(SUBJ_DIR, 'labels_temporal')
    FEATURES_DIR = os.path.join(SUBJ_DIR, 'temporal')

    X_train = np.load(os.path.join(FEATURES_DIR, 'train_features.npy'), allow_pickle=True)
    X_train = np.transpose(X_train, (0, 2, 3, 4, 1))
    y_train = np.load(os.path.join(LABELS_DIR, 'train_labels.npy'), allow_pickle=True)
    y_train = le.transform(y_train)

    print('Train data for subject %d loaded: %s %s' % (i, X_train.shape, y_train.shape))

    X_val = np.load(os.path.join(FEATURES_DIR, 'val_features.npy'), allow_pickle=True)
    X_val = np.transpose(X_val, (0, 2, 3, 4, 1))
    y_val = np.load(os.path.join(LABELS_DIR, 'val_labels.npy'), allow_pickle=True)
    y_val = le.transform(y_val)
    # print(y_val)

    batch_size = 1
    no_epochs = 45
    learning_rates = [0.00005]
    best_val_acc = 0
    for lr in learning_rates:
        model = models.Sequential()
        model.add(layers.Conv3D(8, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu',
                                kernel_initializer='he_uniform', input_shape=(40, 64, 64, 9)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
        # model.add(layers.Dropout(0.35))
        # model.add(layers.Conv3D(64, kernel_size=(3, 3, 3),  activation='relu', kernel_initializer='he_uniform'))
        # model.add(layers.BatchNormalization())
        # model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
        # model.add(layers.Dropout(0.35))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu', kernel_initializer='he_uniform'))
        model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(layers.Dense(NUM_LABELS, activation='softmax'))

        model.summary()

        model.compile(loss=losses.sparse_categorical_crossentropy, optimizer=optimizers.SGD(lr=lr, momentum=0.3),
                      metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=batch_size, epochs=no_epochs, validation_data=(X_val, y_val))
        predictions = np.argmax(model.predict(X_val), axis=1)
        print(predictions, y_val)
        val_acc = accuracy(predictions, y_val)
        print('Validation Accuracy for Learning Rate %0.4f: %f' % (lr, val_acc))

        if (val_acc >= best_val_acc):
            best_val_acc = val_acc
            best_model = model

    best_models.append(best_model)

print('\n\n------------------------------------------\n\n')

# PCA test
test_accuracies = []
for i, model in zip(range(1, 7), best_models):
    SUBJ_DIR = os.path.join(DATA_DIR, 'subj' + str(i))
    LABELS_DIR = os.path.join(SUBJ_DIR, 'labels_temporal')
    FEATURES_DIR = os.path.join(SUBJ_DIR, 'temporal')

    X_test = np.load(os.path.join(FEATURES_DIR, 'test_features.npy'), allow_pickle=True)
    X_test = np.transpose(X_test, (0, 2, 3, 4, 1))
    y_test = np.load(os.path.join(LABELS_DIR, 'test_labels.npy'), allow_pickle=True)

    print('Test data for subject %d loaded: %s %s' % (i, X_test.shape, y_test.shape))
    print('Best Model: %s' % model)

    predictions = model.predict(X_test)
    test_accuracies.append(accuracy(predictions, y_test))
print('Test Accuracies for Each subject', test_accuracies)
print('Mean and Std of Accuracies: %f, %f' % (np.mean(test_accuracies), np.std(test_accuracies)))

# plot means and stds
means = 100 * np.asarray([np.mean(test_accuracies_org), np.mean(test_accuracies_masked), np.mean(test_accuracies_pca),
                          np.mean(test_accuracies_maskedpca)])
stds = 100 * np.asarray([np.std(test_accuracies_org), np.std(test_accuracies_masked), np.std(test_accuracies_pca),
                         np.std(test_accuracies_maskedpca)])
reducs = ['Original', 'Masked', 'PCA', 'Masked+PCA']

fig, ax = plt.subplots()
ax.bar(np.arange(4), means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_yticks(np.arange(0, 105, 10))
ax.set_ylabel('Test Accuracy')
ax.set_xticks(np.arange(4))
ax.set_xticklabels(reducs)
ax.set_title('Mean Test Accuracies for SVM \nwith Different Feature Reduction Pipelines')
ax.grid(True)

# Save the figure and show
plt.tight_layout()
plt.show()
