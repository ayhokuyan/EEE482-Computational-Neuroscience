import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

DATA_DIR = 'data_final'

def accuracy(y_pred, y_real):
    num_true = np.sum(y_pred == y_real)
    return num_true / y_real.shape[0]

def barplot(k_acc, k, string):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(k, k_acc)
    plt.xlabel('k values')
    plt.ylabel('Validation Accuracies')
    plt.title(string)
    plt.show()

# PCA training and validation
best_models = []
for i in range(1, 7):
    SUBJ_DIR = os.path.join(DATA_DIR, 'subj' + str(i))
    LABELS_DIR = os.path.join(SUBJ_DIR, 'labels')
    FEATURES_DIR = os.path.join(SUBJ_DIR, 'PCA')

    X_train = np.load(os.path.join(FEATURES_DIR, 'train_features.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(LABELS_DIR, 'train_labels.npy'), allow_pickle=True)

    print('Train data for subject %d loaded: %s %s' % (i, X_train.shape, y_train.shape))

    X_val = np.load(os.path.join(FEATURES_DIR, 'val_features.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(LABELS_DIR, 'val_labels.npy'), allow_pickle=True)

    k_range = range(1, 26)
    best_val_acc = 0
    k_acc = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn = knn.fit(X_train, y_train)
        predictions = knn.predict(X_val)
        val_acc = accuracy(predictions, y_val)
        k_acc.append(val_acc)

        if (val_acc >= best_val_acc):
            best_val_acc = val_acc
            best_model = knn

    best_models.append(best_model)
    barplot(k_acc, k_range, 'Validation Accuracies for PCA for Subject ' + str(i))
print('\n\n------------------------------------------\n\n')

# PCA test
test_accuracies_pca = []
for i, model in zip(range(1, 7), best_models):
    SUBJ_DIR = os.path.join(DATA_DIR, 'subj' + str(i))
    LABELS_DIR = os.path.join(SUBJ_DIR, 'labels')
    FEATURES_DIR = os.path.join(SUBJ_DIR, 'PCA')

    X_test = np.load(os.path.join(FEATURES_DIR, 'test_features.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(LABELS_DIR, 'test_labels.npy'), allow_pickle=True)

    print('Test data for subject %d loaded: %s %s' % (i, X_test.shape, y_test.shape))
    print('Best Model: %s' % model)

    predictions = model.predict(X_test)
    test_accuracies_pca.append(accuracy(predictions, y_test))
print('Test Accuracies for Each subject', test_accuracies_pca)
print('Mean and Std of Accuracies: %f, %f' % (np.mean(test_accuracies_pca), np.std(test_accuracies_pca)))

# Masked training and validation
best_models = []
for i in range(1, 7):
    SUBJ_DIR = os.path.join(DATA_DIR, 'subj' + str(i))
    LABELS_DIR = os.path.join(SUBJ_DIR, 'labels')
    FEATURES_DIR = os.path.join(SUBJ_DIR, 'masked')

    X_train = np.load(os.path.join(FEATURES_DIR, 'train_features.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(LABELS_DIR, 'train_labels.npy'), allow_pickle=True)

    print('Train data for subject %d loaded: %s %s' % (i, X_train.shape, y_train.shape))

    X_val = np.load(os.path.join(FEATURES_DIR, 'val_features.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(LABELS_DIR, 'val_labels.npy'), allow_pickle=True)

    k_range = range(1, 26)
    best_val_acc = 0
    k_acc = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn = knn.fit(X_train, y_train)
        predictions = knn.predict(X_val)
        val_acc = accuracy(predictions, y_val)
        k_acc.append(val_acc)

        if (val_acc >= best_val_acc):
            best_val_acc = val_acc
            best_model = knn

    best_models.append(best_model)
    barplot(k_acc, k_range, 'Validation Accuracies for Masked for Subject ' + str(i))
print('\n\n------------------------------------------\n\n')

# PCA test
test_accuracies_masked = []
for i, model in zip(range(1, 7), best_models):
    SUBJ_DIR = os.path.join(DATA_DIR, 'subj' + str(i))
    LABELS_DIR = os.path.join(SUBJ_DIR, 'labels')
    FEATURES_DIR = os.path.join(SUBJ_DIR, 'masked')

    X_test = np.load(os.path.join(FEATURES_DIR, 'test_features.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(LABELS_DIR, 'test_labels.npy'), allow_pickle=True)

    print('Test data for subject %d loaded: %s %s' % (i, X_test.shape, y_test.shape))
    print('Best Model: %s' % model)

    predictions = model.predict(X_test)
    test_accuracies_masked.append(accuracy(predictions, y_test))
print('Test Accuracies for Each subject', test_accuracies_masked)
print('Mean and Std of Accuracies: %f, %f' % (np.mean(test_accuracies_masked), np.std(test_accuracies_masked)))

# Masked-PCA training and validation
best_models = []
for i in range(1, 7):
    SUBJ_DIR = os.path.join(DATA_DIR, 'subj' + str(i))
    LABELS_DIR = os.path.join(SUBJ_DIR, 'labels')
    FEATURES_DIR = os.path.join(SUBJ_DIR, 'maskedPCA')

    X_train = np.load(os.path.join(FEATURES_DIR, 'train_features.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(LABELS_DIR, 'train_labels.npy'), allow_pickle=True)

    print('Train data for subject %d loaded: %s %s' % (i, X_train.shape, y_train.shape))

    X_val = np.load(os.path.join(FEATURES_DIR, 'val_features.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(LABELS_DIR, 'val_labels.npy'), allow_pickle=True)

    k_range = range(1, 26)
    best_val_acc = 0
    k_acc = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn = knn.fit(X_train, y_train)
        predictions = knn.predict(X_val)
        val_acc = accuracy(predictions, y_val)
        k_acc.append(val_acc)

        if (val_acc >= best_val_acc):
            best_val_acc = val_acc
            best_model = knn

    best_models.append(best_model)
    barplot(k_acc, k_range, 'Validation Accuracies for Masked PCA for Subject ' + str(i))
print('\n\n------------------------------------------\n\n')

# PCA test
test_accuracies_maskedpca = []
for i, model in zip(range(1, 7), best_models):
    SUBJ_DIR = os.path.join(DATA_DIR, 'subj' + str(i))
    LABELS_DIR = os.path.join(SUBJ_DIR, 'labels')
    FEATURES_DIR = os.path.join(SUBJ_DIR, 'maskedPCA')

    X_test = np.load(os.path.join(FEATURES_DIR, 'test_features.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(LABELS_DIR, 'test_labels.npy'), allow_pickle=True)

    print('Test data for subject %d loaded: %s %s' % (i, X_test.shape, y_test.shape))
    print('Best Model: %s' % model)

    predictions = model.predict(X_test)
    test_accuracies_maskedpca.append(accuracy(predictions, y_test))
print('Test Accuracies for Each subject', test_accuracies_maskedpca)
print('Mean and Std of Accuracies: %f, %f' % (np.mean(test_accuracies_maskedpca), np.std(test_accuracies_maskedpca)))

# Original training and validation
best_models = []
for i in range(1, 7):
    SUBJ_DIR = os.path.join(DATA_DIR, 'subj' + str(i))
    LABELS_DIR = os.path.join(SUBJ_DIR, 'labels')
    FEATURES_DIR = os.path.join(SUBJ_DIR, 'org')

    X_train = np.load(os.path.join(FEATURES_DIR, 'train_features.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(LABELS_DIR, 'train_labels.npy'), allow_pickle=True)

    print('Train data for subject %d loaded: %s %s' % (i, X_train.shape, y_train.shape))

    X_val = np.load(os.path.join(FEATURES_DIR, 'val_features.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(LABELS_DIR, 'val_labels.npy'), allow_pickle=True)

    k_range = range(1, 26)
    best_val_acc = 0
    k_acc = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn = knn.fit(X_train, y_train)
        predictions = knn.predict(X_val)
        val_acc = accuracy(predictions, y_val)
        k_acc.append(val_acc)

        if (val_acc >= best_val_acc):
            best_val_acc = val_acc
            best_model = knn

    best_models.append(best_model)
    barplot(k_acc, k_range, 'Validation Accuracies for Original Data for Subject ' + str(i))
print('\n\n------------------------------------------\n\n')

# PCA test
test_accuracies_org = []
for i, model in zip(range(1, 7), best_models):
    SUBJ_DIR = os.path.join(DATA_DIR, 'subj' + str(i))
    LABELS_DIR = os.path.join(SUBJ_DIR, 'labels')
    FEATURES_DIR = os.path.join(SUBJ_DIR, 'org')

    X_test = np.load(os.path.join(FEATURES_DIR, 'test_features.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(LABELS_DIR, 'test_labels.npy'), allow_pickle=True)

    print('Test data for subject %d loaded: %s %s' % (i, X_test.shape, y_test.shape))
    print('Best Model: %s' % model)

    predictions = model.predict(X_test)
    test_accuracies_org.append(accuracy(predictions, y_test))
print('Test Accuracies for Each subject', test_accuracies_org)
print('Mean and Std of Accuracies: %f, %f' % (np.mean(test_accuracies_org), np.std(test_accuracies_org)))

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
ax.set_title('Mean Test Accuracies for knn with different k values')
ax.grid(True)

# Save the figure and show
plt.tight_layout()
plt.show()
