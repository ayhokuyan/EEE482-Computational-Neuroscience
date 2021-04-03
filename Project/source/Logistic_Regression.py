import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DATA_DIR = 'data_final'

def accuracy(y_pred, y_real):
    num_true = np.sum(y_pred == y_real)
    return num_true / y_real.shape[0]

def barplot(val_acc_tot, string):
    X = np.arange(1, 7)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(X - 0.30, val_acc_tot[0], color='b', width=0.25)
    ax.bar(X - 0.10, val_acc_tot[1], color='g', width=0.25)
    ax.bar(X + 0.10, val_acc_tot[2], color='r', width=0.25)
    ax.bar(X + 0.30, val_acc_tot[3], color='y', width=0.25)
    plt.xlabel('Subjects')
    plt.ylabel('Validation Accuracies')
    plt.title(string)
    plt.legend(["Newton-cg", "Sag", "Saga", "Lbfgs"], loc="upper right")
    plt.grid()
    plt.show()

def barplot2(val_acc_tot, string):
    X = np.arange(1, 7)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(X - 0.10, val_acc_tot[0], color='b', width=0.25)
    ax.bar(X + 0.10, val_acc_tot[1], color='g', width=0.25)
    plt.xlabel('Subjects')
    plt.ylabel('Validation Accuracies')
    plt.title(string)
    plt.legend(["Newton-cg", "Lbfgs"], loc="upper right")
    plt.grid()
    plt.show()

# PCA training and validation
best_models = []
val_acc_tot = np.zeros((4, 6))
for i in range(1, 7):
    SUBJ_DIR = os.path.join(DATA_DIR, 'subj' + str(i))
    LABELS_DIR = os.path.join(SUBJ_DIR, 'labels')
    FEATURES_DIR = os.path.join(SUBJ_DIR, 'PCA')

    X_train = np.load(os.path.join(FEATURES_DIR, 'train_features.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(LABELS_DIR, 'train_labels.npy'), allow_pickle=True)

    print('Train data for subject %d loaded: %s %s' % (i, X_train.shape, y_train.shape))

    X_val = np.load(os.path.join(FEATURES_DIR, 'val_features.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(LABELS_DIR, 'val_labels.npy'), allow_pickle=True)

    solver = ['newton-cg', 'sag', 'saga', 'lbfgs']
    best_val_acc = 0
    k_acc = []
    for j in range(len(solver)):
        logisticRegr = LogisticRegression(solver=solver[j])  # ,max_iter=1000)
        logisticRegr = logisticRegr.fit(X_train, y_train)
        predictions = logisticRegr.predict(X_val)
        val_acc = accuracy(predictions, y_val)
        k_acc.append(val_acc)
        if (val_acc >= best_val_acc):
            best_val_acc = val_acc
            best_model = logisticRegr
        print('Validation Accuracy for solver ' + solver[j] + 'is: ' + str(val_acc))
        val_acc_tot[j][i - 1] = val_acc
    best_models.append(best_model)
barplot(val_acc_tot, 'Validation Accuracies for PCA')
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
val_acc_tot = np.zeros((4, 6))
for i in range(1, 7):
    SUBJ_DIR = os.path.join(DATA_DIR, 'subj' + str(i))
    LABELS_DIR = os.path.join(SUBJ_DIR, 'labels')
    FEATURES_DIR = os.path.join(SUBJ_DIR, 'masked')

    X_train = np.load(os.path.join(FEATURES_DIR, 'train_features.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(LABELS_DIR, 'train_labels.npy'), allow_pickle=True)

    print('Train data for subject %d loaded: %s %s' % (i, X_train.shape, y_train.shape))

    X_val = np.load(os.path.join(FEATURES_DIR, 'val_features.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(LABELS_DIR, 'val_labels.npy'), allow_pickle=True)

    solver = ['newton-cg', 'sag', 'saga', 'lbfgs']
    best_val_acc = 0
    k_acc = []
    for j in range(len(solver)):
        logisticRegr = LogisticRegression(solver=solver[j])  # ,max_iter=1000)
        logisticRegr = logisticRegr.fit(X_train, y_train)
        predictions = logisticRegr.predict(X_val)
        val_acc = accuracy(predictions, y_val)
        k_acc.append(val_acc)
        if (val_acc >= best_val_acc):
            best_val_acc = val_acc
            best_model = logisticRegr
        print('Validation Accuracy for solver ' + solver[j] + 'is: ' + str(val_acc))
        val_acc_tot[j][i - 1] = val_acc
    best_models.append(best_model)
barplot(val_acc_tot, 'Validation Accuracies for Masked')
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
val_acc_tot = np.zeros((4, 6))
for i in range(1, 7):
    SUBJ_DIR = os.path.join(DATA_DIR, 'subj' + str(i))
    LABELS_DIR = os.path.join(SUBJ_DIR, 'labels')
    FEATURES_DIR = os.path.join(SUBJ_DIR, 'maskedPCA')

    X_train = np.load(os.path.join(FEATURES_DIR, 'train_features.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(LABELS_DIR, 'train_labels.npy'), allow_pickle=True)

    print('Train data for subject %d loaded: %s %s' % (i, X_train.shape, y_train.shape))

    X_val = np.load(os.path.join(FEATURES_DIR, 'val_features.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(LABELS_DIR, 'val_labels.npy'), allow_pickle=True)

    solver = ['newton-cg', 'sag', 'saga', 'lbfgs']
    best_val_acc = 0
    k_acc = []
    for j in range(len(solver)):
        logisticRegr = LogisticRegression(solver=solver[j])  # ,max_iter=1000)
        logisticRegr = logisticRegr.fit(X_train, y_train)
        predictions = logisticRegr.predict(X_val)
        val_acc = accuracy(predictions, y_val)
        k_acc.append(val_acc)
        if (val_acc >= best_val_acc):
            best_val_acc = val_acc
            best_model = logisticRegr
        print('Validation Accuracy for solver ' + solver[j] + 'is: ' + str(val_acc))
        val_acc_tot[j][i - 1] = val_acc
    best_models.append(best_model)
barplot(val_acc_tot, 'Validation Accuracies for Masked PCA')
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
val_acc_tot = np.zeros((2, 6))
for i in range(1, 7):
    SUBJ_DIR = os.path.join(DATA_DIR, 'subj' + str(i))
    LABELS_DIR = os.path.join(SUBJ_DIR, 'labels')
    FEATURES_DIR = os.path.join(SUBJ_DIR, 'org')

    X_train = np.load(os.path.join(FEATURES_DIR, 'train_features.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(LABELS_DIR, 'train_labels.npy'), allow_pickle=True)

    print('Train data for subject %d loaded: %s %s' % (i, X_train.shape, y_train.shape))

    X_val = np.load(os.path.join(FEATURES_DIR, 'val_features.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(LABELS_DIR, 'val_labels.npy'), allow_pickle=True)

    solver = ['newton-cg', 'lbfgs']
    best_val_acc = 0
    k_acc = []
    for j in range(len(solver)):
        logisticRegr = LogisticRegression(solver=solver[j])  # ,max_iter=1000)
        logisticRegr = logisticRegr.fit(X_train, y_train)
        predictions = logisticRegr.predict(X_val)
        val_acc = accuracy(predictions, y_val)
        k_acc.append(val_acc)
        if (val_acc >= best_val_acc):
            best_val_acc = val_acc
            best_model = logisticRegr
        print('Validation Accuracy for solver ' + solver[j] + 'is: ' + str(val_acc))
        val_acc_tot[j][i - 1] = val_acc
    best_models.append(best_model)
barplot2(val_acc_tot, 'Validation Accuracies for Original Data')
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
ax.set_title('Mean Test Accuracies for Logistic Regression with Different Solvers')
ax.grid(True)

# Save the figure and show
plt.tight_layout()
plt.show()
