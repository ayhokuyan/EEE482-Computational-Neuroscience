import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

func_labels = np.load('labels.npy', allow_pickle=True)
print(func_labels[0].shape)

all_unique_labels = []
for i in range(6):
    unique_labels = []
    pres_label = ''
    for j in range(len(func_labels[2])):
        if (pres_label != func_labels[2][j]):
            pres_label = func_labels[2][j]
            unique_labels.append(pres_label)
    unique_labels = np.asarray(unique_labels)
    all_unique_labels.append(unique_labels)
all_unique_labels = np.asarray(all_unique_labels)
print(all_unique_labels.shape)

print(func_labels[0])

func_org = np.load('func_org.npy', allow_pickle=True)
print(func_org[0].shape)
for i in range(6):
    func_org[i] = func_org[i].astype('float32')
    func_org[i] = np.transpose(func_org[i], (0, 4, 1, 2, 3))
    func_org[i] = np.asarray(np.split(func_org[i], indices_or_sections=8, axis=1))
    print(func_org[i].shape)
    func_org[i] = np.vstack(func_org[i])
    print(func_org[i].shape)
print(func_org[0].shape)

print(all_unique_labels.shape)
run_list = [12, 12, 12, 12, 11, 12]

labels = []
i = 0
for runs in run_list:
    temp = np.asarray(([all_unique_labels[i]] * runs)).ravel().T
    print(temp.shape)
    labels.append(temp)
    i += 1
# labels = np.asarray(np.hstack(labels))
labels = np.asarray(labels)
print(labels.shape)

SAVE_DIR = 'data_final'
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

for i in range(6):
    # subject save directory
    SUB_SAVE_DIR = os.path.join(SAVE_DIR, 'subj' + str(i + 1))
    if not os.path.exists(SUB_SAVE_DIR):
        os.mkdir(SUB_SAVE_DIR)

    # split org
    X_train, X_test, y_train, y_test = train_test_split(func_org[i], labels[i], test_size=0.2, random_state=24)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=24)

    folders = ['temporal', 'labels_temporal']
    for folder in folders:
        TYPE_SAVE_DIR = os.path.join(SUB_SAVE_DIR, folder)
        if not os.path.exists(TYPE_SAVE_DIR):
            os.mkdir(TYPE_SAVE_DIR)

        if (folder == 'temporal'):
            # save org
            np.save(os.path.join(TYPE_SAVE_DIR, "train_features.npy"), X_train, allow_pickle=True)
            np.save(os.path.join(TYPE_SAVE_DIR, "val_features.npy"), X_val, allow_pickle=True)
            np.save(os.path.join(TYPE_SAVE_DIR, "test_features.npy"), X_test, allow_pickle=True)
        elif (folder == 'labels_temporal'):
            # save labels
            np.save(os.path.join(TYPE_SAVE_DIR, "train_labels.npy"), y_train, allow_pickle=True)
            np.save(os.path.join(TYPE_SAVE_DIR, "val_labels.npy"), y_val, allow_pickle=True)
            np.save(os.path.join(TYPE_SAVE_DIR, "test_labels.npy"), y_test, allow_pickle=True)

# unified subjects (5training 1test subject)
func_org_train = np.vstack(func_org[1:5])
func_org_test = func_org[0]
del func_org
print(func_org_train.shape, func_org_test.shape)
labels_train = np.hstack(labels[1:5])
labels_test = labels[0]
print(labels_train.shape, labels_test.shape)
shuffle_indices = np.random.permutation(labels_train.shape[0])
func_org_train = func_org_train[shuffle_indices]
labels_train = labels_train[shuffle_indices]
SAVE_DIR = os.path.join('data_final', 'subj_uni')
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
np.save(os.path.join(SAVE_DIR, "train_features.npy"), func_org_train, allow_pickle=True)
np.save(os.path.join(SAVE_DIR, "test_features.npy"), func_org_test, allow_pickle=True)
np.save(os.path.join(SAVE_DIR, "train_labels.npy"), labels_train, allow_pickle=True)
np.save(os.path.join(SAVE_DIR, "test_labels.npy"), labels_test, allow_pickle=True)

# mixed subjects
func_org = np.vstack(func_org)
labels = np.hstack(labels)
shuffle_indices = np.random.permutation(func_org.shape[0])
func_org = func_org[shuffle_indices]
labels = labels[shuffle_indices]

X_train, X_test, y_train, y_test = train_test_split(func_org, labels, test_size=0.2, random_state=24)

SAVE_DIR = os.path.join('data_final', 'subj_uni_mixed')
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

np.save(os.path.join(SAVE_DIR, "train_features.npy"), X_train, allow_pickle=True)
np.save(os.path.join(SAVE_DIR, "test_features.npy"), X_test, allow_pickle=True)
np.save(os.path.join(SAVE_DIR, "train_labels.npy"), y_train, allow_pickle=True)
np.save(os.path.join(SAVE_DIR, "test_labels.npy"), y_test, allow_pickle=True)
