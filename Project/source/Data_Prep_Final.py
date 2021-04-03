import numpy as np
from sklearn.decomposition import PCA, NMF
from nilearn.input_data import NiftiMasker
import os
import nibabel as nib
from sklearn.model_selection import train_test_split
from glob import glob

func_org = np.load('func_org.npy', allow_pickle=True)
func_labels = np.load('labels.npy', allow_pickle=True)
print(func_org[4].shape)
for i in range(6):
    func_org[i] = np.reshape(func_org[i], (func_org[i].shape[0], 40 * 64 * 64, 72)).astype('float32')
    func_org[i] = np.transpose(func_org[i], (1, 0, 2))
    func_org[i] = np.reshape(func_org[i], (func_org[i].shape[0], -1))
    func_org[i] = func_org[i].T
    print(func_org[i].shape)
print(func_org.shape)

print(func_labels[0].shape)
run_list = [12, 12, 12, 12, 11, 12]

labels = []
i = 0
for runs in run_list:
    temp = np.asarray(([func_labels[i]] * runs)).ravel().T
    print(temp.shape)
    labels.append(temp)
    i += 1
# labels = np.asarray(np.hstack(labels))
labels = np.asarray(labels)
print(labels.shape)

LABELS = ['scissors', 'face', 'cat', 'scrambledpix', 'bottle', 'chair', 'shoe', 'house']
NUM_LABELS = len(LABELS)

start_indices = np.asarray([6, 21, 35, 49, 63, 78, 92, 106])
all_indices = []
for i in range(9):
    all_indices.append(start_indices + i)
all_indices = np.sort(np.asarray(all_indices).ravel())
print(all_indices)
print(all_indices.shape[0])

DATA_DIR = 'Dataset'
MASK_DIR = 'masks'
SUBJ_COUNT = 6
STIM_LENGTH = 9

masks_list = []
func_masked = []

for subj in range(1, SUBJ_COUNT + 1):
    print('Subject ' + str(subj))

    # Retreive Masks for Subject
    mask_dir = os.path.join(MASK_DIR, 'subj' + str(subj), 'mask4_vt.nii.gz')
    masks_list.append(nib.load(mask_dir))
    masker = NiftiMasker(mask_img=mask_dir, smoothing_fwhm=4,
                         standardize=True, memory="nilearn_cache", memory_level=1)

    # Get FMRI and Label Directories for Subject
    func_dir = os.path.join(DATA_DIR, 'sub-' + str(subj), 'func\\*.gz')
    func_evt_dir = os.path.join(DATA_DIR, 'sub-' + str(subj), 'func\\*.tsv')
    func_files = glob(func_dir)
    func_evt_dir = glob(func_evt_dir)

    # Temp Lists for Subject
    func_masked_temp = []

    run_cnt = 1
    for file, evt in zip(func_files, func_evt_dir):
        print('Run ', run_cnt)
        run_cnt += 1

        # Original FMRI Data
        func_data = nib.load(file)

        # Masked FMRI Data
        func_masked_data = np.asarray(masker.fit_transform(func_data)).T[:, all_indices]
        print('Masked Data', func_masked_data.shape)
        func_masked_temp.append(func_masked_data)

    func_masked.append(np.asarray(func_masked_temp))

func_masked = np.asarray(func_masked)

for i in range(6):
    print(func_masked[i].shape)
    func_masked[i] = np.transpose(func_masked[i], (0, 2, 1))
    print(func_masked[i].shape)
    func_masked[i] = np.vstack(func_masked[i])
    print(func_masked[i].shape)

SAVE_DIR = 'data_final'
data_files = ["train", "test", "val"]
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

    # split masked
    X_train_mask, X_test_mask, y_train_mask, y_test_mask = train_test_split(func_masked[i], labels[i], test_size=0.2,
                                                                            random_state=24)
    X_val_mask, X_test_mask, y_val_mask, y_test_mask = train_test_split(X_test_mask, y_test_mask, test_size=0.5,
                                                                        random_state=24)

    # PCA construction
    pca = PCA(n_components=100, whiten=True)
    pca.fit(X_train)

    # PCA transforms
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    X_val_pca = pca.transform(X_val)

    # PCA masked construction
    pca = PCA(n_components=100, whiten=True)
    pca.fit(X_train_mask)

    # PCA masked transforms
    X_train_pca_mask = pca.transform(X_train_mask)
    X_test_pca_mask = pca.transform(X_test_mask)
    X_val_pca_mask = pca.transform(X_val_mask)

    folders = ['org', 'PCA', 'masked', 'maskedPCA', 'labels']
    for folder in folders:
        TYPE_SAVE_DIR = os.path.join(SUB_SAVE_DIR, folder)
        if not os.path.exists(TYPE_SAVE_DIR):
            os.mkdir(TYPE_SAVE_DIR)

        if (folder == 'org'):
            # save org
            np.save(os.path.join(TYPE_SAVE_DIR, "train_features.npy"), X_train, allow_pickle=True)
            np.save(os.path.join(TYPE_SAVE_DIR, "val_features.npy"), X_val, allow_pickle=True)
            np.save(os.path.join(TYPE_SAVE_DIR, "test_features.npy"), X_test, allow_pickle=True)

        elif (folder == 'PCA'):
            # save PCA
            np.save(os.path.join(TYPE_SAVE_DIR, "train_features.npy"), X_train_pca, allow_pickle=True)
            np.save(os.path.join(TYPE_SAVE_DIR, "val_features.npy"), X_val_pca, allow_pickle=True)
            np.save(os.path.join(TYPE_SAVE_DIR, "test_features.npy"), X_test_pca, allow_pickle=True)

        elif (folder == 'masked'):
            # save masked
            np.save(os.path.join(TYPE_SAVE_DIR, "train_features"), X_train_mask, allow_pickle=True)
            np.save(os.path.join(TYPE_SAVE_DIR, "val_features.npy"), X_val_mask, allow_pickle=True)
            np.save(os.path.join(TYPE_SAVE_DIR, "test_features.npy"), X_test_mask, allow_pickle=True)

        elif (folder == 'maskedPCA'):
            # save masked+PCA
            np.save(os.path.join(TYPE_SAVE_DIR, "train_features.npy"), X_train_pca_mask, allow_pickle=True)
            np.save(os.path.join(TYPE_SAVE_DIR, "val_features.npy"), X_val_pca_mask, allow_pickle=True)
            np.save(os.path.join(TYPE_SAVE_DIR, "test_features.npy"), X_test_pca_mask, allow_pickle=True)

        elif (folder == 'labels'):
            # save labels
            np.save(os.path.join(TYPE_SAVE_DIR, "train_labels.npy"), y_train, allow_pickle=True)
            np.save(os.path.join(TYPE_SAVE_DIR, "val_labels.npy"), y_val, allow_pickle=True)
            np.save(os.path.join(TYPE_SAVE_DIR, "test_labels.npy"), y_test, allow_pickle=True)
