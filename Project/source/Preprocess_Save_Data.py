import numpy as np
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiMasker
import nibabel as nib
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
from sklearn.decomposition import PCA, NMF

LABELS = ['scissors', 'face', 'cat', 'scrambledpix', 'bottle', 'chair', 'shoe', 'house']
NUM_LABELS = len(LABELS)
DATA_DIR = 'Dataset'

start_indices = np.asarray([6, 21, 35, 49, 63, 78, 92, 106])
all_indices = []
for i in range(9):
    all_indices.append(start_indices + i)
all_indices = np.sort(np.asarray(all_indices).ravel())
print(all_indices)
print(all_indices.shape[0])

SUBJ_COUNT = 6
STIM_LENGTH = 9

func_org = []
func_labels = []

for subj in range(1, SUBJ_COUNT + 1):
    print('Subject ' + str(subj))

    # Get FMRI and Label Directories for Subject
    func_dir = os.path.join(DATA_DIR, 'sub-' + str(subj), 'func\\*.gz')
    func_evt_dir = os.path.join(DATA_DIR, 'sub-' + str(subj), 'func\\*.tsv')
    func_files = glob(func_dir)
    func_evt_dir = glob(func_evt_dir)

    # Temp Lists for Subject
    func_org_temp = []
    func_label_temp = []

    run_cnt = 1
    for file, evt in zip(func_files, func_evt_dir):
        print('Run ', run_cnt)
        run_cnt += 1

        # Original FMRI Data
        func_data = nib.load(file)
        print('Original Data', func_data.get_fdata()[:, :, :, all_indices].shape)
        func_org_temp.append(func_data.get_fdata()[:, :, :, all_indices])

        # Read and Filter Labels for Timepoints
        func_label = pd.read_csv(evt, delimiter='\t')
        filt_labels = []
        for i in range(NUM_LABELS):
            filt_labels.append([func_label['trial_type'][i * 12]] * STIM_LENGTH)
        filt_labels = np.asarray(filt_labels).flatten()
        print('Label ', filt_labels.shape)
        func_label_temp.append(filt_labels)

    func_org.append(np.asarray(func_org_temp))
    func_labels.append(filt_labels)

func_org = np.asarray(func_org)
func_labels = np.asarray(func_labels)

print('Original ', func_org[0].shape)
print('Labels ', func_labels[0].shape)

np.save('func_org.npy', func_org, allow_pickle=True)
np.save('Labels.npy', func_labels, allow_pickle=True)
