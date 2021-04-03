import numpy as np
import matplotlib.pyplot as plt
import os

func_labels = np.load('labels.npy', allow_pickle=True)

func_org = np.load('func_org.npy', allow_pickle=True)
print(func_org[0].shape)
for i in range(6):
    func_org[i] = func_org[i].astype('float32')
    func_org[i] = np.transpose(func_org[i], (0,4,1,2,3))
    func_org[i] = np.vstack(func_org[i])
    print(func_org[i].shape)
print(func_org.shape)
func_org = np.vstack(func_org)
print(func_org.shape)

SAVE_DIR = '3D_Data_for_CNN'
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
np.save(os.path.join(SAVE_DIR, '3d_features.npy'), func_org, allow_pickle=True)
