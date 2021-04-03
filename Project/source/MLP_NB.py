import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import math

data_path = ""
os.chdir("data_final/")

subj = "subj"
ways = ["labels", "masked", "maskedPCA", "org", "PCA"]

labels = []
masked = []
maskedPCA = []
org = []
PCA = []

for i in range(6):
    subj_h = subj + str(i + 1)
    p_h = data_path + subj_h + "/"
    for j in (ways):
        path_holder = p_h + j + "/"
        if j == "labels":
            path_h = p_h + j
            labels.append(np.load(path_holder + "train_labels.npy", allow_pickle=True))
            labels.append(np.load(path_holder + "val_labels.npy", allow_pickle=True))
            labels.append(np.load(path_holder + "test_labels.npy", allow_pickle=True))

        if j == "masked":
            path_h = p_h + j
            masked.append(np.load(path_holder + "train_features.npy", allow_pickle=True))
            masked.append(np.load(path_holder + "val_features.npy", allow_pickle=True))
            masked.append(np.load(path_holder + "test_features.npy", allow_pickle=True))

        if j == "maskedPCA":
            path_h = p_h + j
            maskedPCA.append(np.load(path_holder + "train_features.npy", allow_pickle=True))
            maskedPCA.append(np.load(path_holder + "val_features.npy", allow_pickle=True))
            maskedPCA.append(np.load(path_holder + "test_features.npy", allow_pickle=True))

        if j == "org":
            path_h = p_h + j
            org.append(np.load(path_holder + "train_features.npy", allow_pickle=True))
            org.append(np.load(path_holder + "val_features.npy", allow_pickle=True))
            org.append(np.load(path_holder + "test_features.npy", allow_pickle=True))

        if j == "PCA":
            path_h = p_h + j
            PCA.append(np.load(path_holder + "train_features.npy", allow_pickle=True))
            PCA.append(np.load(path_holder + "val_features.npy", allow_pickle=True))
            PCA.append(np.load(path_holder + "test_features.npy", allow_pickle=True))

np_labels = np.array(labels)
np_masked = np.array(masked)
np_masked_PCA = np.array(maskedPCA)
np_org = np.array(org)
np_PCA = np.array(PCA)

def mlp_hyper(train_f, train_l, val_f, val_l, lr, hid_lay, batch_size, k_fold=10):
    accs = []
    models = []
    counter = 1

    for i in (lr):
        for j in (hid_lay):
            for k in (batch_size):
                print(j)
                clf = MLPClassifier(max_iter=500, hidden_layer_sizes=j, learning_rate_init=i, early_stopping=True,
                                    alpha=0.00001, batch_size=k, random_state=1)

                mlp_model = clf.fit(train_f, train_l)

                mlp_preds = mlp_model.predict(val_f)

                correct = 0

                for n in range(val_l.shape[0]):
                    if val_l[n] == mlp_preds[n]:
                        correct += 1

                acc = correct / val_l.shape[0]
                print(acc)
                accs.append(acc)
                models.append(mlp_model)
        counter += 1

    return np.array(models), np.array(accs)

acc_subjs = []
lr_s = [0.0005]
batch_s = [200]
hid_s = [(1000, 750, 500, 250, 100), (1000, 2000, 1000, 100), (200, 200, 200, 200, 100)]
# , (1000,500,250), (1000,750,250,100), (1000,750,100), (1000,2000,1000, 100), (200,200,200,200,100)

accs = []
models = []

for i in range(6):
    print("Accuracy of Subject " + str(i + 1) + " is as follows:")
    print("Masked Accs of subj" + str(i + 1) + ":")
    train_f = np_masked[i * 3]
    val_f = np_masked[(i * 3) + 1]

    train_l = np_labels[i * 3]
    val_l = np_labels[(i * 3) + 1]

    models_h, acc_h = mlp_hyper(train_f, train_l, val_f, val_l, lr_s, hid_s, batch_s)
    accs.append(acc_h)
    models.append(models_h)

    print("MaskedPCA Accs of subj" + str(i + 1) + ":")
    train_f = np_masked_PCA[i * 3]
    val_f = np_masked_PCA[(i * 3) + 1]

    models_h, acc_h = mlp_hyper(train_f, train_l, val_f, val_l, lr_s, hid_s, batch_s)
    accs.append(acc_h)
    models.append(models_h)

    print("Orginal Accs of subj" + str(i + 1) + ":")
    train_f = np_org[i * 3]
    val_f = np_org[(i * 3) + 1]

    models_h, acc_h = mlp_hyper(train_f, train_l, val_f, val_l, lr_s, hid_s, batch_s)
    accs.append(acc_h)
    models.append(models_h)

    print("PCA Accs of subj" + str(i + 1) + ":")
    train_f = np_PCA[i * 3]
    val_f = np_PCA[(i * 3) + 1]

    models_h, acc_h = mlp_hyper(train_f, train_l, val_f, val_l, lr_s, hid_s, batch_s)
    accs.append(acc_h)
    models.append(models_h)

accs = np.array(accs)
models = np.array(models)

accs_1 = []
models_1 = []

accs_2 = []
models_2 = []

accs_3 = []
models_3 = []

accs_4 = []
models_4 = []

accs_5 = []
models_5 = []

accs_6 = []
models_6 = []

for i in range(24):
    t = math.ceil((i + 1) / 4)
    # print(t)
    if t == 1:
        accs_1.append(accs[i])
        models_1.append(models[i])

    elif t == 2:
        accs_2.append(accs[i])
        models_2.append(models[i])

    elif t == 3:
        accs_3.append(accs[i])
        models_3.append(models[i])

    elif t == 4:
        accs_4.append(accs[i])
        models_4.append(models[i])

    elif t == 5:
        accs_5.append(accs[i])
        models_5.append(models[i])

    elif t == 6:
        accs_6.append(accs[i])
        models_6.append(models[i])

accs_1 = np.array(accs_1)
models_1 = np.array(models_1)

accs_2 = np.array(accs_2)
models_2 = np.array(models_2)

accs_3 = np.array(accs_3)
models_3 = np.array(models_3)

accs_4 = np.array(accs_4)
models_4 = np.array(models_4)

accs_5 = np.array(accs_5)
models_5 = np.array(models_5)

accs_6 = np.array(accs_6)
models_6 = np.array(models_6)

table_titles = ["Model 1", "Model 2", "Model 3"]
table_rows = ["Maked", "Masked PCA", "Orginal", "Orginal PCA"]

print("SUBJECT 1 VALIDATION RESULTS")
sub1_table = pd.DataFrame(accs_1, table_rows, table_titles)
sub1_table.head()

print("SUBJECT 2 VALIDATION RESULTS")
sub2_table = pd.DataFrame(accs_2, table_rows, table_titles)
sub2_table.head()

print("SUBJECT 3 VALIDATION RESULTS")
sub3_table = pd.DataFrame(accs_3, table_rows, table_titles)
sub3_table.head()

print("SUBJECT 4 VALIDATION RESULTS")
sub4_table = pd.DataFrame(accs_4, table_rows, table_titles)
sub4_table.head()

print("SUBJECT 5 VALIDATION RESULTS")
sub5_table = pd.DataFrame(accs_5, table_rows, table_titles)
sub5_table.head()

print("SUBJECT 6 VALIDATION RESULTS")
sub6_table = pd.DataFrame(accs_6, table_rows, table_titles)
sub6_table.head()

accs_1 = accs_1.T
X = np.arange(4)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(X + 0.00, accs_1[0], color='b', width=0.25, label="Model 1")
ax.bar(X + 0.25, accs_1[1], color='g', width=0.25, label="Model 2")
ax.bar(X + 0.50, accs_1[2], color='r', width=0.25, label="Model 3")
# ax.bar(X + 0.75, accs_1[3], color = 'o', width = 0.25, label="Orginal PCA")
ax.legend()
ax.set_xticklabels(("", "Maked", "", "Masked PCA", '', "Orginal", "", "Orginal PCA"))
ax.set_ylabel("Estimation Accuracy in Percent (%)")
ax.set_title("Accuracy Comparison of Subject 1 Validation Accuracy for Models 1,2,3...")
ax.set_xlabel("Dataset")

accs_2 = accs_2.T
X = np.arange(4)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(X + 0.00, accs_2[0], color='b', width=0.25, label="Model 1")
ax.bar(X + 0.25, accs_2[1], color='g', width=0.25, label="Model 2")
ax.bar(X + 0.50, accs_2[2], color='r', width=0.25, label="Model 3")
# ax.bar(X + 0.75, accs_1[3], color = 'o', width = 0.25, label="Orginal PCA")
ax.legend()
ax.set_xticklabels(("", "Maked", "", "Masked PCA", '', "Orginal", "", "Orginal PCA"))
ax.set_ylabel("Estimation Accuracy in Percent (%)")
ax.set_title("Accuracy Comparison of Subject 2 Validation Accuracy for Models 1,2,3...")
ax.set_xlabel("Dataset")

accs_3 = accs_3.T
X = np.arange(4)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(X + 0.00, accs_3[0], color='b', width=0.25, label="Model 1")
ax.bar(X + 0.25, accs_3[1], color='g', width=0.25, label="Model 2")
ax.bar(X + 0.50, accs_3[2], color='r', width=0.25, label="Model 3")
# ax.bar(X + 0.75, accs_1[3], color = 'o', width = 0.25, label="Orginal PCA")
ax.legend()
ax.set_xticklabels(("", "Maked", "", "Masked PCA", '', "Orginal", "", "Orginal PCA"))
ax.set_ylabel("Estimation Accuracy in Percent (%)")
ax.set_title("Accuracy Comparison of Subject 3 Validation Accuracy for Models 1,2,3...")
ax.set_xlabel("Dataset")

accs_4 = accs_4.T
X = np.arange(4)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(X + 0.00, accs_4[0], color='b', width=0.25, label="Model 1")
ax.bar(X + 0.25, accs_4[1], color='g', width=0.25, label="Model 2")
ax.bar(X + 0.50, accs_4[2], color='r', width=0.25, label="Model 3")
# ax.bar(X + 0.75, accs_1[3], color = 'o', width = 0.25, label="Orginal PCA")
ax.legend()
ax.set_xticklabels(("", "Maked", "", "Masked PCA", '', "Orginal", "", "Orginal PCA"))
ax.set_ylabel("Estimation Accuracy in Percent (%)")
ax.set_title("Accuracy Comparison of Subject 4 Validation Accuracy for Models 1,2,3...")
ax.set_xlabel("Dataset")

accs_5 = accs_5.T
X = np.arange(4)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(X + 0.00, accs_5[0], color='b', width=0.25, label="Model 1")
ax.bar(X + 0.25, accs_5[1], color='g', width=0.25, label="Model 2")
ax.bar(X + 0.50, accs_5[2], color='r', width=0.25, label="Model 3")
# ax.bar(X + 0.75, accs_1[3], color = 'o', width = 0.25, label="Orginal PCA")
ax.legend()
ax.set_xticklabels(("", "Maked", "", "Masked PCA", '', "Orginal", "", "Orginal PCA"))
ax.set_ylabel("Estimation Accuracy in Percent (%)")
ax.set_title("Accuracy Comparison of Subject 5 Validation Accuracy for Models 1,2,3...")
ax.set_xlabel("Dataset")

accs_6 = accs_6.T
X = np.arange(4)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(X + 0.00, accs_6[0], color='b', width=0.25, label="Model 1")
ax.bar(X + 0.25, accs_6[1], color='g', width=0.25, label="Model 2")
ax.bar(X + 0.50, accs_6[2], color='r', width=0.25, label="Model 3")
# ax.bar(X + 0.75, accs_1[3], color = 'o', width = 0.25, label="Orginal PCA")
ax.legend()
ax.set_xticklabels(("", "Maked", "", "Masked PCA", '', "Orginal", "", "Orginal PCA"))
ax.set_ylabel("Estimation Accuracy in Percent (%)")
ax.set_title("Accuracy Comparison of Subject 6 Validation Accuracy for Models 1,2,3...")
ax.set_xlabel("Dataset")

accs_1 = accs_1.T
accs_2 = accs_2.T
accs_3 = accs_3.T
accs_4 = accs_4.T
accs_5 = accs_5.T
accs_6 = accs_6.T

b_models_1 = []
b_models_2 = []
b_models_3 = []
b_models_4 = []
b_models_5 = []
b_models_6 = []

for i in range(6):
    for j in range(4):
        if i == 0:
            b_h_i = np.argmax(accs_1[j])
            b_models_1.append(models_1[j, b_h_i])

        if i == 1:
            b_h_i = np.argmax(accs_2[j])
            b_models_2.append(models_2[j, b_h_i])

        if i == 2:
            b_h_i = np.argmax(accs_3[j])
            b_models_3.append(models_3[j, b_h_i])

        if i == 3:
            b_h_i = np.argmax(accs_4[j])
            b_models_4.append(models_4[j, b_h_i])

        if i == 4:
            b_h_i = np.argmax(accs_5[j])
            b_models_5.append(models_5[j, b_h_i])

        if i == 5:
            b_h_i = np.argmax(accs_6[j])
            b_models_6.append(models_6[j, b_h_i])

b_models_1 = np.array(b_models_1)
b_models_2 = np.array(b_models_2)
b_models_3 = np.array(b_models_3)
b_models_4 = np.array(b_models_4)
b_models_5 = np.array(b_models_5)
b_models_6 = np.array(b_models_6)
b_models_1[0].get_params()

accs_masked = []
accs_maskedPCA = []
accs_org = []
accs_orgPCA = []

for i in range(6):
    if i == 0:
        model_h = b_models_1
    elif i == 1:
        model_h = b_models_2
    elif i == 2:
        model_h = b_models_3
    elif i == 3:
        model_h = b_models_4
    elif i == 4:
        model_h = b_models_5
    else:
        model_h = b_models_6

    test_f = np_masked[(i * 3) + 2]
    test_l = np_labels[(i * 3) + 2]

    masked_m = model_h[0]

    pre_test = masked_m.predict(test_f)

    correct = 0
    for j in range(test_l.shape[0]):
        if pre_test[j] == test_l[j]:
            correct += 1

    accs_masked.append((correct / test_l.shape[0]))

    test_f = np_masked_PCA[(i * 3) + 2]
    test_l = np_labels[(i * 3) + 2]

    maskedPCA_m = model_h[1]

    pre_test = maskedPCA_m.predict(test_f)

    correct = 0
    for j in range(test_l.shape[0]):
        if pre_test[j] == test_l[j]:
            correct += 1

    accs_maskedPCA.append((correct / test_l.shape[0]))

    test_f = np_org[(i * 3) + 2]
    test_l = np_labels[(i * 3) + 2]

    org_m = model_h[2]

    pre_test = org_m.predict(test_f)

    correct = 0
    for j in range(test_l.shape[0]):
        if pre_test[j] == test_l[j]:
            correct += 1
    accs_org.append((correct / test_l.shape[0]))

    test_f = np_PCA[(i * 3) + 2]
    test_l = np_labels[(i * 3) + 2]

    pca_m = model_h[3]

    pre_test = pca_m.predict(test_f)

    correct = 0
    for j in range(test_l.shape[0]):
        if pre_test[j] == test_l[j]:
            correct += 1

    accs_orgPCA.append(((correct) / test_l.shape[0]))

accs_masked = np.array(accs_masked)
accs_maskedPCA = np.array(accs_maskedPCA)
accs_org = np.array(accs_org)
accs_orgPCA = np.array(accs_orgPCA)

X = np.arange(6)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(X + 0.05, accs_masked, color='b', width=0.20, label="Masked Dataset")
ax.bar(X + 0.25, accs_maskedPCA, color='g', width=0.25, label="Masked PCA Dataset")
ax.bar(X + 0.50, accs_org, color='r', width=0.25, label="Orginal Dataset")
ax.bar(X + 0.70, accs_orgPCA, color='orange', width=0.20, label="Original PCA Dataset")
ax.legend()
ax.set_xticklabels(("", "Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5", "Subject 6"))
ax.set_ylabel("Estimation Accuracy in Percent (%)")
ax.set_title("Test Dataset Accuracy Results for All Subjects for Masked, Masked PCA, Original, Original PCA Datasets")
ax.set_xlabel("Subject Number")

X = np.arange(6)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(X, [accs_masked[0], accs_masked[1], accs_maskedPCA[2], accs_masked[3], accs_masked[4], accs_masked[5]],
       color='green', width=0.5)
ax.set_xticklabels(("", "Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5", "Subject 6"))
ax.set_ylabel("Estimation Accuracy in Percent (%)")
ax.set_title("Test Dataset Accuracy Results for All Subjects For Best Models")
ax.set_xlabel("Subject Number")

table_mlp_test = np.array(
    [accs_masked[0], accs_masked[1], accs_masked[2], accs_masked[3], accs_masked[4], accs_masked[5]])

table_mlp_test = table_mlp_test.reshape(6, 1).T

table_titles = ["Best Model Acc"]
table_rows = ["Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5", "Subject 6"]
test_mlp_accs = np.concatenate(
    (accs_masked.reshape(6, 1), accs_maskedPCA.reshape(6, 1), accs_org.reshape(6, 1), accs_orgPCA.reshape(6, 1)),
    axis=1)

sub3_table = pd.DataFrame(table_mlp_test, table_titles, table_rows)
sub3_table.head()

masked_m = np.mean(accs_masked)
masked_s = np.std(accs_masked) / 2

maskedPCA_m = np.mean(accs_maskedPCA)
maskedPCA_s = np.std(accs_maskedPCA) / 2

org_m = np.mean(accs_org)
org_s = np.std(accs_org) / 2

orgPCA_m = np.mean(accs_orgPCA)
orgPCA_s = np.std(accs_orgPCA) / 2

error = [masked_s, maskedPCA_s, org_s, orgPCA_s]
means = [masked_m, maskedPCA_m, org_m, orgPCA_m]

X = np.arange(4)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(X, [masked_m, maskedPCA_m, org_m, orgPCA_m], yerr=error, color='green', align='center', width=0.5, alpha=0.5,
       capsize=10)
ax.set_xticklabels(("", "Masked", "", "Masked PCA", "", "Original", "", "Original PCA"))
ax.set_ylabel("Estimation Accuracy in Percent (%)")
ax.set_title("Test Dataset Mean Accuracy Results")
ax.set_xlabel("Subject Number")

print(np.array(error * 2))
print(means)

#____________________________________________
# NB Classifier

def naiveBayes_Train(train_f, train_l, val_f, val_l, ):
    clf = GaussianNB()
    model_nb = clf.fit(train_f, train_l)
    pre = model_nb.predict(val_f)

    correct = 0
    for j in range(val_l.shape[0]):
        if val_l[j] == pre[j]:
            correct += 1

    acc_h = correct / val_l.shape[0]
    print(acc_h)
    model = model_nb
    acc = acc_h

    return model, acc

acc_subjs_nb = []
accs_nb = []
models_nb = []

for i in range(6):
    print("Accuracy of Subject " + str(i + 1) + " is as follows:")
    print("Masked Accs of subj" + str(i + 1) + ":")
    train_f = np_masked[i * 3]
    val_f = np_masked[(i * 3) + 1]

    train_l = np_labels[i * 3]
    val_l = np_labels[(i * 3) + 1]

    models_h, acc_h = naiveBayes_Train(train_f, train_l, val_f, val_l)
    accs_nb.append(acc_h)
    models_nb.append(models_h)

    print("MaskedPCA Accs of subj" + str(i + 1) + ":")
    train_f = np_masked_PCA[i * 3]
    val_f = np_masked_PCA[(i * 3) + 1]

    models_h, acc_h = naiveBayes_Train(train_f, train_l, val_f, val_l)
    accs_nb.append(acc_h)
    models_nb.append(models_h)

    print("Orginal Accs of subj" + str(i + 1) + ":")
    train_f = np_org[i * 3]
    val_f = np_org[(i * 3) + 1]

    models_h, acc_h = naiveBayes_Train(train_f, train_l, val_f, val_l)
    accs_nb.append(acc_h)
    models_nb.append(models_h)

    print("PCA Accs of subj" + str(i + 1) + ":")
    train_f = np_PCA[i * 3]
    val_f = np_PCA[(i * 3) + 1]

    models_h, acc_h = naiveBayes_Train(train_f, train_l, val_f, val_l)
    accs_nb.append(acc_h)
    models_nb.append(models_h)

accs_1_nb = []
models_1_nb = []

accs_2_nb = []
models_2_nb = []

accs_3_nb = []
models_3_nb = []

accs_4_nb = []
models_4_nb = []

accs_5_nb = []
models_5_nb = []

accs_6_nb = []
models_6_nb = []

for i in range(24):
    t = math.ceil((i + 1) / 4)
    # print(t)
    if t == 1:
        accs_1_nb.append(accs_nb[i])
        models_1_nb.append(models_nb[i])

    elif t == 2:
        accs_2_nb.append(accs_nb[i])
        models_2_nb.append(models_nb[i])

    elif t == 3:
        accs_3_nb.append(accs_nb[i])
        models_3_nb.append(models_nb[i])

    elif t == 4:
        accs_4_nb.append(accs_nb[i])
        models_4_nb.append(models_nb[i])

    elif t == 5:
        accs_5_nb.append(accs_nb[i])
        models_5_nb.append(models_nb[i])

    elif t == 6:
        accs_6_nb.append(accs_nb[i])
        models_6_nb.append(models_nb[i])

accs_1_nb = np.array(accs_1_nb)
models_1_nb = np.array(models_1_nb)

accs_2_nb = np.array(accs_2_nb)
models_2_nb = np.array(models_2_nb)

accs_3_nb = np.array(accs_3_nb)
models_3_nb = np.array(models_3_nb)

accs_4_nb = np.array(accs_4_nb)
models_4_nb = np.array(models_4_nb)

accs_5_nb = np.array(accs_5_nb)
models_5_nb = np.array(models_5_nb)

accs_6_nb = np.array(accs_6_nb)
models_6_nb = np.array(models_6_nb)

all_val_nb = np.concatenate((accs_1_nb.reshape(4, 1), accs_2_nb.reshape(4, 1), accs_3_nb.reshape(4, 1),
                             accs_4_nb.reshape(4, 1), accs_5_nb.reshape(4, 1), accs_6_nb.reshape(4, 1)), axis=1)
print(all_val_nb.shape)

print("Validation Dataset Accuracy Subject vs. Dataset Table")
table_titles = ["Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5", "Subject 6"]
table_rows = ["Maked", "Masked PCA", "Orginal", "Orginal PCA"]
sub2_table = pd.DataFrame(all_val_nb, table_rows, table_titles)
sub2_table.head()
print(accs_1_nb.shape)

X = np.arange(4)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(X + 0, accs_1_nb, color='b', width=0.15, label="Subject 1")
ax.bar(X + 0.15, accs_2_nb, color='g', width=0.15, label="Subject 2")
ax.bar(X + 0.30, accs_3_nb, color='r', width=0.15, label="Subject 3")
ax.bar(X + 0.45, accs_4_nb, color='orange', width=0.15, label="Subject 4")
ax.bar(X + 0.6, accs_5_nb, color='black', width=0.15, label="Subject 5")
ax.bar(X + 0.75, accs_6_nb, color='yellow', width=0.15, label="Subject 6")
ax.legend()
ax.set_xticklabels(('', "Masked", '', "Masked PCA", '', "Original Dataset", '', '', "Original PCA Dataset"))
ax.set_ylabel("Estimation Accuracy in Percent (%)")
ax.set_title(
    "Validation Dataset Accuracy Results for All Subjects for Masked, Masked PCA, Original, Original PCA Datasets")
ax.set_xlabel("Subject Number")

accs_masked_nb = []
accs_maskedPCA_nb = []
accs_org_nb = []
accs_orgPCA_nb = []

for i in range(6):
    if i == 0:
        model_h = models_1_nb
    elif i == 1:
        model_h = models_2_nb
    elif i == 2:
        model_h = models_3_nb
    elif i == 3:
        model_h = models_4_nb
    elif i == 4:
        model_h = models_5_nb
    else:
        model_h = models_6_nb

    test_f = np_masked[(i * 3) + 2]
    test_l = np_labels[(i * 3) + 2]

    masked_m = model_h[0]

    pre_test = masked_m.predict(test_f)

    correct = 0
    for j in range(test_l.shape[0]):
        if pre_test[j] == test_l[j]:
            correct += 1

    accs_masked_nb.append((correct / test_l.shape[0]))

    test_f = np_masked_PCA[(i * 3) + 2]
    test_l = np_labels[(i * 3) + 2]

    maskedPCA_m = model_h[1]

    pre_test = maskedPCA_m.predict(test_f)

    correct = 0
    for j in range(test_l.shape[0]):
        if pre_test[j] == test_l[j]:
            correct += 1

    accs_maskedPCA_nb.append((correct / test_l.shape[0]))

    test_f = np_org[(i * 3) + 2]
    test_l = np_labels[(i * 3) + 2]

    org_m = model_h[2]

    pre_test = org_m.predict(test_f)

    correct = 0
    for j in range(test_l.shape[0]):
        if pre_test[j] == test_l[j]:
            correct += 1

    accs_org_nb.append((correct / test_l.shape[0]))

    test_f = np_PCA[(i * 3) + 2]
    test_l = np_labels[(i * 3) + 2]

    pca_m = model_h[3]

    pre_test = pca_m.predict(test_f)

    correct = 0
    for j in range(test_l.shape[0]):
        if pre_test[j] == test_l[j]:
            correct += 1

    accs_orgPCA_nb.append((correct / test_l.shape[0]))

X = np.arange(6)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(X + 0.05, accs_masked_nb, color='b', width=0.20, label="Masked Dataset")
ax.bar(X + 0.25, accs_maskedPCA_nb, color='g', width=0.25, label="Masked PCA Dataset")
ax.bar(X + 0.50, accs_org_nb, color='r', width=0.25, label="Orginal Dataset")
ax.bar(X + 0.70, accs_orgPCA_nb, color='orange', width=0.20, label="Original PCA Dataset")
ax.legend()
ax.set_xticklabels(("", "Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5", "Subject 6"))
ax.set_ylabel("Estimation Accuracy in Percent (%)")
ax.set_title("Test Dataset Accuracy Results for All Subjects for Masked, Masked PCA, Original, Original PCA Datasets")
ax.set_xlabel("Subject Number")

accs_masked_nb = np.array(accs_masked_nb)
all_nb_test = np.array(
    [accs_masked_nb[0], accs_org_nb[1], accs_masked_nb[2], accs_masked_nb[3], accs_masked_nb[4], accs_masked_nb[5]])

print(all_nb_test)
all_nb_test = all_nb_test.reshape(6, 1).T

table_titles = ["Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5", "Subject 6"]
table_rows = ["Best Model Acc"]
sub2_table = pd.DataFrame(all_nb_test, table_rows, table_titles)
sub2_table.head()

X = np.arange(6)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(X,
       [accs_masked_nb[0], accs_org_nb[1], accs_masked_nb[2], accs_masked_nb[3], accs_masked_nb[4], accs_masked_nb[5]],
       color='green', width=0.5)
ax.set_xticklabels(("", "Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5", "Subject 6"))
ax.set_ylabel("Estimation Accuracy in Percent (%)")
ax.set_title("Test Dataset Accuracy Results for All Subjects For Best Models")
ax.set_xlabel("Subject Number")

masked_m_nb = np.mean(accs_masked_nb)
masked_s_nb = np.std(accs_masked_nb) / 2

maskedPCA_m_nb = np.mean(accs_maskedPCA_nb)
maskedPCA_s_nb = np.std(accs_maskedPCA_nb) / 2

org_m_nb = np.mean(accs_org_nb)
org_s_nb = np.std(accs_org_nb) / 2

orgPCA_m_nb = np.mean(accs_orgPCA_nb)
orgPCA_s_nb = np.std(accs_orgPCA_nb) / 2

error_nb = [masked_s_nb, maskedPCA_s_nb, org_s_nb, orgPCA_s_nb]
means_nb = [masked_m_nb, maskedPCA_m_nb, org_m_nb, orgPCA_m_nb]

X = np.arange(4)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(X, [masked_m_nb, maskedPCA_m_nb, org_m_nb, orgPCA_m_nb], yerr=error_nb, color='green', align='center', width=0.5,
       alpha=0.5, capsize=10)
ax.set_xticklabels(("", "Masked", "", "Masked PCA", "", "Original", "", "Original PCA"))
ax.set_ylabel("Estimation Accuracy in Percent (%)")
ax.set_title("Test Dataset Mean Accuracy Results")
ax.set_xlabel("Subject Number")
