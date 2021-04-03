import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import os
import matplotlib.pyplot as plt

def create_dictionary(labels, names):
    dictLabels = np.zeros(labels.shape)
    for i in range(len(labels)):
        val = names.index(labels[i])
        # val = np.where(names == str(labels[i][j]))
        dictLabels[i] = val
    return dictLabels

names = ['scissors', 'face', 'cat', 'scrambledpix', 'bottle', 'chair', 'shoe', 'house']
FOLDER_DIR = 'data_final'

# PCA for all subjects
FINAL_ACCS1 = []
for i in range(6):
    SUB_DIR = os.path.join(FOLDER_DIR, 'subj' + str(i + 1))
    PCA_DIR = os.path.join(SUB_DIR, 'PCA')
    LABELS_DIR = os.path.join(SUB_DIR, 'labels')

    trainf_dir = os.path.join(PCA_DIR, 'train_features.npy')
    testf_dir = os.path.join(PCA_DIR, 'test_features.npy')
    valf_dir = os.path.join(PCA_DIR, 'val_features.npy')

    trainl_dir = os.path.join(LABELS_DIR, 'train_labels.npy')
    testl_dir = os.path.join(LABELS_DIR, 'test_labels.npy')
    vall_dir = os.path.join(LABELS_DIR, 'val_labels.npy')

    train_f = np.load(trainf_dir, allow_pickle=True)
    test_f = np.load(testf_dir, allow_pickle=True)
    val_f = np.load(valf_dir, allow_pickle=True)
    train_l = np.load(trainl_dir, allow_pickle=True)
    test_l = np.load(testl_dir, allow_pickle=True)
    val_l = np.load(vall_dir, allow_pickle=True)

    train_l = create_dictionary(np.asarray(train_l), names)
    test_l = create_dictionary(np.asarray(test_l), names)
    val_l = create_dictionary(np.asarray(val_l), names)

    # Start Random Forest Training
    # Forest 1
    forest1 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=100, max_features='sqrt',
                                     max_depth=20, min_samples_leaf=4,
                                     min_samples_split=4, random_state=1)
    forest1.fit(train_f, train_l)
    pred_valid1 = forest1.predict(val_f)
    acc1 = metrics.accuracy_score(val_l, pred_valid1)
    print("Accuracy of subject {}, with Forest 1: ".format(str(i + 1)), acc1)

    # Forest 2
    forest2 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=100, max_features='sqrt',
                                     max_depth=8, min_samples_leaf=4,
                                     min_samples_split=2, random_state=1)
    forest2.fit(train_f, train_l)
    pred_valid2 = forest2.predict(val_f)
    acc2 = metrics.accuracy_score(val_l, pred_valid2)
    print("Accuracy of subject {}, with Forest 2: ".format(str(i + 1)), acc2)

    # Forest 3
    forest3 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=50, max_features='sqrt',
                                     max_depth=20, min_samples_leaf=4,
                                     min_samples_split=4, random_state=1)
    forest3.fit(train_f, train_l)
    pred_valid3 = forest3.predict(val_f)
    acc3 = metrics.accuracy_score(val_l, pred_valid3)
    print("Accuracy of subject {}, with Forest 3: ".format(str(i + 1)), acc3)

    # Forest 4
    forest4 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=500, max_features='sqrt',
                                     max_depth=30, min_samples_leaf=2,
                                     min_samples_split=4, random_state=1)
    forest4.fit(train_f, train_l)
    pred_valid4 = forest4.predict(val_f)
    acc4 = metrics.accuracy_score(val_l, pred_valid4)
    print("Accuracy of subject {}, with Forest 4: ".format(str(i + 1)), acc4)

    # Forest 5
    forest5 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=200, max_features='sqrt',
                                     max_depth=30, min_samples_leaf=2,
                                     min_samples_split=4, random_state=1)
    forest5.fit(train_f, train_l)
    pred_valid5 = forest5.predict(val_f)
    acc5 = metrics.accuracy_score(val_l, pred_valid5)
    print("Accuracy of subject {}, with Forest 5: ".format(str(i + 1)), acc5)

    # Forest 6
    forest6 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=500, max_features='sqrt',
                                     max_depth=30, min_samples_leaf=4,
                                     min_samples_split=4, random_state=1)
    forest6.fit(train_f, train_l)
    pred_valid6 = forest6.predict(val_f)
    acc6 = metrics.accuracy_score(val_l, pred_valid6)
    print("Accuracy of subject {}, with Forest 6: ".format(str(i + 1)), acc6)

    # Forest 7
    forest7 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=500, max_features='sqrt',
                                     max_depth=20, min_samples_leaf=4,
                                     min_samples_split=4, random_state=1)
    forest7.fit(train_f, train_l)
    pred_valid7 = forest7.predict(val_f)
    acc7 = metrics.accuracy_score(val_l, pred_valid7)
    print("Accuracy of subject {}, with Forest 7: ".format(str(i + 1)), acc7)

    # Try the best validation model on the test set
    Forests = [forest1, forest2, forest3, forest4, forest5, forest6, forest7]
    accuracies = np.asarray([acc1, acc2, acc3, acc4, acc5, acc6, acc7])
    max_acc = np.argmax(accuracies)
    best_forest = Forests[max_acc]

    # Try on test set
    pred_test = best_forest.predict(test_f)
    test_acc = metrics.accuracy_score(test_l, pred_test)
    print("Accuracy of subject {}, on test with Forest {}: ".format(str(i + 1), str(max_acc + 1)), test_acc)
    FINAL_ACCS1.append(test_acc)
final1 = np.asarray(FINAL_ACCS1)
print("Mean of Final PCA Accuracies: " + str(np.mean(final1)))

# Masked Data for all subjects
FINAL_ACCS2 = []
for i in range(6):
    SUB_DIR = os.path.join(FOLDER_DIR, 'subj' + str(i + 1))
    masked_DIR = os.path.join(SUB_DIR, 'masked')
    LABELS_DIR = os.path.join(SUB_DIR, 'labels')

    trainf_dir = os.path.join(masked_DIR, 'train_features.npy')
    testf_dir = os.path.join(masked_DIR, 'test_features.npy')
    valf_dir = os.path.join(masked_DIR, 'val_features.npy')

    trainl_dir = os.path.join(LABELS_DIR, 'train_labels.npy')
    testl_dir = os.path.join(LABELS_DIR, 'test_labels.npy')
    vall_dir = os.path.join(LABELS_DIR, 'val_labels.npy')

    train_f = np.load(trainf_dir, allow_pickle=True)
    test_f = np.load(testf_dir, allow_pickle=True)
    val_f = np.load(valf_dir, allow_pickle=True)
    train_l = np.load(trainl_dir, allow_pickle=True)
    test_l = np.load(testl_dir, allow_pickle=True)
    val_l = np.load(vall_dir, allow_pickle=True)

    train_l = create_dictionary(np.asarray(train_l), names)
    test_l = create_dictionary(np.asarray(test_l), names)
    val_l = create_dictionary(np.asarray(val_l), names)

    # Start Random Forest Training
    # Forest 1
    forest1 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=100, max_features='sqrt',
                                     max_depth=20, min_samples_leaf=4,
                                     min_samples_split=4, random_state=1)
    forest1.fit(train_f, train_l)
    pred_valid1 = forest1.predict(val_f)
    acc1 = metrics.accuracy_score(val_l, pred_valid1)
    print("Accuracy of subject {}, with Forest 1: ".format(str(i + 1)), acc1)

    # Forest 2
    forest2 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=100, max_features='sqrt',
                                     max_depth=8, min_samples_leaf=4,
                                     min_samples_split=2, random_state=1)
    forest2.fit(train_f, train_l)
    pred_valid2 = forest2.predict(val_f)
    acc2 = metrics.accuracy_score(val_l, pred_valid2)
    print("Accuracy of subject {}, with Forest 2: ".format(str(i + 1)), acc2)

    # Forest 3
    forest3 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=50, max_features='sqrt',
                                     max_depth=20, min_samples_leaf=4,
                                     min_samples_split=4, random_state=1)
    forest3.fit(train_f, train_l)
    pred_valid3 = forest3.predict(val_f)
    acc3 = metrics.accuracy_score(val_l, pred_valid3)
    print("Accuracy of subject {}, with Forest 3: ".format(str(i + 1)), acc3)

    # Forest 4
    forest4 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=500, max_features='sqrt',
                                     max_depth=30, min_samples_leaf=2,
                                     min_samples_split=4, random_state=1)
    forest4.fit(train_f, train_l)
    pred_valid4 = forest4.predict(val_f)
    acc4 = metrics.accuracy_score(val_l, pred_valid4)
    print("Accuracy of subject {}, with Forest 4: ".format(str(i + 1)), acc4)

    # Forest 5
    forest5 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=200, max_features='sqrt',
                                     max_depth=30, min_samples_leaf=2,
                                     min_samples_split=4, random_state=1)
    forest5.fit(train_f, train_l)
    pred_valid5 = forest5.predict(val_f)
    acc5 = metrics.accuracy_score(val_l, pred_valid5)
    print("Accuracy of subject {}, with Forest 5: ".format(str(i + 1)), acc5)

    # Forest 6
    forest6 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=500, max_features='sqrt',
                                     max_depth=30, min_samples_leaf=4,
                                     min_samples_split=4, random_state=1)
    forest6.fit(train_f, train_l)
    pred_valid6 = forest6.predict(val_f)
    acc6 = metrics.accuracy_score(val_l, pred_valid6)
    print("Accuracy of subject {}, with Forest 6: ".format(str(i + 1)), acc6)

    # Forest 7
    forest7 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=500, max_features='sqrt',
                                     max_depth=20, min_samples_leaf=4,
                                     min_samples_split=4, random_state=1)
    forest7.fit(train_f, train_l)
    pred_valid7 = forest7.predict(val_f)
    acc7 = metrics.accuracy_score(val_l, pred_valid7)
    print("Accuracy of subject {}, with Forest 7: ".format(str(i + 1)), acc7)

    # Try the best validation model on the test set
    Forests = [forest1, forest2, forest3, forest4, forest5, forest6, forest7]
    accuracies = np.asarray([acc1, acc2, acc3, acc4, acc5, acc6, acc7])
    max_acc = np.argmax(accuracies)
    best_forest = Forests[max_acc]

    # Try on test set
    pred_test = best_forest.predict(test_f)
    test_acc = metrics.accuracy_score(test_l, pred_test)
    print("Accuracy of subject {}, on test with Forest {}: ".format(str(i + 1), str(max_acc + 1)), test_acc)
    FINAL_ACCS2.append(test_acc)
final2 = np.asarray(FINAL_ACCS2)
print("Mean of Final Masked Accuracies: " + str(np.mean(final2)))

# Masked PCA Data for all subjects
FINAL_ACCS3 = []
for i in range(6):
    SUB_DIR = os.path.join(FOLDER_DIR, 'subj' + str(i + 1))
    maskedPCA_DIR = os.path.join(SUB_DIR, 'maskedPCA')
    LABELS_DIR = os.path.join(SUB_DIR, 'labels')

    trainf_dir = os.path.join(maskedPCA_DIR, 'train_features.npy')
    testf_dir = os.path.join(maskedPCA_DIR, 'test_features.npy')
    valf_dir = os.path.join(maskedPCA_DIR, 'val_features.npy')

    trainl_dir = os.path.join(LABELS_DIR, 'train_labels.npy')
    testl_dir = os.path.join(LABELS_DIR, 'test_labels.npy')
    vall_dir = os.path.join(LABELS_DIR, 'val_labels.npy')

    train_f = np.load(trainf_dir, allow_pickle=True)
    test_f = np.load(testf_dir, allow_pickle=True)
    val_f = np.load(valf_dir, allow_pickle=True)
    train_l = np.load(trainl_dir, allow_pickle=True)
    test_l = np.load(testl_dir, allow_pickle=True)
    val_l = np.load(vall_dir, allow_pickle=True)

    train_l = create_dictionary(np.asarray(train_l), names)
    test_l = create_dictionary(np.asarray(test_l), names)
    val_l = create_dictionary(np.asarray(val_l), names)

    # Start Random Forest Training
    # Forest 1
    forest1 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=100, max_features='sqrt',
                                     max_depth=20, min_samples_leaf=4,
                                     min_samples_split=4, random_state=1)
    forest1.fit(train_f, train_l)
    pred_valid1 = forest1.predict(val_f)
    acc1 = metrics.accuracy_score(val_l, pred_valid1)
    print("Accuracy of subject {}, with Forest 1: ".format(str(i + 1)), acc1)

    # Forest 2
    forest2 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=100, max_features='sqrt',
                                     max_depth=8, min_samples_leaf=4,
                                     min_samples_split=2, random_state=1)
    forest2.fit(train_f, train_l)
    pred_valid2 = forest2.predict(val_f)
    acc2 = metrics.accuracy_score(val_l, pred_valid2)
    print("Accuracy of subject {}, with Forest 2: ".format(str(i + 1)), acc2)

    # Forest 3
    forest3 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=50, max_features='sqrt',
                                     max_depth=20, min_samples_leaf=4,
                                     min_samples_split=4, random_state=1)
    forest3.fit(train_f, train_l)
    pred_valid3 = forest3.predict(val_f)
    acc3 = metrics.accuracy_score(val_l, pred_valid3)
    print("Accuracy of subject {}, with Forest 3: ".format(str(i + 1)), acc3)

    # Forest 4
    forest4 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=500, max_features='sqrt',
                                     max_depth=30, min_samples_leaf=2,
                                     min_samples_split=4, random_state=1)
    forest4.fit(train_f, train_l)
    pred_valid4 = forest4.predict(val_f)
    acc4 = metrics.accuracy_score(val_l, pred_valid4)
    print("Accuracy of subject {}, with Forest 4: ".format(str(i + 1)), acc4)

    # Forest 5
    forest5 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=200, max_features='sqrt',
                                     max_depth=30, min_samples_leaf=2,
                                     min_samples_split=4, random_state=1)
    forest5.fit(train_f, train_l)
    pred_valid5 = forest5.predict(val_f)
    acc5 = metrics.accuracy_score(val_l, pred_valid5)
    print("Accuracy of subject {}, with Forest 5: ".format(str(i + 1)), acc5)

    # Forest 6
    forest6 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=500, max_features='sqrt',
                                     max_depth=30, min_samples_leaf=4,
                                     min_samples_split=4, random_state=1)
    forest6.fit(train_f, train_l)
    pred_valid6 = forest6.predict(val_f)
    acc6 = metrics.accuracy_score(val_l, pred_valid6)
    print("Accuracy of subject {}, with Forest 6: ".format(str(i + 1)), acc6)

    # Forest 7
    forest7 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=500, max_features='sqrt',
                                     max_depth=20, min_samples_leaf=4,
                                     min_samples_split=4, random_state=1)
    forest7.fit(train_f, train_l)
    pred_valid7 = forest7.predict(val_f)
    acc7 = metrics.accuracy_score(val_l, pred_valid7)
    print("Accuracy of subject {}, with Forest 7: ".format(str(i + 1)), acc7)

    # Try the best validation model on the test set
    Forests = [forest1, forest2, forest3, forest4, forest5, forest6, forest7]
    accuracies = np.asarray([acc1, acc2, acc3, acc4, acc5, acc6, acc7])
    max_acc = np.argmax(accuracies)
    best_forest = Forests[max_acc]

    # Try on test set
    pred_test = best_forest.predict(test_f)
    test_acc = metrics.accuracy_score(test_l, pred_test)
    print("Accuracy of subject {}, on test with Forest {}: ".format(str(i + 1), str(max_acc + 1)), test_acc)
    FINAL_ACCS3.append(test_acc)
final3 = np.asarray(FINAL_ACCS3)
print("Mean of Final Masked PCA Accuracies: " + str(np.mean(final3)))

# ORG Data for all subjects
FINAL_ACCS4 = []
for i in range(6):
    SUB_DIR = os.path.join(FOLDER_DIR, 'subj' + str(i + 1))
    ORG_DIR = os.path.join(SUB_DIR, 'org')
    LABELS_DIR = os.path.join(SUB_DIR, 'labels')

    trainf_dir = os.path.join(ORG_DIR, 'train_features.npy')
    testf_dir = os.path.join(ORG_DIR, 'test_features.npy')
    valf_dir = os.path.join(ORG_DIR, 'val_features.npy')

    trainl_dir = os.path.join(LABELS_DIR, 'train_labels.npy')
    testl_dir = os.path.join(LABELS_DIR, 'test_labels.npy')
    vall_dir = os.path.join(LABELS_DIR, 'val_labels.npy')

    train_f = np.load(trainf_dir, allow_pickle=True)
    test_f = np.load(testf_dir, allow_pickle=True)
    val_f = np.load(valf_dir, allow_pickle=True)
    train_l = np.load(trainl_dir, allow_pickle=True)
    test_l = np.load(testl_dir, allow_pickle=True)
    val_l = np.load(vall_dir, allow_pickle=True)

    train_l = create_dictionary(np.asarray(train_l), names)
    test_l = create_dictionary(np.asarray(test_l), names)
    val_l = create_dictionary(np.asarray(val_l), names)

    # Start Random Forest Training
    # Forest 1
    forest1 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=100, max_features='sqrt',
                                     max_depth=20, min_samples_leaf=4,
                                     min_samples_split=4, random_state=1)
    forest1.fit(train_f, train_l)
    pred_valid1 = forest1.predict(val_f)
    acc1 = metrics.accuracy_score(val_l, pred_valid1)
    print("Accuracy of subject {}, with Forest 1: ".format(str(i + 1)), acc1)

    # Forest 2
    forest2 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=100, max_features='sqrt',
                                     max_depth=8, min_samples_leaf=4,
                                     min_samples_split=2, random_state=1)
    forest2.fit(train_f, train_l)
    pred_valid2 = forest2.predict(val_f)
    acc2 = metrics.accuracy_score(val_l, pred_valid2)
    print("Accuracy of subject {}, with Forest 2: ".format(str(i + 1)), acc2)

    # Forest 3
    forest3 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=50, max_features='sqrt',
                                     max_depth=20, min_samples_leaf=4,
                                     min_samples_split=4, random_state=1)
    forest3.fit(train_f, train_l)
    pred_valid3 = forest3.predict(val_f)
    acc3 = metrics.accuracy_score(val_l, pred_valid3)
    print("Accuracy of subject {}, with Forest 3: ".format(str(i + 1)), acc3)

    # Forest 4
    forest4 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=500, max_features='sqrt',
                                     max_depth=30, min_samples_leaf=2,
                                     min_samples_split=4, random_state=1)
    forest4.fit(train_f, train_l)
    pred_valid4 = forest4.predict(val_f)
    acc4 = metrics.accuracy_score(val_l, pred_valid4)
    print("Accuracy of subject {}, with Forest 4: ".format(str(i + 1)), acc4)

    # Forest 5
    forest5 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=200, max_features='sqrt',
                                     max_depth=30, min_samples_leaf=2,
                                     min_samples_split=4, random_state=1)
    forest5.fit(train_f, train_l)
    pred_valid5 = forest5.predict(val_f)
    acc5 = metrics.accuracy_score(val_l, pred_valid5)
    print("Accuracy of subject {}, with Forest 5: ".format(str(i + 1)), acc5)

    # Forest 6
    forest6 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=500, max_features='sqrt',
                                     max_depth=30, min_samples_leaf=4,
                                     min_samples_split=4, random_state=1)
    forest6.fit(train_f, train_l)
    pred_valid6 = forest6.predict(val_f)
    acc6 = metrics.accuracy_score(val_l, pred_valid6)
    print("Accuracy of subject {}, with Forest 6: ".format(str(i + 1)), acc6)

    # Forest 7
    forest7 = RandomForestClassifier(criterion="entropy", bootstrap=False,
                                     n_estimators=500, max_features='sqrt',
                                     max_depth=20, min_samples_leaf=4,
                                     min_samples_split=4, random_state=1)
    forest7.fit(train_f, train_l)
    pred_valid7 = forest7.predict(val_f)
    acc7 = metrics.accuracy_score(val_l, pred_valid7)
    print("Accuracy of subject {}, with Forest 7: ".format(str(i + 1)), acc7)

    # Try the best validation model on the test set
    Forests = [forest1, forest2, forest3, forest4, forest5, forest6, forest7]
    accuracies = np.asarray([acc1, acc2, acc3, acc4, acc5, acc6, acc7])
    max_acc = np.argmax(accuracies)
    best_forest = Forests[max_acc]

    # Try on test set
    pred_test = best_forest.predict(test_f)
    test_acc = metrics.accuracy_score(test_l, pred_test)
    print("Accuracy of subject {}, on test with Forest {}: ".format(str(i + 1), str(max_acc + 1)), test_acc)
    FINAL_ACCS4.append(test_acc)
final4 = np.asarray(FINAL_ACCS4)
print("Mean of Final ORG Accuracies: " + str(np.mean(final4)))

plt.figure()
objects = ('Original', 'Masked', 'PCA', 'Masked PCA')
y_pos = np.arange(len(objects))
# Calculate Mean
mean1 = np.mean(final1)
mean2 = np.mean(final2)
mean3 = np.mean(final3)
mean4 = np.mean(final4)
# Calculate Standard deviation
std1 = np.std(final1)
std2 = np.std(final2)
std3 = np.std(final3)
std4 = np.std(final4)
performance = [mean4, mean2, mean1, mean3]
error = [std4, std2, std1, std3]

plt.bar(y_pos, performance, yerr=error, align='center', alpha=0.5, capsize=10)
plt.grid(True)
plt.xticks(y_pos, objects)
plt.ylabel('Test Accuracy')
plt.title('Mean Test Accuracies for Random Forest')

plt.show()

print("Random Forest:")
print("1-Org data mean test accuracy and std: mean: {} std: {}".format(str(mean4), str(std4)))
print("")
print("2-pca data mean test accuracy and std: mean: {} std: {}".format(str(mean1), str(std1)))
print("")
print("3-masked data mean test accuracy and std: mean: {} std: {}".format(str(mean2), str(std2)))
print("")
print("4-masked pca data mean test accuracy and std: mean: {} std: {}".format(str(mean3), str(std3)))

methods = [final1, final2, final3, final4]
datas = ['PCA', 'Masked', 'Masked PCA', 'Original']
i = 0
for final in methods:
    plt.figure()
    plt.plot(final)
    plt.xlabel('Subjects')
    plt.xticks(np.arange(len(final)), ['1', '2', '3', '4', '5', '6'])
    plt.ylabel('Accuracy on Test')
    plt.title('The Validation Accuracies of Random Forest\n on {} Data'.format(datas[i]))
    plt.grid(True)
    plt.show()
    i = i + 1
