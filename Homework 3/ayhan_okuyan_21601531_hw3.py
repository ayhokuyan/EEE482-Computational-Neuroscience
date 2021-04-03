import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.stats import norm

question = sys.argv[1]

def ayhan_okuyan_21601531_hw3(question):
    if question == '1' :
        print('Question 1')
        with h5py.File('hw3_data2.mat', 'r') as f:
            print(f.keys())
            yn = f['Yn'].value.T
            xn = f['Xn'].value.T
            yn = yn.astype(float)
            xn = xn.astype(float)
        print('Yn:', yn.shape)
        print('Xn:', xn.shape)

        print('Part A')
        ridge_params = np.logspace(0, 12, num=500, base=10)

        r_2_list_val = []
        r_2_list_test = []
        for coeff in ridge_params:
            sum_r_2_val = 0
            sum_r_2_test = 0
            for i in range(10):
                X_train, y_train, X_val, y_val, X_test, y_test = cvSplit(xn, yn, i)
                w_ridge = findRidgeSol(X_train, y_train, coeff)
                test_pred = np.matmul(X_test, w_ridge)
                val_pred = np.matmul(X_val, w_ridge)
                r_sq_val = calc_r2(y_val, val_pred)
                r_sq_test = calc_r2(y_test, test_pred)
                sum_r_2_val += r_sq_val
                sum_r_2_test += r_sq_test

            r_2_list_val.append(sum_r_2_val / 10)
            r_2_list_test.append(sum_r_2_test / 10)

        opt_lamda = ridge_params[np.argmax(r_2_list_val)]
        print('Optimal Lambda Value:', opt_lamda)
        print('Best Average R^2 Value for Validation', r_2_list_val[np.argmax(r_2_list_val)])
        print('Best Average R^2 Value for Test', r_2_list_test[np.argmax(r_2_list_val)])

        plt.xscale('log')
        plt.xlabel('$\lambda$ (log)')
        plt.ylabel('$R^2$')
        plt.title('Average $R^2$ vs Ridge Coefficient Values')
        plt.grid('on')
        plt.plot(ridge_params, np.asarray(r_2_list_val))
        plt.plot(ridge_params, np.asarray(r_2_list_test))
        plt.legend(['Validation', 'Test'])
        plt.show()

        print('Part B')
        BOOTSTRAP_ITER = 500
        w_boots = []
        for i in range(BOOTSTRAP_ITER):
            x_boot, y_boot = gen_bootstrap(xn, yn)
            w_boot = findRidgeSol(x_boot, y_boot)
            w_boots.append(w_boot)
        w_boots = np.asarray(w_boots)
        print(w_boots.shape)

        w_means = np.mean(w_boots, axis=0)
        w_stds = np.std(w_boots, axis=0)

        print(w_means.shape, w_stds.shape)

        x_ = np.arange(100) + 1
        plt.errorbar(x_, w_means[:, 0], yerr=2 * w_stds[:, 0], ecolor='r', elinewidth=0.4, capsize=2)
        plt.title('Model Weights - OLS')
        plt.xlabel('$i$')
        plt.ylabel('Average Value of $w_i$')
        plt.show()

        z = w_means / w_stds
        p = 2 * (1 - norm.cdf(np.abs(z)))
        significant_OLS = np.argwhere(p < 0.05)
        significant_OLS = significant_OLS[significant_OLS != 0]
        print('Indices of i different than 0:\n', significant_OLS)

        print('Part C')
        BOOTSTRAP_ITER = 500
        w_boots = []
        for i in range(BOOTSTRAP_ITER):
            x_boot, y_boot = gen_bootstrap(xn, yn)
            w_boot = findRidgeSol(x_boot, y_boot, opt_lamda)
            w_boots.append(w_boot)
        w_boots = np.asarray(w_boots)

        w_means = np.mean(w_boots, axis=0)
        w_stds = np.std(w_boots, axis=0)

        x_ = np.arange(100) + 1
        plt.errorbar(x_, w_means[:, 0], yerr=2 * w_stds[:, 0], ecolor='r', elinewidth=0.4, capsize=2)
        plt.title('Model Weights - Ridge w/$\lambda_{opt}$')
        plt.xlabel('$i$')
        plt.ylabel('Average Value of $w_i$')
        plt.show()

        z = w_means / w_stds
        p = 2 * (1 - norm.cdf(np.abs(z)))
        significant_OLS = np.argwhere(p < 0.05).flatten()
        significant_OLS = significant_OLS[significant_OLS != 0]
        print('Indices of i that are different than 0:\n', significant_OLS)

    elif question == '2' :
        print('Question 2')
        with h5py.File('hw3_data3.mat', 'r') as f:
            print(f.keys())
            pop1 = f['pop1'].value
            pop2 = f['pop2'].value
            vox1 = f['vox1'].value
            vox2 = f['vox2'].value
            building = f['building'].value
            face = f['face'].value
        print('pop1:\n', pop1.T)
        print('pop2:\n', pop2.T)
        print('vox1:\n', vox1.T)
        print('vox2:\n', vox2.T)
        print('building:\n', building.T)
        print('face:\n', face.T)

        print('Part A')
        BOOTSTRAP_ITER = 10000
        x_comb = np.vstack((pop1, pop2))
        mean_diffs = []
        for i in range(BOOTSTRAP_ITER):
            x_boot = gen_bootstrap_v2(x_comb)
            pop1_boot = x_boot[0:pop1.shape[0]]
            pop2_boot = x_boot[pop1.shape[0]:x_boot.shape[0]]
            mean_diffs.append(np.mean(pop1_boot) - np.mean(pop2_boot))
        plt.title('Difference of Means (10000 Bootstrap Iterations)')
        plt.xlabel('$(\mu_1-\mu_2)| H_0$')
        plt.ylabel('$P(x|H_0)$')
        plt.hist(mean_diffs, bins=60, density=True)
        plt.show()
        x_bar = np.mean(pop1) - np.mean(pop2)
        std = np.std(mean_diffs)
        mean = np.mean(mean_diffs)
        z = (x_bar - mean) / std
        print('z-value: ', z)
        p = 2 * (1 - norm.cdf(z))
        print('Two-Sided p-value: ', p)

        print('Part B')
        corr_vector = []
        for i in range(BOOTSTRAP_ITER):
            vox1_temp, vox2_temp = gen_bootstrap_vox(vox1, vox2)
            corr_temp = np.corrcoef(vox1_temp.T, vox2_temp.T)
            corr_vector.append(corr_temp[0, 1])
        plt.title('Identical Resampling (10000 Bootstrap Iterations)')
        plt.xlabel('$\\rho | H_0$')
        plt.ylabel('$P(x|H_0)$')
        plt.hist(corr_vector, bins=60, density=False)
        plt.show()
        print('Mean correlation:', np.mean(corr_vector))
        CONFIDENCE = 95
        lower_bound, upper_bound = confidence_interval(np.asarray(corr_vector))
        print('%2d%% confidence interval: (%1.5f, %1.5f)' % (CONFIDENCE, lower_bound, upper_bound))
        corr_zero = len(np.where(np.isclose(corr_vector, 0))[0])
        print('Number of zero-correlation values:', corr_zero)

        print('Part C')
        corr_vector = []
        for i in range(BOOTSTRAP_ITER):
            vox1_temp = gen_bootstrap_v2(vox1)
            vox2_temp = gen_bootstrap_v2(vox2)
            corr_temp = np.corrcoef(vox1_temp.T, vox2_temp.T)
            corr_vector.append(corr_temp[0, 1])
        plt.title('Independent Resampling (10000 Bootstrap Iterations)')
        plt.xlabel('$\\rho | H_0$')
        plt.ylabel('$P(x|H_0)$')
        plt.hist(corr_vector, bins=60, density=False)
        plt.show()
        print('Mean correlation:', np.mean(corr_vector))
        x_bar = np.corrcoef(vox1.T, vox2.T)[0, 1]
        std = np.std(corr_vector)
        mean = np.mean(corr_vector)
        z = (x_bar - mean) / std
        print('z-value: ', z)
        p = 1 - norm.cdf(z)
        print('One-Sided p-value: ', p)

        print('Part D')
        mean_diffs = []
        for i in range(BOOTSTRAP_ITER):
            dif_vect = gen_bootstrap_fb(face, building)
            mean_diffs.append(np.mean(dif_vect)[0])
        plt.title('Difference of Means Pairwise Resampling (10000 Bootstrap Iterations)')
        plt.xlabel('$(\mu_{face}-\mu_{building})| H_0$')
        plt.ylabel('$P(x|H_0)$')
        plt.hist(mean_diffs, bins=60, density=True)
        plt.show()
        x_bar = np.mean(face) - np.mean(building)
        std = np.std(mean_diffs)
        mean = np.mean(mean_diffs)
        z = (x_bar - mean) / std
        print('z-value: ', z)
        p = 2 * (1 - norm.cdf(np.abs(z)))
        print('Two-Sided p-value: ', p)

        print('Part E')
        x_comb = np.vstack((face, building))
        mean_diffs = []
        for i in range(BOOTSTRAP_ITER):
            x_boot = gen_bootstrap_v2(x_comb)
            face_boot = x_boot[0:face.shape[0]]
            build_boot = x_boot[face.shape[0]:x_boot.shape[0]]
            mean_diffs.append(np.mean(face_boot) - np.mean(build_boot))
        plt.title('Difference of Means (10000 Bootstrap Iterations)')
        plt.xlabel('$(\mu_1-\mu_2)| H_0$')
        plt.ylabel('$P(x|H_0)$')
        plt.hist(mean_diffs, bins=60, density=True)
        plt.show()
        x_bar = np.mean(face) - np.mean(building)
        std = np.std(mean_diffs)
        mean = np.mean(mean_diffs)
        z = (x_bar - mean) / std
        print('z-value: ', z)
        p = 2 * (1 - norm.cdf(np.abs(z)))
        print('Two-Sided p-value: ', p)



#Question 1 Functions
def findRidgeSol(X, y, r_coeff=0):
    xTx = np.matmul(X.T, X)
    xTy = np.matmul(X.T, y)
    return np.matmul(np.linalg.inv(xTx + r_coeff * np.identity(xTx.shape[0])), xTy)

def cvSplit(X,y,turn):
    FOLD = 10
    num_sample = X.shape[0]
    num_fold = int(num_sample/FOLD)
    val_ind = turn*num_fold
    test_ind = (turn+1)*num_fold
    val_end = test_ind
    if(test_ind >= num_sample):
        test_ind = 0
    train_ind = test_ind + num_fold
    test_end = train_ind
    if(train_ind >= num_sample):
        train_ind = 0
    X_val_set = X[val_ind:val_end]
    y_val_set = y[val_ind:val_end]
    X_test_set = X[test_ind:test_end]
    y_test_set = y[test_ind:test_end]
    if(train_ind == 2*num_fold):
        X_train_set = X[train_ind:num_sample]
        y_train_set = y[train_ind:num_sample]
    elif(train_ind == 0):
        X_train_set = X[0:val_ind]
        y_train_set = y[0:val_ind]
    elif(train_ind == num_fold):
        X_train_set = X[train_ind:val_ind]
        y_train_set = y[train_ind:val_ind]
    else:
        X_train_set = np.vstack((X[0:val_ind], X[train_ind:num_sample]))
        y_train_set = np.vstack((y[0:val_ind], y[train_ind:num_sample]))
    return(X_train_set,y_train_set,X_val_set,y_val_set,X_test_set,y_test_set)

def calc_r2(y_true, y_pred):
    pearson = np.corrcoef(y_true.T,y_pred.T)[0,1]
    return pearson**2

def gen_bootstrap(X,y):
    num_sample = X.shape[0]
    seq = [np.random.randint(0,num_sample) for i in range(num_sample)]
    X_boot = X[seq]
    y_boot = y[seq]
    return (X_boot, y_boot)

#Question 2 Functions
def gen_bootstrap_v2(X):
    num_sample = X.shape[0]
    seq = [np.random.randint(0,num_sample) for i in range(num_sample)]
    X_boot = X[seq]
    return X_boot

def gen_bootstrap_vox(x,y):
    num_sample = x.shape[0]
    seq = [np.random.randint(0,num_sample) for i in range(num_sample)]
    x_boot = x[seq]
    y_boot = y[seq]
    return (x_boot,y_boot)

def confidence_interval(x, percent=95):
    low_end = np.percentile(x, (100-percent)/2)
    high_end = np.percentile(x, percent+ (100-percent)/2)
    return low_end,high_end

def gen_bootstrap_fb(x,y):
    num_sample = x.shape[0]
    seq = [np.random.randint(0,num_sample) for i in range(num_sample)]
    x_boot = x[seq]
    y_boot = y[seq]
    dif_vector = []
    for i in range(num_sample):
        x_temp = x_boot[i]
        y_temp = y_boot[i]
        x_dif = x_temp - y_temp
        opt = [0,0,x_dif,-1*x_dif]
        dif_vector.append(np.random.choice(opt))
    return np.asarray(dif_vector)

ayhan_okuyan_21601531_hw3(question)



