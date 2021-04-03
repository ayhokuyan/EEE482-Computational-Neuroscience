import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sklearn.decomposition as decomp
from tqdm import tqdm

question = sys.argv[1]

def ayhan_okuyan_21601531_hw4(question):
    if question == '1' :
        print('Question 1')
        with h5py.File('hw4_data1.mat', 'r') as file:
            keys = list(file.keys())
            print(keys)
            faces = np.asarray(file['faces'])
        print(faces.shape)

        print('Part A')

        plt.imshow(faces[:, 0].reshape((32, 32)).T, cmap='bone')
        plt.axis('off')
        plt.title('Sample Stimuli')
        plt.show()

        pca_data = decomp.PCA(100, whiten=True)
        pca_data.fit(faces.T)

        plt.plot(pca_data.explained_variance_ratio_)
        plt.xlabel('Principal Components')
        plt.ylabel('Proportion of Variance Explained')
        plt.title('Proportion of Explained Variance\n for Each Individual Principle Component')
        plt.grid()
        plt.show()

        print('First 25 Principal Components')
        plt.figure(figsize=(9, 9))
        for i in range(1, 26):
            plt.subplot(5, 5, i)
            plt.imshow(pca_data.components_[i - 1].reshape((32, 32)).T, cmap='bone')
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()

        print('Part B')
        faces_rec10 = (faces - pca_data.mean_.reshape(1024, 1)).T.dot(pca_data.components_[0:10].T).dot(
            pca_data.components_[0:10]) + pca_data.mean_
        faces_rec25 = (faces - pca_data.mean_.reshape(1024, 1)).T.dot(pca_data.components_[0:25].T).dot(
            pca_data.components_[0:25]) + pca_data.mean_
        faces_rec50 = (faces - pca_data.mean_.reshape(1024, 1)).T.dot(pca_data.components_[0:50].T).dot(
            pca_data.components_[0:50]) + pca_data.mean_
        print('Original Images')
        plt.figure(figsize=(9, 9))
        for i in range(1, 37):
            plt.subplot(6, 6, i)
            plt.imshow(faces.T[i - 1].reshape((32, 32)).T, cmap='bone')
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()
        print('Reconstructed Images with 10 PCs')
        plt.figure(figsize=(9, 9))
        for i in range(1, 37):
            plt.subplot(6, 6, i)
            plt.imshow(faces_rec10[i - 1].reshape((32, 32)).T, cmap='bone')
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()
        print('Reconstructed Images with 25 PCs')
        plt.figure(figsize=(9, 9))
        for i in range(1, 37):
            plt.subplot(6, 6, i)
            plt.imshow(faces_rec25[i - 1].reshape((32, 32)).T, cmap='bone')
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()
        print('Reconstructed Images with 50 PCs')
        plt.figure(figsize=(9, 9))
        for i in range(1, 37):
            plt.subplot(6, 6, i)
            plt.imshow(faces_rec50[i - 1].reshape((32, 32)).T, cmap='bone')
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()

        mse_10 = np.mean((faces.T - faces_rec10) ** 2)
        std_10 = np.std(np.mean((faces.T - faces_rec10) ** 2, axis=1))
        print('Reconstruction MSE Loss with %d PCs:\nMean: %f\nStd: %f ' % (10, mse_10, std_10))
        mse_25 = np.mean((faces.T - faces_rec25) ** 2)
        std_25 = np.std(np.mean((faces.T - faces_rec25) ** 2, axis=1))
        print('Reconstruction MSE Loss with %d PCs:\nMean: %f\nStd: %f ' % (25, mse_25, std_25))
        mse_50 = np.mean((faces.T - faces_rec50) ** 2)
        std_50 = np.std(np.mean((faces.T - faces_rec50) ** 2, axis=1))
        print('Reconstruction MSE Loss with %d PCs:\nMean: %f\nStd: %f ' % (50, mse_50, std_50))

        print('Part C')

        rng = np.random.seed(10)
        ica10 = decomp.FastICA(10, whiten=True, random_state=rng)
        ica25 = decomp.FastICA(25, whiten=True, random_state=rng)
        ica50 = decomp.FastICA(50, whiten=True, random_state=rng)
        ica10.fit(faces.T)
        ica25.fit(faces.T)
        ica50.fit(faces.T)

        print('10 ICs')
        plt.figure(figsize=(9, 3.5))
        for i in range(1, 11):
            plt.subplot(2, 5, i)
            plt.imshow(ica10.components_[i - 1].reshape((32, 32)).T, cmap='bone')
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()
        print('25 ICs')
        plt.figure(figsize=(9, 9))
        for i in range(1, 26):
            plt.subplot(5, 5, i)
            plt.imshow(ica25.components_[i - 1].reshape((32, 32)).T, cmap='bone')
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()
        print('50 ICs')
        plt.figure(figsize=(18, 9))
        for i in range(1, 51):
            plt.subplot(5, 10, i)
            plt.imshow(ica50.components_[i - 1].reshape((32, 32)).T, cmap='bone')
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()

        faces_ica10 = ica10.fit_transform(faces.T).dot(ica10.mixing_.T) + ica10.mean_
        faces_ica25 = ica25.fit_transform(faces.T).dot(ica25.mixing_.T) + ica25.mean_
        faces_ica50 = ica50.fit_transform(faces.T).dot(ica50.mixing_.T) + ica50.mean_

        print('Reconstructed Images with 10 ICs')
        plt.figure(figsize=(9, 9))
        for i in range(1, 37):
            plt.subplot(6, 6, i)
            plt.imshow(faces_ica10[i - 1].reshape((32, 32)).T, cmap='bone')
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()
        print('Reconstructed Images with 25 ICs')
        plt.figure(figsize=(9, 9))
        for i in range(1, 37):
            plt.subplot(6, 6, i)
            plt.imshow(faces_ica25[i - 1].reshape((32, 32)).T, cmap='bone')
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()
        print('Reconstructed Images with 50 ICs')
        plt.figure(figsize=(9, 9))
        for i in range(1, 37):
            plt.subplot(6, 6, i)
            plt.imshow(faces_ica50[i - 1].reshape((32, 32)).T, cmap='bone')
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()

        mse_10 = np.mean((faces.T - faces_ica10) ** 2)
        std_10 = np.std(np.mean((faces.T - faces_ica10) ** 2, axis=1))
        print('Reconstruction MSE Loss with %d ICs:\nMean: %f\nStd: %f ' % (10, mse_10, std_10))
        mse_25 = np.mean((faces.T - faces_ica25) ** 2)
        std_25 = np.std(np.mean((faces.T - faces_ica25) ** 2, axis=1))
        print('Reconstruction MSE Loss with %d ICs:\nMean: %f\nStd: %f ' % (25, mse_25, std_25))
        mse_50 = np.mean((faces.T - faces_ica50) ** 2)
        std_50 = np.std(np.mean((faces.T - faces_ica50) ** 2, axis=1))
        print('Reconstruction MSE Loss with %d ICs:\nMean: %f\nStd: %f ' % (50, mse_50, std_50))

        print('Part D')
        faces_nn = faces + np.abs(np.min(faces))
        nmf10 = decomp.NMF(n_components=10, solver='mu', max_iter=1000)
        nmf25 = decomp.NMF(n_components=25, solver='mu', max_iter=1000)
        nmf50 = decomp.NMF(n_components=50, solver='mu', max_iter=1000)
        nmf10.fit(faces_nn.T)
        nmf25.fit(faces_nn.T)
        nmf50.fit(faces_nn.T)

        print('10 MFs')
        plt.figure(figsize=(9, 3.5))
        for i in range(1, 11):
            plt.subplot(2, 5, i)
            plt.imshow(nmf10.components_[i - 1].reshape((32, 32)).T, cmap='bone')
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()
        print('25 MFs')
        plt.figure(figsize=(9, 9))
        for i in range(1, 26):
            plt.subplot(5, 5, i)
            plt.imshow(nmf25.components_[i - 1].reshape((32, 32)).T, cmap='bone')
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()
        print('50 MFs')
        plt.figure(figsize=(18, 9))
        for i in range(1, 51):
            plt.subplot(5, 10, i)
            plt.imshow(nmf50.components_[i - 1].reshape((32, 32)).T, cmap='bone')
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()

        faces_nmf10 = nmf10.fit_transform(faces_nn.T).dot(nmf10.components_) - np.abs(np.min(faces))
        faces_nmf25 = nmf25.fit_transform(faces_nn.T).dot(nmf25.components_) - np.abs(np.min(faces))
        faces_nmf50 = nmf50.fit_transform(faces_nn.T).dot(nmf50.components_) - np.abs(np.min(faces))

        print('Reconstructed Images with 10 MFs')
        plt.figure(figsize=(9, 9))
        for i in range(1, 37):
            plt.subplot(6, 6, i)
            plt.imshow(faces_nmf10[i - 1].reshape((32, 32)).T, cmap='bone')
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()
        print('Reconstructed Images with 25 MFs')
        plt.figure(figsize=(9, 9))
        for i in range(1, 37):
            plt.subplot(6, 6, i)
            plt.imshow(faces_nmf25[i - 1].reshape((32, 32)).T, cmap='bone')
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()
        print('Reconstructed Images with 50 MFs')
        plt.figure(figsize=(9, 9))
        for i in range(1, 37):
            plt.subplot(6, 6, i)
            plt.imshow(faces_nmf50[i - 1].reshape((32, 32)).T, cmap='bone')
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()

        mse_10 = np.mean((faces.T - faces_nmf10) ** 2)
        std_10 = np.std(np.mean((faces.T - faces_nmf10) ** 2, axis=1))
        print('Reconstruction MSE Loss with %d MFs:\nMean: %f\nStd: %f ' % (10, mse_10, std_10))
        mse_25 = np.mean((faces.T - faces_nmf25) ** 2)
        std_25 = np.std(np.mean((faces.T - faces_nmf25) ** 2, axis=1))
        print('Reconstruction MSE Loss with %d MFs:\nMean: %f\nStd: %f ' % (25, mse_25, std_25))
        mse_50 = np.mean((faces.T - faces_nmf50) ** 2)
        std_50 = np.std(np.mean((faces.T - faces_nmf50) ** 2, axis=1))
        print('Reconstruction MSE Loss with %d MFs:\nMean: %f\nStd: %f ' % (50, mse_50, std_50))

    elif question == '2' :
        print('Question 2')



        print('Part A')
        x_range = np.arange(-15, 15, 0.01)
        mean_vector = np.arange(-10, 11, 1)
        STD = 1
        A = 1
        for mean in mean_vector:
            plt.plot(x_range, tuning_curve(x_range, A, mean, STD))
        plt.title('Tuning Curves of A Population of Neurons')
        plt.xlabel('Stimulus')
        plt.ylabel('Neuron Activity')
        plt.show()

        stim = -1
        plt.plot(mean_vector, tuning_curve(stim, A, mean_vector, STD))
        plt.title('Population Response to Stimulus $x=-1$\nvs. Preffered Stimuli')
        plt.xlabel('Reffered Stimulus')
        plt.ylabel('Response')
        plt.show()

        print('Part B')
        NUM_TRIAL = 200
        stim_interval = np.arange(-5, 5.01, 0.01)

        stim_vector = []
        est_stim_vector = []
        resp_vector = []
        error_vector = []

        for trial in range(NUM_TRIAL):
            trial_stim = np.random.choice(stim_interval)
            trial_resp = tuning_curve(trial_stim, A, mean_vector, STD)
            noise = np.random.normal(0, STD / 20, len(mean_vector))
            trial_resp += noise

            estimated_stim = wta_decoder(mean_vector, trial_resp)
            trial_error = np.abs(trial_stim - estimated_stim)

            stim_vector.append(trial_stim)
            est_stim_vector.append(estimated_stim)
            error_vector.append(trial_error)
            resp_vector.append(trial_resp)

        plt.figure(figsize=(12, 5))
        plt.scatter(np.arange(NUM_TRIAL), stim_vector, c='r', s=10)
        plt.scatter(np.arange(NUM_TRIAL), est_stim_vector, c='c', s=10)
        plt.legend(['stimulus', 'estimate'])
        plt.yticks(np.arange(-5, 6, 1))
        plt.grid(axis='y')
        plt.xlabel('Trial')
        plt.ylabel('Stimulus')
        plt.title('Actual and Winner-Take-All Estimated Stimuli Across 200 Trials')
        plt.show()

        wta_error_mean = np.mean(error_vector)
        wta_error_std = np.std(error_vector)
        print('Mean Error in WTA Estimation: %f \nError standard deviation in WTA Estimation: %f' % (
        wta_error_mean, wta_error_std))

        print('Part C')
        est_stim_vector_MLE = []
        error_vector_MLE = []

        for resp, stim in zip(resp_vector, stim_vector):
            estimated_stim = MLE_decoder(resp, mean_vector)

            est_stim_vector_MLE.append(estimated_stim)
            error_vector_MLE.append(np.abs(stim - estimated_stim))

        plt.figure(figsize=(12, 5))
        plt.scatter(np.arange(NUM_TRIAL), stim_vector, c='r', s=10)
        plt.scatter(np.arange(NUM_TRIAL), est_stim_vector_MLE, c='c', s=10)
        plt.legend(['stimulus', 'estimate'])
        plt.yticks(np.arange(-5, 6, 1))
        plt.grid(axis='y')
        plt.xlabel('Trial')
        plt.ylabel('Stimulus')
        plt.title('Actual and MLE Estimated Stimuli Across 200 Trials')
        plt.show()

        mle_error_mean = np.mean(error_vector_MLE)
        mle_error_std = np.std(error_vector_MLE)
        print('Mean Error in MLE Estimation: %f \nError standard deviation in MLE Estimation: %f' % (
        mle_error_mean, mle_error_std))

        print('Part D')
        est_stim_vector_MAP = []
        error_vector_MAP = []

        for resp, stim in zip(resp_vector, stim_vector):
            estimated_stim = MAP_decoder(resp, mean_vector)

            est_stim_vector_MAP.append(estimated_stim)
            error_vector_MAP.append(np.abs(stim - estimated_stim))

        plt.figure(figsize=(12, 5))
        plt.scatter(np.arange(NUM_TRIAL), stim_vector, c='r', s=10)
        plt.scatter(np.arange(NUM_TRIAL), est_stim_vector_MAP, c='c', s=10)
        plt.legend(['stimulus', 'estimate'])
        plt.yticks(np.arange(-5, 6, 1))
        plt.grid(axis='y')
        plt.xlabel('Trial')
        plt.ylabel('Stimulus')
        plt.title('Actual and MAP Estimated Stimuli Across 200 Trials')
        plt.show()

        map_error_mean = np.mean(error_vector_MAP)
        map_error_std = np.std(error_vector_MAP)
        print('Mean Error in MAP Estimation: %f \nError standard deviation in MAP Estimation: %f' % (
        map_error_mean, map_error_std))

        print('Part E')
        std_vector = [0.1, 0.2, 0.5, 1, 2, 5]
        error_all_trials = []
        for trial in tqdm(range(NUM_TRIAL)):
            trial_stim = np.random.choice(stim_interval)
            error_single_trial = []
            for std in std_vector:
                trial_resp = tuning_curve(trial_stim, A, mean_vector, std)
                noise = np.random.normal(0, 1 / 20, len(mean_vector))
                trial_resp += noise

                estimated_stim = MLE_decoder(trial_resp, mean_vector)
                error_single_trial.append(np.abs(estimated_stim - trial_stim))
            error_all_trials.append(error_single_trial)
        error_all_trials = np.asarray(error_all_trials)
        print(error_all_trials.shape)

        plt.figure(figsize=(16, 6))
        for i in range(1, 7):
            plt.subplot(2, 3, i)
            error_vals = error_all_trials[:, i - 1]
            plt.plot(np.arange(NUM_TRIAL), error_vals)
            mean = np.mean(error_vals)
            std = np.std(error_vals)
            plt.plot([mean] * NUM_TRIAL)
            plt.fill_between(np.arange(NUM_TRIAL), np.max([mean - std, 0]), mean + std, alpha=0.5, facecolor='green')
            plt.title('std = %0.1f (mean=%0.3f, std=%0.3f)' % (std_vector[i - 1], mean, std))
        plt.subplots_adjust(wspace=0.12, hspace=0.25)
        plt.show()

        means = np.mean(error_all_trials, axis=0)
        stds = np.std(error_all_trials, axis=0)
        plt.errorbar(std_vector, means, yerr=stds, marker='x', markerfacecolor='r', ecolor='r', elinewidth=0.5)
        plt.title('Mean and Standard Deviation of Absolute Errors vs. Standard Deviation')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Error')
        plt.show()


#Question 2 Functions
def tuning_curve(x, A, mean, std):
      return A * np.exp(-(x - mean) ** 2 / (2 * std ** 2))
tuning_curve = np.vectorize(tuning_curve)

def wta_decoder(pref_stim, resp):
    highest = np.argmax(resp)
    return pref_stim[highest]

def ols_error(x, responses, means, A=1, std=1):
    return np.sum((responses - tuning_curve(x,A,means,std))**2)

def MLE_decoder(response, means, A=1, std=1, stim_interval=np.arange(-5,5.01,0.01)):
    errors = []
    for stim in stim_interval:
        errors.append(ols_error(stim, response,means))
    idx = np.argmin(errors)
    return stim_interval[idx]

def map_error(x, responses, means, A=1, std=1):
    return np.sum((responses - tuning_curve(x,A,means,std))**2)/(2*(std/20)**2) + x**2/(2*2.5**2)

def MAP_decoder(response, means, A=1, std=1, stim_interval=np.arange(-5,5.01,0.01)):
    errors = []
    for stim in stim_interval:
        errors.append(map_error(stim, response, means))
    idx = np.argmin(errors)
    return stim_interval[idx]

ayhan_okuyan_21601531_hw4(question)



