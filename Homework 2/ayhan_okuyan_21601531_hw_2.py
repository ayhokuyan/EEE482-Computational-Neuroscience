import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from mpl_toolkits.mplot3d import axes3d, Axes3D
from PIL import Image
from scipy import signal

question = sys.argv[1]

def ayhan_okuyan_21601531_hw2(question):
    if question == '1' :
        print('Question 1')

        data = io.loadmat('c2p3.mat')
        counts = data['counts']
        stim = data['stim']
        #print(counts.shape)
        #print(stim.shape)

        print('Part A')
        sta_images = np.zeros((10, 16, 16))

        for spk in range(len(counts) - 10):
            if counts[spk + 10] > 0:
                for i in range(10):
                    sta_images[i, :, :] += stim[:, :, spk + i] * counts[spk + 10]

        min_sta_val = np.min(sta_images)
        max_sta_val = np.max(sta_images)

        for i in range(sta_images.shape[0]):
            plt.imshow(sta_images[i, :, :], cmap='gray', vmin=min_sta_val, vmax=max_sta_val)
            plt.title('STA - %d steps before a spike' % (10 - (i)))
            plt.axis('off')
            plt.show()

        print('Part B')
        #print(sta_images.shape)
        sta_sum = np.sum(sta_images, axis=1).T
        plt.imshow(sta_sum, cmap='gray')
        plt.title('Row Summed STA Images')
        plt.xlabel('Time')
        plt.ylabel('Space')
        plt.show()

        sta_sum = np.sum(sta_images, axis=2).T
        plt.imshow(sta_sum, cmap='gray')
        plt.title('Column Summed STA Images')
        plt.xlabel('Time')
        plt.ylabel('Space')
        plt.show()

        print('Part C')
        projections = []

        for spk in range(len(counts) - 1):
            projections.append(np.sum(stim[:, :, spk] * sta_images[9, :, :]))
        projections /= np.max(projections)

        plt.hist(projections, bins=100)
        plt.title('Histogram of Normalized Projections')
        plt.show()

        nonzero_projections = []

        for spk in range(len(counts) - 1):
            if counts[spk + 1] > 0:
                nonzero_projections.append(np.sum(stim[:, :, spk] * sta_images[9, :, :]))
        nonzero_projections /= np.max(nonzero_projections)
        plt.hist(nonzero_projections, bins=100)
        plt.title('Histogram of Normalized Projections for Nonzero Spikes')
        plt.show()

        plt.hist([np.asarray(projections), np.asarray(nonzero_projections)], bins=55)
        plt.title('Bar Plot Comparisons for All and Nonzero Projections')
        plt.show()

    elif question == '2' :
        print('Question 2')

        print('Part A')
        receptive_field = dog_receptive_field(2, 4)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.linspace(-10, 10, 21)
        Y = X
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, receptive_field, cmap='Spectral', edgecolor='none')
        plt.title("Difference of Gaussians Receptive Field (Width=21)")
        plt.show(fig)

        plt.imshow(receptive_field, cmap='Spectral')
        plt.show()

        print('Part B')
        img = Image.open('hw2_image.bmp')
        img = np.asarray(img)[:, :, 0]
        #print(img.shape)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()

        out = signal.convolve2d(img, receptive_field, mode='same')
        plt.imshow(out, cmap='gray')
        plt.axis('off')
        plt.show()

        print('Part C')
        out_thr = np.copy(out)
        #print(np.min(out_thr), np.max(out_thr))
        threshold_value = -2
        out_thr[out_thr > threshold_value] = 1
        out_thr[out_thr <= threshold_value] = 0
        plt.imshow(out_thr, cmap='gray')
        plt.title('Edge Detection Thresholded at -2')
        plt.axis('off')
        plt.show()

        out_thr = np.copy(out)
        threshold_value = 0
        out_thr[out_thr > threshold_value] = 1
        out_thr[out_thr <= threshold_value] = 0
        plt.imshow(out_thr, cmap='gray')
        plt.title('Edge Detection Thresholded at 0')
        plt.axis('off')
        plt.show()

        out_thr = np.copy(out)
        threshold_value = 2
        out_thr[out_thr <= threshold_value] = 0
        out_thr[out_thr > threshold_value] = 1
        plt.imshow(out_thr, cmap='gray')
        plt.title('Edge Detection Thresholded at 2')
        plt.axis('off')
        plt.show()

        out_thr = np.copy(out)
        threshold_value = 5
        out_thr[out_thr <= threshold_value] = 0
        out_thr[out_thr > threshold_value] = 1
        plt.imshow(out_thr, cmap='gray')
        plt.title('Edge Detection Thresholded at 5')
        plt.axis('off')
        plt.show()

        print('Part D')
        theta = np.pi / 2
        sigma_l = sigma_w = 3
        lamda = 6
        phi = 0
        gabor_rec_field = gabor_receptive_field(sigma_l, sigma_w, theta, lamda, phi)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.linspace(-10, 10, 21)
        Y = X
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, gabor_rec_field, cmap='RdGy', edgecolor='none')
        plt.title("Gabor Receptive Field (Width=21,theta=pi/2)")
        plt.show(fig)

        plt.imshow(gabor_rec_field, cmap='RdGy')
        plt.show()

        print('Part E')
        out = signal.convolve2d(img, gabor_rec_field, mode='same')
        plt.imshow(out, cmap='gray')
        plt.axis('off')
        plt.show()

        out_thr = np.copy(out)
        #print(np.min(out_thr), np.max(out_thr))
        threshold_value = 0
        out_thr[out_thr > threshold_value] = 1
        out_thr[out_thr <= threshold_value] = 0
        plt.imshow(out_thr, cmap='gray')
        plt.title('Edge Detection Thresholded at -150')
        plt.axis('off')
        plt.show()

        print('Part F')
        theta_list = [0, np.pi / 6, np.pi / 3, np.pi / 2]
        theta_list_str = ['0', 'pi/6', 'pi/3', 'pi/2']
        sigma_l = sigma_w = 3
        lamda = 6
        phi = 0

        ind = 0
        out_f = np.zeros(out.shape)
        for el in theta_list:
            gabor_temp = gabor_receptive_field(sigma_l, sigma_w, el, lamda, phi)

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            X = np.linspace(-10, 10, 21)
            Y = X
            X, Y = np.meshgrid(X, Y)
            ax.plot_surface(X, Y, gabor_temp, cmap='RdGy', edgecolor='none')
            plt.title("Gabor Receptive Field, theta=%s" % theta_list_str[ind])
            ind += 1
            plt.show(fig)

            plt.imshow(gabor_temp, cmap='RdGy')
            plt.show()

            out_temp = signal.convolve2d(img, gabor_temp, mode='same')
            plt.imshow(out_temp, cmap='gray')
            plt.axis('off')
            plt.show()
            out_f += out_temp

        plt.imshow(out_f, cmap='gray')
        plt.axis('off')
        plt.show()

        out_ff = np.copy(out_f)
        #print(np.max(out_ff))
        #print(np.min(out_ff))
        thr_val = 0
        out_ff[out_ff <= thr_val] = 0
        out_ff[out_ff > thr_val] = 1
        plt.imshow(out_ff, cmap='gray')
        plt.axis('off')
        plt.show()

#Functions for Question 2
def dif_of_gauss(x, y, std_c, std_s):
    central_gaussian = (0.5/(np.pi*std_c**2))*np.exp(-(x**2+y**2)/(2*std_c**2))
    surround_gaussian = (0.5/(np.pi*std_s**2))*np.exp(-(x**2+y**2)/(2*std_s**2))
    return central_gaussian - surround_gaussian


def dog_receptive_field(std_c, std_s, width=21):
    receptive_field = np.zeros((width, width))
    min_edge = int(-(width - 1) / 2)
    max_edge = int((width - 1) / 2)
    x_ind = 0
    y_ind = 0
    for i in range(min_edge, max_edge, 1):
        for j in range(min_edge, max_edge, 1):
            receptive_field[x_ind, y_ind] = dif_of_gauss(i, j, std_c, std_s)
            x_ind += 1
        y_ind += 1
        x_ind = 0
    return receptive_field

def gabor(x, theta, sigma_l, sigma_w, lamda, phi):
    k_theta = np.asarray([np.sin(theta), np.cos(theta)])
    k_orth_theta = np.asarray([np.cos(theta), -np.sin(theta)])
    gaussian_part_l = -(np.dot(k_theta, x))**2/(2*sigma_l**2)
    gaussian_part_w = -(np.dot(k_orth_theta, x))**2/(2*sigma_w**2)
    orientation_part = np.cos((2*np.pi*np.dot(k_orth_theta,x)/lamda)+phi)
    return np.exp(gaussian_part_l+gaussian_part_w)*orientation_part


def gabor_receptive_field(sigma_l, sigma_w, theta, lamda, phi, width=21):
    receptive_field = np.zeros((width, width))
    min_edge = int(-(width - 1) / 2)
    max_edge = int((width - 1) / 2)
    x_ind = 0
    y_ind = 0
    for i in range(min_edge, max_edge, 1):
        for j in range(min_edge, max_edge, 1):
            receptive_field[x_ind, y_ind] = gabor(np.asarray([i, j]), theta, sigma_l, sigma_w, lamda, phi)
            x_ind += 1
        y_ind += 1
        x_ind = 0
    return receptive_field


ayhan_okuyan_21601531_hw2(question)



