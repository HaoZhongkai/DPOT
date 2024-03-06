#!/usr/bin/env python
#-*- coding:utf-8 _*-
import torch
import h5py
import matplotlib.pyplot as plt


# data_path = './../../data/large/ns2d_pdb_M1e-1_eta1e-2_zeta1e-2/test/data_0.hdf5'
data_path = '../../data/large/cfdbench/ns2d_cdb_test.hdf5'
x = h5py.File(data_path, "r")['data'][:,:,1:,:]

print('load finished')
import numpy as np
import matplotlib.pyplot as plt



def visualize_channels(data):
    channel_names = ['Flow Velocity X', 'Flow Velocity Y', 'mask', ]
    n_channels = data.shape[3]

    for c in range(n_channels):
        fig, axs = plt.subplots(4, 5, figsize=(20, 16))
        for i in range(4):
            for j in range(5):
                time_step_idx = i * 5 + j
                if time_step_idx < data.shape[2]:
                    im = axs[i, j].imshow(data[:, :, time_step_idx, c], cmap='viridis')
                    axs[i, j].set_title(f"Time Step: {time_step_idx + 1}")
                    axs[i, j].axis('off')
        fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal', pad=0.05)
        plt.suptitle(f"Channel: {channel_names[c]}")
        plt.show()


def visualize_histograms(data):
    channel_names = ['Flow Velocity X', 'Flow Velocity Y', 'Mask']
    n_channels = data.shape[3]
    for c in range(n_channels):
        plt.figure(figsize=(10, 6))
        plt.hist(data[:, :, :, c].flatten(), bins=50, color='blue', alpha=0.7)
        plt.title(f"Histogram for {channel_names[c]}")
        plt.xlabel(channel_names[c])
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()


def compute_statistics(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    return mean_val, std_val, min_val, max_val


def visualize_statistics(mean, std, min_val, max_val):
    labels = ["Mean", "Std. Dev.", "Min", "Max"]
    values = [mean, std, min_val, max_val]
    plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
    plt.title('Macro Statistics of Data')
    for i, v in enumerate(values):
        plt.text(i, v + 0.05, "{:.2f}".format(v), ha='center')
    plt.show()


# 可视化
visualize_channels(x)
visualize_histograms(x)

# 计算并可视化统计量
mean_val, std_val, min_val, max_val = compute_statistics(x)
visualize_statistics(mean_val, std_val, min_val, max_val)



