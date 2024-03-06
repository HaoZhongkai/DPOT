#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.ndimage

#### FNO data
# path = '/root/files/pdessl/data/large/ns2d_1e-5_test.hdf5'
# path = '/root/files/pdessl/data/large/pdebench/ns2d_pdb_M1e-1_eta1e-2_zeta1e-2/test'
# path = '/root/files/pdessl/data/large/pdebench/ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_rand_512/test'
# path = '/root/files/pdessl/data/large/pdebench/dr_pdb/test'
# path = '/root/files/pdessl/data/large/pdebench/swe_pdb/test'
# path = '/root/files/pdessl/data/large/cfdbench/ns2d_cdb_test.hdf5'
# path = '/root/files/pdessl/data/large/pdearena/ns2d_pda/test'
path = '/home/zhongkai/files/ml4phys/mgn/pdessl/data/large/superbench/cosmo_2048/test_1/cosmo_test_1.hdf5'

data_id = 5

all_data = h5py.File(path, 'r')['data']
# all_data = all_data[data_id]
all_data = all_data[data_id][...,0]

# all_data = h5py.File(path+'/data_{}.hdf5'.format(data_id), 'r')['data'][...,0]
start_idx = 45
n_frames = 3
for i in range(n_frames):
    x = all_data[:,:,start_idx+i]
    if x.shape[0] < 128:
        x = scipy.ndimage.zoom(x, (128/x.shape[0], 128/x.shape[1]), order=3)

    # x = np.random.randn(128, 128)
    plt.figure()
    # plt.imshow(x,cmap='viridis')
    plt.imshow(x,cmap='plasma')
    # plt.imshow(x,cmap='inferno')
    # plt.imshow(x,cmap='magma')
    # plt.imshow(x,cmap='cividis')
    # plt.imshow(x,cmap='seismic')
    # plt.imshow(x,cmap='Spectral')
    # plt.imshow(x,cmap='hsv')
    # plt.imshow(x,cmap='twilight')
    # plt.imshow(x,cmap='rainbow')
    # plt.imshow(x,cmap='jet')
    # plt.imshow(x,cmap='terrain')
    plt.axis('off')
    # plt.tight_layout()
    plt.savefig('data_example_figure_{}_{}.png'.format(data_id, i),bbox_inches='tight',pad_inches=0)
    plt.show()

# print(x.shape)




