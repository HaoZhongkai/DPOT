#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import torch
import torch.nn.functional as F
import time
import numpy as np
import pickle
import os
import h5py
from functools import partial
from typing import Sequence
# from sklearn.preprocessing import QuantileTransformer


from utils.make_master_file import DATASET_DICT
from utils.normalizer import init_normalizer, UnitTransformer, PointWiseUnitTransformer, MinMaxTransformer, TorchQuantileTransformer, IdentityTransformer
from torch.utils.data import Dataset
from utils.make_master_file import DATASET_DICT
from utils.utilities import downsample, resize







class MixedTemporalDataset(Dataset):
    # _num_datasets = 0
    # _num_channels = 0
    def __init__(self, data_names, n_list = None, res = 128,t_in = 10, t_ar = 1, n_channels = None, normalize=False,train=True,data_weights=None):
        '''
        Dataset class for training pretraining multiple datasets
        :param data_names: names of datasets, specified in make_master_file.py
        :param n_list: num of training samples per dataset, should corresponds to the order of data_names
        :param res: input resolution for the model, 64/128/256/512/1024
        :param t_in: input timesteps, 10 for default
        :param t_ar: steps for auto-regressive pretraining, 1 for default
        :param n_channels: number of channels for dataset, if None, it auto reads max number of channels from config file, should be specified for test dataset
        :param normalize: if normalize data,  reversible instance normalization is implemented in each model
        :param train: if it is train dataset or (in distribution) test dataset
        '''
        # set global configs
        # if train:
        #     MixedTemporalDataset._num_datasets = len(data_names)
        #     MixedTemporalDataset._num_channels = max([DATASET_DICT[name]['n_channels'] for name in data_names])
        self.data_names = data_names if isinstance(data_names, list) else [data_names]
        self.data_weights = data_weights if data_weights is not None else [1] * len(self.data_names)
        self.num_datasets = len(data_names)
        self.t_in = t_in
        self.t_ar = t_ar
        self.train = train
        self.res = res
        self.n_sizes = n_list if n_list is not None else [DATASET_DICT[name]['train_size'] if train else DATASET_DICT[name]['test_size'] for name in self.data_names]
        self.weighted_sizes = [size * weight for size, weight in zip(self.n_sizes, self.data_weights)]
        # self.cumulative_sizes = np.cumsum(self.n_sizes)
        self.cumulative_sizes = np.cumsum(self.weighted_sizes)

        self.t_tests = [DATASET_DICT[name]['t_test'] for name in self.data_names]
        self.downsamples = [DATASET_DICT[name]['downsample'] for name in self.data_names]
        # self.n_channels = MixedTemporalDataset._num_channels
        self.n_channels = max([DATASET_DICT[name]['n_channels'] for name in self.data_names]) if n_channels is None else n_channels

        self.data_files = []
        for name in self.data_names:
            if DATASET_DICT[name]['scatter_storage']:
                def open_hdf5_file(path, idx):
                    return h5py.File(f'{path}/data_{idx}.hdf5', 'r')['data'][:]
                path = DATASET_DICT[name]['train_path'] if train else DATASET_DICT[name]['test_path']
                self.data_files.append(partial(open_hdf5_file, path))
                # if DATASET_DICT[name]['scatter_storage']:
                #     if train:
                #         self.data_files.append(lambda x, name=name:h5py.File(DATASET_DICT[name]['train_path'] + '/data_{}.hdf5'.format(x),'r')['data'])
                #     else:
                #         self.data_files.append(lambda x, name=name:h5py.File(DATASET_DICT[name]['test_path'] + '/data_{}.hdf5'.format(x),'r')['data'])
            else:
                self.data_files.append(h5py.File(DATASET_DICT[name]['train_path'] if train else DATASET_DICT[name]['test_path'], 'r'))
            # self.data_files = [h5py.File(DATASET_DICT[name]['train_path'] if train else DATASET_DICT[name]['test_path'], 'r') for name in self.data_names]


        self.normalize = normalize
        self.normalizers = []
        if normalize:
            print('Using normalizer for inputs')
            for data in self.data_files:
                self.normalizers.append(UnitTransformer(torch.from_numpy(data['data'][:500]).float()))    ### use 500 for normalization


    def pad_data(self, x):
        '''
        pad data to unified shape
        :param x: H, W, T, C
        :return:  H', W', T', C'
        '''
        H, W, T, C = x.shape
        x = x.view(H, W, -1).permute(2, 0, 1) # Cmax, H, W
        x = F.interpolate(x.unsqueeze(0), size=(self.res, self.res),mode='bilinear').squeeze(0).permute(1, 2, 0)
        x = x.view(*x.shape[:2], T, C)
        x_new = torch.ones([*x.shape[:-1], self.n_channels])
        x_new[..., :x.shape[-1]] = x  # H, W, T, Cmax

        return x_new

    def get_target_mask(self, x, size_orig):
        '''
        :param x: single data, H, W, T, C
        :param size_orig: original size of x
        :return: masks for evaluation (by resolution)
        '''
        msk = torch.zeros(*x.shape[:2], 1, x.shape[-1])    ## target mask shape H,W,1,C
        kx, ky = x.shape[0] // size_orig[0], x.shape[1] // size_orig[1]
        if kx ==0 or ky == 0:
            # print('warnings: target resolution < data resolution')
            kx = 1 if kx ==0 else kx
            ky = 1 if ky == 0 else ky
        msk[::kx, ::ky, :, :size_orig[-1]] = 1

        return msk

    def __len__(self):
        return self.cumulative_sizes[-1]




    def __getitem__(self, idx):
        '''
        Logic of getitem: first find which dataset idx is in, then reshape it to H,W,T,C,
            for training dataset, we random sample start timestep
            for test dataset, we return the whole trajectory
        :param idx: id in the whole dataset
        :return: data slice
        '''
        dataset_idx = int(np.searchsorted(self.cumulative_sizes, idx + 1))

        if dataset_idx == 0:
            data_idx = idx
        else:
            data_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        data_idx //= self.data_weights[dataset_idx]
        # t_0 = time.time()
        sample = torch.from_numpy(self.data_files[dataset_idx](data_idx)[:] if callable(self.data_files[dataset_idx]) else self.data_files[dataset_idx]['data'][data_idx][:]).float()
        # sample = torch.from_numpy(np.array(self.data_files[dataset_idx]['data'][data_idx],dtype=np.float32))
        if sample.ndim == 3:    ### augment channel dim
            sample = sample.unsqueeze(-1)

        # print(time.time() - t_0)
        orig_size = list(sample.shape)
        orig_size[-1] = DATASET_DICT[self.data_names[dataset_idx]]['pred_channels'] if 'pred_channels' in DATASET_DICT[self.data_names[dataset_idx]].keys() else orig_size[-1]
        sample = self.pad_data(sample)


        if self.train:  ## sample [0, t_in] and [t_in, t_in+ t_ar] for training ,trucated if too long
            start_idx = np.random.randint(max(sample.shape[-2] - (self.t_in + self.t_ar) + 1, 1))
            x, y = sample[..., start_idx: start_idx + self.t_in,:], sample[..., start_idx + self.t_in: min(start_idx + self.t_in + self.t_ar, sample.shape[-2]),:]
            # msk = msk[...,start_idx + self.t_in: min(start_idx + self.t_in + self.t_ar, sample.shape[-2]),:]
            msk = torch.ones([*x.shape[:2], 1, x.shape[-1]])
        else: ## test datasets returns full trajectory
            start_idx = 0
            x, y = sample[..., start_idx:start_idx + self.t_in,:], sample[..., self.t_in:self.t_in + self.t_tests[dataset_idx],:]
            # msk = msk[..., self.t_in:self.t_in + self.t_tests[dataset_idx],:]
            msk = self.get_target_mask(sample, orig_size)

        if self.normalize:
            # x = self.normalizers[int(dataset_idx)].transform(x, inverse=False)
            x = (x.unsqueeze(0) - self.normalizers[int(dataset_idx)].mean[..., start_idx: start_idx + self.t_in,:]) / (self.normalizers[int(dataset_idx)].std[..., start_idx: start_idx + self.t_in,:] + 1e-6)
            x = x.squeeze()

        ### downsample
        if self.downsamples[dataset_idx] != (1, 1):
            x, y = x[::self.downsamples[dataset_idx][0],::self.downsamples[dataset_idx][1]], y[::self.downsamples[dataset_idx][0],::self.downsamples[dataset_idx][1]]

        idx_cls = torch.LongTensor([dataset_idx])   #TODO(hzk): now return relative idx in given datasets, finally we need global idx
        return x, y, msk, idx_cls







class MixedMaskedDataset(Dataset):
    # _num_datasets = 0
    # _num_channels = 0
    def __init__(self, data_names, n_list = None, res = 128,t_in = 10, t_ar = 1, n_channels = None, normalize=False,train=True,data_weights=None):
        '''
        Dataset class for training pretraining multiple datasets
        :param data_names: names of datasets, specified in make_master_file.py
        :param n_list: num of training samples per dataset, should corresponds to the order of data_names
        :param res: input resolution for the model, 64/128/256/512/1024
        :param t_in: input timesteps, 10 for default
        :param t_ar: steps for auto-regressive pretraining, 1 for default
        :param n_channels: number of channels for dataset, if None, it auto reads max number of channels from config file, should be specified for test dataset
        :param normalize: if normalize data,  reversible instance normalization is implemented in each model
        :param train: if it is train dataset or (in distribution) test dataset
        '''
        # set global configs
        # if train:
        #     MixedTemporalDataset._num_datasets = len(data_names)
        #     MixedTemporalDataset._num_channels = max([DATASET_DICT[name]['n_channels'] for name in data_names])
        self.data_names = data_names if isinstance(data_names, list) else [data_names]
        self.data_weights = data_weights if data_weights is not None else [1] * len(self.data_names)
        self.num_datasets = len(data_names)
        self.t_in = t_in
        self.t_ar = t_ar
        self.train = train
        self.res = res
        self.n_sizes = n_list if n_list is not None else [DATASET_DICT[name]['train_size'] if train else DATASET_DICT[name]['test_size'] for name in self.data_names]
        self.weighted_sizes = [size * weight for size, weight in zip(self.n_sizes, self.data_weights)]
        # self.cumulative_sizes = np.cumsum(self.n_sizes)
        self.cumulative_sizes = np.cumsum(self.weighted_sizes)

        self.t_tests = [DATASET_DICT[name]['t_test'] for name in self.data_names]
        self.downsamples = [DATASET_DICT[name]['downsample'] for name in self.data_names]
        # self.n_channels = MixedTemporalDataset._num_channels
        self.n_channels = max([DATASET_DICT[name]['n_channels'] for name in self.data_names]) if n_channels is None else n_channels

        self.data_files = []
        for name in self.data_names:
            if DATASET_DICT[name]['scatter_storage']:
                def open_hdf5_file(path, idx):
                    return h5py.File(f'{path}/data_{idx}.hdf5', 'r')['data'][:]
                path = DATASET_DICT[name]['train_path'] if train else DATASET_DICT[name]['test_path']
                self.data_files.append(partial(open_hdf5_file, path))
                # if DATASET_DICT[name]['scatter_storage']:
                #     if train:
                #         self.data_files.append(lambda x, name=name:h5py.File(DATASET_DICT[name]['train_path'] + '/data_{}.hdf5'.format(x),'r')['data'])
                #     else:
                #         self.data_files.append(lambda x, name=name:h5py.File(DATASET_DICT[name]['test_path'] + '/data_{}.hdf5'.format(x),'r')['data'])
            else:
                self.data_files.append(h5py.File(DATASET_DICT[name]['train_path'] if train else DATASET_DICT[name]['test_path'], 'r'))
            # self.data_files = [h5py.File(DATASET_DICT[name]['train_path'] if train else DATASET_DICT[name]['test_path'], 'r') for name in self.data_names]


        self.normalize = normalize
        self.normalizers = []
        if normalize:
            print('Using normalizer for inputs')
            for data in self.data_files:
                self.normalizers.append(UnitTransformer(torch.from_numpy(data['data'][:500]).float()))    ### use 500 for normalization


    def pad_data(self, x):
        '''
        pad data to unified shape
        :param x: H, W, T, C
        :return:  H', W', T', C'
        '''
        H, W, T, C = x.shape
        x = x.view(H, W, -1).permute(2, 0, 1) # Cmax, H, W
        x = F.interpolate(x.unsqueeze(0), size=(self.res, self.res),mode='bilinear').squeeze(0).permute(1, 2, 0)
        x = x.view(*x.shape[:2], T, C)
        x_new = torch.ones([*x.shape[:-1], self.n_channels])    # use 1 for void padding
        x_new[..., :x.shape[-1]] = x  # H, W, T, Cmax

        return x_new

    def get_target_mask(self, x, size_orig):
        '''
        :param x: single data, H, W, T, C
        :param size_orig: original size of x
        :return: masks for evaluation (by resolution)
        '''
        msk = torch.zeros(*x.shape[:2], 1, x.shape[-1])    ## target mask shape H,W,1,C
        kx, ky = x.shape[0] // size_orig[0], x.shape[1] // size_orig[1]
        if kx ==0 or ky == 0:
            # print('warnings: target resolution < data resolution')
            kx = 1 if kx ==0 else kx
            ky = 1 if ky == 0 else ky
        msk[::kx, ::ky, :, :size_orig[-1]] = 1

        return msk

    def get_masked_input(self, x):
        '''
        :param x:  single data, H, W, T, C
        :param size_orig:  original size of x
        :return: masked input, TODO: downsampling resolution
        '''
        x_new = x.clone()
        x_new[:,:,-1,:] = -1
        return x_new


    def __len__(self):
        return self.cumulative_sizes[-1]




    def __getitem__(self, idx):
        '''
        Logic of getitem: first find which dataset idx is in, then reshape it to H,W,T,C,
            for training dataset, we random sample start timestep
            for test dataset, we return the whole trajectory
        :param idx: id in the whole dataset
        :return: data slice
        '''
        dataset_idx = int(np.searchsorted(self.cumulative_sizes, idx + 1))

        if dataset_idx == 0:
            data_idx = idx
        else:
            data_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        data_idx //= self.data_weights[dataset_idx]
        # t_0 = time.time()
        sample = torch.from_numpy(self.data_files[dataset_idx](data_idx)[:] if callable(self.data_files[dataset_idx]) else self.data_files[dataset_idx]['data'][data_idx][:]).float()
        # sample = torch.from_numpy(np.array(self.data_files[dataset_idx]['data'][data_idx],dtype=np.float32))
        if sample.ndim == 3:    ### augment channel dim
            sample = sample.unsqueeze(-1)

        # print(time.time() - t_0)
        orig_size = list(sample.shape)
        sample = self.pad_data(sample)


        if self.train:  ## sample [0, t_in] and [t_in, t_in+ t_ar] for training ,trucated if too long
            start_idx = np.random.randint(max(sample.shape[-2] - self.t_in + 1, 1))
            x = sample[..., start_idx: start_idx + self.t_in,:]
            # msk = msk[...,start_idx + self.t_in: min(start_idx + self.t_in + self.t_ar, sample.shape[-2]),:]
            x_msk = self.get_masked_input(x)
            # x_msk = x


            target_msk = torch.ones([*x.shape[:2], 1, x.shape[-1]])
        else: ## test datasets returns full trajectory
            x_msk, x = sample[...,:self.t_in,:], sample[..., self.t_in-1:self.t_in + self.t_tests[dataset_idx],:]
            target_msk = self.get_target_mask(sample, orig_size)
            x_msk = self.get_masked_input(x_msk)
        ### downsample
        if self.downsamples[dataset_idx] != (1, 1):
            x_msk, x = x_msk[::self.downsamples[dataset_idx][0],::self.downsamples[dataset_idx][1]], x[::self.downsamples[dataset_idx][0],::self.downsamples[dataset_idx][1]]

        idx_cls = torch.LongTensor([dataset_idx])   #TODO(hzk): now return relative idx in given datasets, finally we need global idx
        return x_msk, x, target_msk, idx_cls



class SteadyDataset2D(Dataset):
    def __init__(self, data_name, n_train=None, res=128, n_channels = None, normalize=False, train=True):
        '''
        :param data_name:
        :param n_train:
        :param res:
        :param t_in:
        :param t_ar:
        :param n_channels:
        :param normalize:
        :param train:
        '''
        self.data_name = data_name
        self.n_size = n_train if n_train is not None else DATASET_DICT[data_name]['train_size'] if train else DATASET_DICT[data_name]['test_size']
        self.train = train
        self.res = res
        self.n_channels = DATASET_DICT[data_name]['n_channels'] if n_channels is None else n_channels
        self.downsample = DATASET_DICT[data_name]['downsample']



        if DATASET_DICT[self.data_name]['scatter_storage']:
            def open_hdf5_file(path, idx, name):
                return h5py.File(f'{path}/data_{idx}.hdf5', 'r')[name][:]

            path = DATASET_DICT[self.data_name]['train_path'] if train else DATASET_DICT[self.data_name]['test_path']
            self.data_files = partial(open_hdf5_file, path)
        else:
            self.data_files = h5py.File(DATASET_DICT[self.data_name]['train_path'] if train else DATASET_DICT[self.data_name]['test_path'], 'r')

    def pad_data(self, x):
        '''
        pad data to unified shape
        :param x: H, W, T, C
        :return:  H', W', T', C'
        '''
        H, W, C = x.shape
        x = x.view(H, W, -1).permute(2, 0, 1)  # Cmax, H, W, L
        x = F.interpolate(x.unsqueeze(0), size=(self.res, self.res), mode='bilinear').squeeze(0).permute(1, 2, 0).unsqueeze(-2)
        # x = resize(x, [self.res, self.res])
        x_new = torch.ones([*x.shape[:-1], self.n_channels])
        x_new[..., :x.shape[-1]] = x  # H, W, T, Cmax

        return x_new


    def shuffle_channels(self, x, y):
        idx1, idx2 = torch.randperm(x.shape[-1])[:2]
        x[..., [idx1, idx2]] = x[..., [idx2, idx1]]
        y[...,[idx1, idx2]] = y[..., [idx2, idx1]]
        return x, y


    def get_target_mask(self, x, size_orig):
        '''
        :param x: single data, H, W, T, C
        :param size_orig: original size of x
        :return: masks for evaluation (by resolution)
        '''
        msk = torch.zeros(*x.shape[:2], 1, x.shape[-1])  ## target mask shape H,W,1,C
        kx, ky = x.shape[0] // size_orig[0], x.shape[1] // size_orig[1]
        if kx == 0 or ky == 0:
            # print('warnings: target resolution < data resolution')
            kx = 1 if kx == 0 else kx
            ky = 1 if ky == 0 else ky
        msk[::kx, ::ky, :, :size_orig[-1]] = 1

        return msk


    def __getitem__(self, idx):
        '''
        Logic of getitem:  reshape data to H,W,L,T,C,
            for training dataset, we random sample start timestep
            for test dataset, we return the whole trajectory
        :param idx: id in the whole dataset
        :return: data slice
        '''
        # t_0 = time.time()
        sample_x = torch.from_numpy(self.data_files(idx,name='x')[:] if callable(self.data_files) else self.data_files['x'][idx]).float()
        sample_y = torch.from_numpy(self.data_files(idx,name='y')[:] if callable(self.data_files) else self.data_files['y'][idx]).float()

        # sample = torch.from_numpy(np.array(self.data_files[dataset_idx]['data'][data_idx],dtype=np.float32))
        if sample_x.ndim == 2:    ### augment channel dim
            sample_x = sample_x.unsqueeze(-1)
            sample_y = sample_y.unsqueeze(-1)


        # sample_x, sample_y = self.shuffle_channels(sample_x, sample_y)

        # print(time.time() - t_0)
        orig_size = list(sample_x.shape)
        orig_size[-1] = DATASET_DICT[self.data_name]['pred_channels'] if 'pred_channels' in DATASET_DICT[self.data_name].keys() else orig_size[-1]
        x, y = self.pad_data(sample_x), self.pad_data(sample_y)


        if self.train:  ## sample [0, t_in] and [t_in, t_in+ t_ar] for training ,trucated if too long
            msk = torch.ones([*x.shape[:2], 1, x.shape[-1]])
        else: ## test datasets returns full trajectory
            msk = self.get_target_mask(x, orig_size)


        ### downsample
        if self.downsample != (1, 1, 1):
            x, y = x[::self.downsample[0],::self.downsample[1]], y[::self.downsample[0],::self.downsample[1]]

        # idx_cls = torch.LongTensor([dataset_idx])
        return x, y, msk

    def __len__(self):
        return self.n_size



class TemporalDataset3D(Dataset):
    def __init__(self, data_name, n_train=None, res=128, t_in=10, t_ar = 1, n_channels = None, normalize=False, train=True):
        '''

        :param data_name:
        :param n_train:
        :param res:
        :param t_in:
        :param t_ar:
        :param n_channels:
        :param normalize:
        :param train:
        '''
        self.data_name = data_name
        self.n_size = n_train if n_train is not None else DATASET_DICT[data_name]['train_size'] if train else DATASET_DICT[data_name]['test_size']
        self.train = train
        self.res = res
        self.t_in = t_in
        self.t_ar = t_ar
        self.t_test = DATASET_DICT[data_name]['t_test']
        self.n_channels = DATASET_DICT[data_name]['n_channels'] if n_channels is None else n_channels
        self.downsample = DATASET_DICT[data_name]['downsample']



        if DATASET_DICT[self.data_name]['scatter_storage']:
            def open_hdf5_file(path, idx):
                return h5py.File(f'{path}/data_{idx}.hdf5', 'r')['data'][:]

            path = DATASET_DICT[self.data_name]['train_path'] if train else DATASET_DICT[self.data_name]['test_path']
            self.data_files = partial(open_hdf5_file, path)
        else:
            self.data_files = h5py.File(DATASET_DICT[self.data_name]['train_path'] if train else DATASET_DICT[self.data_name]['test_path'], 'r')

    def pad_data(self, x):
        '''
        pad data to unified shape
        :param x: H, W, T, C
        :return:  H', W', T', C'
        '''
        H, W, L, T, C = x.shape
        x = x.view(H, W, L, -1).permute(3, 0, 1, 2)  # Cmax, H, W, L
        x = F.interpolate(x.unsqueeze(0), size=(self.res, self.res, self.res), mode='trilinear').squeeze(0).permute(1, 2, 3, 0)
        x = x.view(*x.shape[:3], T, C)
        x_new = torch.ones([*x.shape[:-1], self.n_channels])
        x_new[..., :x.shape[-1]] = x  # H, W, T, Cmax

        return x_new

    def get_target_mask(self, x, size_orig):
        '''
        :param x: single data, H, W, T, C
        :param size_orig: original size of x
        :return: masks for evaluation (by resolution)
        '''
        msk = torch.zeros(*x.shape[:3], 1, x.shape[-1])  ## target mask shape H,W,1,C
        kx, ky, kz = x.shape[0] // size_orig[0], x.shape[1] // size_orig[1], x.shape[2] // size_orig[2]
        if kx == 0 or ky == 0 or kz == 0:
            # print('warnings: target resolution < data resolution')
            kx = 1 if kx == 0 else kx
            ky = 1 if ky == 0 else ky
            kz = 1 if kz == 0 else kz
        msk[::kx, ::ky, ::kz, :, :size_orig[-1]] = 1

        return msk


    def __getitem__(self, idx):
        '''
        Logic of getitem:  reshape data to H,W,L,T,C,
            for training dataset, we random sample start timestep
            for test dataset, we return the whole trajectory
        :param idx: id in the whole dataset
        :return: data slice
        '''



        # t_0 = time.time()
        sample = torch.from_numpy(self.data_files(idx)[:] if callable(self.data_files) else self.data_files['data'][idx][:]).float()
        # sample = torch.from_numpy(np.array(self.data_files[dataset_idx]['data'][data_idx],dtype=np.float32))
        if sample.ndim == 4:    ### augment channel dim
            sample = sample.unsqueeze(-1)

        # print(time.time() - t_0)
        orig_size = list(sample.shape)
        orig_size[-1] = DATASET_DICT[self.data_name]['pred_channels'] if 'pred_channels' in DATASET_DICT[self.data_name].keys() else orig_size[-1]
        sample = self.pad_data(sample)


        if self.train:  ## sample [0, t_in] and [t_in, t_in+ t_ar] for training ,trucated if too long
            start_idx = np.random.randint(max(sample.shape[-2] - (self.t_in + self.t_ar) + 1, 1))
            x, y = sample[..., start_idx: start_idx + self.t_in,:], sample[..., start_idx + self.t_in: min(start_idx + self.t_in + self.t_ar, sample.shape[-2]),:]
            # msk = msk[...,start_idx + self.t_in: min(start_idx + self.t_in + self.t_ar, sample.shape[-2]),:]
            msk = torch.ones([*x.shape[:3], 1, x.shape[-1]])
        else: ## test datasets returns full trajectory
            start_idx = 0
            x, y = sample[..., start_idx:start_idx + self.t_in,:], sample[..., self.t_in:self.t_in + self.t_test,:]
            # msk = msk[..., self.t_in:self.t_in + self.t_tests[dataset_idx],:]
            msk = self.get_target_mask(sample, orig_size)


        ### downsample
        if self.downsample != (1, 1, 1):
            x, y = x[::self.downsample[0],::self.downsample[1],::self.downsample[2]], y[::self.downsample[0],::self.downsample[1],::self.downsample[2]]

        # idx_cls = torch.LongTensor([dataset_idx])
        return x, y, msk

    def __len__(self):
        return self.n_size

#
# def load_dataset(path):
#     '''
#     Auxiliary function for reading dataset
#     :param path:
#     :return:
#     '''
#     if path.endswith('.pkl'):
#         data = pickle.load(open(path, 'rb'))
#     elif path.endswith('.npy') or path.endswith('.npz'):
#         fp = np.load(path)
#         x = fp['x']
#         y = fp['y']
#         theta = None if fp['theta'].ndim == 0 else fp['theta']
#         data = {'x': x, 'y': y, 'theta': theta}
#     elif path.endswith('.hdf5'):
#         with h5py.File(path, 'r') as fp:
#             x = np.array(fp['x'],dtype=np.float32)
#             y = np.array(fp['y'],dtype=np.float32)
#             theta = None if fp['theta'].ndim == 0 else np.array(fp['theta'],dtype=np.float32)
#             data = {'x': x, 'y': y, 'theta': theta}
#     else:
#         raise ValueError
#     return data


# class GridDataset(Dataset):
#     def __init__(self, name, data=None, data_index=None, downsample_x=(0,), downsample_y=(0,)):
#         super(GridDataset, self).__init__()
#
#         if name not in DATASET_DICT.keys():
#             raise NotImplementedError
#
#         self.meta_info = DATASET_DICT[name]
#         self.downsample_x = downsample_x if downsample_x[0] else self.meta_info['default_downsample_x']
#         self.downsample_y = downsample_y if downsample_y[0] else self.meta_info['default_downsample_y']
#         self.scattered_storage =  ('scatter_stored' in self.meta_info.keys()) and self.meta_info['scatter_stored']
#         self.enable_grid = False
#
#         ### process dataset, initialize attributes
#         if self.scattered_storage:
#             self.data_index = list(data_index)
#             self.path_str = self.meta_info['path']
#
#             ### get shapes
#             x0, y0, theta0 = self.__getitem__(0)
#
#             self.gridsize_x = x0.shape[:-2] if self.meta_info['temporal'] else x0.shape[:-1]
#             self.gridsize_y = y0.shape[:-2] if self.meta_info['temporal'] else y0.shape[:-1]
#
#
#
#         else:
#             if data is None:
#                 data = load_dataset(self.meta_info['path'])
#
#             self.x, self.y = torch.from_numpy(data['x']), torch.from_numpy(data['y'])
#             self.theta = None if data['theta'] == None else torch.from_numpy(data['theta'])
#
#             #### downsample
#             self.x = self.__downsample(self.downsample_x, attr_name='x')
#             self.y = self.__downsample(self.downsample_y, attr_name='y')
#
#             self.gridsize_x = self.x.shape[1:-2] if self.meta_info['temporal'] else self.x.shape[1:-1]
#             self.gridsize_y = self.y.shape[1:-2] if self.meta_info['temporal'] else self.y.shape[1:-1]
#
#
#
#
#
#     def __len__(self):
#         if self.scattered_storage:
#             return len(self.data_index)
#         else:
#             return self.x.shape[0]
#
#     def __getitem__(self, idx):
#         if self.scattered_storage:
#             data = np.load(os.path.join(self.path_str,'data_{}.npz'.format(self.data_index[idx])))
#             x, y = torch.from_numpy(data['x']).unsqueeze(0), torch.from_numpy(data['y']).unsqueeze(0)
#             if hasattr(self, 'x_normalizer'):
#                 x, y = self.x_normalizer.transform(x, inverse=False), self.y_normalizer.transform(y, inverse=False)
#             x, y = self.__downsample(self.downsample_x, data=x), self.__downsample(self.downsample_y, data=y)
#             if self.enable_grid:
#                 x = self.auto_load_grid(data=x)
#             if self.meta_info['theta_dim'] == 0:
#                 theta = torch.zeros([])
#             else:
#                 theta = self.theta_normalizer.transform(torch.from_numpy(data['theta']).unsqueeze(0),inverse=False).squeeze(0)
#             return x.squeeze(0), y.squeeze(0), theta
#         else:
#             if self.theta is None:
#                 return self.x[idx], self.y[idx], torch.zeros([])
#             else:
#                 return self.x[idx], self.y[idx], self.theta[idx]
#
#
#
#
#     #### downscale dataset, support up to 4 dim, must pass either attr_name or data
#     def __downsample(self, downsample, data=None, attr_name=None):
#         if data is None:
#             if attr_name is not None:
#                 data = getattr(self, attr_name)
#             else:
#                 raise ValueError
#         downsample = downsample * self.meta_info['space_dim'] if isinstance(downsample, list) and len(downsample)==1 else downsample
#         if self.meta_info['space_dim'] == 1:
#             if isinstance(downsample, int):
#                 data = data[:,::downsample]
#             else:
#                 data = data[:,::downsample[0]]
#         elif self.meta_info['space_dim'] == 2:
#             if isinstance(downsample, int):
#                 data = data[:,::downsample, ::downsample]
#             else:
#                 data = data[:,::downsample[0],::downsample[1]]
#         elif self.meta_info['space_dim'] == 3:
#             if isinstance(downsample, int):
#                 data = data[:, ::downsample, ::downsample,:: downsample]
#             else:
#                 data = data[:, ::downsample[0], ::downsample[1], ::downsample[2]]
#         elif self.meta_info['space_dim'] == 4:
#             if isinstance(downsample, int):
#                 data = data[:, ::downsample, ::downsample, :: downsample, :: downsample]
#             else:
#                 data = data[:, ::downsample[0], ::downsample[1], ::downsample[2], ::downsample[3]]
#         else:
#             raise ValueError
#
#
#         if attr_name=='x':
#             self.x = data
#         elif attr_name == 'y':
#             self.y = data
#
#         return data
#
#     def get_normalizer(self, type):
#
#         # restore from file
#         if self.scattered_storage:
#             normalizer_data = np.load(os.path.join(self.path_str, 'normalizer_data.npz'))
#             if type == 'unit':
#                 x1, x2, y1, y2, t1, t2 = normalizer_data['unit_mean_x'], normalizer_data['unit_std_x'], normalizer_data['unit_mean_y'], normalizer_data['unit_std_y'], normalizer_data['unit_mean_theta'], normalizer_data['unit_std_theta']
#             elif type == 'pointunit':
#                 x1, x2, y1, y2, t1, t2 = normalizer_data['pointunit_mean_x'], normalizer_data['pointunit_std_x'], normalizer_data['pointunit_mean_y'], normalizer_data['pointunit_std_y'], normalizer_data['pointunit_mean_theta'], normalizer_data['pointunit_std_theta']
#             elif type == 'minmax':
#                 x1, x2, y1, y2, t1, t2 = normalizer_data['minmax_min_x'], normalizer_data['minmax_max_x'], normalizer_data['minmax_min_y'], normalizer_data['minmax_max_y'], normalizer_data['minmax_min_theta'], normalizer_data['minmax_max_theta']
#             else:
#                 x1, x2, y1, y2, t1, t2 = None, None, None, None, None, None
#             self.x_normalizer, self.y_normalizer = init_normalizer(type, x1, x2, eps=1e-7), init_normalizer(type, y1, y2, eps=1e-7)
#             self.theta_normalizer = init_normalizer(type, t1, t2, eps=1e-7) if self.meta_info['theta_dim'] else None
#
#
#         else:
#             if type in ['unit', 'pointunit','minmax','none']:
#                 if type == 'unit':
#                     normalizer = UnitTransformer
#                 elif type == 'pointunit':
#                     normalizer = partial(PointWiseUnitTransformer, temporal=self.meta_info['temporal'])
#                 elif type == 'minmax':
#                     normalizer = MinMaxTransformer
#                 else:
#                     normalizer = IdentityTransformer
#
#
#                 self.x_normalizer = normalizer(self.x, eps=1e-7)
#                 self.y_normalizer = normalizer(self.y, eps=1e-7)
#                 self.theta_normalizer = None if self.theta is None else normalizer(self.theta, eps=1e-7)
#             # elif type == 'quantile':
#             #
#             #     x_normalizer_numpy = QuantileTransformer(output_distribution='normal')
#             #     x_normalizer_numpy = x_normalizer_numpy.fit(self.x.reshape(-1, self.x.shape[-1]))
#             #     x_normalizer = TorchQuantileTransformer(x_normalizer_numpy.output_distribution, x_normalizer_numpy.references_, x_normalizer_numpy.quantiles_)
#             #
#             #     y_normalizer_numpy = QuantileTransformer(output_distribution='normal')
#             #     y_normalizer_numpy = y_normalizer_numpy.fit(self.y.reshape(-1, self.x.shape[-1]))
#             #     y_normalizer = TorchQuantileTransformer(y_normalizer_numpy.output_distribution, y_normalizer_numpy.references_, y_normalizer_numpy.quantiles_)
#             #
#             #     if self.theta is not None:
#             #         theta_normalizer_numpy = QuantileTransformer(output_distribution='normal')
#             #         theta_normalizer_numpy = theta_normalizer_numpy.fit(self.theta.reshape(-1, self.theta.shape[-1]))
#             #         theta_normalizer = TorchQuantileTransformer(theta_normalizer_numpy.output_distribution, theta_normalizer_numpy.references_, theta_normalizer_numpy.quantiles_)
#             #     else:
#             #         theta_normalizer = None
#             else:
#                 raise NotImplementedError
#
#         return self.x_normalizer, self.y_normalizer, self.theta_normalizer
#
#     def apply_normalizer(self, x_normalizer=None, y_normalizer=None, theta_normalizer=None):
#         if x_normalizer is not None:
#             self.x_normalizer = x_normalizer
#             if not self.scattered_storage:
#                 self.x = x_normalizer.transform(self.x, inverse=False)
#         if y_normalizer is not None:
#             self.y_normalizer = y_normalizer
#             if not self.scattered_storage:
#                 self.y = y_normalizer.transform(self.y, inverse=False)
#         if theta_normalizer is not None:
#             self.theta_normalizer = theta_normalizer
#             if not self.scattered_storage:
#                 self.theta = theta_normalizer.transform(self.theta, inverse=False)
#         return
#
#     @staticmethod
#     def get_splits(meta_info):
#         all_ids = list(range(meta_info['size']))
#         train_num, valid_num, test_num = meta_info['split']
#         return all_ids[:train_num],  all_ids[train_num+test_num:], all_ids[train_num:train_num+test_num]
#
#     ###
#     ### assume datatype torch, assert grid before x
#     def auto_load_grid(self, data=None):
#         if data is None:
#             if self.scattered_storage:
#                 self.enable_grid = True
#                 return
#             else:
#                 set_globally = True
#                 data = self.x
#         else:
#             set_globally = False
#         space_dim = self.meta_info['space_dim']
#         if space_dim == 1:
#             grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]))
#             grid = torch.unsqueeze(grid[0], dim=-1)
#         elif space_dim == 2:
#             grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]), torch.linspace(0, 1, data.shape[2]))
#             grid = torch.stack(grid, dim=-1)
#         elif space_dim == 3:
#             grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]), torch.linspace(0, 1, data.shape[2]),torch.linspace(0,1, data.shape[3]))
#             grid = torch.stack(grid, dim=-1)
#         elif space_dim == 4:
#             grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]), torch.linspace(0, 1, data.shape[2]),torch.linspace(0,1, data.shape[3]),torch.linspace(0,1, data.shape[4]))
#             grid = torch.stack(grid, dim=-1)
#         else:
#             raise ValueError('dim should be 1, 2, 3 or 4.')
#         if self.meta_info['temporal']:
#             grid = grid.unsqueeze(-2)
#             data = torch.cat([torch.tile(grid.unsqueeze(0),[data.shape[0]] + [1] * space_dim + [data.shape[-2], 1]), data],dim=-1)
#         else:
#             data = torch.cat([torch.tile(grid.unsqueeze(0),[data.shape[0]] + [1] * grid.ndim), data],dim=-1)
#         if set_globally:
#             self.x = data
#         return data
#




# class GridSubDataset(GridDataset):
#     r"""
#     Subset of a dataset at specified indices.
#
#     Args:
#         dataset (Dataset): The whole Dataset
#         indices (sequence): Indices in the whole set selected for subset
#     """
#
#     def __init__(self, dataset: GridDataset, indices: Sequence):
#         self.dataset = dataset
#         self.indices = indices
#
#         ### set status variables
#         self.meta_info = self.dataset.meta_info
#         self.downsample_x = self.dataset.downsample_x
#         self.downsample_y = self.dataset.downsample_y
#         self.gridsize_x = self.dataset.gridsize_x
#         self.gridsize_y = self.dataset.gridsize_y
#
#         self.x = self.dataset.x[self.indices]
#         self.y = self.dataset.y[self.indices]
#         self.theta = self.dataset.theta[self.indices] if self.dataset.theta is not None else None
#
#
#
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx], self.theta[idx] if self.theta is not None else torch.zeros([])
#
#     def __len__(self):
#         return len(self.indices)