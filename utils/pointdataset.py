#!/usr/bin/env python
#-*- coding:utf-8 _*-
import torch
import numpy as np
import pickle
import os
import h5py
from functools import partial
from typing import Sequence
from sklearn.preprocessing import QuantileTransformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


from opb.utils.utilities import MultipleTensors
from opb.utils.make_master_file import DATASET_DICT
from opb.utils.normalizer import init_normalizer, UnitTransformer, PointWiseUnitTransformer, MinMaxTransformer, TorchQuantileTransformer, IdentityTransformer, cal_normalizer_efficient




def load_dataset(path):
    # path = os.path.join('..', path)
    if path.endswith('.pkl'):
        data_list = pickle.load(open(path, 'rb'))
    elif path.endswith('.hdf5'):
        with h5py.File(path, 'r') as fp:
            data_list = []
            for key, item in fp.items():
                x = np.array(item['x'],dtype=np.float32)
                y = np.array(item['y'],dtype=np.float32)
                theta = None if item['theta'].ndim == 0 else np.array(item['theta'], dtype=np.float32)
                fn = []
                if 'fn' in item:
                    for fn_ in item['fn']:
                        fn.append(np.array(fn_, dtype=np.float32))
                else:
                    fn = None
                data_list.append({'x': x, 'y': y, 'theta': theta, 'fn': fn})


    else:
        raise ValueError
    return data_list


def load_single_data(path):
    if path.endswith('.pkl'):
        data = pickle.load(open(path,'rb'))
    elif path.endswith('.hdf5'):
        data = {}
        with h5py.File(path, 'r') as fp:
            data['x'] = np.array(fp['x'],dtype=np.float32)
            data['y'] = np.array(fp['y'],dtype=np.float32)
            data['theta'] = None if fp['theta'].ndim == 0 else np.array(fp['theta'],dtype=np.float32)
            data['fn'] = []
            if 'fn' in data:
                for item in data['fn'].items():
                    data['fn'].append(np.array(item,dtype=np.float32))
            else:
                data['fn'] = None

        return data







def collate_op(items):
    transposed = zip(*items)
    batched = []
    for sample in transposed:
        if isinstance(sample[0], torch.Tensor):
            batched.append(torch.stack(sample))
        elif isinstance(sample[0], MultipleTensors):
            sample_ = MultipleTensors([pad_sequence([sample[i][j] for i in range(len(sample))]).permute(1,0,2) for j in range(len(sample[0]))])
            batched.append(sample_)
        else:
            raise NotImplementedError
    return batched



class PointDataset(Dataset):
    def __init__(self, name, data_list=None, data_index=None, downsample_x=(0,), train=True, max_nodes=-1):
        super(PointDataset, self).__init__()

        if name not in DATASET_DICT.keys():
            raise NotImplementedError

        self.meta_info = DATASET_DICT[name]
        self.downsample_x = downsample_x if downsample_x[0] else self.meta_info['default_downsample_x']
        self.scattered_storage = ('scatter_stored' in self.meta_info.keys()) and self.meta_info['scatter_stored']
        self.train = train
        self.max_sample_nodes = max_nodes

        ### process dataset, initialize attributes
        if self.scattered_storage:
            self.data_index = list(data_index)
            self.path_str = self.meta_info['path']

            ### get shapes
            x0, y0, theta0, fn0 = self.__getitem__(0)

            self.fn_shape = [x.shape[-1] for x in fn0] if isinstance(fn0, MultipleTensors) else 0
            self.num_fns = 0 if self.fn_shape ==0  else len(self.fn_shape)



        else:
            if data_list is None:
                self.data_list = load_dataset(self.meta_info['path']) ## list using hdf5 groups or python list
            else:
                self.data_list = data_list
            self.data = []
            self.theta = [] if self.meta_info['theta_dim'] else None
            for i in range(len(self.data_list)):
                self.data_list[i]['x'] = torch.from_numpy(self.data_list[i]['x'])   # N, C or N, T, C
                self.data_list[i]['y'] = torch.from_numpy(self.data_list[i]['y'])
                if self.meta_info['theta_dim']:
                    self.theta.append(self.data_list[i]['theta'])
                if self.data_list[i]['fn'] is not None:
                    self.data_list[i]['fn'] = MultipleTensors(self.data_list[i]['fn'])
                    self.fn_shape = [x.shape[-1] for x in self.data_list[i]['fn']]
                else:
                    self.fn_shape = 0

            self.num_fns = 0 if self.fn_shape == 0 else len(self.fn_shape)

            #### downsample x, y and fn (if available), note that x, y must be simultanously downsampled
            self.__downsample(self.downsample_x)



    def __len__(self):
        if self.scattered_storage:
            return len(self.data_index)
        else:
            return len(self.data_list)

    def __getitem__(self, idx):
        if self.scattered_storage:
            data = load_single_data(os.path.join(self.path_str,'data_{}.hdf5'.format(self.data_index[idx])))
            x, y, fn = torch.from_numpy(data['x']), torch.from_numpy(data['y']), MultipleTensors([])
            if hasattr(self, 'x_normalizer'):
                x, y = self.x_normalizer.transform(x, inverse=False), self.y_normalizer.transform(y, inverse=False)
            x, y = self.__downsample(self.downsample_x, data=x), self.__downsample(self.downsample_y, data=y)
            if self.meta_info['theta_dim'] == 0:
                theta = torch.zeros([])
            else:
                theta = self.theta_normalizer.transform(torch.from_numpy(data['theta']).unsqueeze(0),inverse=False).squeeze(0)
        else:
            x, y = self.data_list[idx]['x'], self.data_list[idx]['y']
            theta = self.theta[idx] if self.theta is not None else torch.zeros([])
            fn = self.data_list[idx]['fn'] if self.num_fns else torch.zeros([])

        ### set maximum number of nodes for training settings
        if self.train and self.max_sample_nodes > 0:
            n_ids = torch.randperm(x.shape[0])[:self.max_sample_nodes]
            x, y = x[n_ids], y[n_ids]
            for j in range(self.num_fns):
                n_ids_j = torch.randperm(fn[j].shape[0])[:self.max_sample_nodes]
                fn[j] = fn[j][n_ids_j]

        num_nodes = torch.LongTensor([x.shape[0]])

        return x, y, theta, fn, num_nodes




    #### downscale dataset, support up to 4 dim, must pass either attr_name or data
    def __downsample(self, downsample, data=None):
        downsample = downsample * self.meta_info['space_dim'] if isinstance(downsample, list) and len(downsample)==1 else downsample
        downsample = [downsample] if isinstance(downsample, int) else downsample
        assert len(downsample) <= (self.num_fns + 1)

        if data is None:
            for i in range(len(self)):
                self.data_list[i]['x'] = self.data_list[i]['x'][::downsample[0]]
                self.data_list[i]['y'] = self.data_list[i]['y'][::downsample[0]]
                for j in range(min(self.num_fns, len(downsample)-1)):
                    self.data_list[i]['fn'][j] = self.data_list[i]['fn'][j][::downsample[j + 1]]
            return
        else:
            data['x'] = data['x'][::downsample[0]]
            data['y'] = data['y'][::downsample[0]]

            for j in range(min(self.num_fns, len(downsample) - 1)):
                data['fn'][j] = data['fn'][j][::downsample[j + 1]]
            return data

    def get_normalizer(self, type):

        # restore from file
        if self.scattered_storage:
            normalizer_data = np.load(os.path.join(self.path_str, 'normalizer_data.npz'))
            if type == 'unit':
                x1, x2, y1, y2, t1, t2 = normalizer_data['unit_mean_x'], normalizer_data['unit_std_x'], normalizer_data['unit_mean_y'], normalizer_data['unit_std_y'], normalizer_data['unit_mean_theta'], normalizer_data['unit_std_theta']
            elif type == 'minmax':
                x1, x2, y1, y2, t1, t2 = normalizer_data['minmax_min_x'], normalizer_data['minmax_max_x'], normalizer_data['minmax_min_y'], normalizer_data['minmax_max_y'], normalizer_data['minmax_min_theta'], normalizer_data['minmax_max_theta']
            else:
                x1, x2, y1, y2, t1, t2 = None, None, None, None, None, None
            self.x_normalizer, self.y_normalizer = init_normalizer(type, x1, x2, eps=1e-7), init_normalizer(type, y1, y2, eps=1e-7)
            self.theta_normalizer = init_normalizer(type, t1, t2, eps=1e-7) if self.meta_info['theta_dim'] else None


        else:

            if type == 'unit':
                normalizer = UnitTransformer
            elif type == 'minmax':
                normalizer = MinMaxTransformer
            elif type == 'none':
                normalizer = IdentityTransformer
            else:
                raise NotImplementedError

            self.x_normalizer = cal_normalizer_efficient(type, [data['x'] for data in self.data_list], eps=1e-7)
            self.y_normalizer = cal_normalizer_efficient(type, [data['y'] for data in self.data_list], eps=1e-7)
            self.theta_normalizer = None if self.theta is None else normalizer(self.theta, eps=1e-7)



        return self.x_normalizer, self.y_normalizer, self.theta_normalizer

    def apply_normalizer(self, x_normalizer=None, y_normalizer=None, theta_normalizer=None):
        if self.scattered_storage:
            self.x_normalizer = x_normalizer
            self.y_normalizer = y_normalizer
            self.theta_normalizer = theta_normalizer
        else:
            with torch.no_grad():
                for i in range(len(self.data_list)):
                    if x_normalizer is not None:
                        self.data_list[i]['x'] = x_normalizer.transform(self.data_list[i]['x'], inverse=False)
                    if y_normalizer is not None:
                        self.data_list[i]['y'] = y_normalizer.transform(self.data_list[i]['y'], inverse=False)
                if theta_normalizer is not None:
                    self.theta = theta_normalizer.transform(self.theta, inverse=False)
        return

    @staticmethod
    def get_splits(meta_info):
        all_ids = list(range(meta_info['size']))
        train_num, valid_num, test_num = meta_info['split']
        return all_ids[:train_num],  all_ids[train_num+test_num:], all_ids[train_num:train_num+test_num]


if __name__ == "__main__":
    # dataset = PointDataset('elas2d',downsample_x=2,downsample_y=2,max_nodes=400)
    # x,y, theta, fn = dataset[0]
    # print(x.shape, y.shape, theta.shape, fn.shape)
    pass

