#!/usr/bin/env python  
#-*- coding:utf-8 _*-

import numpy as np
import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from data_generation.cfdbench import get_auto_dataset

delta_cavity = 0.1
delta_cylinder = 0.1
delta_tube = 0.1
delta_dam = 0.1
train_data_cavity, dev_data_cavity, test_data_cavity = get_auto_dataset(
    data_dir=Path('../../data/large/cfdbench'),
    data_name='cavity_prop_bc_geo',
    delta_time=0.1,
    norm_props=True,
    norm_bc=True,
)
train_data_cylinder, dev_data_cylinder, test_data_cylinder = get_auto_dataset(
    data_dir=Path('../../data/large/cfdbench'),
    data_name='cylinder_prop_bc_geo',
    delta_time=0.1,
    norm_props=True,
    norm_bc=True,
)
# train_data_dam, dev_data_dam, test_data_dam = get_auto_dataset(
#     data_dir=Path('../../data/large/cfdbench'),
#     data_name='dam_prop',
#     delta_time=0.1,
#     norm_props=True,
#     norm_bc=True,
# )
train_data_tube, dev_data_tube, test_data_tube = get_auto_dataset(
    data_dir=Path('../../data/large/cfdbench'),
    data_name='tube_prop_bc_geo',
    delta_time=0.1,
    norm_props=True,
    norm_bc=True,
)

cavity_lens = [data.shape[0] for data in train_data_cavity.all_features]
cylinder_lens = [data.shape[0] for data in train_data_cylinder.all_features]
tube_lens = [data.shape[0] for data in train_data_tube.all_features]
# dam_lens = [data.shape[0] for data in train_data_dam.all_features]

train_cavity_feats, train_cylinder_feats, train_tube_feats = train_data_cavity.all_features, train_data_cylinder.all_features, train_data_tube.all_features
test_cavity_feats, test_cylinder_feats, test_tube_feats = test_data_cavity.all_features, test_data_cylinder.all_features, test_data_tube.all_features

train_feats = train_cavity_feats + train_cylinder_feats + train_tube_feats
test_feats = test_cavity_feats + test_cylinder_feats + test_tube_feats


print(cavity_lens)
print(cylinder_lens)
print(tube_lens)
# print(dam_lens)

infer_steps = 20

def split_trajectory(data_list, time_step, grid_size=64):
    traj_split = []
    for i, x in enumerate(data_list):
        T = x.shape[0]
        num_segments = int(np.ceil(T / time_step))
        padded_length = num_segments * time_step
        padded_array = np.zeros((padded_length, *x.shape[1:]))

        # Copy the original data into the padded array
        padded_array[:T, ...] = x

        # If needed, pad the last segment with the last frame of the original array
        if T % time_step != 0:
            last_frame = x[-1, ...]
            padded_array[T:, ...] = last_frame

        # Reshape the array into segments
        padded_array = F.interpolate(torch.from_numpy(padded_array),size=(grid_size,grid_size),mode='bilinear',align_corners=True).numpy()
        padded_array = padded_array.reshape((num_segments, time_step, *padded_array.shape[1:]))

        traj_split.append(padded_array)

    traj_split = np.concatenate(traj_split, axis=0)
    return traj_split


train_data = split_trajectory(train_feats, infer_steps,grid_size=64)
test_data = split_trajectory(test_feats, infer_steps,grid_size=64)
train_data, test_data = train_data.transpose(0,3,4,1,2), test_data.transpose(0, 3, 4, 1, 2) # B, X, Y, T, C
print(train_data.shape, test_data.shape)

with h5py.File('../../data/large/cfdbench/ns2d_cdb_train.hdf5','w') as fp:
    fp.create_dataset('data',data=train_data,compression=None)



with h5py.File('../../data/large/cfdbench/ns2d_cdb_test.hdf5','w') as fp:
    fp.create_dataset('data',data=test_data,compression=None)


