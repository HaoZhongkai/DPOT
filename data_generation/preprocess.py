#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import pickle
import numpy as np
import os
import random
import scipy
import scipy.io
import h5py
from tqdm import tqdm
import torch
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from data_generation.cfdbench import get_auto_dataset



def preprocess_mat():
    data = h5py.File('/home/haozhongkai/files/ml4phys/mgn/pdessl/data/ns2d/ns_V1e-3_N5000_T50.mat')
    data = np.array(data['u'])
    data = np.transpose(data, (3,1,2,0))
    train_u = data[:4800]
    test_u = data[4800:]
    print(train_u.shape, test_u.shape)
    pickle.dump(train_u, open('/home/haozhongkai/files/ml4phys/mgn/pdessl/data/ns2d/ns2d_1e-3_train.pkl','wb'))
    pickle.dump(test_u, open('/home/haozhongkai/files/ml4phys/mgn/pdessl/data/ns2d/ns2d_1e-3_test.pkl','wb'))




def save_hdf5():
    import pickle
    import h5py
    import os

    # 文件名列表
    file_names = [
        "ns2d_1e-3_test.pkl", "ns2d_1e-3_train.pkl",
        "ns2d_1e-4_test.pkl", "ns2d_1e-4_train.pkl",
        "ns2d_1e-5_test.pkl", "ns2d_1e-5_train.pkl"
    ]

    for fname in file_names:
        with open(os.path.join('/datasets/opb/pretrain',fname), 'rb') as f:
            data = pickle.load(f)

        hdf5_name = os.path.splitext(fname)[0] + '.hdf5'

        with h5py.File(os.path.join('/datasets/opb/pretrain',hdf5_name), 'w') as hf:
            hf.create_dataset('data', data=data)

    print("Conversion completed!")


### run with root
def process_pdebench_data(path='/data/pdebench/164693',save_name='/data/pdebench/ns2d_pdb_M1_eta1e-8_zeta1e-8',n_train=9000, n_test=1000):
    ### link: https://darus.uni-stuttgart.de/file.xhtml?fileId=164693&version=3.0
    ### keys: Vx, Vy, Vz, density, pressure, t-coordinate, x-coordinate, y-coordinate, z-coordinate
    os.mkdir(save_name)
    os.mkdir(save_name + '/train')
    os.mkdir(save_name + '/test' )
    print('path created')
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        keys.sort()
        print(keys)
        vx = f['Vx']
        vy = f['Vy']
        # vz = f['Vz']
        density = f['density']
        pressure = f['pressure']
        t = f['t-coordinate']
        x = f['x-coordinate']
        y = f['y-coordinate']
        # z = f['z-coordinate']

        vx = np.array(vx, dtype=np.float32)
        vy = np.array(vy, dtype=np.float32)
        # vz = np.array(vz, dtype=np.float32)
        density = np.array(density, dtype=np.float32)
        pressure = np.array(pressure, dtype=np.float32)

        t = np.array(t, dtype=np.float32)    ###, t, x are equispaced
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        # z = np.array(z, dtype=np.float32)
        print('Content loaded:', vx.shape, density.shape, pressure.shape, t.shape, x.shape, y.shape)

        ## storage: x: u(t0), y: u(t1~t20), order: [B, T, X, Y ,C]
        data = np.stack([vx, vy, density, pressure],axis=-1).transpose(0,2,3,1,4)
        # X = data[:,0:1]
        # Y = data[:,1:]
        print(data.shape)   # B, X, Y, T, C
    del vx, vy,  density, pressure


    def split_data(N):

        all_ids = list(range(N))
        test_size = N // 10
        test_ids = random.sample(all_ids, test_size)
        train_ids = [id_ for id_ in all_ids if id_ not in test_ids]

        return train_ids, test_ids

    # train_ids, test_ids = split_data(10000)
    train_ids, test_ids = np.arange(int(9/10 * data.shape[0])), np.arange(int(9/10 * data.shape[0]),data.shape[0])
    print('train ids',train_ids)
    print('test ids',test_ids)

    for i in range(n_train):
        with h5py.File(save_name + '/train/data_{}.hdf5'.format(i),'w') as f:
            f.create_dataset('data', data=data[train_ids[i]], compression=None)
        print('task @ {} saved, shape {}'.format(i, data[i].shape))
    for i in range(n_test):
        start = data.shape[0] - n_test
        with h5py.File(save_name + '/test/data_{}.hdf5'.format(i),'w') as f:
            f.create_dataset('data', data=data[test_ids[i]], compression=None)
            # f.create_dataset('data', data=data[start + i], compression=None)
        print('task @ {} saved, shape {}'.format(i, data[i].shape))


    print('file saved')

### Shallow water PDE
def process_swe_pdebench(path, save_name, n_train=900, n_test=100):
    ## t: 0~ 5, [101], x, y: -1~1. [128]
    os.mkdir(save_name)
    os.mkdir(save_name + '/train')
    os.mkdir(save_name + '/test')
    print('path created')
    data = []
    with h5py.File(path, 'r') as fp:
        for i in range(len(fp.keys())):
            data.append(fp["{0:0=4d}/data".format(i)])


        data = np.stack(data, axis=0).transpose(0,2,3,1,4)  # 1000,128,128,101,2
        print(data.shape)

    train_ids, test_ids = np.arange(int(n_train)), np.arange(n_train, n_train + n_test)
    print('train ids', train_ids)
    print('test ids', test_ids)

    for i in range(n_train):
        with h5py.File(save_name + '/train/data_{}.hdf5'.format(i), 'w') as f:
            f.create_dataset('data', data=data[train_ids[i]], compression=None)
        print('task @ {} saved, shape {}'.format(i, data[i].shape))
    for i in range(n_test):
        start = data.shape[0] - n_test
        with h5py.File(save_name + '/test/data_{}.hdf5'.format(i), 'w') as f:
            f.create_dataset('data', data=data[test_ids[i]], compression=None)
            # f.create_dataset('data', data=data[start + i], compression=None)
        print('task @ {} saved, shape {}'.format(i, data[i].shape))

    print('file saved')
    return


### Diffusion Reaction PDE
def process_dr_pdebench(path, save_name, n_train=900, n_test=100):
    ## t: 0~1, [101], x, y: -2.5~2.5 [128]
    os.mkdir(save_name)
    os.mkdir(save_name + '/train')
    os.mkdir(save_name + '/test')
    print('path created')
    data = []
    with h5py.File(path, 'r') as fp:
        for i in range(len(fp.keys())):
            data.append(fp["{0:0=4d}/data".format(i)])


        data = np.stack(data, axis=0).transpose(0,2,3,1,4)  # 1000,128,128,101,2
        print(data.shape)

    train_ids, test_ids = np.arange(int(n_train)), np.arange(n_train, n_train + n_test)
    print('train ids', train_ids)
    print('test ids', test_ids)

    for i in range(n_train):
        with h5py.File(save_name + '/train/data_{}.hdf5'.format(i), 'w') as f:
            f.create_dataset('data', data=data[train_ids[i]], compression=None)
        print('task @ {} saved, shape {}'.format(i, data[i].shape))
    for i in range(n_test):
        start = data.shape[0] - n_test
        with h5py.File(save_name + '/test/data_{}.hdf5'.format(i), 'w') as f:
            f.create_dataset('data', data=data[test_ids[i]], compression=None)
            # f.create_dataset('data', data=data[start + i], compression=None)
        print('task @ {} saved, shape {}'.format(i, data[i].shape))

    print('file saved')
    return


### run with root
def process_pdebench3d_data(path,save_name,n_train=90, n_test=10):
    ### link: https://darus.uni-stuttgart.de/file.xhtml?fileId=164693&version=3.0
    ### keys: Vx, Vy, Vz, density, pressure, t-coordinate, x-coordinate, y-coordinate, z-coordinate
    if not os.path.exists(save_name):
        os.mkdir(save_name)
    os.mkdir(save_name + '/train')
    os.mkdir(save_name + '/test' )
    print('path created')
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        keys.sort()
        print(keys)
        vx = f['Vx']
        vy = f['Vy']
        vz = f['Vz']
        density = f['density']
        pressure = f['pressure']
        t = f['t-coordinate']
        x = f['x-coordinate']
        y = f['y-coordinate']
        z = f['z-coordinate']

        vx = np.array(vx, dtype=np.float32)
        vy = np.array(vy, dtype=np.float32)
        vz = np.array(vz, dtype=np.float32)
        density = np.array(density, dtype=np.float32)
        pressure = np.array(pressure, dtype=np.float32)

        t = np.array(t, dtype=np.float32)    ###, t, x are equispaced
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        z = np.array(z, dtype=np.float32)
        print('Content loaded:', vx.shape, density.shape, pressure.shape, t.shape, x.shape, y.shape)

        ## storage: x: u(t0), y: u(t1~t20), order: [B, T, X, Y, Z ,C]
        data = np.stack([vx, vy, vz, pressure, density],axis=-1).transpose(0,2,3,4,1,5)
        # X = data[:,0:1]
        # Y = data[:,1:]
        print(data.shape)   # B, X, Y, T, C
    del vx, vy,  density, pressure

    def split_data(N):

        all_ids = list(range(N))
        test_size = N // 10
        test_ids = random.sample(all_ids, test_size)
        train_ids = [id_ for id_ in all_ids if id_ not in test_ids]

        return train_ids, test_ids

    # train_ids, test_ids = split_data(10000)
    train_ids, test_ids = np.arange(int(9/10 * data.shape[0])), np.arange(int(9/10 * data.shape[0]),data.shape[0])
    print('train ids',train_ids)
    print('test ids',test_ids)

    for i in range(n_train):
        with h5py.File(save_name + '/train/data_{}.hdf5'.format(i),'w') as f:
            f.create_dataset('data', data=data[train_ids[i]], compression=None)
        print('task @ {} saved, shape {}'.format(i, data[i].shape))
    for i in range(n_test):
        start = data.shape[0] - n_test
        with h5py.File(save_name + '/test/data_{}.hdf5'.format(i),'w') as f:
            f.create_dataset('data', data=data[test_ids[i]], compression=None)
            # f.create_dataset('data', data=data[start + i], compression=None)
        print('task @ {} saved, shape {}'.format(i, data[i].shape))


    print('file saved')




def preprocess_ns2d(load_path='data/large/pdearena/NavierStokes-2D',
                    save_path='data/large/pdearena/ns2d_pda'):
    """
    Preprocess the Navier-Stokes 2D dataset from PDEArena

    there are 3 channels in the dataset:
        u, vx, vy
    data shape: (N, 128, 128, 14, 3)
    """
    LOAD_PATH = load_path
    SAVE_PATH_TEST = save_path + '/test'
    SAVE_PATH_TRAIN = save_path + '/train'

    # Create new folders if SAVE_PATH does not exist
    os.makedirs(SAVE_PATH_TEST, exist_ok=True)
    os.makedirs(SAVE_PATH_TRAIN, exist_ok=True)

    test_tot = 0
    train_tot = 0

    # Traverse the file in LOAD_PATH
    for root, dirs, files in os.walk(LOAD_PATH):
        for file in tqdm(files):
            # Skip the file if it is not a HDF5 file
            if not file.endswith('.h5'):
                continue
            # Open the file
            try:
                with h5py.File(os.path.join(root, file), 'r') as f:
                    if 'test' in file:
                        key = 'test'
                        path = SAVE_PATH_TEST
                    elif 'train' in file:
                        key = 'train'
                        path = SAVE_PATH_TRAIN
                    elif 'valid' in file:
                        key = 'valid'
                        path = SAVE_PATH_TRAIN
                    else:
                        raise ValueError('Unknown file type {}!'.format(file))

                    u = f[key]['u'][:]
                    vx = f[key]['vx'][:]
                    vy = f[key]['vy'][:]

                    out = np.stack([u, vx, vy], axis=-1)
                    out = np.transpose(out, (0, 2, 3, 1, 4))

                    # Create the destination file
                    for data in out:
                        if key == 'test':
                            idx = test_tot
                            test_tot += 1
                        else:
                            idx = train_tot
                            train_tot += 1
                        dst_file = 'data_{}.hdf5'.format(idx)
                        save_path = os.path.join(path, dst_file)
                        with h5py.File(save_path, 'w') as g:
                            # Write data as a hdf5 dataset
                            # with key 'data'
                            g.create_dataset('data', data=data)
            except Exception as e:
                print('Error in file {}: {}'.format(file, e))
                continue


def preprocess_ns2d_cond():
    """
    Preprocess the Navier-Stokes 2D conditioned
        dataset from PDEArena

    there are 3 channels in the dataset:
        u, vx, vy
    data shape: (N, 128, 128, 56, 3)
    """
    preprocess_ns2d(
        load_path='data/large/pdearena/NavierStokes-2D-conditoned',
        save_path='data/large/pdearena/ns2d_cond_pda'
    )


def preprocess_shallow_water():
    """
    Preprocess the Shallow Water dataset from PDEArena

    there are 5 channels in the dataset:
        u, v, div, vor, pres
    data shape: (N, 96, 192, 88, 5)
    """
    LOAD_PATH = 'data/large/pdearena/ShallowWater-2D'
    SAVE_PATH_TEST = 'data/large/pdearena/sw2d_pda/test'
    SAVE_PATH_TRAIN = 'data/large/pdearena/sw2d_pda/train'

    # Create new folders if SAVE_PATH does not exist
    os.makedirs(SAVE_PATH_TEST, exist_ok=True)
    os.makedirs(SAVE_PATH_TRAIN, exist_ok=True)

    test_tot = 0
    train_tot = 0

    # Traverse the file in LOAD_PATH
    for root, dirs, files in tqdm(os.walk(LOAD_PATH)):
        for file in files:
            # Skip the file if it is not a HDF5 file
            if not file.endswith('.nc'):
                continue
            # Open the file
            try:
                with h5py.File(os.path.join(root, file), 'r') as f:
                    if 'test' in root:
                        key = 'test'
                        path = SAVE_PATH_TEST
                    elif 'train' in root:
                        key = 'train'
                        path = SAVE_PATH_TRAIN
                    elif 'valid' in root:
                        key = 'valid'
                        path = SAVE_PATH_TRAIN
                    else:
                        raise ValueError('Unknown file type {}!'.format(file))

                    u = f['u'][:]
                    u = u[:, 0, ...]
                    v = f['v'][:]
                    v = v[:, 0, ...]
                    div = f['div'][:]
                    div = div[:, 0, ...]
                    vor = f['vor'][:]
                    vor = vor[:, 0, ...]
                    pres = f['pres'][:]

                    data = np.stack([u, v, div, vor, pres], axis=-1)
                    data = np.transpose(data, (1, 2, 0, 3))

                    # Create the destination file
                    if key == 'test':
                        idx = test_tot
                        test_tot += 1
                    else:
                        idx = train_tot
                        train_tot += 1
                    dst_file = 'data_{}.hdf5'.format(idx)
                    save_path = os.path.join(path, dst_file)
                    with h5py.File(save_path, 'w') as g:
                        # Write data as a hdf5 dataset
                        # with key 'data'
                        g.create_dataset('data', data=data)
            except Exception as e:
                print('Error in file {}: {}'.format(file, e))
                continue




def preprocess_cfdbench_data():
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

    with h5py.File('./../data/cfdbench/ns2d_cdb_train.hdf5','w') as fp:
        fp.create_dataset('data',data=train_data,compression=None)



    with h5py.File('./../data/cfdbench/ns2d_cdb_test.hdf5','w') as fp:
        fp.create_dataset('data',data=test_data,compression=None)


if __name__ == '__main__':


    #### FNO datasets
    # preprocess_mat()

    #### PDEBench datasets
    process_pdebench_data(path='./../data/164687',save_name='./../data/pdebench/ns2d_pdb_M1e-1_eta1e-2_zeta1e-2',n_train=9000, n_test=1000)
    process_pdebench_data(path='./../data/164688',save_name='./../data/pdebench/ns2d_pdb_M1e-1_eta1e-1_zeta1e-1',n_train=9000, n_test=1000)
    process_pdebench_data(path='./../data/164690',save_name='./../data/pdebench/ns2d_pdb_M1_eta1e-2_zeta1e-2',n_train=9000, n_test=1000)
    process_pdebench_data(path='./../data/164691',save_name='./../data//pdebench/ns2d_pdb_M1_eta1e-1_zeta1e-1',n_train=9000, n_test=1000)
    process_pdebench_data(path='./../data/164685',save_name='./../data/pdebench/ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_turb_512',n_train=900, n_test=100)
    process_pdebench_data(path='./../data/164686',save_name='./../data/pdebench/ns2d_pdb_M1_eta1e-8_zeta1e-8_turb_512',n_train=900, n_test=100)
    process_pdebench_data(path='./../data/164689',save_name='./../data/pdebench/ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_rand_512',n_train=900, n_test=100)
    process_pdebench_data(path='./../data/164692',save_name='./../data/pdebench/ns2d_pdb_M1_eta1e-8_zeta1e-8_rand_512',n_train=900, n_test=100)
    process_swe_pdebench(path='./../data/133021',save_name='./../data/pdebench/swe_pdb',n_train=900, n_test=100)
    process_dr_pdebench(path='./../data/133017',save_name='./../data/pdebench/dr_pdb',n_train=900, n_test=100)
    process_pdebench3d_data(path='./../data/164693',save_name='./../data/pdebench/ns3d_pdb_M1_rand',n_train=90, n_test=10)
    process_pdebench3d_data(path='./../data/173286',save_name='./../data/pdebench/ns3d_pdb_M1e-1_rand',n_train=90, n_test=10)
    process_pdebench3d_data(path='./../data/164694',save_name='./../data/pdebench/ns3d_pdb_M1_turb',n_train=540, n_test=60)

    #### PDEArena datasets
    preprocess_ns2d()
    preprocess_ns2d_cond()
    preprocess_shallow_water()


    #### CFDBench datasets
    preprocess_cfdbench_data()
