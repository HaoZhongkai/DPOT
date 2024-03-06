#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import pickle
import numpy as np
import os
import random
import scipy
import scipy.io
import h5py
def preprocess_ns2d():
    def preprocess(path):
        x = pickle.load(open(path,'rb'))
        a = x[0]
        u = x[1]
        y = np.concatenate([a[...,2:],u],axis=-1)
        print(y.shape)
        pickle.dump(y,open(path,'wb'))

    # preprocess('/home/haozhongkai/files/ml4phys/mgn/pdessl/data/ns2d/ns2d_1e-4_test.pkl')
    # preprocess('/home/haozhongkai/files/ml4phys/mgn/pdessl/data/ns2d/ns2d_1e-4_train.pkl')
    # preprocess('/home/haozhongkai/files/ml4phys/mgn/pdessl/data/ns2d/ns2d_1e-5_test.pkl')
    preprocess('/home/haozhongkai/files/ml4phys/mgn/pdessl/data/ns2d/ns2d_1e-5_train.pkl')


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
        # 读取pickle文件
        with open(os.path.join('/datasets/opb/pretrain',fname), 'rb') as f:
            data = pickle.load(f)

        # 创建对应的hdf5文件名
        hdf5_name = os.path.splitext(fname)[0] + '.hdf5'

        # 将数据写入hdf5文件
        with h5py.File(os.path.join('/datasets/opb/pretrain',hdf5_name), 'w') as hf:
            hf.create_dataset('data', data=data)

    print("Conversion completed!")



def save_hdf5_g45():
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
        # 读取pickle文件
        with open(os.path.join('./../../data/ns2d',fname), 'rb') as f:
            data = pickle.load(f)

        # 创建对应的hdf5文件名
        hdf5_name = os.path.splitext(fname)[0] + '.hdf5'

        # 将数据写入hdf5文件
        with h5py.File(os.path.join('./../../data/large',hdf5_name), 'w') as hf:
            hf.create_dataset('data', data=data)

    print("Conversion completed!")

### run with root
def process_pdebench_data(path='/datasets/opb/griddataset/pdebench/164693',save_name='/datanas/opb/pretrain/ns2d_pdb_M1_eta1e-8_zeta1e-8',n_train=9000, n_test=1000):
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
    ### save as a large hdf5
    # with h5py.File(save_name + '_train.hdf5' ,'w') as f:
    #     f.create_dataset('data', data=data[:n_train],chunks=(1, *data.shape[1:]),compression=None)
    # with h5py.File(save_name + '_test.hdf5' ,'w') as f:
    #     f.create_dataset('data', data=data[-n_test:],chunks=(1, *data.shape[1:]),compression=None)
    #

    def split_data(N):
        """
        随机划分数据为9:1的比例并返回id

        参数:
        - N: 数据的总数量

        返回:
        - train_ids: 训练集的id列表
        - test_ids: 测试集的id列表
        """
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
    ### save as a large hdf5
    # with h5py.File(save_name + '_train.hdf5' ,'w') as f:
    #     f.create_dataset('data', data=data[:n_train],chunks=(1, *data.shape[1:]),compression=None)
    # with h5py.File(save_name + '_test.hdf5' ,'w') as f:
    #     f.create_dataset('data', data=data[-n_test:],chunks=(1, *data.shape[1:]),compression=None)
    #

    def split_data(N):
        """
        随机划分数据为9:1的比例并返回id

        参数:
        - N: 数据的总数量

        返回:
        - train_ids: 训练集的id列表
        - test_ids: 测试集的id列表
        """
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


# preprocess_ns2d()
# preprocess_mat()
# save_hdf5()

### unfinished: 93(3d), 90(2d)
# process_pdebench_data(path='/datasets/opb/griddataset/pdebench/164693',save_path='/datanas/opb/pretrain/ns2d_pdb_M1_eta1e-8_zeta1e-8.hdf5')

### finished
# process_pdebench_data(path='/datanas/opb/griddataset/pdebench/164687',save_name='/datasets/opb/pretrain/ns2d_pdb_M1e-1_eta1e-2_zeta1e-2',n_train=9000, n_test=1000)
# process_pdebench_data(path='/datanas/opb/griddataset/pdebench/164688',save_name='/datasets/opb/pretrain/ns2d_pdb_M1e-1_eta1e-1_zeta1e-1',n_train=9000, n_test=1000)
# process_pdebench_data(path='/datanas/opb/griddataset/pdebench/164690',save_name='/datasets/opb/pretrain/ns2d_pdb_M1_eta1e-2_zeta1e-2',n_train=9000, n_test=1000)
# process_pdebench_data(path='/datanas/opb/griddataset/pdebench/164691',save_name='/datasets/opb/pretrain/ns2d_pdb_M1_eta1e-1_zeta1e-1',n_train=9000, n_test=1000)


# save_hdf5_g45()

#### on g45
# process_pdebench_data(path='./../../data/large/164687',save_name='./../../data/large/pdebench/ns2d_pdb_M1e-1_eta1e-2_zeta1e-2',n_train=9000, n_test=1000)
# process_pdebench_data(path='./../../data/large/164688',save_name='./../../data/large/pdebench/ns2d_pdb_M1e-1_eta1e-1_zeta1e-1',n_train=9000, n_test=1000)
# process_pdebench_data(path='./../../data/large/164690',save_name='./../../data/large/pdebench/ns2d_pdb_M1_eta1e-2_zeta1e-2',n_train=9000, n_test=1000)
# process_pdebench_data(path='./../../data/large/164691',save_name='./../../data/large//pdebench/ns2d_pdb_M1_eta1e-1_zeta1e-1',n_train=9000, n_test=1000)
# process_pdebench_data(path='./../../data/large/pdebench/164685',save_name='./../../data/large/pdebench/ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_turb_512',n_train=900, n_test=100)
# process_pdebench_data(path='./../../data/large/pdebench/164686',save_name='./../../data/large/pdebench/ns2d_pdb_M1_eta1e-8_zeta1e-8_turb_512',n_train=900, n_test=100)
# process_pdebench_data(path='./../../data/large/164689',save_name='./../../data/large/pdebench/ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_rand_512',n_train=900, n_test=100)
# process_pdebench_data(path='./../../data/large/164692',save_name='./../../data/large/pdebench/ns2d_pdb_M1_eta1e-8_zeta1e-8_rand_512',n_train=900, n_test=100)
# process_swe_pdebench(path='./../../data/large/pdebench/133021',save_name='./../../data/large/pdebench/swe_pdb',n_train=900, n_test=100)
# process_dr_pdebench(path='./../../data/large/pdebench/133017',save_name='./../../data/large/pdebench/dr_pdb',n_train=900, n_test=100)
# process_pdebench3d_data(path='./../../data/large/pdebench/164693',save_name='./../../data/large/pdebench/ns3d_pdb_M1_rand',n_train=90, n_test=10)
# process_pdebench3d_data(path='./../../data/large/pdebench/173286',save_name='./../../data/large/pdebench/ns3d_pdb_M1e-1_rand',n_train=90, n_test=10)
process_pdebench3d_data(path='./../../data/large/pdebench/164694',save_name='./../../data/large/pdebench/ns3d_pdb_M1_turb',n_train=540, n_test=60)












