import h5py
import os
import numpy as np
from tqdm import tqdm


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

if __name__ == '__main__':
    preprocess_ns2d()
    preprocess_ns2d_cond()
    preprocess_shallow_water()


# PATH = "data/large/pdearena/ShallowWater-2D/test/seed=19324/run0001/output.nc"
# # open and print data shape
# with h5py.File(PATH, 'r') as f:
#     print(f)
#     for key in f.keys():
#         print(key, f[key])
#         # f[key] is a HDF5 group
#         # print all the members of the group
#         # for kkey in f[key].keys():
#         #     print(f[key][kkey])
#         #     # Extract numpy array from the dataset
#         #     data = f[key][kkey][:]
#         #     print(data.shape)


# PATH2 = "data/large/pdebench/ns2d_pdb_M1_eta1e-1_zeta1e-1/train/data_0.hdf5"

# open and print the shape
# with h5py.File(PATH2, 'r') as f:
#     print(f)
#     for key in f.keys():
#         print(key, f[key])
