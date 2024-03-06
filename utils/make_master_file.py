### script for writing meta information of datasets into master.csv
### for node property prediction datasets.
import os
import h5py
import pandas as pd


DATASET_DICT = {}
DATASET_LIST = []

### classic benchmark
name = 'ns2d_fno_1e-5'
DATASET_DICT[name] = {'train_path': './data/large/ns2d_1e-5_train.hdf5', 'test_path': './data/large/ns2d_1e-5_test.hdf5'}
DATASET_DICT[name]['train_size'] = 1000
DATASET_DICT[name]['test_size'] = 200
DATASET_DICT[name]['scatter_storage']= False
DATASET_DICT[name]['t_test'] = 10   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 20
DATASET_DICT[name]['in_size'] = (64, 64)
DATASET_DICT[name]['n_channels'] = 1
DATASET_DICT[name]['downsample'] = (1, 1)


name = 'ns2d_fno_1e-4'
DATASET_DICT[name] = {'train_path': './data/large/ns2d_1e-4_train.hdf5', 'test_path': './data/large/ns2d_1e-4_test.hdf5'}
DATASET_DICT[name]['train_size'] = 9800
DATASET_DICT[name]['test_size'] = 200
DATASET_DICT[name]['scatter_storage'] = False
DATASET_DICT[name]['t_test'] = 20   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 30
DATASET_DICT[name]['in_size'] = (64, 64)
DATASET_DICT[name]['n_channels'] = 1
DATASET_DICT[name]['downsample'] = (1, 1)



name = 'ns2d_fno_1e-3'
DATASET_DICT[name] = {'train_path': './data/large/ns2d_1e-3_train.hdf5', 'test_path': './data/large/ns2d_1e-3_test.hdf5'}
DATASET_DICT[name]['train_size'] = 1000
DATASET_DICT[name]['test_size'] = 200
DATASET_DICT[name]['scatter_storage'] = False
DATASET_DICT[name]['t_test'] = 20   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 50
DATASET_DICT[name]['in_size'] = (64, 64)
DATASET_DICT[name]['n_channels'] = 1
DATASET_DICT[name]['downsample'] = (1, 1)


name = 'ns2d_pdb_M1_eta1e-1_zeta1e-1'
# DATASET_DICT[name] = {'train_path': './data/large/ns2d_pdb_M1_eta1e-2_zeta1e-2_train.hdf5', 'test_path': './data/large/ns2d_pdb_M1_eta1e-2_zeta1e-2_test.hdf5'}
DATASET_DICT[name] = {'train_path': './data/large/pdebench/ns2d_pdb_M1_eta1e-1_zeta1e-1/train', 'test_path': './data/large/pdebench/ns2d_pdb_M1_eta1e-1_zeta1e-1/test'}
# DATASET_DICT[name] = {'train_path': '/datasets/opb/pretrain/ns2d_pdb_M1_eta1e-2_zeta1e-2/train', 'test_path': '/datasets/opb/pretrain/ns2d_pdb_M1_eta1e-2_zeta1e-2/test'}
DATASET_DICT[name]['train_size'] = 9000
DATASET_DICT[name]['test_size'] = 200       ### default 200, maximum 1000
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 11   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 21
DATASET_DICT[name]['in_size'] = (128, 128)
DATASET_DICT[name]['n_channels'] = 4
DATASET_DICT[name]['downsample'] = (1, 1)


name = 'ns2d_pdb_M1_eta1e-2_zeta1e-2'
# DATASET_DICT[name] = {'train_path': './data/large/ns2d_pdb_M1_eta1e-2_zeta1e-2_train.hdf5', 'test_path': './data/large/ns2d_pdb_M1_eta1e-2_zeta1e-2_test.hdf5'}
DATASET_DICT[name] = {'train_path': './data/large/pdebench/ns2d_pdb_M1_eta1e-2_zeta1e-2/train', 'test_path': './data/large/pdebench/ns2d_pdb_M1_eta1e-2_zeta1e-2/test'}
# DATASET_DICT[name] = {'train_path': '/datasets/opb/pretrain/ns2d_pdb_M1_eta1e-2_zeta1e-2/train', 'test_path': '/datasets/opb/pretrain/ns2d_pdb_M1_eta1e-2_zeta1e-2/test'}
DATASET_DICT[name]['train_size'] = 9000
DATASET_DICT[name]['test_size'] = 200       ### default 200, maximum 1000
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 11   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 21
DATASET_DICT[name]['in_size'] = (128, 128)
DATASET_DICT[name]['n_channels'] = 4
DATASET_DICT[name]['downsample'] = (1, 1)


name = 'ns2d_pdb_M1e-1_eta1e-1_zeta1e-1'
DATASET_DICT[name] = {'train_path': './data/large/pdebench/ns2d_pdb_M1e-1_eta1e-1_zeta1e-1/train', 'test_path': './data/large/pdebench/ns2d_pdb_M1e-1_eta1e-1_zeta1e-1/test'}
DATASET_DICT[name]['train_size'] = 9000
DATASET_DICT[name]['test_size'] = 200       ### default 200, maximum 1000
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 11   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 21
DATASET_DICT[name]['in_size'] = (128, 128)
DATASET_DICT[name]['n_channels'] = 4
DATASET_DICT[name]['downsample'] = (1, 1)


name = 'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2'
DATASET_DICT[name] = {'train_path': './data/large/pdebench/ns2d_pdb_M1e-1_eta1e-2_zeta1e-2/train', 'test_path': './data/large/pdebench/ns2d_pdb_M1e-1_eta1e-2_zeta1e-2/test'}
DATASET_DICT[name]['train_size'] = 9000
DATASET_DICT[name]['test_size'] = 200       ### default 200, maximum 1000
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 11   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 21
DATASET_DICT[name]['in_size'] = (128, 128)
DATASET_DICT[name]['n_channels'] = 4
DATASET_DICT[name]['downsample'] = (1, 1)

# # Superbench
# SUPER_BENCH_PATH = 'data/large/superbench'
# # Read .h5 file and get the shape
# def get_shape(path):
#     with h5py.File(path, 'r') as f:
#         # There will be only one dataset in the file
#         for key in f.keys():
#             return f[key].shape
# # Get the folder names in the SUPER_BENCH_PATH
# datasets = os.listdir(SUPER_BENCH_PATH)
# datasets = [dataset for dataset in datasets if dataset != 'superbench_v1']
# for dataset in datasets:
#     # Get all the files in SUPER_BENCH_PATH/dataset/train
#     train_files = os.listdir(os.path.join(SUPER_BENCH_PATH, dataset, 'train'))
#     for train_file in train_files:
#         # Get the shape of the file
#         train_shape = get_shape(os.path.join(SUPER_BENCH_PATH, dataset, 'train', train_file))
#         # Get all other files
#         sub_folders = ['test_1', 'test_2', 'valid_1', 'valid_2']
#
#         for sub_folder in sub_folders:
#             test_files = os.listdir(
#                 os.path.join(SUPER_BENCH_PATH, dataset, sub_folder))
#
#             # Initialize the dataset dict
#             for test_file in test_files:
#                 # Get the shape of the file
#                 test_shape = get_shape(
#                     os.path.join(SUPER_BENCH_PATH, dataset, sub_folder, test_file))
#                 name = dataset + '_' + test_file.replace('.hdf5', '')
#                 DATASET_DICT[name] = {
#                     'train_path': os.path.join(SUPER_BENCH_PATH,
#                         dataset, 'train', train_file),
#                     'test_path': os.path.join(SUPER_BENCH_PATH,
#                         dataset, sub_folder, test_file)}
#                 DATASET_DICT[name]['train_size'] = train_shape[0]
#                 DATASET_DICT[name]['test_size'] = test_shape[0]
#                 DATASET_DICT[name]['scatter_storage'] = False
#                 DATASET_DICT[name]['t_test'] = 40
#                 DATASET_DICT[name]['t_in'] = 10
#                 DATASET_DICT[name]['t_total'] = 50
#                 DATASET_DICT[name]['in_size'] = (train_shape[1], train_shape[2])
#                 DATASET_DICT[name]['n_channels'] = train_shape[4]
#                 DATASET_DICT[name]['downsample'] = (1, 1)


name = 'ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_turb_512'
DATASET_DICT[name] = {'train_path': './data/large/pdebench/ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_turb_512/train', 'test_path': './data/large/pdebench/ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_turb_512/test'}
DATASET_DICT[name]['train_size'] = 900
DATASET_DICT[name]['test_size'] = 20       ### default 200, maximum 1000
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 11   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 21
DATASET_DICT[name]['in_size'] = (512, 512)
DATASET_DICT[name]['n_channels'] = 4
DATASET_DICT[name]['downsample'] = (1, 1)


name = 'ns2d_pdb_M1_eta1e-8_zeta1e-8_turb_512'
DATASET_DICT[name] = {'train_path': './data/large/pdebench/ns2d_pdb_M1_eta1e-8_zeta1e-8_turb_512/train', 'test_path': './data/large/pdebench/ns2d_pdb_M1_eta1e-8_zeta1e-8_turb_512/test'}
DATASET_DICT[name]['train_size'] = 900
DATASET_DICT[name]['test_size'] = 20       ### default 200, maximum 1000
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 11   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 21
DATASET_DICT[name]['in_size'] = (512, 512)
DATASET_DICT[name]['n_channels'] = 4
DATASET_DICT[name]['downsample'] = (1, 1)


name = 'ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_rand_512'
DATASET_DICT[name] = {'train_path': './data/large/pdebench/ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_rand_512/train', 'test_path': './data/large/pdebench/ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_rand_512/test'}
DATASET_DICT[name]['train_size'] = 900
DATASET_DICT[name]['test_size'] = 20       ### default 200, maximum 1000
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 11   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 21
DATASET_DICT[name]['in_size'] = (512, 512)
DATASET_DICT[name]['n_channels'] = 4
DATASET_DICT[name]['downsample'] = (1, 1)


name = 'ns2d_pdb_M1_eta1e-8_zeta1e-8_rand_512'
DATASET_DICT[name] = {'train_path': './data/large/pdebench/ns2d_pdb_M1_eta1e-8_zeta1e-8_rand_512/train', 'test_path': './data/large/pdebench/ns2d_pdb_M1_eta1e-8_zeta1e-8_rand_512/test'}
DATASET_DICT[name]['train_size'] = 900
DATASET_DICT[name]['test_size'] = 20       ### default 200, maximum 1000
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 11   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 21
DATASET_DICT[name]['in_size'] = (512, 512)
DATASET_DICT[name]['n_channels'] = 4
DATASET_DICT[name]['downsample'] = (1, 1)

name = 'ns3d_pdb_M1_rand'
DATASET_DICT[name] = {'train_path': './data/large/pdebench/ns3d_pdb_M1_rand/train', 'test_path': './data/large/pdebench/ns3d_pdb_M1_rand/test'}
DATASET_DICT[name]['train_size'] = 90
DATASET_DICT[name]['test_size'] = 10       ### default 200, maximum 1000
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 11   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 21
DATASET_DICT[name]['in_size'] = (128, 128, 128)
DATASET_DICT[name]['n_channels'] = 5
DATASET_DICT[name]['downsample'] = (1, 1, 1)



name = 'ns3d_pdb_M1e-1_rand'
DATASET_DICT[name] = {'train_path': './data/large/pdebench/ns3d_pdb_M1e-1_rand/train', 'test_path': './data/large/pdebench/ns3d_pdb_M1e-1_rand/test'}
DATASET_DICT[name]['train_size'] = 90
DATASET_DICT[name]['test_size'] = 10       ### default 200, maximum 1000
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 11   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 21
DATASET_DICT[name]['in_size'] = (128, 128, 128)
DATASET_DICT[name]['n_channels'] = 5
DATASET_DICT[name]['downsample'] = (1, 1, 1)


name = 'ns3d_pdb_M1_turb'
DATASET_DICT[name] = {'train_path': './data/large/pdebench/ns3d_pdb_M1_turb/train', 'test_path': './data/large/pdebench/ns3d_pdb_M1_turb/test'}
DATASET_DICT[name]['train_size'] = 540
DATASET_DICT[name]['test_size'] = 60       ### default 200, maximum 1000
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 11   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 21
DATASET_DICT[name]['in_size'] = (64, 64, 64)
DATASET_DICT[name]['n_channels'] = 5
DATASET_DICT[name]['downsample'] = (1, 1, 1)


name = 'swe_pdb'
DATASET_DICT[name] = {'train_path': './data/large/pdebench/swe_pdb/train', 'test_path': './data/large/pdebench/swe_pdb/test'}
DATASET_DICT[name]['train_size'] = 900
DATASET_DICT[name]['test_size'] = 60       ### default 60, maximum 100
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 91   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 101
DATASET_DICT[name]['in_size'] = (128, 128)
DATASET_DICT[name]['n_channels'] = 1
DATASET_DICT[name]['downsample'] = (1, 1)

name = 'dr_pdb'
DATASET_DICT[name] = {'train_path': './data/large/pdebench/dr_pdb/train', 'test_path': './data/large/pdebench/dr_pdb/test'}
DATASET_DICT[name]['train_size'] = 900
DATASET_DICT[name]['test_size'] = 60       ### default 200, maximum 100
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 91   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 101
DATASET_DICT[name]['in_size'] = (128, 128)
DATASET_DICT[name]['n_channels'] = 2
DATASET_DICT[name]['downsample'] = (1, 1)


name = 'cfdbench'
DATASET_DICT[name] = {'train_path': './data/large/cfdbench/ns2d_cdb_train.hdf5', 'test_path': './data/large/cfdbench/ns2d_cdb_test.hdf5'}
DATASET_DICT[name]['train_size'] = 9000
DATASET_DICT[name]['test_size'] = 1000       ### default 200, maximum 1000
DATASET_DICT[name]['scatter_storage'] = False
DATASET_DICT[name]['t_test'] = 20   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 20
DATASET_DICT[name]['in_size'] = (64, 64)
DATASET_DICT[name]['n_channels'] = 3
DATASET_DICT[name]['pred_channels'] = 2
DATASET_DICT[name]['downsample'] = (1, 1)




name = 'ns2d_cond_pda'
DATASET_DICT[name] = {'train_path': './data/large/pdearena/ns2d_cond_pda/train', 'test_path': './data/large/pdearena/ns2d_cond_pda/test'}
DATASET_DICT[name]['train_size'] = 3100
DATASET_DICT[name]['test_size'] = 200       ### default 200, maximum 1000
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 46   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 56
DATASET_DICT[name]['in_size'] = (128, 128)
DATASET_DICT[name]['n_channels'] = 3
DATASET_DICT[name]['downsample'] = (1, 1)

name = 'ns2d_pda'
DATASET_DICT[name] = {'train_path': './data/large/pdearena/ns2d_pda/train', 'test_path': './data/large/pdearena/ns2d_pda/test'}
DATASET_DICT[name]['train_size'] = 6500
DATASET_DICT[name]['test_size'] = 650       ### default 650, maximum 1300
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 4   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 14
DATASET_DICT[name]['in_size'] = (128, 128)
DATASET_DICT[name]['n_channels'] = 3
DATASET_DICT[name]['downsample'] = (1, 1)


name = 'sw2d_pda'
DATASET_DICT[name] = {'train_path': './data/large/pdearena/sw2d_pda/train', 'test_path': './data/large/pdearena/sw2d_pda/test'}
DATASET_DICT[name]['train_size'] = 7000
DATASET_DICT[name]['test_size'] = 400       ### default 400, maximum 1400
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 78   ## predict 10 timesteps for testing
DATASET_DICT[name]['t_in'] = 10     ## use 10 as prefix steps, not necessary used
DATASET_DICT[name]['t_total'] = 88
DATASET_DICT[name]['in_size'] = (96, 192)
DATASET_DICT[name]['n_channels'] = 5
DATASET_DICT[name]['downsample'] = (1, 1)



pd.DataFrame(DATASET_DICT).to_csv('dataset_config.csv')