# Pretraining  Neural Operators on Multiphysics Data

Code for pretraining neural operators on multiple PDE datasets. We will update more details soon.

![fig1](/resources/dpot.jpg)



### Usage 

##### Dataset Protocol

All datasets are stored using hdf5 format, containing  `data`  field. Some datasets are stored with individual hdf5 files, others are stored within a single hdf5 file.

In `data_generation/preprocess.py`,  we have the script for preprocessing the datasets from each source. Download the original file from these sources and preprocess them to `/data` folder.

| Dataset       | Link                                                         |
| ------------- | ------------------------------------------------------------ |
| FNO data      | [Here](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) |
| PDEBench data | [Here](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986) |
| PDEArena data | [Here](https://microsoft.github.io/pdearena/datadownload/)   |
| CFDbench data | [Here](https://cloud.tsinghua.edu.cn/d/435413b55dea434297d1/) |

In `utils/make_master_file.py` , we have all dataset configurations. When new datasets are merged, you should add a configuration dict. It stores all relative paths so that you could run on any places. 

```bash
mkdir data
```

##### Single GPU Pre-training

Now we have a single GPU pretraining code script `train_temporal.py`, you could start it by 

```bash
python train_temporal.py --model FNO --train_paths ns2d_fno_1e-5 --test_paths ns2d_fno_1e-5 --gpu 0 
```

to start a training process.

Or you could start it by writing a configuration file in `configs/ns2d.yaml` and start it by automatically using free GPUs with

```bash
python trainer.py --config_file ns2d.yaml
```

##### Multiple GPU Pre-training

```bash
python parallel_trainer.py --config_file ns2d_parallel.yaml
```

##### Configuration file

Now I use yaml as the configuration file. You could specify parameters for args. If you want to run multiple tasks, you could move parameters into the `tasks` ,

```yaml
model: DPOT
width: 512
tasks:
 lr: [0.001,0.0001]
 batch_size: [256, 32] 
```

This means that you start 2 tasks if you submit this configuration to `trainer.py`. 

##### Requirement

Install the following packages via conda-forge

```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install matplotlib scikit-learn scipy pandas h5py -c conda-forge
conda install timm einops tensorboard -c conda-forge
```

### Code Structure

- `README.md`
- `train_temporal.py`: main code of single GPU pre-training auto-regressive model 
- `trainer.py`: framework of auto scheduling training tasks for parameter tuning
- `utils/`
  - `criterion.py`:  loss functions of relative error
  - `griddataset.py`: dataset of mixture of temporal uniform grid dataset
  - `make_master_file.py`: datasets config file
  - `normalizer`: normalization methods (#TODO: implement instance reversible norm)
  - `optimizer`: Adam/AdamW/Lamb optimizer supporting complex numbers
  - `utilities.py`: other auxiliary functions
- `configs/`: configuration files for pre-training or fine-tuning
- `models/`
  - `dpot.py`:         DPOT model
  - `fno.py`:          FNO with group normalization
  - `mlp.py`
- `data_generation/`:  Some code for preprocessing data (ask hzk if you want to use them)
  - `darcy/`
  - `ns2d/`