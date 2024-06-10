## DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training (ICML'2024)

Code for [paper](https://arxiv.org/pdf/2403.03542) DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training (ICML'2024). It  pretrains neural operator transformers (from **7M** to **1B**)  on multiple PDE datasets. Pre-trained weights could be found at https://huggingface.co/hzk17/DPOT.

![fig1](/resources/dpot.jpg)

Our pre-trained DPOT achieves the state-of-the-art performance on multiple PDE datasets and could be used for finetuning on different types of downstream PDE problems.

![fig2](/resources/dpot_result.jpg)



### Usage 

##### Pre-trained models

We have five pre-trained checkpoints of different sizes. Pre-trained weights are at https://huggingface.co/hzk17/DPOT.

| Size   | Attention dim | MLP dim | Layers | Heads | Model size |
| ------ | ------------- | ------- | ------ | ----- | ---------- |
| Tiny   | 512           | 512     | 4      | 4     | 7M         |
| Small  | 1024          | 1024    | 6      | 8     | 30M        |
| Medium | 1024          | 4096    | 12     | 8     | 122M       |
| Large  | 1536          | 6144    | 24     | 16    | 509M       |
| Huge   | 2048          | 8092    | 27     | 8     | 1.03B      |

Here is an example code of loading pre-trained model.
```python
model = DPOTNet(img_size=128, patch_size=8, mixing_type='afno', in_channels=4, in_timesteps=10, out_timesteps=1, out_channels=4, normalize=False, embed_dim=512, modes=32, depth=4, n_blocks=4, mlp_ratio=1, out_layer_dim=32, n_cls=12)
model.load_state_dict(torch.load('model_Ti.pth')['model'])
```



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
python train_temporal.py --model DPOT --train_paths ns2d_fno_1e-5 --test_paths ns2d_fno_1e-5 --gpu 0 
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



### Citation

If you use DPOT in your research, please use the following BibTeX entry.

```
@article{hao2024dpot,
  title={DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training},
  author={Hao, Zhongkai and Su, Chang and Liu, Songming and Berner, Julius and Ying, Chengyang and Su, Hang and Anandkumar, Anima and Song, Jian and Zhu, Jun},
  journal={arXiv preprint arXiv:2403.03542},
  year={2024}
}
```