# AdaCVD

ToDo: Introduction.

## Overview

![AdaCVD Project Overview](fig-overview.png)

## 🛠️ Installation

This project uses `micromamba` — a fast, minimal installer for `conda` environments. It's a lightweight alternative to `conda` that supports the same environment and package management commands. If you're already using `conda`, you can adapt the steps accordingly, but we recommend `micromamba` for faster setups. If you don’t have `micromamba` installed, follow the [official installation guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

### 1. Create and Activate the Environment

```bash
micromamba create -n adacvd python=3.10
micromamba activate adacvd
```

### 2. Install PyTorch with CUDA Support

```bash
micromamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
> Adjust `pytorch-cuda=12.1` if your system uses a different CUDA version.

### 3. Install Remaining Dependencies

```bash
micromamba install -f environment.yaml
```

### 4. Verify CUDA Availability

```bash
python -c "import torch; assert torch.cuda.is_available()"
```

### 5. Install the Project in Editable Mode

```bash
pip install -e .
```

## Data

This project uses data from the UK Biobank.For a detailed description of data preprocessing, see [`docuentation/ukb_data_preparation.md`](documentation/ukb_data_preparation.md).

## Model Training & Inference

### Training

To start training the model, use the following command:

```bash
accelerate launch pandora/training/train_model.py --train_dir={base_dir} --device=cuda
```

Replace `{base_dir}` with the path to your training directory containing the configuration file (such as [`config/training/train_settings.yaml`](config/training/train_settings.yaml)). All model checkpoints and predictions will be stored in this file.

> Note: `accelerate launch` enables multi-GPU training when multiple GPUs are available.

### Inference

ToDo.

## Baselines

To train the baseline models, use the following command:

```bash
python exploration/ml_baselines/train_ml_baseline.py --model lgbm --config config/training/train_settings.yaml
```

Adjust the model or config file depending on which model you want to run.

## Evaluation

ToDo.

##

